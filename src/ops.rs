
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::layout::{self, Cluster, Replicated, Slice};
use crate::device::{full, shared, sliding};
use crate::{Chip, EMBED_SCALE, LOGIT_SOFTCAP};

#[device(chip = 1)]
pub fn embed_token(
    ctx: &mut Context,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = embedding_table.dma_gather_scaled(offset);
    let row: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = row.to_dm(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = ctx
        .main
        .begin(row.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), EMBED_SCALE)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    result.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn sliding_project_qkv(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let q_weight = sliding::projection::load_query_weight(ctx, q_weight);

    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, x);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &x, input_rms_weight);

    // Replicating x to every slice through the switch or a DM-to-DM DMA costs 54-62k cycles;
    // staging the vector in HBM and loading it back replicated runs at HBM DMA speed. x goes as
    // two f8 pieces of x times a power of two (their sum is exact), which the projections
    // contract as f8 x f8 with no lookup pass; the head RMSNorms that follow are
    // scale-invariant, so the factor is never undone.
    // One copy, not sixteen: a copy axis the source lacks does not replicate the store (V50).
    let x2_hbm = shared::mlp::stage_x_hi_lo_qkv_hbm(ctx, &x);
    // Replicating x onto the 512 slices as one HBM load costs 512 DMA descriptors that all read
    // the same 7.7 KB: on hardware that is ~45k cycles, not the 18k of the static model (V154).
    // Instead eight copies per cluster come from HBM (16 descriptors) and a ring-32 switch
    // broadcast fills each copy's 32 slices (V157: qkv 151k -> 108k, 3/3 PASS). The copies are
    // loaded and switched on a real cluster axis: a pass on the dummy axis BothClusters serves
    // one cluster only (V155).
    let x8: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, 1 # 32], m![Dummy2, H]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, Dummy256 / 8], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![Dummy8, Dummy256 / 8], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, layout::BothClusters, Replicated, m![Dummy2, H]> = unsafe { x.reshape() };
    let k_weight = sliding::projection::load_kv_weight(ctx, k_weight);
    let v_weight = sliding::projection::load_kv_weight(ctx, v_weight);

    // q, k and v come back one head per slice; the head-wise RMSNorms and RoPE stay in that
    // layout (no transposes in or broadcasts out) and the outputs are written from it.
    let q = sliding::projection::project_query(ctx, &x, &q_weight);
    let (k, v) = sliding::projection::project_key_value(ctx, &x, &k_weight, &v_weight);

    // The projections' per-channel weight scales are folded into the head RMSNorms (their
    // loads are eight descriptors in the head layout instead of 512 in the projection layout).
    let q = sliding::rmsnorm::normalize_query_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &q,
        q_weight_scale,
        q_rms_weight,
    );
    let k = sliding::rmsnorm::normalize_key_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &k,
        k_weight_scale,
        k_rms_weight,
    );
    let v = sliding::rmsnorm::normalize_value_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &v,
        v_weight_scale,
    );

    let (q, k) = sliding::rope::apply_rope_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &q,
        &k,
        rope_offset,
        cos,
        sin,
    );

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

/// Measurement variant: the qkv tail with the head norms and RoPE sharing one buffer (V239).
/// V240 sweep: the qkv contractions with sequential lanes.
#[device(chip = 1)]
pub fn sliding_project_qkv_seq(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let q_weight = sliding::projection::load_query_weight(ctx, q_weight);

    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, x);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &x, input_rms_weight);

    // Replicating x to every slice through the switch or a DM-to-DM DMA costs 54-62k cycles;
    // staging the vector in HBM and loading it back replicated runs at HBM DMA speed. x goes as
    // two f8 pieces of x times a power of two (their sum is exact), which the projections
    // contract as f8 x f8 with no lookup pass; the head RMSNorms that follow are
    // scale-invariant, so the factor is never undone.
    // One copy, not sixteen: a copy axis the source lacks does not replicate the store (V50).
    let x2_hbm = shared::mlp::stage_x_hi_lo_qkv_hbm(ctx, &x);
    // Replicating x onto the 512 slices as one HBM load costs 512 DMA descriptors that all read
    // the same 7.7 KB: on hardware that is ~45k cycles, not the 18k of the static model (V154).
    // Instead eight copies per cluster come from HBM (16 descriptors) and a ring-32 switch
    // broadcast fills each copy's 32 slices (V157: qkv 151k -> 108k, 3/3 PASS). The copies are
    // loaded and switched on a real cluster axis: a pass on the dummy axis BothClusters serves
    // one cluster only (V155).
    let x8: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, 1 # 32], m![Dummy2, H]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, Dummy256 / 8], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![Dummy8, Dummy256 / 8], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, layout::BothClusters, Replicated, m![Dummy2, H]> = unsafe { x.reshape() };
    let k_weight = sliding::projection::load_kv_weight(ctx, k_weight);
    let v_weight = sliding::projection::load_kv_weight(ctx, v_weight);

    // q, k and v come back one head per slice; the head-wise RMSNorms and RoPE stay in that
    // layout (no transposes in or broadcasts out) and the outputs are written from it.
    let q = sliding::projection::project_query_seq(ctx, &x, &q_weight);
    let (k, v) = sliding::projection::project_key_value_seq(ctx, &x, &k_weight, &v_weight);

    // The projections' per-channel weight scales are folded into the head RMSNorms (their
    // loads are eight descriptors in the head layout instead of 512 in the projection layout).
    let q = sliding::rmsnorm::normalize_query_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &q,
        q_weight_scale,
        q_rms_weight,
    );
    let k = sliding::rmsnorm::normalize_key_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &k,
        k_weight_scale,
        k_rms_weight,
    );
    let v = sliding::rmsnorm::normalize_value_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &v,
        v_weight_scale,
    );

    let (q, k) = sliding::rope::apply_rope_heads::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &q,
        &k,
        rope_offset,
        cos,
        sin,
    );

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

/// Measurement variant: the qkv tail with the head norms and RoPE sharing one buffer (V239).
#[device(chip = 1)]
pub fn sliding_project_qkv_v239(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let q_weight = sliding::projection::load_query_weight(ctx, q_weight);

    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, x);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &x, input_rms_weight);

    // Replicating x to every slice through the switch or a DM-to-DM DMA costs 54-62k cycles;
    // staging the vector in HBM and loading it back replicated runs at HBM DMA speed. x goes as
    // two f8 pieces of x times a power of two (their sum is exact), which the projections
    // contract as f8 x f8 with no lookup pass; the head RMSNorms that follow are
    // scale-invariant, so the factor is never undone.
    // One copy, not sixteen: a copy axis the source lacks does not replicate the store (V50).
    let x2_hbm = shared::mlp::stage_x_hi_lo_qkv_hbm(ctx, &x);
    // Replicating x onto the 512 slices as one HBM load costs 512 DMA descriptors that all read
    // the same 7.7 KB: on hardware that is ~45k cycles, not the 18k of the static model (V154).
    // Instead eight copies per cluster come from HBM (16 descriptors) and a ring-32 switch
    // broadcast fills each copy's 32 slices (V157: qkv 151k -> 108k, 3/3 PASS). The copies are
    // loaded and switched on a real cluster axis: a pass on the dummy axis BothClusters serves
    // one cluster only (V155).
    let x8: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, 1 # 32], m![Dummy2, H]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![Dummy8, Dummy256 / 8], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![Dummy8, Dummy256 / 8], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, layout::BothClusters, Replicated, m![Dummy2, H]> = unsafe { x.reshape() };
    let k_weight = sliding::projection::load_kv_weight(ctx, k_weight);
    let v_weight = sliding::projection::load_kv_weight(ctx, v_weight);

    // q, k and v come back one head per slice; the head-wise RMSNorms and RoPE stay in that
    // layout (no transposes in or broadcasts out) and the outputs are written from it.
    let q = sliding::projection::project_query(ctx, &x, &q_weight);
    let (k, v) = sliding::projection::project_key_value(ctx, &x, &k_weight, &v_weight);

    // V239: the three head RMSNorms and the RoPE share one four-row buffer, so the three sqrt
    // passes collapse into one and the RoPE's rotate-half and sin passes halve. V218 priced a
    // pass at ~600 real cycles; this is five passes fewer.
    let q_scale_vrf = sliding::rmsnorm::load_channel_scale_query::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        q_weight_scale,
    );
    let k_scale_vrf = sliding::rmsnorm::load_channel_scale_row::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        k_weight_scale,
    );
    let v_scale_vrf = sliding::rmsnorm::load_channel_scale_row::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        v_weight_scale,
    );

    let mut mean_square: DmTensor<f32, Chip, layout::HeadClusters, layout::HeadSlicesPerCluster, m![Dummy2, Gs, 1 # 8]> =
        DmTensor::new();
    sliding::rmsnorm::head_mean_square_query(ctx, &q, &q_scale_vrf, &mut mean_square);
    sliding::rmsnorm::head_mean_square_row(ctx, &k, &k_scale_vrf, 0, &mut mean_square);
    sliding::rmsnorm::head_mean_square_row(ctx, &v, &v_scale_vrf, 1, &mut mean_square);
    let rms = sliding::rmsnorm::head_rms_all(ctx, &mean_square);

    let q_rms_vrf = sliding::rmsnorm::head_rms_vrf_query(ctx, &rms);
    let k_rms_vrf = sliding::rmsnorm::head_rms_vrf_row(ctx, &rms, 0);
    let v_rms_vrf = sliding::rmsnorm::head_rms_vrf_row(ctx, &rms, 1);

    let q_weight_vrf =
        sliding::rmsnorm::load_head_norm_weight::<layout::HeadClusters, layout::HeadSlicesPerCluster>(ctx, q_rms_weight);
    let k_weight_vrf =
        sliding::rmsnorm::load_head_norm_weight::<layout::HeadClusters, layout::HeadSlicesPerCluster>(ctx, k_rms_weight);

    let mut qk: DmTensor<bf16, Chip, layout::HeadClusters, layout::HeadSlicesPerCluster, m![Dummy2, Gs, Ds]> =
        DmTensor::new();
    sliding::rmsnorm::head_normalize_query(ctx, &q, &q_scale_vrf, &q_weight_vrf, &q_rms_vrf, &mut qk);
    sliding::rmsnorm::head_normalize_row(ctx, &k, &k_scale_vrf, &k_weight_vrf, &k_rms_vrf, 0, &mut qk);
    let v = sliding::rmsnorm::head_normalize_value_row(ctx, &v, &v_scale_vrf, &v_rms_vrf);

    let (q, k) = sliding::rope::apply_rope_rows::<layout::HeadClusters, layout::HeadSlicesPerCluster>(
        ctx,
        &qk,
        rope_offset,
        cos,
        sin,
    );

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn full_project_qkv(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(ctx, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::projection::project_query(ctx, &x, q_weight, q_weight_scale);
    let k_raw: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> =
        full::projection::project_key(ctx, &x, k_weight, k_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = full::rmsnorm::normalize_query(ctx, &q, q_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_value(ctx, &k_raw);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_key(ctx, &k_raw, k_rms_weight);

    let (q, k) = full::rope::apply_rope(ctx, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn sliding_attention(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    sliding::attention::attend(ctx, q, k, v, mask, out_hbm);
}

#[device(chip = 1)]
pub fn sliding_attention_output(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The attention output already lives in HBM as [Ns, Gs, Ds] = [Qs]; project_output loads
    // each slice's Qs chunk straight from there instead of broadcasting x through the switch.
    let x: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let x_hbm = sliding::projection::project_output(ctx, x, o_weight);
    // Both operands of the post-attention RMSNorm are loaded straight into its reducing layout.
    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, &x_hbm);
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_scaled_reduced::<Cluster>(ctx, &x, o_weight_scale, post_attn_rms_weight, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

/// V240 sweep: the attention-output contraction with sequential lanes.
/// V240 sweep: the O-weight rows in three tiles (44/44/32) instead of two.
#[device(chip = 1)]
pub fn sliding_attention_output_t3(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The attention output already lives in HBM as [Ns, Gs, Ds] = [Qs]; project_output loads
    // each slice's Qs chunk straight from there instead of broadcasting x through the switch.
    let x: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let x_hbm = sliding::projection::project_output_t3(ctx, x, o_weight);
    // Both operands of the post-attention RMSNorm are loaded straight into its reducing layout.
    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, &x_hbm);
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_scaled_reduced::<Cluster>(ctx, &x, o_weight_scale, post_attn_rms_weight, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

/// V240 sweep: the attention-output contraction with sequential lanes.
#[device(chip = 1)]
pub fn sliding_attention_output_seq(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The attention output already lives in HBM as [Ns, Gs, Ds] = [Qs]; project_output loads
    // each slice's Qs chunk straight from there instead of broadcasting x through the switch.
    let x: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let x_hbm = sliding::projection::project_output_seq(ctx, x, o_weight);
    // Both operands of the post-attention RMSNorm are loaded straight into its reducing layout.
    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, &x_hbm);
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_scaled_reduced::<Cluster>(ctx, &x, o_weight_scale, post_attn_rms_weight, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_first_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_first_page(ctx, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_next_page(ctx, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_output(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![Gf]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::attention::divide_by_softmax_sum(ctx, &x, running_sum);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> = layout::broadcast_full_heads(ctx, &x);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        full::projection::project_output(ctx, &x, o_weight, o_weight_scale);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(ctx, &x, post_attn_rms_weight);

    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut ctx.tdma);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(ctx, &x, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn decoder_feedforward(
    ctx: &mut Context,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    // The residual is loaded once, straight into the RMSNorm reducing layout, and serves both
    // the pre-FF normalization and the final residual add.
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &residual, pre_ff_rms_weight);

    // Replicate x to every slice by way of HBM: a DM-to-DM scatter runs at ~70 B/cycle
    // (54k cycles), an HBM-to-DM replicated load at ~3x that. x goes as two f8 pieces (their
    // sum is bf16 x exactly) so the projections can run f8 x f8 contractions on the raw f4 lookup.
    let (x2_hbm, erf_scale, out_scale) =
        shared::mlp::stage_x_hi_lo_hbm_full(ctx, &x, up_global_scale, gate_global_scale);
    // The up/gate stage runs on whole rows (V181): each slice's f4 rows and block scales are one
    // contiguous HBM segment each; a segmented load costs twice per byte on hardware (V174).
    let x = shared::mlp::feedforward_v181(
        ctx,
        &x2_hbm,
        &erf_scale,
        &out_scale,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        down_global_scale,
    );

    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_gate_reduced::<Cluster>(ctx, &x, post_ff_rms_weight, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

/// Measurement variant: decoder_feedforward_v238.
/// Measurement variant: the down projection on four column chunks (V237).
/// V240 sweep: the up/gate contraction with interleaved lanes.
#[device(chip = 1)]
pub fn decoder_feedforward_v240(
    ctx: &mut Context,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    // The residual is loaded once, straight into the RMSNorm reducing layout, and serves both
    // the pre-FF normalization and the final residual add.
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &residual, pre_ff_rms_weight);

    // Replicate x to every slice by way of HBM: a DM-to-DM scatter runs at ~70 B/cycle
    // (54k cycles), an HBM-to-DM replicated load at ~3x that. x goes as two f8 pieces (their
    // sum is bf16 x exactly) so the projections can run f8 x f8 contractions on the raw f4 lookup.
    let (x2_hbm, erf_scale, out_scale) =
        shared::mlp::stage_x_hi_lo_hbm_full(ctx, &x, up_global_scale, gate_global_scale);
    // The up/gate stage runs on whole rows (V181): each slice's f4 rows and block scales are one
    // contiguous HBM segment each; a segmented load costs twice per byte on hardware (V174).
    let x = shared::mlp::feedforward_v240(
        ctx,
        &x2_hbm,
        &erf_scale,
        &out_scale,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        down_global_scale,
    );

    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_gate_reduced::<Cluster>(ctx, &x, post_ff_rms_weight, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

/// Measurement variant: decoder_feedforward_v238.
/// Measurement variant: the down projection on four column chunks (V237).
#[device(chip = 1)]
pub fn decoder_feedforward_v237(
    ctx: &mut Context,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    // The residual is loaded once, straight into the RMSNorm reducing layout, and serves both
    // the pre-FF normalization and the final residual add.
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &residual, pre_ff_rms_weight);

    // Replicate x to every slice by way of HBM: a DM-to-DM scatter runs at ~70 B/cycle
    // (54k cycles), an HBM-to-DM replicated load at ~3x that. x goes as two f8 pieces (their
    // sum is bf16 x exactly) so the projections can run f8 x f8 contractions on the raw f4 lookup.
    let (x2_hbm, erf_scale, out_scale) =
        shared::mlp::stage_x_hi_lo_hbm_full(ctx, &x, up_global_scale, gate_global_scale);
    // The up/gate stage runs on whole rows (V181): each slice's f4 rows and block scales are one
    // contiguous HBM segment each; a segmented load costs twice per byte on hardware (V174).
    let x = shared::mlp::feedforward_v237(
        ctx,
        &x2_hbm,
        &erf_scale,
        &out_scale,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        down_global_scale,
    );

    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_gate_reduced::<Cluster>(ctx, &x, post_ff_rms_weight, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

/// Measurement variant: decoder_feedforward_v238.
#[device(chip = 1)]
pub fn decoder_feedforward_v238(
    ctx: &mut Context,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    // The residual is loaded once, straight into the RMSNorm reducing layout, and serves both
    // the pre-FF normalization and the final residual add.
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    let x = shared::rmsnorm::normalize_reduced_f32::<Cluster>(ctx, &residual, pre_ff_rms_weight);

    // Replicate x to every slice by way of HBM: a DM-to-DM scatter runs at ~70 B/cycle
    // (54k cycles), an HBM-to-DM replicated load at ~3x that. x goes as two f8 pieces (their
    // sum is bf16 x exactly) so the projections can run f8 x f8 contractions on the raw f4 lookup.
    let (x2_hbm, erf_scale, out_scale) =
        shared::mlp::stage_x_hi_lo_hbm_full(ctx, &x, up_global_scale, gate_global_scale);
    // The up/gate stage runs on whole rows (V181): each slice's f4 rows and block scales are one
    // contiguous HBM segment each; a segmented load costs twice per byte on hardware (V174).
    let x = shared::mlp::feedforward_v238(
        ctx,
        &x2_hbm,
        &erf_scale,
        &out_scale,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        down_global_scale,
    );

    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_gate_reduced::<Cluster>(ctx, &x, post_ff_rms_weight, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn final_norm_and_logits(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![W]>,
) {
    let x: DmTensor<bf16, Chip, shared::lm_head::Cluster, Slice, m![H]> = input.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &x, rms_weight);

    let logits = shared::lm_head::logits(ctx, &x, lm_head_weight);

    let scaled: DmTensor<
        f32,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = ctx
        .main
        .begin(logits.view())
        .fetch::<m![W / 16 % 32], m![W % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_div(LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .commit_trim::<m![W % 8]>()
        .commit();

    let capped: DmTensor<
        bf16,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![W / 8 % 64], m![W % 8]>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit();

    capped.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}
