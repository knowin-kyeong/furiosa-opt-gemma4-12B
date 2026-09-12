
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
    // V299: every weight row is its own HBM read (rows interleaved within each head's 64 slices), -8.8% on hardware.
    let q_weight = sliding::projection::load_query_weight_hi(ctx, q_weight);

    // V292: x is staged on both clusters (8 copies x 8 chunks in every 32-slice sub-ring) and replicated on
    // chip by one ring-32 all-gather: no HBM hop, so no ExplicitSync idles the DMA queue (7/8 jobs, -1.3%).
    let x = shared::xsw::load_blocks(ctx, x);
    let x = shared::xsw::normalize_blocks_f32(ctx, &x, input_rms_weight);
    let x2 = shared::xsw::stage_x_hi_lo_blocks(ctx, &x);
    let x: DmTensor<f8e4m3, Chip, layout::BothClusters, Replicated, m![Dummy2, H]> = shared::xsw::replicate_blocks(ctx, &x2);
    let k_weight = sliding::projection::load_kv_weight_hi(ctx, k_weight);
    let v_weight = sliding::projection::load_kv_weight_hi(ctx, v_weight);

    // q, k and v come back one head per slice; the head-wise RMSNorms and RoPE stay in that
    // layout (no transposes in or broadcasts out) and the outputs are written from it.
    let q = sliding::projection::project_query_hi(ctx, &x, &q_weight);
    let (k, v) = sliding::projection::project_key_value_hi(ctx, &x, &k_weight, &v_weight);

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
    let x = shared::rmsnorm::load_reducing_aligned::<Cluster>(ctx, &x_hbm);
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
    // V293: the pre-FF norm, the hi/lo split and the geglu scalars run on both clusters (8 copies x 8 chunks in
    // every 32-slice sub-ring) and x is replicated on chip: no x2 HBM hop, so no ExplicitSync idles the DMA queue
    // (13/16 Arena jobs, -1.8k). The post-FF tail keeps cluster 0's block 0, which is exactly ReducingSlices.
    let residual_b = shared::xsw::load_blocks(ctx, residual_hbm);
    let x = shared::xsw::normalize_blocks_f32_fused(ctx, &residual_b, pre_ff_rms_weight);

    // Replicate x to every slice by way of HBM: a DM-to-DM scatter runs at ~70 B/cycle
    // (54k cycles), an HBM-to-DM replicated load at ~3x that. x goes as two f8 pieces (their
    // sum is bf16 x exactly) so the projections can run f8 x f8 contractions on the raw f4 lookup.
    let (x2, erf_b, out_b) = shared::xsw::stage_x_hi_lo_full_blocks(ctx, &x, up_global_scale, gate_global_scale);
    let x_rep = shared::xsw::replicate_blocks(ctx, &x2);
    let erf_all = shared::xsw::broadcast_scalar_blocks(ctx, erf_b);
    // V306: out_scale stays on cluster 0 block 0 (= ReducingSlices) and is applied in the tail multiply.
    let out_tail: DmTensor<f32, Chip, Cluster, shared::rmsnorm::ReducingSlices, m![1 # 8]> = unsafe { out_b.reshape() };
    // The up/gate stage runs on whole rows (V181): each slice's f4 rows and block scales are one
    // contiguous HBM segment each; a segmented load costs twice per byte on hardware (V174).
    let x = shared::mlp::feedforward_vtp(
        ctx,
        x_rep,
        erf_all,
        out_tail,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        down_global_scale,
    );

    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual: DmTensor<bf16, Chip, Cluster, shared::rmsnorm::ReducingSlices, m![H % 480]> = unsafe { residual_b.reshape() };
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

/// V323 probe: production layout, 1920 / 1920 rows, 120-row groups.
#[device(chip = 1)]
pub fn probe_attn_load_sym(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![H % 120, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![1], m![H % 120, Qs % 256 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 32, m![H % 120, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 120], m![Qs % 256 = 32]>()
        .collect::<m![H % 120], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V323 probe: 2048 / 1792 rows (53.3% on cluster 0), 128-row groups.
#[device(chip = 1)]
pub fn probe_attn_load_a53(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H # 4096 / 2048], m![H # 4096 % 2048 / 128, Qs / 256], m![H # 4096 % 128, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H # 4096 / 2048], m![H # 4096 % 2048 / 128, Qs / 256], m![1], m![H # 4096 % 128, Qs % 256 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 32, m![H # 4096 % 128, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H # 4096 % 128], m![Qs % 256 = 32]>()
        .collect::<m![H # 4096 % 128], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V323 probe: 2560 / 1280 rows (66.7% on cluster 0), 160-row groups.
#[device(chip = 1)]
pub fn probe_attn_load_a67(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H # 5120 / 2560], m![H # 5120 % 2560 / 160, Qs / 256], m![H # 5120 % 160, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H # 5120 / 2560], m![H # 5120 % 2560 / 160, Qs / 256], m![1], m![H # 5120 % 160, Qs % 256 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 32, m![H # 5120 % 160, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H # 5120 % 160], m![Qs % 256 = 32]>()
        .collect::<m![H # 5120 % 160], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V335 probe (control): clusters split by column halves (Qs / 2048), 32 row groups x 8 column chunks per cluster.
/// Same per-slice runs as `stk`, but each cluster still reads both HBM stacks (alternating 256-column chunks).
#[device(chip = 1)]
pub fn probe_attn_load_ctl(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![Qs / 2048], m![H / 120, Qs / 256 % 8], m![H % 120, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![Qs / 2048], m![H / 120, Qs / 256 % 8], m![1], m![H % 120, Qs % 256 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 32, m![H % 120, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 120], m![Qs % 256 = 32]>()
        .collect::<m![H % 120], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V335 probe (stack split): clusters split by 256-column chunk parity (Qs / 256 % 2). Chunk c of row r starts at
/// r * 4096 + c * 256, so its HBM stack bit (address bit 8) is c & 1 on a 512-aligned base: each cluster reads one stack.
#[device(chip = 1)]
pub fn probe_attn_load_stk(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![Qs / 256 % 2], m![H / 120, Qs / 512], m![H % 120, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![Qs / 256 % 2], m![H / 120, Qs / 512], m![1], m![H % 120, Qs % 256 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 32, m![H % 120, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 120], m![Qs % 256 = 32]>()
        .collect::<m![H % 120], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V339 probe (reference): production tile structure -- rows 96 + 24 per 120-row group, both tiles on both clusters.
#[device(chip = 1)]
pub fn probe_attn_load_pt2(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let t0: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![H % 120 = 96, Qs % 256]> = o_weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let t1: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![H % 120 = 24, Qs % 256]> = o_weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);
    let _k0: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![1], m![H % 120 = 96, Qs % 256 = 32]> = ctx
        .sub
        .begin(t0.view().tile::<m![Qs % 256], 32, m![H % 120 = 96, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 120 = 96], m![Qs % 256 = 32]>()
        .collect::<m![H % 120 = 96], m![Qs % 256 = 32]>()
        .to_trf();
    let _k1: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 120 % 16, Qs / 256], m![1], m![H % 120 = 24, Qs % 256 = 32]> = ctx
        .sub
        .begin(t1.view().tile::<m![Qs % 256], 32, m![H % 120 = 24, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 120 = 24], m![Qs % 256 = 32]>()
        .collect::<m![H % 120 = 24], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V339 probe (uneven tiles, cluster 0 60%): tile0 = rows 0..3072 split evenly (96-row groups on both clusters), tile1 = rows
/// 3072..3840 on cluster 0 only (48-row groups). Cluster 1's slices carry 96 rows, cluster 0's 144 (V321: the cluster-1 lag
/// goes away only when its per-slice bytes shrink).
#[device(chip = 1)]
pub fn probe_attn_load_u60(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let t0: DmTensor<f8e4m3, Chip, m![H = 3072 / 1536], m![H = 3072 % 1536 / 96, Qs / 256], m![H % 96, Qs % 256]> = o_weight
        .view()
        .tile::<m![H], 3072, m![H = 3072 # 3840, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let t1: DmTensor<f8e4m3, Chip, m![1 # 2], m![H = 768 / 48, Qs / 256], m![H % 48, Qs % 256]> = o_weight
        .view()
        .tile::<m![H], 768, m![H = 768 # 3840, Qs]>(3072)
        .to_dm(&mut ctx.tdma);
    let _k0: TrfTensor<f8e4m3, Chip, m![H = 3072 / 1536], m![H = 3072 % 1536 / 96, Qs / 256], m![1], m![H % 96, Qs % 256 = 32]> = ctx
        .sub
        .begin(t0.view().tile::<m![Qs % 256], 32, m![H % 96, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 96], m![Qs % 256 = 32]>()
        .collect::<m![H % 96], m![Qs % 256 = 32]>()
        .to_trf();
    let _k1: TrfTensor<f8e4m3, Chip, m![1 # 2], m![H = 768 / 48, Qs / 256], m![1], m![H % 48, Qs % 256 = 32]> = ctx
        .sub
        .begin(t1.view().tile::<m![Qs % 256], 32, m![H % 48, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 48], m![Qs % 256 = 32]>()
        .collect::<m![H % 48], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V339 probe (uneven tiles, cluster 0 53.3%): tile0 = rows 0..3584 split evenly (112-row groups), tile1 = rows 3584..3840 on
/// cluster 0 only (16-row groups): cluster 0's slices carry 128 rows (exactly one 32 KB DM page), cluster 1's 112.
#[device(chip = 1)]
pub fn probe_attn_load_u53(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let t0: DmTensor<f8e4m3, Chip, m![H = 3584 / 1792], m![H = 3584 % 1792 / 112, Qs / 256], m![H = 3584 % 112, Qs % 256]> = o_weight
        .view()
        .tile::<m![H], 3584, m![H = 3584 # 3840, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let t1: DmTensor<f8e4m3, Chip, m![1 # 2], m![H = 256 / 16, Qs / 256], m![H % 16, Qs % 256]> = o_weight
        .view()
        .tile::<m![H], 256, m![H = 256 # 3840, Qs]>(3584)
        .to_dm(&mut ctx.tdma);
    let _k0: TrfTensor<f8e4m3, Chip, m![H = 3584 / 1792], m![H = 3584 % 1792 / 112, Qs / 256], m![1], m![H = 3584 % 112, Qs % 256 = 32]> = ctx
        .sub
        .begin(t0.view().tile::<m![Qs % 256], 32, m![H = 3584 % 112, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H = 3584 % 112], m![Qs % 256 = 32]>()
        .collect::<m![H = 3584 % 112], m![Qs % 256 = 32]>()
        .to_trf();
    let _k1: TrfTensor<f8e4m3, Chip, m![1 # 2], m![H = 256 / 16, Qs / 256], m![1], m![H % 16, Qs % 256 = 32]> = ctx
        .sub
        .begin(t1.view().tile::<m![Qs % 256], 32, m![H % 16, Qs % 256 = 32 # 256]>(0))
        .fetch::<m![H % 16], m![Qs % 256 = 32]>()
        .collect::<m![H % 16], m![Qs % 256 = 32]>()
        .to_trf();
}

/// V341 probe: 512-column chunks (each read = 2 aligned granules, both HBM stacks), 32 row groups x 8 chunks, 60 rows per slice.
#[device(chip = 1)]
pub fn probe_attn_load_c512(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 60 % 32, Qs / 512], m![H % 60, Qs % 512]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 60 % 32, Qs / 512], m![1], m![H % 60, Qs % 512 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 512], 32, m![H % 60, Qs % 512 = 32 # 512]>(0))
        .fetch::<m![H % 60], m![Qs % 512 = 32]>()
        .collect::<m![H % 60], m![Qs % 512 = 32]>()
        .to_trf();
}

/// V341 probe: 1024-column chunks (each read = 4 granules), 64 row groups x 4 chunks, 30 rows per slice.
#[device(chip = 1)]
pub fn probe_attn_load_c1024(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 30 % 64, Qs / 1024], m![H % 30, Qs % 1024]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 30 % 64, Qs / 1024], m![1], m![H % 30, Qs % 1024 = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 1024], 32, m![H % 30, Qs % 1024 = 32 # 1024]>(0))
        .fetch::<m![H % 30], m![Qs % 1024 = 32]>()
        .collect::<m![H % 30], m![Qs % 1024 = 32]>()
        .to_trf();
}

/// V341 probe: whole-row reads interleaved within each cluster (V299/V321 form: slice = row % 256, element = row / 256), one read =
/// one 4,096 B row; `H # 4096 / 2048` gives cluster 0 eight rows per slice and cluster 1 seven (per-slice asymmetry, V321 qa9-like).
#[device(chip = 1)]
pub fn probe_attn_load_r53i(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H # 4096 / 2048], m![H # 4096 % 2048 % 256], m![H # 4096 % 2048 / 256, Qs]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H # 4096 / 2048], m![H # 4096 % 2048 % 256], m![1], m![H # 4096 % 2048 / 256, Qs = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs], 32, m![H # 4096 % 2048 / 256, Qs = 32 # 4096]>(0))
        .fetch::<m![H # 4096 % 2048 / 256], m![Qs = 32]>()
        .collect::<m![H # 4096 % 2048 / 256], m![Qs = 32]>()
        .to_trf();
}

/// V341 probe: whole-row reads interleaved within each cluster on 128 live slices (slice = row % 128, element = row / 128 = 15 rows),
/// symmetric 1920 / 1920 -- the V286 direction (fewer live slices) combined with one-row reads.
#[device(chip = 1)]
pub fn probe_attn_load_r15i(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H % 128, 1 # 2], m![H % 1920 / 128, Qs]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H % 128, 1 # 2], m![1], m![H % 1920 / 128, Qs = 32]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs], 32, m![H % 1920 / 128, Qs = 32 # 4096]>(0))
        .fetch::<m![H % 1920 / 128], m![Qs = 32]>()
        .collect::<m![H % 1920 / 128], m![Qs = 32]>()
        .to_trf();
}

// V344 probes: the production 256 B column chunks on fewer live slices per cluster (V286: halving qkv's live slices
// made its row loads ~20% faster). Every keep-alive stages 3,840 elements per slice, as `probe_attn_load_sym` does,
// so the sub-context staging costs the same in every arm.

/// V344 probe: 128 live slices per cluster, 240 rows each, padding innermost (even slices live).
#[device(chip = 1)]
pub fn probe_attn_load_h2i(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 240 % 8, Qs / 256, 1 # 2], m![H % 240, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 240 % 8, Qs / 256, 1 # 2], m![1], m![H % 240, Qs % 256 = 16]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 16, m![H % 240, Qs % 256 = 16 # 256]>(0))
        .fetch::<m![H % 240], m![Qs % 256 = 16]>()
        .collect::<m![H % 240], m![Qs % 256 = 16]>()
        .to_trf();
}

/// V344 probe: 128 live slices per cluster, 240 rows each, padding outermost (slices 0..128 live).
#[device(chip = 1)]
pub fn probe_attn_load_h2o(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![1 # 2, H / 240 % 8, Qs / 256], m![H % 240, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![1 # 2, H / 240 % 8, Qs / 256], m![1], m![H % 240, Qs % 256 = 16]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 16, m![H % 240, Qs % 256 = 16 # 256]>(0))
        .fetch::<m![H % 240], m![Qs % 256 = 16]>()
        .collect::<m![H % 240], m![Qs % 256 = 16]>()
        .to_trf();
}

/// V344 probe: 128 live slices per cluster, 240 rows each, padding in the middle (blocks of 16 live slices alternate).
#[device(chip = 1)]
pub fn probe_attn_load_h2c(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 240 % 8, 1 # 2, Qs / 256], m![H % 240, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 240 % 8, 1 # 2, Qs / 256], m![1], m![H % 240, Qs % 256 = 16]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 16, m![H % 240, Qs % 256 = 16 # 256]>(0))
        .fetch::<m![H % 240], m![Qs % 256 = 16]>()
        .collect::<m![H % 240], m![Qs % 256 = 16]>()
        .to_trf();
}

/// V344 probe: 64 live slices per cluster, 480 rows each, padding innermost.
#[device(chip = 1)]
pub fn probe_attn_load_h4i(ctx: &mut Context, o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) {
    let w: DmTensor<f8e4m3, Chip, m![H / 1920], m![H / 480 % 4, Qs / 256, 1 # 4], m![H % 480, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, m![H / 1920], m![H / 480 % 4, Qs / 256, 1 # 4], m![1], m![H % 480, Qs % 256 = 8]> = ctx
        .sub
        .begin(w.view().tile::<m![Qs % 256], 8, m![H % 480, Qs % 256 = 8 # 256]>(0))
        .fetch::<m![H % 480], m![Qs % 256 = 8]>()
        .collect::<m![H % 480], m![Qs % 256 = 8]>()
        .to_trf();
}
