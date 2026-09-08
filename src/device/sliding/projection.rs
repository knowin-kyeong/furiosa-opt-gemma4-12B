
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{BothClusters, Cluster, Replicated, Slice};

// Both clusters do real work: the query rows are split across the two clusters and then
// 256 slices per cluster, 8 rows each.
type QueryClusters = m![Qs / 2048];
type QueryRows = m![Qs / 8 % 256];

/// The query weight, dequantized to bf16 in its projection layout. Issued by the caller
/// before anything that depends on x: the scheduler orders DMA by the program order of the
/// consumer, so the lookup pass being first puts the 13.5k-cycle load at the head of the
/// queue, and the pass itself runs while x is normalized, staged and replicated.
pub(crate) type QueryWeight = DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8, H]>;

pub(crate) fn load_query_weight(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QueryWeight {
    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);
    ctx.main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, BothClusters, Replicated, m![H]>,
    weight_f8: &QueryWeight,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    // x is replicated onto every slice of both clusters.
    let x: DmTensorView<'_, bf16, Chip, QueryClusters, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QueryClusters, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 16], m![H % 16]>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    // Gather each cluster's half onto its Slice layout through the switch, then join the two
    // halves through HBM (two descriptors; storing straight from 512 slices of 16 B costs 37k).
    let halves: DmTensor<bf16, Chip, QueryClusters, Slice, m![Qs % 2048]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 8 # 16]>()
        .switch::<Slice, m![Qs / 8 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 8 % 256], m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let mut q_hbm: HbmTensor<bf16, Chip, m![Qs]> = HbmTensor::new();
    halves.view().to_hbm_view(&mut ctx.tdma, q_hbm.view_mut());
    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = q_hbm.to_dm(&mut ctx.tdma);

    unsafe { output.reshape() }
}

// Both clusters do real work on the K/V projections: rows split across the clusters, then
// 256 slices per cluster, 4 rows each.
type KvClusters = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];

/// A K or V weight in its projection layout. (Splitting its lookup out like the query's
/// does not help: the scheduler still issues the K/V loads only around their contractions.)
pub(crate) type KvWeight = DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]>;

pub(crate) fn load_kv_weight(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>) -> KvWeight {
    weight.to_dm(&mut ctx.tdma)
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, KvClusters, KvRows, m![1], m![H]>,
    weight_f8: &KvWeight,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![Ps % 4 # 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let halves: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 4 # 16]>()
        .switch::<Slice, m![Ps / 4 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 4 % 256], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    let mut kv_hbm: HbmTensor<bf16, Chip, m![Ps]> = HbmTensor::new();
    halves.view().to_hbm_view(&mut ctx.tdma, kv_hbm.view_mut());
    kv_hbm.to_dm(&mut ctx.tdma)
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, BothClusters, Replicated, m![H]>,
    k_weight: &KvWeight,
    v_weight: &KvWeight,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, KvClusters, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvClusters, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // eight chunk partials are summed across slices within a cluster.
    let weight_f8: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns, m![H % 60, Qs % 512]> =
        weight.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns, m![Qs % 512]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns, m![1], m![Qs % 512]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 32], m![Qs % 16]>()
        .to_trf();
    // The per-channel scale rides along the contraction epilogue: one value per row, staged
    // as a padded packet per time step in the VRF.
    let weight_scale: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = weight_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows, m![H % 60, 1 # 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![H % 60], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60], m![1 # 8]>()
        .to_vrf();

    let contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 60, Qs / 32 % 16], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60, Qs / 16 % 32], m![Qs % 16]>()
        .contract_outer::<m![H % 60, Qs / 32 % 16], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60]>()
        .contract_lane::<m![H % 60], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60, 1 # 2], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![H % 60], m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 15], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    // Gather the [H] vector from both clusters through HBM (each cluster writes its half,
    // then the Slice layout is loaded back); the channel scale is already applied.
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm.to_dm(&mut ctx.tdma)
}

type TwoClusters = m![H / 1920];
type HiddenRows = m![H / 60 % 32, 1 # 8];
type HiddenRowsByColumns = m![H / 60 % 32, Qs / 512];

fn apply_output_channel_scale(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // The [H] f32 scale (15 KB) does not fit the 8 KB VRF, so scale in two 1920-wide tiles.
    const TILE: usize = 1920;
    const TILES: usize = H::SIZE / TILE;

    let weight_scale: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = weight_scale.to_dm(&mut ctx.tdma);
    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = DmTensor::new();

    for i in 0..TILES {
        let x_tile = x.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(TILE * i);
        let scale_tile = weight_scale.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(TILE * i);

        let scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H = 1920]> = ctx
            .sub
            .begin(scale_tile)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .to_vrf();

        ctx.main
            .begin(x_tile)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H = 1920 / 4], m![H = 1920 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_concat::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_final()
            .cast::<bf16, m![H = 1920 % 8 # 16]>()
            .commit_trim::<m![H = 1920 % 8]>()
            .commit_view(output.view_mut().tile::<m![H], 1920, m![H = 1920 #{!} 3840]>(TILE * i));
    }

    output
}
