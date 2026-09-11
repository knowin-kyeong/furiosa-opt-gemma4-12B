
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};
use crate::hi_lo_trunc_fns;
use crate::device::layout::{BothClusters, Cluster, HeadClusters, HeadSlicesPerCluster, Replicated, Slice};

// Both clusters do real work: the query rows are split across the two clusters and then
// 256 slices per cluster, 8 rows each.
type QueryClusters = m![Qs / 2048];
type QueryRows = m![Qs / 8 % 256];

/// The query weight in its projection layout, f8 as stored: it is contracted as f8 x f8 against
/// the two f8 pieces of x (no lookup pass).
pub(crate) type QueryWeight = DmTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![Qs % 8, H]>;

pub(crate) fn load_query_weight(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QueryWeight {
    weight.to_dm(&mut ctx.tdma)
}

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
    weight_f8: &QueryWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> {
    // x (two f8 pieces whose sum is bf16 x times a power of two) is replicated onto every
    // slice of both clusters. Each weight packet is streamed twice (the Dummy2 time axis) so
    // the Time Reducer adds the dot products with the two pieces.
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Qs % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 64, Dummy2], m![H % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    // The per-channel weight scale is applied by the query RMSNorm that follows (loaded there
    // in the head layout with eight descriptors instead of 512 here).
    // Each cluster holds four heads spread over 64 slices x 8 rows each; a ring-64 gather
    // puts every head on one slice, the layout the query RMSNorm and RoPE work in, without
    // leaving the cluster (the two clusters then post-process their four heads in parallel).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

// Both clusters do real work on the K/V projections: rows split across the clusters, then
// 256 slices per cluster, 4 rows each.
type KvClusters = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];

/// A K or V weight in its projection layout, f8 as stored.
pub(crate) type KvWeight = DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]>;

pub(crate) fn load_kv_weight(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>) -> KvWeight {
    weight.to_dm(&mut ctx.tdma)
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![Dummy2, H]>,
    weight_f8: &KvWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]> {
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Ps % 4, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    // The per-channel weight scale is applied by the head RMSNorm that follows.
    // Ring-64 gather to one head per slice within the cluster (see project_query).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
    k_weight: &KvWeight,
    v_weight: &KvWeight,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
) {
    let x: DmTensorView<'_, f8e4m3, Chip, KvClusters, KvRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let k = project_one_kv_matrix(ctx, &x_trf, k_weight);
    let v = project_one_kv_matrix(ctx, &x_trf, v_weight);

    (k, v)
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // x as two f8 pieces of x * s (see shared/f8split.rs), made once on eight slices and staged
    // through HBM so that each slice loads its column chunk of both pieces with one descriptor;
    // the tiles then contract f8 x f8 with no lookup pass, and the post-attention RMSNorm
    // absorbs s.
    let xs: DmTensor<bf16, Chip, Cluster, XSlices256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let (x_hi, x_lo) = hi_lo_x256(ctx, &xs, 16f32);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Qs / 256, Dummy2, Qs % 256]> = HbmTensor::new();
    x_hi.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    x_lo.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(1));

    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// V198: attention output on the run length V197 measured as fastest. Qs is split across 16
/// chunks of 256 columns instead of 8 of 512, so every weight run is exactly one 256-byte
/// granule and each slice needs half as much of x. Rows per slice double to 120 to keep all
/// 256 slices live; the reduce still runs over the innermost slice axis (V186's rule).
type HiddenRows256 = m![H / 120 % 16, 1 # 16];
type HiddenRowsByColumns256 = m![H / 120 % 16, Qs / 256];
type XSlices256 = m![1 # 16, Qs / 256];
hi_lo_trunc_fns!(hi_lo_x256, Cluster, XSlices256, Qs, 256, 16, 32, 64);
hi_lo_trunc_fns!(hi_lo_x_direct, TwoClusters, HiddenRowsByColumns256, Qs, 256, 16, 32, 64);

type TwoClusters = m![H / 1920];
type HiddenRows = m![H / 60 % 32, 1 # 8];
type HiddenRowsByColumns = m![H / 60 % 32, Qs / 512];
/// The attention output on eight slices, 512 elements each, for the f8 split.
type XSlices = m![1 # 32, Qs / 512];
hi_lo_trunc_fns!(hi_lo_x, Cluster, XSlices, Qs, 512, 32, 64, 128);

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

// V240 sweep variants: lane mode and tile count.
pub(crate) fn project_query_seq(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
    weight_f8: &QueryWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> {
    // x (two f8 pieces whose sum is bf16 x times a power of two) is replicated onto every
    // slice of both clusters. Each weight packet is streamed twice (the Dummy2 time axis) so
    // the Time Reducer adds the dot products with the two pieces.
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Qs % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 64, Dummy2], m![H % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Sequential)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    // The per-channel weight scale is applied by the query RMSNorm that follows (loaded there
    // in the head layout with eight descriptors instead of 512 here).
    // Each cluster holds four heads spread over 64 slices x 8 rows each; a ring-64 gather
    // puts every head on one slice, the layout the query RMSNorm and RoPE work in, without
    // leaving the cluster (the two clusters then post-process their four heads in parallel).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
fn project_one_kv_matrix_seq(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![Dummy2, H]>,
    weight_f8: &KvWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]> {
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Ps % 4, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Sequential)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    // The per-channel weight scale is applied by the head RMSNorm that follows.
    // Ring-64 gather to one head per slice within the cluster (see project_query).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}
pub(crate) fn project_key_value_seq(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
    k_weight: &KvWeight,
    v_weight: &KvWeight,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
) {
    let x: DmTensorView<'_, f8e4m3, Chip, KvClusters, KvRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let k = project_one_kv_matrix_seq(ctx, &x_trf, k_weight);
    let v = project_one_kv_matrix_seq(ctx, &x_trf, v_weight);

    (k, v)
}
pub(crate) fn project_output_seq(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // x as two f8 pieces of x * s (see shared/f8split.rs), made once on eight slices and staged
    // through HBM so that each slice loads its column chunk of both pieces with one descriptor;
    // the tiles then contract f8 x f8 with no lookup pass, and the post-attention RMSNorm
    // absorbs s.
    let xs: DmTensor<bf16, Chip, Cluster, XSlices256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let (x_hi, x_lo) = hi_lo_x256(ctx, &xs, 16f32);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Qs / 256, Dummy2, Qs % 256]> = HbmTensor::new();
    x_hi.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    x_lo.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(1));

    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}
pub(crate) fn project_output_t3(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 44, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 44, m![H / 120, H % 120 = 44 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 44, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 44, m![H / 120, H % 120 = 44 # 120, Qs]>(44)
        .to_dm(&mut ctx.tdma);
    let tile2: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // x as two f8 pieces of x * s (see shared/f8split.rs), made once on eight slices and staged
    // through HBM so that each slice loads its column chunk of both pieces with one descriptor;
    // the tiles then contract f8 x f8 with no lookup pass, and the post-attention RMSNorm
    // absorbs s.
    let xs: DmTensor<bf16, Chip, Cluster, XSlices256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let (x_hi, x_lo) = hi_lo_x256(ctx, &xs, 16f32);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Qs / 256, Dummy2, Qs % 256]> = HbmTensor::new();
    x_hi.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    x_lo.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(1));

    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 44, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 44, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 44, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 44]>()
        .contract_lane::<m![H % 120 = 44], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 44]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 44 / 4], m![H % 120 = 44 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 44 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 44, m![H % 120 = 44 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 44, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 44, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 44, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 44]>()
        .contract_lane::<m![H % 120 = 44], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 44]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 44 / 4], m![H % 120 = 44 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 44 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 44, m![H % 120 = 44 #{!} 120]>(44));

    ctx.main
        .begin(tile2.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// V244 gating probe: can a vector chain follow a `switch`, and does its VRF operand use the
/// post-switch slice mapping? If yes, the per-channel weight scale can be folded into the
/// ring-64 gather, the head norms stop needing a scale VRF, and V239's merged tail becomes
/// reachable (RULES 10.0k (5)).
pub(crate) fn project_query_gather_scaled(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
    weight_f8: &QueryWeight,
    channel_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> {
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Qs % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 64, Dummy2], m![H % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    // The scale in the head layout, i.e. the mapping the switch writes into.
    let channel_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { channel_scale.view().reshape() };
    let scale_dm: DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> =
        channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .switch::<HeadSlicesPerCluster, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// V248: `project_output` with the contraction's fetch packet at 32 elements instead of 64, i.e.
/// the same reduction split differently between Packet and Time. The book's cost model says this
/// is a wash -- fetch costs `Time x (Packet / read_size)` and main's read_size is at most 32 B, so
/// 64 f8 is one packet of two reads and 32 f8 is two packets of one -- and V234 showed 128 is
/// rejected outright. This measures the model rather than trusting it.
pub(crate) fn project_output_p32(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // x as two f8 pieces of x * s (see shared/f8split.rs), made once on eight slices and staged
    // through HBM so that each slice loads its column chunk of both pieces with one descriptor;
    // the tiles then contract f8 x f8 with no lookup pass, and the post-attention RMSNorm
    // absorbs s.
    let xs: DmTensor<bf16, Chip, Cluster, XSlices256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let (x_hi, x_lo) = hi_lo_x256(ctx, &xs, 16f32);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Qs / 256, Dummy2, Qs % 256]> = HbmTensor::new();
    x_hi.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    x_lo.view()
        .to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut().tile::<m![Dummy2], 1, m![Qs / 256, Dummy2 = 1 #{!} 2, Qs % 256]>(1));

    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 32 % 8, Dummy2], m![Qs % 32]>()
        .collect::<m![H % 120 = 88, Qs / 32 % 8, Dummy2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 32 % 8, Dummy2], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 32 % 8, Dummy2], m![Qs % 32]>()
        .collect::<m![H % 120 = 32, Qs / 32 % 8, Dummy2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 32 % 8, Dummy2], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// The two f8 pieces of `x * s` committed into one `[Dummy2, Qs % 256]` buffer, in the layout the
/// O-weight contraction already reads. Committing into tiles rather than returning two tensors is
/// what lets the TRF stage them as one operand with no copy pass (`hi_lo_trunc_fns` returns two).
fn hi_lo_pair_direct(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]>,
    s: f32,
) -> DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> {
    let (x_hi, x_lo) = hi_lo_x_direct(ctx, x, s);
    let mut pair: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Dummy2, Qs % 256]> =
        DmTensor::new();
    ctx.main
        .begin(x_hi.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .commit_trim::<m![Qs % 32]>()
        .commit_view(pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    ctx.main
        .begin(x_lo.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .commit_trim::<m![Qs % 32]>()
        .commit_view(pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(1));
    pair
}

pub(crate) fn project_output_direct(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x = hi_lo_pair_direct(ctx, &xs, 16f32);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

pub(crate) fn project_output_split_store(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x = hi_lo_pair_direct(ctx, &xs, 16f32);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    // V253: store each tile's rows as its contraction finishes instead of the whole [H] at the
    // end. The epilogue's critical path is store -> 1,600 cycles of HBM read-after-write -> load,
    // and only the *last* store sits on it; splitting 120 rows into 88 + 32 shortens that last
    // store from 2,040 to ~544 while the 88-row store overlaps tile1's weight load.
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction
        .view()
        .tile::<m![H % 120], 60, m![H % 120 = 60 # 120]>(0)
        .to_hbm_view(
            &mut ctx.tdma,
            gathered_hbm.view_mut().tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 #{!} 120]>(0),
        );
    contraction
        .view()
        .tile::<m![H % 120], 60, m![H % 120 = 60 # 120]>(60)
        .to_hbm_view(
            &mut ctx.tdma,
            gathered_hbm.view_mut().tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 #{!} 120]>(60),
        );
    gathered_hbm
}

pub(crate) fn project_output_gathered(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x = hi_lo_pair_direct(ctx, &xs, 16f32);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    // V254: four row groups gathered onto one slice before the store. The contraction leaves
    // sixteen live slices per cluster holding 120 values each, so the store is 64 descriptors of
    // 120 B -- 2,040 cycles at util 0.003 for 7,680 B, the worst item left in the kernel. Ringing
    // four of them together (the live slices sit sixteen apart, so the ring is 64, not the 256 the
    // old 60-row layout needed) leaves 8 descriptors of 960 B. The ring delivers packet-major, so
    // a repack pass puts H back in order before the store, as `gather_pack_full` does.
    let ringed: DmTensor<bf16, Chip, TwoClusters, m![H / 480 % 4, 1 # 64], m![H / 8 % 15, H / 120 % 4, H % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![H / 8 % 15], m![H % 8 # 16]>()
        .switch::<m![H / 480 % 4, 1 # 64], m![H / 8 % 15, H / 120 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![H / 8 % 15, H / 120 % 4], m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let packed: DmTensor<bf16, Chip, TwoClusters, m![H / 480 % 4, 1 # 64], m![H % 480]> = ctx
        .main
        .begin(ringed.view())
        .fetch::<m![H / 120 % 4, H / 8 % 15], m![H % 8 # 16]>()
        .collect::<m![H / 8 % 60], m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    packed.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}


const H_F32_OUT: f32 = H::SIZE as f32;
pub(crate) fn project_output_norm(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x = hi_lo_pair_direct(ctx, &xs, 16f32);
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Dummy2, Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Dummy2, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // V255 (the V45 slot): the post-attention RMSNorm runs where the contraction leaves its
    // result, so the [H] vector never round-trips through HBM. What crosses the clusters is one
    // f32 scalar instead of 7,680 bytes: store 2,040 + reload 546 become a 4-byte store and a
    // 4-byte load, and the read-after-write latency between them is covered by the three operand
    // loads that have to happen anyway.
    //
    // The reduce order is what blocked this twice before. After the chunk reduce the live slices
    // are the row groups -- the *outer* slice axis -- and the VRU only reduces the innermost one
    // (V186). Ringing the sixteen per-slice partials onto one slice first turns the cross-slice
    // sum into an intra-slice one, and it moves 16 x 32 B, not the whole vector.
    let scale_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> =
        channel_scale.to_dm(&mut ctx.tdma);
    let gamma_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> =
        rms_weight.to_dm(&mut ctx.tdma);
    let resid_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> =
        residual_hbm.to_dm(&mut ctx.tdma);

    let scale_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows256, m![H % 120]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    // Per-slice sum of squares of x * scale.
    let ms: DmTensor<f32, Chip, TwoClusters, HiddenRows256, m![1 # 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // The sixteen row-group partials onto one slice, then summed there.
    let ringed: DmTensor<f32, Chip, TwoClusters, m![1 # 256], m![H / 120 % 16, 1 # 8]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![1 # 256], m![H / 120 % 16]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 })
        .collect::<m![H / 120 % 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let half: DmTensor<f32, Chip, TwoClusters, m![1 # 256], m![1 # 8]> = ctx
        .main
        .begin(ringed.view())
        .fetch::<m![H / 120 % 16], m![1 # 8]>()
        .collect::<m![H / 120 % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // One f32 across the clusters, instead of the whole [H].
    let mut halves_hbm: HbmTensor<f32, Chip, m![H / 1920, 1 # 8]> = HbmTensor::new();
    half.view().to_hbm_view(&mut ctx.tdma, halves_hbm.view_mut());
    let halves: DmTensor<f32, Chip, TwoClusters, HiddenRows256, m![H / 1920, 1 # 8]> =
        halves_hbm.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, TwoClusters, HiddenRows256, m![1 # 8]> = ctx
        .main
        .begin(halves.view())
        .fetch::<m![H / 1920], m![1 # 8]>()
        .collect::<m![H / 1920], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32_OUT)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, TwoClusters, HiddenRows256, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let gamma_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows256, m![H % 120]> = ctx
        .sub
        .begin(gamma_dm.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let resid_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows256, m![H % 120]> = ctx
        .sub
        .begin(resid_dm.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows256, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gamma_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &resid_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}


// ---------------------------------------------------------------------------------------------
// V256: the K/V projections as a chunked reduction, to price the structure before qkv's whole
// projection moves to it.
//
// Today every slice holds whole rows and therefore needs *all* of H, which is why x has to be
// replicated onto 512 slices -- `ops.rs:78`, a ring-32 broadcast costing 7,943 static Main, the
// largest single Main item in qkv. Giving each slice a *chunk* of H instead lets it load its own
// piece of x straight from the staging scratch, which is what made V251 worth 7.9% on attn_out.
//
// This variant converts K and V only. q keeps the broadcast, so the broadcast saving is *not* in
// the measurement -- what is measured is the cost of the structure itself: the weight run drops
// from 15,360 B to 1,920 B, a `vector_inter_slice_reduce` appears, and the head gather changes
// from ring-64-stride-1 to ring-32-stride-2. V215 and V208 disagree on the sign of the run-length
// term for qkv, so this is the cheapest honest way to settle it: if K/V come out neutral or
// better, the full conversion (which does delete the broadcast) is clearly worth building.
//
// A 1,920 B run is the alignment-clean choice: aligned it touches 8 granules (1,920 / 256 = 7.5
// -> 8) and starting at byte 128 it ends at 2,048 exactly, so it touches 8 either way.
type KvRowsChunked = m![Ps / 8 % 128, H / 1920 % 2];
type KvRowsReduced = m![Ps / 8 % 128, 1 # 2];

pub(crate) type KvWeightChunked = DmTensor<f8e4m3, Chip, KvClusters, KvRowsChunked, m![Ps % 8, H % 1920]>;

pub(crate) fn load_kv_weight_chunked(
    ctx: &mut Context,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> KvWeightChunked {
    weight.to_dm(&mut ctx.tdma)
}

fn project_one_kv_chunked(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRowsChunked, m![1], m![Dummy2, H % 1920]>,
    weight_f8: &KvWeightChunked,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]> {
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRowsReduced, m![Ps % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 8, H / 64 % 30, Dummy2], m![H % 64]>()
        .collect::<m![Ps % 8, H / 64 % 30, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Ps % 8, H / 64 % 30, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<KvRowsReduced, m![Ps % 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    // Eight rows per slice over 128 live slices two apart, so a head is 32 of them.
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Ds / 8, 1 # 2], m![Ds % 8]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn project_key_value_chunked(
    ctx: &mut Context,
    x2_hbm: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    k_weight: &KvWeightChunked,
    v_weight: &KvWeightChunked,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
) {
    // Each slice loads only its own 1,920-column chunk of both f8 pieces: two runs of 1,920 B,
    // no broadcast.
    let x: DmTensor<f8e4m3, Chip, KvClusters, KvRowsChunked, m![Dummy2, H % 1920]> =
        x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, KvClusters, KvRowsChunked, m![1], m![Dummy2, H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32 % 60], m![H % 32]>()
        .collect::<m![Dummy2, H / 32 % 60], m![H % 32]>()
        .to_trf();

    let k = project_one_kv_chunked(ctx, &x_trf, k_weight);
    let v = project_one_kv_chunked(ctx, &x_trf, v_weight);

    (k, v)
}

pub(crate) fn project_output_one_piece(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}


/// V259: attn_out tile split re-swept now that V257 removed the Dummy2 replay -- the
/// contraction is half the Main work it was when V203 chose 88/32, so the split that hides the
/// load best may have moved.
pub(crate) fn project_output_1p_104(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 104, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 16, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 104, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 104]>()
        .contract_lane::<m![H % 120 = 104], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 104]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 104 / 4], m![H % 120 = 104 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 104 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 104, m![H % 120 = 104 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 16, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 16]>()
        .contract_lane::<m![H % 120 = 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 16 / 4], m![H % 120 = 16 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 16 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 16, m![H % 120 = 16 #{!} 120]>(104));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}


pub(crate) fn project_output_1p_72(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 72, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 48, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 72, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 72]>()
        .contract_lane::<m![H % 120 = 72], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 72]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 72 / 4], m![H % 120 = 72 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 72 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 72, m![H % 120 = 72 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 48, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 48, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 48, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 48]>()
        .contract_lane::<m![H % 120 = 48], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 48]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 48 / 4], m![H % 120 = 48 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 48 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 48, m![H % 120 = 48 #{!} 120]>(72));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}


pub(crate) fn project_output_1p_split(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    // V260: rows 0..60 are complete once tile0 (88 rows) has contracted, so that half is stored
    // while tile1 is still loading; only the second half sits on the critical path. The split has
    // to divide the 120-row block evenly -- an offset tile that reaches less far than the buffer
    // is an `unpad`, which the HBM side has no API for -- so it is 60/60 rather than the 88/32
    // tile boundary. V253 measured 60/60 with *both* stores left at the end, which is why it only
    // saw the cost of the extra command.
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H / 120, H % 120]> = HbmTensor::new();
    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));
    contraction
        .view()
        .tile::<m![H % 120], 60, m![H % 120 = 60 # 120]>(0)
        .to_hbm_view(
            &mut ctx.tdma,
            gathered_hbm.view_mut().tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 #{!} 120]>(0),
        );
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 32, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 32, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 32, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 32]>()
        .contract_lane::<m![H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 32, m![H % 120 = 32 #{!} 120]>(88));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    // V260: the store split on the *tile* boundary, 88 + 32, so tile0's rows go out while tile1
    // is still loading. V253 split 60/60 -- which does not line up with the tiles -- so its first
    // store still had to wait for tile1 and it only measured the cost of an extra command.
    // rows 60..120, the half that tile1 completes
    contraction
        .view()
        .tile::<m![H % 120], 60, m![H % 120 = 60 # 120]>(60)
        .to_hbm_view(
            &mut ctx.tdma,
            gathered_hbm.view_mut().tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 #{!} 120]>(60),
        );
    unsafe { gathered_hbm.reshape() }
}

/// E5/V266: attn_out O-weight tiles 104/16 instead of 88/32, re-swept after V257 halved the contraction.
pub(crate) fn project_output_e5_104(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 104, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 16, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 104, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 104]>()
        .contract_lane::<m![H % 120 = 104], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 104]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 104 / 4], m![H % 120 = 104 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 104 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 104, m![H % 120 = 104 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 16, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 16]>()
        .contract_lane::<m![H % 120 = 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 16 / 4], m![H % 120 = 16 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 16 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 16, m![H % 120 = 16 #{!} 120]>(104));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// E5/V266: attn_out O-weight tiles 96/24 instead of 88/32, re-swept after V257 halved the contraction.
pub(crate) fn project_output_e5_96(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 96, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 96]>()
        .contract_lane::<m![H % 120 = 96], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 96]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 24, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 24]>()
        .contract_lane::<m![H % 120 = 24], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120]>(96));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// E5/V266: attn_out O-weight tiles 72/48 instead of 88/32, re-swept after V257 halved the contraction.
pub(crate) fn project_output_e5_72(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 72, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 72, m![H / 120, H % 120 = 72 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 48, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 48, m![H / 120, H % 120 = 48 # 120, Qs]>(72)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 72, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 72]>()
        .contract_lane::<m![H % 120 = 72], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 72]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 72 / 4], m![H % 120 = 72 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 72 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 72, m![H % 120 = 72 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 48, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 48, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 48, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 48]>()
        .contract_lane::<m![H % 120 = 48], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 48]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 48 / 4], m![H % 120 = 48 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 48 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 48, m![H % 120 = 48 #{!} 120]>(72));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// E5c/V269: attn_out O-weight tiles 100/20, refining V266 around 96/24.
pub(crate) fn project_output_e5_100(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 100, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 100, m![H / 120, H % 120 = 100 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 20, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 20, m![H / 120, H % 120 = 20 # 120, Qs]>(100)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 100, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 100, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 100, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 100]>()
        .contract_lane::<m![H % 120 = 100], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 100]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 100 / 4], m![H % 120 = 100 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 100 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 100, m![H % 120 = 100 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 20, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 20, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 20, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 20]>()
        .contract_lane::<m![H % 120 = 20], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 20]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 20 / 4], m![H % 120 = 20 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 20 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 20, m![H % 120 = 20 #{!} 120]>(100));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// E5c/V269: attn_out O-weight tiles 92/28, refining V266 around 96/24.
pub(crate) fn project_output_e5_92(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 92, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 92, m![H / 120, H % 120 = 92 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 28, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 28, m![H / 120, H % 120 = 28 # 120, Qs]>(92)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 92, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 92, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 92, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 92]>()
        .contract_lane::<m![H % 120 = 92], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 92]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 92 / 4], m![H % 120 = 92 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 92 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 92, m![H % 120 = 92 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 28, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 28, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 28, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 28]>()
        .contract_lane::<m![H % 120 = 28], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 28]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 28 / 4], m![H % 120 = 28 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 28 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 28, m![H % 120 = 28 #{!} 120]>(92));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// X1/V271: 96/24 tiles with the contraction output stored at 256 B-aligned offsets.
pub(crate) fn project_output_e5_96_x1(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]> {
    // Both clusters do real work: the hidden rows are split across the two clusters and
    // then across 32 row groups per cluster, and Qs across 8 column chunks, so each of the
    // 512 slices owns 60 rows x 512 columns (30 KB f8) and needs only an eighth of x. The
    // rows come in two tiles, 44 then 16, issued up front into distinct buffers, so the
    // contraction of one tile overlaps the loads of the rest. The eight chunk
    // partials are summed across slices within a cluster; the per-channel weight scale is
    // applied by the post-attention RMSNorm (rmsnorm::normalize_add_scaled_reduced), which
    // keeps its load out of the front of the DMA queue.
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // V251: the f8 split runs where the contraction needs it, on all 512 slices, instead of on
    // sixteen slices with an HBM round trip in between. The old path was load 535 -> split ->
    // store 399 -> store 399 -> load 933, and the 527 cycles the DMA engine sat idle waiting for
    // the split were what delayed the whole O-weight stream to cycle 2,464. Loading x straight
    // into the contraction's own layout costs the same 933 (each slice still reads 512 B: 256
    // bf16 now instead of 2 x 256 f8) and drops the three other commands entirely.
    // s = 16 is safe without measuring: the attention output is a convex combination of the
    // value rows, which the value RMSNorm bounds by sqrt(Ds) = 16, so |x s| <= 256 < 448.
    // V257: one f8 piece of x instead of two. attn_out's tolerance is the loosest of the three
    // (atol 0.05 against qkv's 0.04 and ffn's 0.01) and the single-piece question has only ever
    // been asked of the two tighter kernels (V137/V138 on qkv, V142 on ffn). Dropping the second
    // piece removes the `Dummy2` replay axis from the contraction -- so each weight packet is
    // fetched once, not twice -- one of the two split passes, and half of x's bytes.
    //
    // The piece is rounded, not truncated: `hi_lo_trunc_fns` masks the mantissa because the *pair*
    // has to sum exactly, and that constraint is gone here, so a plain cast is strictly better.
    let xs: DmTensor<bf16, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, HiddenRowsByColumns256, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, HiddenRows256, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 96, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 96]>()
        .contract_lane::<m![H % 120 = 96], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 96]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 24, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 24]>()
        .contract_lane::<m![H % 120 = 24], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows256, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120]>(96));

    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    // X1/V271: each slice writes its 240 B at a 256 B boundary (16 B of padding per row group).
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());
    gathered_hbm
}

/// Q2a: x staged into the TRF once, in the query layout; the K/V projections relabel their weights to it.
pub(crate) fn stage_x_trf_query(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]>,
) -> TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![Dummy2, H]> {
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    ctx.sub
        .begin(x)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf()
}

/// Q2: `project_query` against a TRF copy of x on any cluster/slice labels.
pub(crate) fn project_query_trf_on<C: M, S: M>(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, C, S, m![1], m![Dummy2, H]>,
    weight_f8: &QueryWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Gs, Ds]> {
    // x (two f8 pieces whose sum is bf16 x times a power of two) is replicated onto every
    // slice of both clusters. Each weight packet is streamed twice (the Dummy2 time axis) so
    // the Time Reducer adds the dot products with the two pieces.
    let weight: DmTensorView<'_, f8e4m3, Chip, C, S, m![Qs % 8, H]> = unsafe { weight_f8.view().reshape() };
    let contraction: DmTensor<bf16, Chip, C, S, m![Qs % 8]> = ctx
        .main
        .begin(weight)
        .fetch::<m![Qs % 8, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Qs % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    // The per-channel weight scale is applied by the query RMSNorm that follows (loaded there
    // in the head layout with eight descriptors instead of 512 here).
    // Each cluster holds four heads spread over 64 slices x 8 rows each; a ring-64 gather
    // puts every head on one slice, the layout the query RMSNorm and RoPE work in, without
    // leaving the cluster (the two clusters then post-process their four heads in parallel).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Q2: `project_one_kv_matrix` against a TRF copy of x on any cluster/slice labels.
fn project_one_kv_matrix_on<C: M, S: M>(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, C, S, m![1], m![Dummy2, H]>,
    weight_f8: &KvWeight,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]> {
    let weight: DmTensorView<'_, f8e4m3, Chip, C, S, m![Ps % 4, H]> = unsafe { weight_f8.view().reshape() };
    let contraction: DmTensor<bf16, Chip, C, S, m![Ps % 4]> = ctx
        .main
        .begin(weight)
        .fetch::<m![Ps % 4, H / 64, Dummy2], m![H % 64]>()
        .collect::<m![Ps % 4, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    // The per-channel weight scale is applied by the head RMSNorm that follows.
    // Ring-64 gather to one head per slice within the cluster (see project_query).
    let scaled: DmTensorView<'_, bf16, Chip, HeadClusters, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

/// Q2: `project_key_value` against a TRF copy of x on any cluster/slice labels.
pub(crate) fn project_key_value_trf_on<C: M, S: M>(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, C, S, m![1], m![Dummy2, H]>,
    k_weight: &KvWeight,
    v_weight: &KvWeight,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlicesPerCluster, m![Ds]>,
) {
    let k = project_one_kv_matrix_on(ctx, x_trf, k_weight);
    let v = project_one_kv_matrix_on(ctx, x_trf, v_weight);

    (k, v)
}
