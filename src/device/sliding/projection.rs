
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

// ---------------------------------------------------------------------------------------------
// V219: what does one vector pass actually cost?
//
// V218 found that deleting the three head RMSNorms saves 6,833 real cycles while the static
// schedule only charges 1,263 - a 5.4x overrun. But dividing each ablation's excess by its
// instruction delta gives 172, 56 and 13 cycles, so there is no single per-pass constant in that
// data, and the removals are not even additive. Measure the slope directly instead: N minimal
// vector passes (one f32 scalar per slice, one AddF) chained head to tail, N = 0/4/8/16. Their
// result is dropped, so the kernel stays numerically correct and the jobs report PASS.
//
// `i8` runs the same eight passes with no dependency between them. If serial and independent cost
// the same, the price is in-order issue; if independent is much cheaper, it is dependency latency
// and only fusion that shortens the chain will pay.
// ---------------------------------------------------------------------------------------------

pub(crate) fn pass_chain_0(ctx: &mut Context) {
    // Garbage DM masked to +0.0, so the chain below is finite whatever was in the buffer.
    let scratch: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = DmTensor::new();
    let seed: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(scratch.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_logic(LogicBinaryOpF32::BitAnd, 0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
}

pub(crate) fn pass_chain_4(ctx: &mut Context) {
    // Garbage DM masked to +0.0, so the chain below is finite whatever was in the buffer.
    let scratch: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = DmTensor::new();
    let seed: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(scratch.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_logic(LogicBinaryOpF32::BitAnd, 0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p0: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p1: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p0.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p2: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p1.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p3: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p2.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _tail = p3;
}

pub(crate) fn pass_chain_8(ctx: &mut Context) {
    // Garbage DM masked to +0.0, so the chain below is finite whatever was in the buffer.
    let scratch: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = DmTensor::new();
    let seed: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(scratch.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_logic(LogicBinaryOpF32::BitAnd, 0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p0: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p1: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p0.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p2: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p1.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p3: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p2.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p4: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p3.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p5: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p4.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p6: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p5.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p7: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p6.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _tail = p7;
}

pub(crate) fn pass_chain_16(ctx: &mut Context) {
    // Garbage DM masked to +0.0, so the chain below is finite whatever was in the buffer.
    let scratch: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = DmTensor::new();
    let seed: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(scratch.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_logic(LogicBinaryOpF32::BitAnd, 0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p0: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p1: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p0.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p2: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p1.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p3: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p2.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p4: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p3.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p5: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p4.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p6: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p5.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p7: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p6.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p8: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p7.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p9: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p8.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p10: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p9.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p11: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p10.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p12: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p11.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p13: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p12.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p14: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p13.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let p15: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(p14.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _tail = p15;
}

pub(crate) fn pass_indep_8(ctx: &mut Context) {
    // Garbage DM masked to +0.0, so the chain below is finite whatever was in the buffer.
    let scratch: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = DmTensor::new();
    let seed: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(scratch.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_logic(LogicBinaryOpF32::BitAnd, 0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let q0: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q0 = q0;
    let q1: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 2f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q1 = q1;
    let q2: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 3f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q2 = q2;
    let q3: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 4f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q3 = q3;
    let q4: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q4 = q4;
    let q5: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 6f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q5 = q5;
    let q6: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 7f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q6 = q6;
    let q7: DmTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx
        .main
        .begin(seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 8f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let _q7 = q7;
}
