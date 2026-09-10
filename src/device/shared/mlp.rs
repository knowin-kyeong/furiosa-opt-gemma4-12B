
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{C, Dummy2, Dummy256, Dummy8, H, L};
use crate::device::layout::{Cluster, Slice};
use crate::device::shared::rmsnorm::{self, ReducingSlices};
use crate::{hi_lo_fns, max_square_fns, pow2_scale_fns, stage_packet_fns};

const INVSQRT2: f32 = 0.70710678118f32;

/// Both clusters do real work on the up/gate projections: the L rows are split across the
/// two clusters, then 128 row groups per cluster, and H across two 1920-column halves
/// (512 slices x 60 rows x 1920 columns). The two half partials are summed across slices.
type UpGateClusters = m![L / 7680];
type UpGateRowsSplit = m![L / 60 % 128, 1 # 2];
/// The same rows with the reduced result replicated on both slices of a pair, so every slice is live.
type UpGateRowsPairs = m![L / 60 % 128, Dummy2];
type UpGateRowsByColumns = m![L / 60 % 128, H / 1920];
/// Eight row groups per slice after the ring-16 gather ahead of the geglu output store.
type UpGateRowsGathered = m![L / 480 % 16, 1 # 16];


pow2_scale_fns!(pow2_scale_reducing, Cluster, ReducingSlices);
pow2_scale_fns!(pow2_scale_gathered_all, UpGateClusters, m![Dummy256]);
stage_packet_fns!(stage_packet_reducing, Cluster, ReducingSlices);
stage_packet_fns!(stage_packet_gathered, UpGateClusters, UpGateRowsGathered);
stage_packet_fns!(stage_packet_pairs, UpGateClusters, UpGateRowsPairs);
stage_packet_fns!(stage_packet_down, DownClusters, DownRowsByColumns);
max_square_fns!(max_square_reducing, Cluster, ReducingSlices, H, 480, 30, 60, 120);
max_square_fns!(max_square_gathered, UpGateClusters, UpGateRowsGathered, L, 480, 30, 60, 120);
hi_lo_fns!(hi_lo_reducing, Cluster, ReducingSlices, H, 480, 15, 30, 60, 120);
hi_lo_fns!(hi_lo_gathered, UpGateClusters, UpGateRowsGathered, L, 480, 15, 30, 60, 120);

/// The FFN input as two f8 pieces of x * s (see `hi_lo_fns`; s is chosen from max x^2 over the
/// vector), written to one HBM scratch laid out so that every slice's column half of both pieces
/// is one contiguous segment. The geglu's two scalars are also derived here, with 1/s folded in
/// (the projections come out multiplied by s): `s_gate / (s sqrt 2)` for the erf argument and
/// `s_up s_gate / (2 s^2)` for the output factor. All of it runs in the RMSNorm's reducing layout
/// (8 slices x 480: no tiles, small VRFs).
pub(crate) fn stage_x_hi_lo_hbm(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (
    HbmTensor<f8e4m3, Chip, m![H / 1920, Dummy2, H % 1920]>,
    HbmTensor<f32, Chip, m![1 # 8]>,
    HbmTensor<f32, Chip, m![1 # 8]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let m_local = max_square_reducing(ctx, &x);
    let m_all: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, inv_s) = pow2_scale_reducing(ctx, &m_all);
    let s_vrf = stage_packet_reducing(ctx, &s);
    let inv_s_vrf = stage_packet_reducing(ctx, &inv_s);

    let (x_hi, x_lo) = hi_lo_reducing(ctx, &x, &s_vrf);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![H / 1920, Dummy2, H % 1920]> = HbmTensor::new();
    x_hi.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![H / 1920, Dummy2 = 1 #{!} 2, H % 1920]>(0),
    );
    x_lo.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![H / 1920, Dummy2 = 1 #{!} 2, H % 1920]>(1),
    );

    // The geglu scalars.
    let s_up: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = up_global_scale.to_dm(&mut ctx.tdma);
    let s_gate: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = gate_global_scale.to_dm(&mut ctx.tdma);
    let s_gate_vrf = stage_packet_reducing(ctx, &s_gate);
    let erf_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_gate.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2_vrf = stage_packet_reducing(ctx, &half_inv_s2);
    let out_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_up.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_gate_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &half_inv_s2_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let erf_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { erf_scale.reshape() };
    let out_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { out_scale.reshape() };
    let mut erf_hbm: HbmTensor<f32, Chip, m![1 # 8]> = HbmTensor::new();
    erf_one.view().to_hbm_view(&mut ctx.tdma, erf_hbm.view_mut());
    let mut out_hbm: HbmTensor<f32, Chip, m![1 # 8]> = HbmTensor::new();
    out_one.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
    (x2_hbm, erf_hbm, out_hbm)
}

/// The QKV input as two f8 pieces of x * s (see `stage_x_hi_lo_hbm`), staged in HBM once. The
/// projections' outputs come out multiplied by s; the head RMSNorms that follow are
/// scale-invariant (eps aside), so nothing undoes it.
///
/// V15..V41 wrote this sixteen times so that the replicated load could spread over HBM channels,
/// by giving the destination a copy axis the source did not have. That does not replicate a store
/// (V50): one copy was written and fifteen were read out of uninitialised HBM. Making the copies
/// real costs one store descriptor set each, which the Core issues in order and which outweighs
/// the load it saves at every copy count (V49).
pub(crate) fn stage_x_hi_lo_qkv_hbm(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> HbmTensor<f8e4m3, Chip, m![Dummy2, H]> {
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let m_local = max_square_reducing(ctx, &x);
    let m_all: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, _inv_s) = pow2_scale_reducing(ctx, &m_all);
    let s_vrf = stage_packet_reducing(ctx, &s);

    let (x_hi, x_lo) = hi_lo_reducing(ctx, &x, &s_vrf);
    // m1: the two pieces are gathered into one buffer on the slices that already hold them, so
    // the staging costs one DMA command instead of two.
    let mut x2: DmTensor<f8e4m3, Chip, Cluster, ReducingSlices, m![Dummy2, H % 480]> = DmTensor::new();
    ctx.main
        .begin(x_hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(0));
    ctx.main
        .begin(x_lo.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(1));
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Dummy2, H]> = HbmTensor::new();
    x2.view().to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut());
    x2_hbm
}
/// The geglu output (gathered, eight row groups per slice) as two f8 pieces of x * s_c, s_c chosen
/// per cluster from the cluster-wide max x^2 (each slice's max is broadcast to every slice of the
/// cluster over a ring-256 switch and reduced there). Written to an HBM scratch laid out so that a
/// down slice's column chunk of both pieces is one contiguous segment, plus 1/s_c per cluster for
/// the down projection's epilogue.
fn stage_geglu_hi_lo_hbm(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]>,
) -> (HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]>, HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>) {
    let m_local = max_square_gathered(ctx, x);
    let m_every: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![L / 480 % 16, 1 # 8]> = ctx
        .main
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![L / 480 % 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]> = ctx
        .sub
        .begin(m_every.view())
        .fetch::<m![L / 480 % 16], m![1 # 8]>()
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let (s_all, inv_s_all) = pow2_scale_gathered_all(ctx, &m_all);
    let s: DmTensor<f32, Chip, UpGateClusters, UpGateRowsGathered, m![1 # 8]> = unsafe { s_all.reshape() };
    let s_vrf = stage_packet_gathered(ctx, &s);

    let (x_hi, x_lo) = hi_lo_gathered(ctx, x, &s_vrf);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]> = HbmTensor::new();
    x_hi.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![L / 1920, Dummy2 = 1 #{!} 2, L % 1920]>(0),
    );
    x_lo.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![L / 1920, Dummy2 = 1 #{!} 2, L % 1920]>(1),
    );

    let inv_s_one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = unsafe { inv_s_all.reshape() };
    let mut inv_s_hbm: HbmTensor<f32, Chip, m![L / 7680, 1 # 8]> = HbmTensor::new();
    inv_s_one.view().to_hbm_view(&mut ctx.tdma, inv_s_hbm.view_mut());
    (x2_hbm, inv_s_hbm)
}

/// A scalar from HBM onto every slice of the up/gate pair layout: two descriptors (one per
/// cluster) and a ring-256 broadcast, instead of 256 descriptors.
fn broadcast_scalar_pairs(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![1 # 8]>,
) -> VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPairs, m![1 # 8]> {
    let one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]> = ctx
        .main
        .begin(one.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPairs, m![1 # 8]> = unsafe { all.reshape() };
    stage_packet_pairs(ctx, &all)
}

/// 1/s_c of the geglu pieces onto every down slice: each slice's column chunk came from cluster
/// c/4, so the two values are loaded onto two slices per cluster and broadcast with a
/// permutation over a ring of 256 to the slices whose chunk they scale.
fn broadcast_inv_s_down(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>,
) -> VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> {
    let two: DmTensor<f32, Chip, DownClusters, m![1 # 128, L / 7680], m![1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, DownClusters, m![Dummy256 / 8, L / 7680, Dummy8 / 2], m![1 # 8]> = ctx
        .main
        .begin(two.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256 / 8, L / 7680, Dummy8 / 2], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> = unsafe { all.reshape() };
    stage_packet_down(ctx, &all)
}

/// The up/gate pass-A helpers for one tile height: load `$rows` packed rows x each slice's
/// 1920-column half and contract them with x into per-16-column-block partial sums, written into
/// the tile of a 60-row partials buffer (so pass B can tile the rows independently).
///
/// Pass A streams the f4 -> f8 lookup straight into an f8 x f8 contraction (f32 accumulate):
/// x is held in the TRF as two f8 pieces, `x_hi = f8(x)` and `x_lo = f8(x - x_hi)`, whose sum is
/// the bf16 x exactly, and each weight packet is streamed twice (the `Dummy2` time axis) so the
/// Time Reducer adds the two dot products. Every pass reloads the 4 KB lookup table (838 cycles
/// of DMA) and every tile load has ~550 cycles of fixed DMA cost, so the tiles are as tall as the
/// contraction allows; the pass-B tiles stay at 16 rows (the scale VRF holds 16 x 120 f32).
macro_rules! up_gate_contract_fns {
    ($load:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]> {
            packed
                .view()
                .tile::<m![L % 60], $rows, m![L / 60, L % 60 = $rows # 60, H]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        /// Pass A: per-16-column-block partial dot products of `$rows` packed rows with x.
        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![Dummy2, H % 1920]>,
            packed: &DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]>,
            offset: usize,
            partials: &mut DmTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]>,
        ) {
            ctx.main
                .begin(packed.view())
                .fetch::<m![L % 60 = $rows, H / 64 % 30, Dummy2], m![H % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![L % 60 = $rows, H / 64 % 30, Dummy2, H / 32 % 2], m![H % 32]>()
                .contract_outer::<m![L % 60 = $rows, H / 64 % 30, Dummy2], m![H % 64], _, _, _>(x_trf)
                .contract_packet::<m![H / 16 % 4]>()
                .contract_time::<m![L % 60 = $rows, H / 64 % 30]>()
                .contract_lane::<m![L % 60 = $rows, H / 64 % 30], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![H / 16 % 4]>()
                .commit_view(partials.view_mut().tile::<m![L % 60], $rows, m![L % 60 = $rows #{!} 60, H / 16 % 120]>(offset));
        }
    };
}

/// Pass B for one tile height: block scales, column reduction and the cross-slice sum of the
/// column chunks, over a `$rows`-row tile of the partials buffer.
macro_rules! up_gate_reduce_fns {
    ($reduce:ident, $rows:literal) => {
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]>,
            scale_all: &DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H / 16 % 120]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![L % 60], $rows, m![L % 60 = $rows # 60, H / 16 % 120]>(offset))
                .fetch::<m![L % 60 = $rows], m![H / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = $rows, H / 128 % 15], m![H / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view().tile::<m![L % 60], $rows, m![L % 60 = $rows # 60, H / 16 % 120]>(offset))
                .fetch::<m![L % 60 = $rows, H / 128 % 15], m![H / 16 % 8]>()
                .collect::<m![L % 60 = $rows, H / 128 % 15], m![H / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L % 60 = $rows, H / 64 % 30], m![H / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<H, m![L % 60 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<UpGateRowsPairs, m![L % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![L % 60 = $rows / 4], m![L % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![L % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![L % 60], $rows, m![L % 60 = $rows #{!} 60]>(offset));
        }
    };
}
up_gate_contract_fns!(load_up_gate_rows_60, contract_up_gate_rows_60, 60);
up_gate_reduce_fns!(reduce_up_gate_rows_16, 16);
up_gate_reduce_fns!(reduce_up_gate_rows_12, 12);

pub(crate) fn feedforward(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![H / 1920, Dummy2, H % 1920]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    let up0 = load_up_gate_rows_60(ctx, up_weight_packed, 0);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate0 = load_up_gate_rows_60(ctx, gate_weight_packed, 0);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_8(ctx, down_weight_packed, 48);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // Each slice needs only its 1920-wide half of x (both f8 pieces, one DMA).
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![Dummy2, H % 1920]> = x2.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![Dummy2, H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32 % 60], m![H % 32]>()
        .collect::<m![Dummy2, H / 32 % 60], m![H % 32]>()
        .to_trf();

    // Each tile is contracted into per-block partial sums as it lands (pass A) and the block
    // scales are applied to those partials (pass B); see the tile helpers.
    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]> = DmTensor::new();
    let mut up_partials: DmTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> = DmTensor::new();
    let mut gate_partials: DmTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> = DmTensor::new();
    contract_up_gate_rows_60(ctx, &x_trf, &up0, 0, &mut up_partials);
    contract_up_gate_rows_60(ctx, &x_trf, &gate0, 0, &mut gate_partials);
    reduce_up_gate_rows_12(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_rows_12(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_rows_12(ctx, &up_partials, &up_scale, 12, &mut up);
    reduce_up_gate_rows_12(ctx, &gate_partials, &gate_scale, 12, &mut gate);
    reduce_up_gate_rows_12(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_rows_12(ctx, &gate_partials, &gate_scale, 24, &mut gate);
    reduce_up_gate_rows_12(ctx, &up_partials, &up_scale, 36, &mut up);
    reduce_up_gate_rows_12(ctx, &gate_partials, &gate_scale, 36, &mut gate);
    reduce_up_gate_rows_12(ctx, &up_partials, &up_scale, 48, &mut up);
    reduce_up_gate_rows_12(ctx, &gate_partials, &gate_scale, 48, &mut gate);

    // geglu runs in the up/gate reduce layout (see geglu_split); its output is staged through
    // HBM (see V7). Storing 60 rows from each of 256 slices costs 4.5k cycles of descriptors,
    // so eight row groups are first gathered onto one slice over a ring of 16 (the live slices
    // sit two apart): store 1,870, switch 503. Wider rings save nothing more on the store and
    // cost more in the switch.
    let x = geglu_split(ctx, up, gate, erf_scale, out_scale);
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsGathered, m![L / 4 % 15, L / 60 % 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 2 })
        .collect::<m![L / 4 % 15, L / 60 % 8], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_8(ctx, &x_trf, &down3);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);
    let p = contract_down_rows_4(ctx, &x_trf, &down4);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// geglu on up and gate as the up/gate projections leave them: one row group of 60 per slice
/// (replicated on both slices of a pair), both clusters. 60 f32 do not fill 8-wide packets, so
/// every 4-element packet is padded to 8 and the vector passes work on the live half (the V18
/// scale-packet pattern). The two scalars (erf argument factor and output factor, with the global
/// scales and 1/s folded in) come from the HBM scratches the input staging wrote.
pub(crate) fn geglu_split(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]>,
    gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]> {
    let erf_scale_vrf = broadcast_scalar_pairs(ctx, erf_scale);
    let out_scale_vrf = broadcast_scalar_pairs(ctx, out_scale);

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPairs, m![L % 60]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 15, 1 # 2], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &erf_scale_vrf)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPairs, m![L / 4 % 15, L % 4 # 8]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 15, 1 # 2], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &out_scale_vrf)
        .vector_widen_concat::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit()
}

/// Both clusters do real work on the down projection: hidden rows are split across the
/// two clusters, then 32 row groups per cluster, and L across 8 column chunks (512 slices x
/// 60 rows x 1920 columns). The chunk partials are summed across slices within a cluster.
pub(crate) type DownClusters = m![H / 1920];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

/// The down tile helpers for one tile height (see `up_gate_tile_fns`): 16, 8 and 4 rows. The
/// last tile is the small one so that little work trails the final weight load.
macro_rules! down_tile_fns {
    ($load:ident, $contract:ident, $reduce:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]> {
            packed
                .view()
                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        /// Pass A: per-16-column-block partial dot products of `$rows` packed rows with x.
        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]>,
            packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]>,
        ) -> DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L / 16 % 120]> {
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 = $rows, L / 64 % 30, Dummy2, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64], _, _, _>(x_trf)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 = $rows, L / 64 % 30]>()
                .contract_lane::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![L / 16 % 4]>()
                .commit()
        }

        /// Pass B: block scales, column reduction and the cross-slice sum of the column chunks.
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L / 16 % 120]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]>,
            inv_s_vrf: &VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L / 16 % 120]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 120]>(offset))
                .fetch::<m![H % 60 = $rows], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view())
                .fetch::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), inv_s_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<DownRows, m![H % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![H % 60 = $rows / 4], m![H % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![H % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));
        }
    };
}
down_tile_fns!(load_down_rows_16, contract_down_rows_16, reduce_down_rows_16, 16);
down_tile_fns!(load_down_rows_8, contract_down_rows_8, reduce_down_rows_8, 8);
down_tile_fns!(load_down_rows_4, contract_down_rows_4, reduce_down_rows_4, 4);
down_tile_fns!(load_down_rows_12, contract_down_rows_12, reduce_down_rows_12, 12);

// ---------------------------------------------------------------------------------------------
// V181: up/gate with each slice holding 30 whole rows and all of H. A slice's weight (57.6 KB)
// and block scale (7.2 KB) are then one contiguous segment each instead of 60 (a segmented
// load costs twice per byte on hardware, V174), there is no cross-slice sum of column halves,
// and x is replicated to every slice by the ring broadcast that fixed qkv (V157). 30 rows do
// not fit the 4-row transpose, so pass B commits one scalar packet per row and the rows are
// packed into a dense vector after the ring-16 gather ahead of the geglu-output staging.
type UpGateRowsFull = m![L / 30 % 256];
stage_packet_fns!(stage_packet_full, UpGateClusters, UpGateRowsFull);

fn broadcast_scalar_full(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![1 # 8]>,
) -> VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]> {
    let one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]> = ctx
        .main
        .begin(one.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]> = unsafe { all.reshape() };
    stage_packet_full(ctx, &all)
}

/// Pass A over all 30 rows and all of H: per-16-column-block partial dot products.
fn contract_up_gate_full(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]>,
    packed: &DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]>,
) -> DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> {
    ctx.main
        .begin(packed.view())
        .fetch::<m![L % 30, H / 64, Dummy2], m![H % 64]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 30, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![L % 30, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![L % 30, H / 64]>()
        .contract_lane::<m![L % 30, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>()
        .commit()
}

/// Pass B for one tile of `$rows` rows: block scales and the column reduction, one f32 scalar
/// packet per row (no transpose, so any tile height works).
macro_rules! up_gate_reduce_full_fns {
    ($reduce:ident, $rows:literal) => {
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]>,
            scale_all: &DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]>,
            offset: usize,
            out: &mut DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30 = $rows, H / 16]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![L % 30], $rows, m![L % 30 = $rows # 30, H / 16]>(offset))
                .fetch::<m![L % 30 = $rows], m![H / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 30 = $rows, H / 128], m![H / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view().tile::<m![L % 30], $rows, m![L % 30 = $rows # 30, H / 16]>(offset))
                .fetch::<m![L % 30 = $rows, H / 128], m![H / 16 % 8]>()
                .collect::<m![L % 30 = $rows, H / 128], m![H / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L % 30 = $rows, H / 64], m![H / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<H, m![L % 30 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit_view(out.view_mut().tile::<m![L % 30], $rows, m![L % 30 = $rows #{!} 30, 1 # 8]>(offset));
        }
    };
}
up_gate_reduce_full_fns!(reduce_up_gate_full_8, 8);
up_gate_reduce_full_fns!(reduce_up_gate_full_6, 6);

/// geglu on the row-scalar packets of one slice (up and gate rows of the same slice).
fn geglu_full(
    ctx: &mut Context,
    up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
    gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
) -> DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> {
    let erf_scale_vrf = broadcast_scalar_full(ctx, erf_scale);
    let out_scale_vrf = broadcast_scalar_full(ctx, out_scale);

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &erf_scale_vrf)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &out_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Ring-16 gather of 16 slices' row packets onto one slice, then the 480 scalar packets packed
/// into a dense bf16 vector by the transpose engine (480 rows are a multiple of 4).
fn gather_pack_full(
    ctx: &mut Context,
    g: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]> {
    // The ring delivers packet-major: every slice's packet r before packet r + 1 (V167 read it
    // slice-major and came out permuted), so the gathered buffer is [L % 30, L / 30 % 16] and the
    // packing pass fetches it back in L order.
    let gathered: DmTensor<f32, Chip, UpGateClusters, UpGateRowsGathered, m![L % 30, L / 30 % 16, 1 # 8]> = ctx
        .main
        .begin(g.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .switch::<UpGateRowsGathered, m![L % 30, L / 30 % 16]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![L % 30, L / 30 % 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    ctx.main
        .begin(gathered.view())
        .fetch::<m![L / 30 % 16, L % 30], m![1 # 8]>()
        .collect::<m![L % 480], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 480 / 4], m![L % 480 % 4 # 16]>()
        .commit_trim::<m![L % 480 % 4]>()
        .commit()
}

pub(crate) fn stage_x_hi_lo_hbm_full(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (
    HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    HbmTensor<f32, Chip, m![1 # 8]>,
    HbmTensor<f32, Chip, m![1 # 8]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let m_local = max_square_reducing(ctx, &x);
    let m_all: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, inv_s) = pow2_scale_reducing(ctx, &m_all);
    let s_vrf = stage_packet_reducing(ctx, &s);
    let inv_s_vrf = stage_packet_reducing(ctx, &inv_s);

    let (x_hi, x_lo) = hi_lo_reducing(ctx, &x, &s_vrf);
    // m1: the two pieces are gathered into one buffer on the slices that already hold them, so
    // the staging costs one DMA command instead of two.
    let mut x2: DmTensor<f8e4m3, Chip, Cluster, ReducingSlices, m![Dummy2, H % 480]> = DmTensor::new();
    ctx.main
        .begin(x_hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(0));
    ctx.main
        .begin(x_lo.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(1));
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Dummy2, H]> = HbmTensor::new();
    x2.view().to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut());

    // The geglu scalars.
    let s_up: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = up_global_scale.to_dm(&mut ctx.tdma);
    let s_gate: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = gate_global_scale.to_dm(&mut ctx.tdma);
    let s_gate_vrf = stage_packet_reducing(ctx, &s_gate);
    let erf_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_gate.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2_vrf = stage_packet_reducing(ctx, &half_inv_s2);
    let out_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_up.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_gate_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &half_inv_s2_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let erf_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { erf_scale.reshape() };
    let out_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { out_scale.reshape() };
    let mut erf_hbm: HbmTensor<f32, Chip, m![1 # 8]> = HbmTensor::new();
    erf_one.view().to_hbm_view(&mut ctx.tdma, erf_hbm.view_mut());
    let mut out_hbm: HbmTensor<f32, Chip, m![1 # 8]> = HbmTensor::new();
    out_one.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
    (x2_hbm, erf_hbm, out_hbm)
}
// ---------------------------------------------------------------------------------------------
// V237: the down projection with L in four 3,840-column chunks instead of eight of 1,920.
//
// V211 measured the alignment penalty the eight-chunk layout pays: a chunk's weight run is
// 1,920 nibbles = 960 B and chunk c starts at c * 960 B, so six chunks out of eight begin in the
// middle of a 256-byte granule and every run touches five granules to use four (509 vs 399
// B/cycle over the same 29.5 MB, 15,951 cycles). Four chunks make the run 1,920 B at offsets
// c * 1,920 B: aligned runs touch 8 granules (1,920 / 256 = 7.5 -> 8) and misaligned ones start
// at byte 128 and end at byte 2,048 exactly, so they touch 8 as well. The stream drops from
// 1.267 fetched bytes per useful byte to 1.067. The block scales double their run too (120 ->
// 240 B): still under the 256 B floor, but there are half as many runs.
//
// The cost is that only 128 of the 256 slices stay live, so each one contracts twice the
// columns -- and V235 established that ffn's MainContext hides behind the DMA stream, which is
// exactly the trade this makes. x per slice doubles for the same reason (3,840 -> 7,680 B).
//
// Pass B is capped by the scale VRF (8 KB): 240 f32 columns per row allows 8 rows per tile, not
// 16, so pass A keeps the tile heights of the weight loads and pass B tiles the partials buffer
// independently, the way V181's up/gate stage does.
// A DM allocation has to span all 256 slices, so the half of them this layout does not use is
// spent as a dead `1 # 2` axis (the same idiom `DownRows` uses to leave seven of eight idle).
// The chunk axis stays innermost because the VRU reduces over it (V186).
pub(crate) type DownRows4 = m![H / 60 % 32, 1 # 2, 1 # 4];
pub(crate) type DownRowsByColumns4 = m![H / 60 % 32, 1 # 2, L / 3840];

stage_packet_fns!(stage_packet_down4, DownClusters, DownRowsByColumns4);

macro_rules! down4_contract_fns {
    ($load:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns4, m![H % 60 = $rows, L % 3840]> {
            packed
                .view()
                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        /// Pass A: per-16-column-block partial dot products, written into the tile of a 60-row
        /// partials buffer so pass B can tile the rows at the scale VRF's height instead.
        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns4, m![1], m![Dummy2, L % 3840]>,
            packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns4, m![H % 60 = $rows, L % 3840]>,
            offset: usize,
            partials: &mut DmTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![H % 60, L / 16 % 240]>,
        ) {
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 60 = $rows, L / 64 % 60, Dummy2], m![L % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 = $rows, L / 64 % 60, Dummy2, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 = $rows, L / 64 % 60, Dummy2], m![L % 64], _, _, _>(x_trf)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 = $rows, L / 64 % 60]>()
                .contract_lane::<m![H % 60 = $rows, L / 64 % 60], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![L / 16 % 4]>()
                .commit_view(partials.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60, L / 16 % 240]>(offset));
        }
    };
}
down4_contract_fns!(load_down4_rows_16, contract_down4_rows_16, 16);
down4_contract_fns!(load_down4_rows_12, contract_down4_rows_12, 12);

macro_rules! down4_reduce_fns {
    ($reduce:ident, $rows:literal) => {
        /// Pass B: block scales, column reduction and the cross-slice sum of the column chunks.
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![H % 60, L / 16 % 240]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns4, m![H % 60, L / 16 % 240]>,
            inv_s_vrf: &VrfTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![1 # 8]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, DownClusters, DownRows4, m![H % 60]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![H % 60 = $rows, L / 16 % 240]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 240]>(offset))
                .fetch::<m![H % 60 = $rows], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 240]>(offset))
                .fetch::<m![H % 60 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .collect::<m![H % 60 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 = $rows, L / 64 % 60], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), inv_s_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<DownRows4, m![H % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![H % 60 = $rows / 4], m![H % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![H % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));
        }
    };
}
down4_reduce_fns!(reduce_down4_rows_8, 8);
down4_reduce_fns!(reduce_down4_rows_4, 4);

/// `stage_geglu_hi_lo_hbm` with the scratch laid out for four column chunks.
fn stage_geglu_hi_lo_hbm4(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]>,
) -> (HbmTensor<f8e4m3, Chip, m![L / 3840, Dummy2, L % 3840]>, HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>) {
    let m_local = max_square_gathered(ctx, x);
    let m_every: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![L / 480 % 16, 1 # 8]> = ctx
        .main
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![L / 480 % 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]> = ctx
        .sub
        .begin(m_every.view())
        .fetch::<m![L / 480 % 16], m![1 # 8]>()
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let (s_all, inv_s_all) = pow2_scale_gathered_all(ctx, &m_all);
    let s: DmTensor<f32, Chip, UpGateClusters, UpGateRowsGathered, m![1 # 8]> = unsafe { s_all.reshape() };
    let s_vrf = stage_packet_gathered(ctx, &s);

    let (x_hi, x_lo) = hi_lo_gathered(ctx, x, &s_vrf);
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![L / 3840, Dummy2, L % 3840]> = HbmTensor::new();
    x_hi.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![L / 3840, Dummy2 = 1 #{!} 2, L % 3840]>(0),
    );
    x_lo.view().to_hbm_view(
        &mut ctx.tdma,
        x2_hbm.view_mut().tile::<m![Dummy2], 1, m![L / 3840, Dummy2 = 1 #{!} 2, L % 3840]>(1),
    );

    let inv_s_one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = unsafe { inv_s_all.reshape() };
    let mut inv_s_hbm: HbmTensor<f32, Chip, m![L / 7680, 1 # 8]> = HbmTensor::new();
    inv_s_one.view().to_hbm_view(&mut ctx.tdma, inv_s_hbm.view_mut());
    (x2_hbm, inv_s_hbm)
}

/// `broadcast_inv_s_down` for the four-chunk layout: a down slice's chunk came from up/gate
/// cluster `chunk / 2`, which is bit 1 of the 128-slice index instead of bit 2 of 256.
fn broadcast_inv_s_down4(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>,
) -> VrfTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![1 # 8]> {
    let two: DmTensor<f32, Chip, DownClusters, m![1 # 128, L / 7680], m![1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, DownClusters, m![Dummy256 / 8, Dummy8 / 4, L / 7680, Dummy2], m![1 # 8]> = ctx
        .main
        .begin(two.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256 / 8, Dummy8 / 4, L / 7680, Dummy2], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![1 # 8]> = unsafe { all.reshape() };
    stage_packet_down4(ctx, &all)
}

pub(crate) fn feedforward_v181(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_8(ctx, down_weight_packed, 48);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_8(ctx, &x_trf, &down3);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);
    let p = contract_down_rows_4(ctx, &x_trf, &down4);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V238: the down projection in four tiles (16/16/16/12) instead of five (16/16/16/8/4).
/// Every pass A reloads the 4 KB f4 lookup table (838 cycles of DMA) and every tile load has
/// ~550 cycles of fixed DMA cost, so one tile fewer is ~1.4k of static DMA and one pass fewer
/// on each of MainContext and SubContext (V218 priced a pass at ~600 real cycles).
pub(crate) fn feedforward_v238(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_12(ctx, down_weight_packed, 48);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_12(ctx, &x_trf, &down3);
    reduce_down_rows_12(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V237: `feedforward_v181` with the down projection on four 3,840-column chunks (see
/// `DownRowsByColumns4`). Everything up to and including the geglu is unchanged; only the
/// geglu-output scratch layout, the inv_s broadcast and the down stage move.
pub(crate) fn feedforward_v237(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down4_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns4, m![H % 60, L / 16 % 240]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down4_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down4_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down4_rows_12(ctx, down_weight_packed, 48);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm4(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down4(ctx, &inv_s_hbm);

    // Each slice loads only its 3840-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns4, m![Dummy2, L % 3840]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns4, m![1], m![Dummy2, L % 3840]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 120], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 120], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows4, m![H % 60]> = DmTensor::new();
    let mut down_partials: DmTensor<f32, Chip, DownClusters, DownRowsByColumns4, m![H % 60, L / 16 % 240]> =
        DmTensor::new();
    contract_down4_rows_16(ctx, &x_trf, &down0, 0, &mut down_partials);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 0, &mut down);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 8, &mut down);
    contract_down4_rows_16(ctx, &x_trf, &down1, 16, &mut down_partials);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 16, &mut down);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 24, &mut down);
    contract_down4_rows_16(ctx, &x_trf, &down2, 32, &mut down_partials);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 32, &mut down);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 40, &mut down);
    contract_down4_rows_12(ctx, &x_trf, &down3, 48, &mut down_partials);
    reduce_down4_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 48, &mut down);
    reduce_down4_rows_4(ctx, &down_partials, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}


/// V240: the up/gate contraction with the lanes interleaved instead of sequential.
fn contract_up_gate_full_int(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]>,
    packed: &DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]>,
) -> DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> {
    ctx.main
        .begin(packed.view())
        .fetch::<m![L % 30, H / 64, Dummy2], m![H % 64]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 30, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![L % 30, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![L % 30, H / 64]>()
        .contract_lane::<m![L % 30, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Interleaved)
        .commit_trim::<m![H / 16 % 4]>()
        .commit()
}

/// Pass B for one tile of `$rows` rows: block scales and the column reduction, one f32 scalar
/// packet per row (no transpose, so any tile height works).
/// V240: `feedforward_v181` with the up/gate lanes interleaved.
pub(crate) fn feedforward_v240(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_8(ctx, down_weight_packed, 48);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full_int(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full_int(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_8(ctx, &x_trf, &down3);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);
    let p = contract_down_rows_4(ctx, &x_trf, &down4);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

// ---------------------------------------------------------------------------------------------
// V241: the down projection's pass A in two 30-row tiles instead of five, with pass B tiling the
// partials buffer on its own.
//
// Pass A and pass B are tied together today only because one macro generates both at the same
// tile height, and pass B is capped at 16 rows by the 8 KB scale VRF. Pass A has no such cap: it
// writes into a tile of a 60-row partials buffer, exactly as V181's up/gate stage does. Every
// pass A reloads the 4 KB f4 lookup table (838 cycles of DMA) and every tile load carries ~550
// cycles of fixed DMA cost, so five tiles cost ~6.9k of static DMA where two cost ~2.8k. What it
// gives up is pipelining: the first contraction now waits for 30 rows of weight instead of 16.
macro_rules! down8_contract_fns {
    ($load:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]> {
            packed
                .view()
                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]>,
            packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]>,
            offset: usize,
            partials: &mut DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]>,
        ) {
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 = $rows, L / 64 % 30, Dummy2, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64], _, _, _>(x_trf)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 = $rows, L / 64 % 30]>()
                .contract_lane::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![L / 16 % 4]>()
                .commit_view(partials.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60, L / 16 % 120]>(offset));
        }
    };
}
down8_contract_fns!(load_down8_rows_30, contract_down8_rows_30, 30);

macro_rules! down8_reduce_fns {
    ($reduce:ident, $rows:literal) => {
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]>,
            inv_s_vrf: &VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L / 16 % 120]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 120]>(offset))
                .fetch::<m![H % 60 = $rows], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 120]>(offset))
                .fetch::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), inv_s_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<DownRows, m![H % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![H % 60 = $rows / 4], m![H % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![H % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));
        }
    };
}
down8_reduce_fns!(reduce_down8_rows_16, 16);
down8_reduce_fns!(reduce_down8_rows_12, 12);

/// V241: `feedforward_v181` with the down pass A in two 30-row tiles.
pub(crate) fn feedforward_v241(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down8_rows_30(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down8_rows_30(ctx, down_weight_packed, 30);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let mut down_partials: DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        DmTensor::new();
    contract_down8_rows_30(ctx, &x_trf, &down0, 0, &mut down_partials);
    reduce_down8_rows_16(ctx, &down_partials, &down_scale, &inv_s_vrf, 0, &mut down);
    reduce_down8_rows_12(ctx, &down_partials, &down_scale, &inv_s_vrf, 16, &mut down);
    contract_down8_rows_30(ctx, &x_trf, &down1, 30, &mut down_partials);
    reduce_down8_rows_16(ctx, &down_partials, &down_scale, &inv_s_vrf, 28, &mut down);
    reduce_down8_rows_16(ctx, &down_partials, &down_scale, &inv_s_vrf, 44, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V242 sweep: down tiles 8/16/16/16/4 - a small first tile so the contraction starts early, and a small last one so little trails the final load.
pub(crate) fn feedforward_s1(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_8(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 8);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 24);
    let down3 = load_down_rows_16(ctx, down_weight_packed, 40);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_8(ctx, &x_trf, &down0);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 8, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 24, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down3);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 40, &mut down);
    let p = contract_down_rows_4(ctx, &x_trf, &down4);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V242 sweep: down tiles 4/8/16/16/16 - smallest first, growing.
pub(crate) fn feedforward_s2(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_4(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_8(ctx, down_weight_packed, 4);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 12);
    let down3 = load_down_rows_16(ctx, down_weight_packed, 28);
    let down4 = load_down_rows_16(ctx, down_weight_packed, 44);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_4(ctx, &x_trf, &down0);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_8(ctx, &x_trf, &down1);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 4, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 12, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down3);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 28, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down4);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 44, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V245: the down block scales loaded as whole rows and distributed by a switch.
///
/// The schedule prices this exactly: `up_scale` and `down_scale` are the same 3.69 MB, but
/// up/gate holds whole rows (7,200 B runs, util 0.556, 4,880 cycles) while down holds each
/// slice's 120-block column chunk (120 B runs, util 0.277, 9,774 cycles). A run under 256 B
/// fetches a whole granule for a fraction of it, and down pays that penalty 60 times per slice.
/// Loading the scale in the row layout (32 live slices, 57,600 B contiguous each) and moving
/// each chunk to its slice with one switch keeps the bytes and drops the penalty: ~4.9k of
/// static DMA, which at ffn's 2.1x model-to-hardware ratio is ~10k real.
fn load_down_scale_by_rows(
    ctx: &mut Context,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> {
    let rows: DmTensor<f8e4m3, Chip, DownClusters, DownRows, m![H % 60, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let rows: DmTensorView<'_, f8e4m3, Chip, DownClusters, DownRows, m![H % 60, L / 1920, L / 16 % 120]> =
        unsafe { rows.view().reshape() };
    ctx.main
        .begin(rows)
        .fetch::<m![H % 60, L / 1920, L / 128 % 15], m![L / 16 % 8]>()
        .switch::<DownRowsByColumns, m![H % 60, L / 128 % 15]>(SwitchConfig::Broadcast1 { slice1: 1, slice0: 8 })
        .collect::<m![H % 60, L / 128 % 15], m![L / 16 % 8]>()
        .commit_trim::<m![L / 16 % 8]>()
        .commit()
}

/// V245: `feedforward_v181` with the down block scales loaded as whole rows.
pub(crate) fn feedforward_v245(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale = load_down_scale_by_rows(ctx, down_weight_scale);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_8(ctx, down_weight_packed, 48);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_8(ctx, &x_trf, &down3);
    reduce_down_rows_8(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);
    let p = contract_down_rows_4(ctx, &x_trf, &down4);
    reduce_down_rows_4(ctx, &p, &down_scale, &inv_s_vrf, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

// ---------------------------------------------------------------------------------------------
// V246: the down projection on four 3,840-column chunks with 30 rows per slice.
//
// V237 tried four chunks with 60 rows and lost 17% because 32 row groups x 4 chunks is only 128
// slices: each one contracted twice the columns, and V235's "ffn's Main hides behind the stream"
// turned out to be one-directional -- below the envelope it buys nothing, above it costs full
// price. Sixty-four row groups x four chunks is 256 slices, so the work per slice is exactly what
// it is today (30 x 3,840 = 60 x 1,920) and only the run lengths change:
//
//   down weight  960 B -> 1,920 B runs   util 0.701 -> ~0.88   (~ -6.3k static)
//   block scale  120 B -> 240 B runs     util 0.277 -> ~0.55   (~ -4.9k static)
//   x per slice  3,840 -> 7,680 B                              (~ +2.2k static)
//
// The 1,920 B run is the reason: aligned it touches 8 granules (1,920 / 256 = 7.5 -> 8) and
// misaligned it starts at byte 128 and ends at 2,048, so it touches 8 as well. The 960 B run
// touches 5 for 4 bytes' worth in six chunks out of eight, which is what V211 measured at 27%.
//
// Thirty rows do not fit the 4-row transpose, so pass B commits one scalar packet per row and a
// ring-2 gather pairs the row groups back into the 60-row dense vector the tail already expects
// -- the same shape V181 uses for up/gate.
pub(crate) type DownRows30 = m![H / 30 % 64, 1 # 4];
pub(crate) type DownRowsByColumns30 = m![H / 30 % 64, L / 3840];

stage_packet_fns!(stage_packet_down30, DownClusters, DownRowsByColumns30);

macro_rules! down30_contract_fns {
    ($load:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns30, m![H % 30 = $rows, L % 3840]> {
            packed
                .view()
                .tile::<m![H % 30], $rows, m![H / 30, H % 30 = $rows # 30, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns30, m![1], m![Dummy2, L % 3840]>,
            packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns30, m![H % 30 = $rows, L % 3840]>,
            offset: usize,
            partials: &mut DmTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![H % 30, L / 16 % 240]>,
        ) {
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 30 = $rows, L / 64 % 60, Dummy2], m![L % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 30 = $rows, L / 64 % 60, Dummy2, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 30 = $rows, L / 64 % 60, Dummy2], m![L % 64], _, _, _>(x_trf)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 30 = $rows, L / 64 % 60]>()
                .contract_lane::<m![H % 30 = $rows, L / 64 % 60], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![L / 16 % 4]>()
                .commit_view(partials.view_mut().tile::<m![H % 30], $rows, m![H % 30 = $rows #{!} 30, L / 16 % 240]>(offset));
        }
    };
}
down30_contract_fns!(load_down30_rows_8, contract_down30_rows_8, 8);
down30_contract_fns!(load_down30_rows_6, contract_down30_rows_6, 6);

/// Pass B: block scales, column reduction and the cross-slice sum of the four column chunks,
/// committed as one scalar packet per row (no transpose, so 6-row tiles are fine).
macro_rules! down30_reduce_fns {
    ($reduce:ident, $rows:literal) => {
        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![H % 30, L / 16 % 240]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns30, m![H % 30, L / 16 % 240]>,
            inv_s_vrf: &VrfTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![1 # 8]>,
            offset: usize,
            out: &mut DmTensor<f32, Chip, DownClusters, DownRows30, m![H % 30, 1 # 8]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![H % 30 = $rows, L / 16 % 240]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![H % 30], $rows, m![H % 30 = $rows # 30, L / 16 % 240]>(offset))
                .fetch::<m![H % 30 = $rows], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 30 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view().tile::<m![H % 30], $rows, m![H % 30 = $rows # 30, L / 16 % 240]>(offset))
                .fetch::<m![H % 30 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .collect::<m![H % 30 = $rows, L / 128 % 30], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 30 = $rows, L / 64 % 60], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), inv_s_vrf)
                .vector_intra_slice_reduce::<L, m![H % 30 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<DownRows30, m![H % 30 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit_view(out.view_mut().tile::<m![H % 30], $rows, m![H % 30 = $rows #{!} 30, 1 # 8]>(offset));
        }
    };
}
down30_reduce_fns!(reduce_down30_rows_8, 8);
down30_reduce_fns!(reduce_down30_rows_6, 6);

/// The row-scalar packets of two neighbouring row groups gathered onto one slice and packed into
/// the dense 60-row bf16 vector the post-FF RMSNorm already consumes (`gather_pack_full`'s shape).
fn gather_pack_down30(
    ctx: &mut Context,
    g: &DmTensor<f32, Chip, DownClusters, DownRows30, m![H % 30, 1 # 8]>,
) -> DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> {
    let gathered: DmTensor<f32, Chip, DownClusters, DownRows, m![H % 30, H / 30 % 2, 1 # 8]> = ctx
        .main
        .begin(g.view())
        .fetch::<m![H % 30], m![1 # 8]>()
        .switch::<DownRows, m![H % 30, H / 30 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 4 })
        .collect::<m![H % 30, H / 30 % 2], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    ctx.main
        .begin(gathered.view())
        .fetch::<m![H / 30 % 2, H % 30], m![1 # 8]>()
        .collect::<m![H % 60], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 / 4], m![H % 60 % 4 # 16]>()
        .commit_trim::<m![H % 60 % 4]>()
        .commit()
}

/// `broadcast_inv_s_down` for the 30-row four-chunk layout: a down slice's chunk came from
/// up/gate cluster `chunk / 2`, which is bit 1 of the slice index (slice = row_group * 4 + chunk).
fn broadcast_inv_s_down30(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>,
) -> VrfTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![1 # 8]> {
    let two: DmTensor<f32, Chip, DownClusters, m![1 # 128, L / 7680], m![1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, DownClusters, m![Dummy256 / 4, L / 7680, Dummy2], m![1 # 8]> = ctx
        .main
        .begin(two.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256 / 4, L / 7680, Dummy2], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![1 # 8]> = unsafe { all.reshape() };
    stage_packet_down30(ctx, &all)
}

/// V246: `feedforward_v181` with the down projection on 30 rows x four column chunks.
pub(crate) fn feedforward_v246(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down30_rows_8(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns30, m![H % 30, L / 16 % 240]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down30_rows_8(ctx, down_weight_packed, 8);
    let down2 = load_down30_rows_8(ctx, down_weight_packed, 16);
    let down3 = load_down30_rows_6(ctx, down_weight_packed, 24);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm4(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down30(ctx, &inv_s_hbm);

    // Each slice loads only its 3840-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns30, m![Dummy2, L % 3840]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns30, m![1], m![Dummy2, L % 3840]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 120], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 120], m![L % 32]>()
        .to_trf();

    let mut packets: DmTensor<f32, Chip, DownClusters, DownRows30, m![H % 30, 1 # 8]> = DmTensor::new();
    let mut down_partials: DmTensor<f32, Chip, DownClusters, DownRowsByColumns30, m![H % 30, L / 16 % 240]> =
        DmTensor::new();
    contract_down30_rows_8(ctx, &x_trf, &down0, 0, &mut down_partials);
    reduce_down30_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 0, &mut packets);
    contract_down30_rows_8(ctx, &x_trf, &down1, 8, &mut down_partials);
    reduce_down30_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 8, &mut packets);
    contract_down30_rows_8(ctx, &x_trf, &down2, 16, &mut down_partials);
    reduce_down30_rows_8(ctx, &down_partials, &down_scale, &inv_s_vrf, 16, &mut packets);
    contract_down30_rows_6(ctx, &x_trf, &down3, 24, &mut down_partials);
    reduce_down30_rows_6(ctx, &down_partials, &down_scale, &inv_s_vrf, 24, &mut packets);
    let down = gather_pack_down30(ctx, &packets);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// V248: the geglu's two scalars in one HBM scratch, so the down-stream pays one load and one
/// ring-256 switch instead of two of each. `broadcast_scalar_full` costs a 521-cycle load plus a
/// switch pass with a 719-cycle config, and it runs twice for two f32 packets.
fn broadcast_scalar_pair_full(
    ctx: &mut Context,
    v: &HbmTensor<f32, Chip, m![Dummy2, 1 # 8]>,
) -> (
    VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]>,
    VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]>,
) {
    let one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![Dummy2, 1 # 8]> = v.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![Dummy2, 1 # 8]> = ctx
        .main
        .begin(one.view())
        .fetch::<m![Dummy2], m![1 # 8]>()
        .switch::<m![Dummy256], m![Dummy2]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Dummy2], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, 1 # 8]> = unsafe { all.reshape() };
    let first: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]> = ctx
        .sub
        .begin(all.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let second: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![1 # 8]> = ctx
        .sub
        .begin(all.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, 1 # 8]>(1))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    (first, second)
}

/// `geglu_full` taking both scalars from one scratch.
fn geglu_full_pair(
    ctx: &mut Context,
    up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
    gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]>,
    scales: &HbmTensor<f32, Chip, m![Dummy2, 1 # 8]>,
) -> DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> {
    let (erf_scale_vrf, out_scale_vrf) = broadcast_scalar_pair_full(ctx, scales);

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &erf_scale_vrf)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &out_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// V248: `stage_x_hi_lo_hbm_full` writing the two geglu scalars into one scratch.
pub(crate) fn stage_x_hi_lo_hbm_pair(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (
    HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    HbmTensor<f32, Chip, m![Dummy2, 1 # 8]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let m_local = max_square_reducing(ctx, &x);
    let m_all: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, inv_s) = pow2_scale_reducing(ctx, &m_all);
    let s_vrf = stage_packet_reducing(ctx, &s);
    let inv_s_vrf = stage_packet_reducing(ctx, &inv_s);

    let (x_hi, x_lo) = hi_lo_reducing(ctx, &x, &s_vrf);
    // m1: the two pieces are gathered into one buffer on the slices that already hold them, so
    // the staging costs one DMA command instead of two.
    let mut x2: DmTensor<f8e4m3, Chip, Cluster, ReducingSlices, m![Dummy2, H % 480]> = DmTensor::new();
    ctx.main
        .begin(x_hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(0));
    ctx.main
        .begin(x_lo.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(x2.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(1));
    let mut x2_hbm: HbmTensor<f8e4m3, Chip, m![Dummy2, H]> = HbmTensor::new();
    x2.view().to_hbm_view(&mut ctx.tdma, x2_hbm.view_mut());

    // The geglu scalars.
    let s_up: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = up_global_scale.to_dm(&mut ctx.tdma);
    let s_gate: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = gate_global_scale.to_dm(&mut ctx.tdma);
    let s_gate_vrf = stage_packet_reducing(ctx, &s_gate);
    let erf_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_gate.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_s_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let half_inv_s2_vrf = stage_packet_reducing(ctx, &half_inv_s2);
    let out_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(s_up.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_gate_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &half_inv_s2_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let erf_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { erf_scale.reshape() };
    let out_one: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = unsafe { out_scale.reshape() };
    let mut scales_hbm: HbmTensor<f32, Chip, m![Dummy2, 1 # 8]> = HbmTensor::new();
    erf_one
        .view()
        .to_hbm_view(&mut ctx.tdma, scales_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(0));
    out_one
        .view()
        .to_hbm_view(&mut ctx.tdma, scales_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(1));
    (x2_hbm, scales_hbm)
}

/// V248: the four-tile down stage with the geglu scalars broadcast once.
pub(crate) fn feedforward_v248(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    scales: &HbmTensor<f32, Chip, m![Dummy2, 1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_12(ctx, down_weight_packed, 48);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full_pair(ctx, up, gate, scales);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_down_rows_12(ctx, &x_trf, &down3);
    reduce_down_rows_12(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

// ---------------------------------------------------------------------------------------------
// V250: the down projection splitting L across the clusters instead of H.
//
// Today up/gate splits L across the two clusters (`UpGateClusters = m![L / 7680]`) and down splits
// H (`DownClusters = m![H / 1920]`). Because of that mismatch every down slice needs a column chunk
// of the *whole* L, half of which was produced on the other cluster -- and a cross-cluster move has
// to go through HBM. That is the only reason the geglu output is staged out and back, which the
// schedule prices at 7,522 cycles of static DMA: stores 1,653 + 1,653 + 755 and loads 2,934 + 527.
//
// Splitting L for down too puts producer and consumer on the same cluster, so the geglu output
// reaches the down slices over the switch and never leaves the chip, and `inv_s` -- which is
// already per-cluster -- stops needing its own scratch entirely.
//
// The shape is chosen so nothing V237/V246 killed comes back: 64 row groups x 4 chunks is still
// 256 live slices, still 60 rows per slice, still 1,920-column chunks and therefore still the same
// 960 B weight runs and the same x bytes per slice. Only the cluster axis moves.
//
// The price is that each cluster now holds a *partial* [H] instead of a complete half, so the two
// have to be added: each stores its [H], both load both halves back, and the add folds into the
// global-scale pass the tail already runs.
pub(crate) type DownClustersL = m![L / 7680];
pub(crate) type DownRowsL = m![H / 60 % 64, 1 # 4];
pub(crate) type DownRowsByColumnsL = m![H / 60 % 64, L / 1920 % 4];

/// The geglu output as two f8 pieces in one buffer, so a single switch carries both.
fn hi_lo_gathered_pair(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]>,
    s_vrf: &VrfTensor<f32, Chip, UpGateClusters, UpGateRowsGathered, m![1 # 8]>,
) -> DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsGathered, m![Dummy2, L % 480]> {
    let (x_hi, x_lo) = hi_lo_gathered(ctx, x, s_vrf);
    let mut pair: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsGathered, m![Dummy2, L % 480]> =
        DmTensor::new();
    ctx.main
        .begin(x_hi.view())
        .fetch::<m![L / 32 % 15], m![L % 32]>()
        .collect::<m![L / 32 % 15], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, L % 480]>(0));
    ctx.main
        .begin(x_lo.view())
        .fetch::<m![L / 32 % 15], m![L % 32]>()
        .collect::<m![L / 32 % 15], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, L % 480]>(1));
    pair
}

/// The geglu pieces from the 16 gathered slices to the 256 down slices, without touching HBM:
/// first four gathered slices concatenate into one 1,920-column chunk, then the four chunks are
/// broadcast over the 64 row groups that all need them.
fn geglu_to_down_slices(
    ctx: &mut Context,
    pair: &DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsGathered, m![Dummy2, L % 480]>,
) -> DmTensor<f8e4m3, Chip, DownClustersL, DownRowsByColumnsL, m![Dummy2, L % 1920]> {
    // The ring delivers packet-major -- every slice's packet r before packet r + 1 (V167 read it
    // slice-major and came out permuted) -- so the gathered buffer is [packet, source slice] and
    // the second pass's fetch reads it back in L order, which costs nothing.
    let chunks: DmTensor<f8e4m3, Chip, UpGateClusters, m![1 # 64, L / 1920 % 4], m![Dummy2, L / 32 % 15, L / 480 % 4, L % 32]> = ctx
        .main
        .begin(pair.view())
        .fetch::<m![Dummy2, L / 32 % 15], m![L % 32]>()
        .switch::<m![1 # 64, L / 1920 % 4], m![Dummy2, L / 32 % 15, L / 480 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![Dummy2, L / 32 % 15, L / 480 % 4], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    // L % 1920 = (L / 480 % 4) * 480 + (L / 32 % 15) * 32 + (L % 32), so this fetch order makes the
    // broadcast's output element natural L order with no repacking pass.
    let all: DmTensor<f8e4m3, Chip, UpGateClusters, m![Dummy256 / 4, L / 1920 % 4], m![Dummy2, L / 480 % 4, L / 32 % 15, L % 32]> = ctx
        .main
        .begin(chunks.view())
        .fetch::<m![Dummy2, L / 480 % 4, L / 32 % 15], m![L % 32]>()
        .switch::<m![Dummy256 / 4, L / 1920 % 4], m![Dummy2, L / 480 % 4, L / 32 % 15]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Dummy2, L / 480 % 4, L / 32 % 15], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    unsafe { all.reshape() }
}

macro_rules! downl_tile_fns {
    ($load:ident, $contract:ident, $reduce:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60 = $rows, L % 1920]> {
            packed
                .view()
                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<f8e4m3, Chip, DownClustersL, DownRowsByColumnsL, m![1], m![Dummy2, L % 1920]>,
            packed: &DmTensor<f4e2m1, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60 = $rows, L % 1920]>,
        ) -> DmTensor<f32, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60 = $rows, L / 16 % 120]> {
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 = $rows, L / 64 % 30, Dummy2, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 = $rows, L / 64 % 30, Dummy2], m![L % 64], _, _, _>(x_trf)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 = $rows, L / 64 % 30]>()
                .contract_lane::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![L / 16 % 4]>()
                .commit()
        }

        fn $reduce(
            ctx: &mut Context,
            partials: &DmTensor<f32, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60 = $rows, L / 16 % 120]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60, L / 16 % 120]>,
            inv_s_vrf: &VrfTensor<f32, Chip, DownClustersL, DownRowsByColumnsL, m![1 # 8]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, DownClustersL, DownRowsL, m![H % 60]>,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60 = $rows, L / 16 % 120]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 120]>(offset))
                .fetch::<m![H % 60 = $rows], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(partials.view())
                .fetch::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 = $rows, L / 64 % 30], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), inv_s_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<DownRowsL, m![H % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![H % 60 = $rows / 4], m![H % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![H % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));
        }
    };
}
downl_tile_fns!(load_downl_rows_16, contract_downl_rows_16, reduce_downl_rows_16, 16);
downl_tile_fns!(load_downl_rows_12, contract_downl_rows_12, reduce_downl_rows_12, 12);

stage_packet_fns!(stage_packet_downl, DownClustersL, DownRowsByColumnsL);

/// The two clusters' partial [H] sums, added and scaled in one pass. With L split across the
/// clusters each one produces a partial sum over its own half, so this replaces the single-owner
/// store the H-split used; the global scale folds in where it already ran.
pub(crate) fn combine_down_partials(
    ctx: &mut Context,
    partials: &HbmTensor<bf16, Chip, m![L / 7680, H]>,
    down_global_scale: &DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let a: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = partials
        .view()
        .tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H]>(0)
        .to_dm(&mut ctx.tdma);
    let b: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = partials
        .view()
        .tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H]>(1)
        .to_dm(&mut ctx.tdma);

    let b_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(a.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &b_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// V250: the four-tile down stage with L split across the clusters.
pub(crate) fn feedforward_v250(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (
    HbmTensor<bf16, Chip, m![L / 7680, H]>,
    DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>,
) {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_downl_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClustersL, DownRowsByColumnsL, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_downl_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_downl_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_downl_rows_12(ctx, down_weight_packed, 48);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);

    // The geglu output reaches the down slices over the switch: producer and consumer are now on
    // the same cluster, so nothing goes through HBM and `inv_s` stays where it was computed.
    let m_local = max_square_gathered(ctx, &x);
    let m_every: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![L / 480 % 16, 1 # 8]> = ctx
        .main
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![L / 480 % 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]> = ctx
        .sub
        .begin(m_every.view())
        .fetch::<m![L / 480 % 16], m![1 # 8]>()
        .collect::<m![L / 480 % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let (s_all, inv_s_all) = pow2_scale_gathered_all(ctx, &m_all);
    let s: DmTensor<f32, Chip, UpGateClusters, UpGateRowsGathered, m![1 # 8]> = unsafe { s_all.reshape() };
    let s_vrf = stage_packet_gathered(ctx, &s);
    let pair = hi_lo_gathered_pair(ctx, &x, &s_vrf);
    let x = geglu_to_down_slices(ctx, &pair);

    let inv_s: DmTensor<f32, Chip, DownClustersL, DownRowsByColumnsL, m![1 # 8]> = unsafe { inv_s_all.reshape() };
    let inv_s_vrf = stage_packet_downl(ctx, &inv_s);

    let x_trf: TrfTensor<f8e4m3, Chip, DownClustersL, DownRowsByColumnsL, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClustersL, DownRowsL, m![H % 60]> = DmTensor::new();
    let p = contract_downl_rows_16(ctx, &x_trf, &down0);
    reduce_downl_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_downl_rows_16(ctx, &x_trf, &down1);
    reduce_downl_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_downl_rows_16(ctx, &x_trf, &down2);
    reduce_downl_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let p = contract_downl_rows_12(ctx, &x_trf, &down3);
    reduce_downl_rows_12(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);

    // Each cluster owns a partial [H] over its half of L; the caller adds the two halves in the
    // pass that already applies the global scale.
    let mut partials_hbm: HbmTensor<bf16, Chip, m![L / 7680, H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, partials_hbm.view_mut());

    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    (partials_hbm, down_global_scale)
}


/// V260: the down output stored in two pieces so the first fills the trailing-compute hole.
pub(crate) fn feedforward_v260(
    ctx: &mut Context,
    x2: &HbmTensor<f8e4m3, Chip, m![Dummy2, H]>,
    erf_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_scale: &HbmTensor<f32, Chip, m![1 # 8]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    // V181: whole rows per slice - one contiguous segment each for the f4 rows and the block
    // scales - and x replicated to every slice by 8 copies per cluster plus a ring-32 switch.
    let up_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_w: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_12(ctx, down_weight_packed, 48);

    // V206: sixty-four copies per cluster and a ring of 4, not eight copies and a ring of 32.
    // The switch is pure movement on MainContext, and ffn's MainContext (83.5k static cycles) is
    // the resource that actually binds this kernel, so trading ring cycles for DMA descriptors
    // pays: 3/3 Arena jobs, -16,351 cycles (-5.3%). The static makespan predicts the opposite
    // (+144), which is the point - the schedule believes this work hides behind the weight
    // stream and on hardware it does not. qkv is a different case (its Main is small next to its
    // stream, and its x2 region is re-read by every copy), so qkv keeps ring 32.
    let x8: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, 1 # 4], m![Dummy2, H]> = x2.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, m![C, Dummy256 / 64], m![Dummy2, H]> = ctx
        .main
        .begin(x8.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .switch::<m![C, Dummy256 / 64], m![Dummy2, H / 32]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![Dummy2, H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsFull, m![1], m![Dummy2, H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();

    let up_partials = contract_up_gate_full(ctx, &x_trf, &up_w);
    let gate_partials = contract_up_gate_full(ctx, &x_trf, &gate_w);
    let mut up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsFull, m![L % 30, 1 # 8]> = DmTensor::new();
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 0, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 0, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 8, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 8, &mut gate);
    reduce_up_gate_full_8(ctx, &up_partials, &up_scale, 16, &mut up);
    reduce_up_gate_full_8(ctx, &gate_partials, &gate_scale, 16, &mut gate);
    reduce_up_gate_full_6(ctx, &up_partials, &up_scale, 24, &mut up);
    reduce_up_gate_full_6(ctx, &gate_partials, &gate_scale, 24, &mut gate);

    let g = geglu_full(ctx, up, gate, erf_scale, out_scale);
    let x = gather_pack_full(ctx, &g);
    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm(ctx, &x);
    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);

    // Each slice loads only its 1920-wide chunk of the geglu output (both f8 pieces) from HBM.
    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![Dummy2, L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .collect::<m![Dummy2, L / 32 % 60], m![L % 32]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let p = contract_down_rows_16(ctx, &x_trf, &down0);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down1);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);
    let p = contract_down_rows_16(ctx, &x_trf, &down2);
    reduce_down_rows_16(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);
    let mut down_hbm: HbmTensor<bf16, Chip, m![H / 60, H % 60]> = HbmTensor::new();
    down.view()
        .tile::<m![H % 60], 48, m![H % 60 = 48 # 60]>(0)
        .to_hbm_view(
            &mut ctx.tdma,
            down_hbm.view_mut().tile::<m![H % 60], 48, m![H / 60, H % 60 = 48 #{!} 60]>(0),
        );
    let p = contract_down_rows_12(ctx, &x_trf, &down3);
    reduce_down_rows_12(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    // V260: the first three tiles' rows go out while the last tile is still contracting. The
    // schedule shows a 3,245-cycle hole at the end of the weight stream where the DMA engine
    // waits for that last pass A (3,863 Main) and pass B, and the 2,510-cycle output store then
    // runs after it. Storing rows 0..48 into that hole leaves only ~500 cycles of store on the
    // critical path. The split is aligned with the tile boundary on purpose -- V253 tried this on
    // attn_out with a 60/60 split that did *not* line up with its 88/32 tiles, so the first store
    // still had to wait for the second tile and the experiment only measured the extra command.
    down.view()
        .tile::<m![H % 60], 12, m![H % 60 = 12 # 60]>(48)
        .to_hbm_view(
            &mut ctx.tdma,
            down_hbm.view_mut().tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 #{!} 60]>(48),
        );
    let down_hbm: HbmTensor<bf16, Chip, m![H]> = unsafe { down_hbm.reshape() };
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}
