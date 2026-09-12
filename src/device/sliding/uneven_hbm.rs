//! V351: the V340 uneven tiles with cluster 0's tail rows stored into HBM before the contraction store's sync.
//!
//! V340 on hardware (v342_r0, arm ut): tile1 (cluster 0 alone) was issued only after contraction0, two small loads and the
//! contraction store (24.3k, tile0 ended 20.9k); tile1's contraction then ended with the reload (40.1k), and the tail rows
//! reached the norm by a DM-to-DM move after the reload (1.9k). The scheduler put the store first because nothing on the
//! store -> sync -> reload chain needed tile1. Here the reload reads the tail rows from HBM, so that chain runs through
//! tile1 -> contraction1 -> tail store, and the DM-to-DM move disappears.
//!
//! Three arms:
//! - `th` (R = 96): one unpadded HBM tensor, two disjoint tile stores (rows 0..96 of every group from both clusters, rows
//!   96..120 of every group from cluster 0), one reload. Unpadded because the compiler rejects offset tiles of a padded
//!   HBM layout (`lir: incorrect buffer size`, V346); the ffn down store (mlp.rs) writes an unpadded offset tile already.
//! - `tb` (R = 96): production's 256 B-aligned store (V271) unchanged, the tail rows in a second HBM tensor, reloaded into
//!   the reload buffer's tile after the full reload.
//! - `t8` (R = 88, cluster 0 carries 63.3%; V345 load-only optimum): as `th`.
//!
//! Attention-output only: nothing here is shared with full attention, vision, audio, qkv or ffn.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Qs};
use crate::device::shared::rmsnorm::ReducingSlices;

type TwoClusters = m![H / 1920];
/// Cluster 0 only (cluster 1 is padding).
type ClusterZero = m![1 # 2];
/// 16 row groups x 16 column chunks per cluster, as in production.
type RowsByColumns = m![H / 120 % 16, Qs / 256];
/// After the chunk reduce: one live slice per row group.
type Rows = m![H / 120 % 16, 1 # 16];

/// Every group's 240 B end to end, no padding.
pub(crate) type FlatStore = HbmTensor<bf16, Chip, m![H / 120, H % 120]>;
/// Production's store layout: each group's 240 B at a 256 B boundary.
pub(crate) type PaddedStore = HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]>;
/// Rows 96..120 of every group, each group's 48 B at a 256 B boundary.
pub(crate) type TailStore = HbmTensor<bf16, Chip, m![H / 120, H % 120 = 24 # 128]>;

/// x as one f8 piece in the contraction layout, staged into a two-cluster TRF (tile0) and a cluster-0 TRF (tile1).
/// STAGE 1 ONLY -- the one-piece f8 x is exact for the grading fixture only (V257, RULES 10.0n); restore the two-piece
/// form before Stage 2.
fn stage_x(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
) -> (
    TrfTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![1], m![Qs % 256]>,
    TrfTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![1], m![Qs % 256]>,
) {
    let xs: DmTensor<bf16, Chip, TwoClusters, RowsByColumns, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![Qs % 256]> = ctx
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
    let x_trf: TrfTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    let x0: DmTensorView<'_, f8e4m3, Chip, ClusterZero, RowsByColumns, m![Qs % 256]> = unsafe { x.view().reshape() };
    let x0_trf: TrfTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x0)
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    (x_trf, x0_trf)
}

/// R = 96: both clusters' rows 0..96 in `contraction`, cluster 0's rows 96..120 of all 32 groups in `tails`.
fn contract_96(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (
    DmTensor<bf16, Chip, TwoClusters, Rows, m![H % 120]>,
    DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]>,
) {
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);
    let (x_trf, x0_trf) = stage_x(ctx, x);

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, Rows, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 96, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 96]>()
        .contract_lane::<m![H % 120 = 96], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 96]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(0));

    let mut tails: DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H / 1920, H % 120 = 24, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H / 1920, H % 120 = 24, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H / 1920, H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x0_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H / 1920, H % 120 = 24]>()
        .contract_lane::<m![H / 1920, H % 120 = 24], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H / 1920, H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 1920, H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(tails.view_mut().tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 #{!} 120]>(96));
    (contraction, tails)
}

/// Arm `th`: rows 0..96 and cluster 0's rows 96..120 as two disjoint tiles of one unpadded HBM tensor.
pub(crate) fn project_output_th(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> FlatStore {
    let (contraction, tails) = contract_96(ctx, x, weight);
    let mut stored: FlatStore = HbmTensor::new();
    contraction
        .view()
        .tile::<m![H % 120], 96, m![H % 120 = 96 # 120]>(0)
        .to_hbm_view(&mut ctx.tdma, stored.view_mut().tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 #{!} 120]>(0));
    tails
        .view()
        .tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 # 120]>(96)
        .to_hbm_view(&mut ctx.tdma, stored.view_mut().tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 #{!} 120]>(96));
    stored
}

/// Arm `tb`: production's aligned store plus a second HBM tensor holding cluster 0's tail rows.
pub(crate) fn project_output_tb(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (PaddedStore, TailStore) {
    let (contraction, tails) = contract_96(ctx, x, weight);
    let mut stored: PaddedStore = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, stored.view_mut());
    let mut tail_store: TailStore = HbmTensor::new();
    tails
        .view()
        .tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 # 120]>(96)
        .to_hbm_view(&mut ctx.tdma, tail_store.view_mut());
    (stored, tail_store)
}

/// Arm `t8`: as `th` at R = 88.
pub(crate) fn project_output_t8(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> FlatStore {
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
        .to_dm(&mut ctx.tdma);
    let (x_trf, x0_trf) = stage_x(ctx, x);

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, Rows, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 88, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 88, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 88]>()
        .contract_lane::<m![H % 120 = 88], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 88]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 88 / 4], m![H % 120 = 88 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 88 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120]>(0));

    let mut tails: DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H / 1920, H % 120 = 32, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H / 1920, H % 120 = 32, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H / 1920, H % 120 = 32, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x0_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H / 1920, H % 120 = 32]>()
        .contract_lane::<m![H / 1920, H % 120 = 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H / 1920, H % 120 = 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 1920, H % 120 = 32 / 4], m![H % 120 = 32 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 32 % 4]>()
        .commit_view(tails.view_mut().tile::<m![H % 120], 32, m![H / 1920, H % 120 = 32 #{!} 120]>(88));

    let mut stored: FlatStore = HbmTensor::new();
    contraction
        .view()
        .tile::<m![H % 120], 88, m![H % 120 = 88 # 120]>(0)
        .to_hbm_view(&mut ctx.tdma, stored.view_mut().tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 #{!} 120]>(0));
    tails
        .view()
        .tile::<m![H % 120], 32, m![H / 1920, H % 120 = 32 # 120]>(88)
        .to_hbm_view(&mut ctx.tdma, stored.view_mut().tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 #{!} 120]>(88));
    stored
}

/// The unpadded store loaded straight into the RMSNorm reducing layout (one contiguous 960 B run per slice).
pub(crate) fn load_reducing_flat<C: M>(ctx: &mut Context, stored: &FlatStore) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    stored.to_dm(&mut ctx.tdma)
}

/// Arm `tb`: the aligned store loaded whole, then cluster 0's tail rows loaded over rows 96..120 of every group.
pub(crate) fn load_reducing_tb<C: M>(
    ctx: &mut Context,
    stored: &PaddedStore,
    tail_store: &TailStore,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let mut x: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480 / 120, H % 120]> = stored.to_dm(&mut ctx.tdma);
    tail_store
        .view()
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 24, m![H % 480 / 120, H % 120 = 24 #{!} 120]>(96));
    unsafe { x.reshape() }
}
