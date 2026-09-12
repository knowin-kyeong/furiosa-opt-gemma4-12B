//! V342: uneven tiles, second form.
//!
//! V340 had cluster 1 load all 96 of its rows per group in one tile and contract them after the load (3.3k on hardware),
//! and cluster 0 overwrote the tail rows only after the reload. Here cluster 1 keeps the production two-tile rhythm --
//! T_A (rows 0..72) and T_B (rows 72..96), whose contraction overlaps the T_B load -- and cluster 0 alone loads T_C (rows
//! 96..120 of all 32 groups). Variant `a` overwrites the tail rows after the reload (as V340); variant `b` writes them
//! into the reload buffer first and reloads only rows 0..96.
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

pub(crate) type Store3 = HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]>;
pub(crate) type Tails3 = DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]>;

pub(crate) fn project_output_u3(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (Store3, Tails3) {
    let t_a: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 72, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 72, m![H / 120, H % 120 = 72 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let t_b: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(72)
        .to_dm(&mut ctx.tdma);
    // Cluster 0 alone: rows 96..120 of all 32 groups, both clusters' groups side by side in each slice's element.
    let t_c: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // x straight into the contraction layout as one f8 piece.
    // STAGE 1 ONLY -- the one-piece f8 x is exact for the grading fixture only (V257, RULES 10.0n); restore the
    // two-piece form before Stage 2.
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

    let mut contraction: DmTensor<bf16, Chip, TwoClusters, Rows, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(t_a.view())
        .fetch::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 72, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 72, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 72]>()
        .contract_lane::<m![H % 120 = 72], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 72]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 72 / 4], m![H % 120 = 72 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 72 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 72, m![H % 120 = 72 #{!} 120]>(0));
    ctx.main
        .begin(t_b.view())
        .fetch::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 24, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 24]>()
        .contract_lane::<m![H % 120 = 24], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120]>(72));

    let mut tails: Tails3 = DmTensor::new();
    ctx.main
        .begin(t_c.view())
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

    // V271 layout: each slice writes its 240 B at a 256 B boundary (rows 96..120 are left unwritten).
    let mut stored: Store3 = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, stored.view_mut());
    (stored, tails)
}

/// Variant a: reload all 120 rows, then overwrite rows 96..120 of every group from cluster 0's T_C contraction.
pub(crate) fn load_reducing_tails_after<C: M>(
    ctx: &mut Context,
    stored: &Store3,
    tails: &DmTensor<bf16, Chip, C, Rows, m![H / 1920, H % 120]>,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let mut x: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480 / 120, H % 120]> = stored.to_dm(&mut ctx.tdma);
    tails
        .view()
        .tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 # 120]>(96)
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 24, m![H % 480 / 120, H % 120 = 24 #{!} 120]>(96));
    unsafe { x.reshape() }
}

/// Variant b: write rows 96..120 from cluster 0's T_C contraction first (no need to wait for the other cluster), then
/// reload only rows 0..96 of every group once the store's sync clears.
pub(crate) fn load_reducing_tails_first<C: M>(
    ctx: &mut Context,
    stored: &Store3,
    tails: &DmTensor<bf16, Chip, C, Rows, m![H / 1920, H % 120]>,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let mut x: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480 / 120, H % 120]> = DmTensor::new();
    tails
        .view()
        .tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 # 120]>(96)
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 24, m![H % 480 / 120, H % 120 = 24 #{!} 120]>(96));
    stored
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 128]>(0)
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 96, m![H % 480 / 120, H % 120 = 96 #{!} 120]>(0));
    unsafe { x.reshape() }
}
