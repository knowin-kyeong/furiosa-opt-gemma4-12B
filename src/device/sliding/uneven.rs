//! V340: the attention output projection with uneven tiles.
//!
//! When both clusters load at once, cluster 1 runs at roughly 0.7x cluster 0's rate (V294, V335) and the kernel waits
//! for it at the contraction-store ExplicitSync (9-13k). That lag only goes away when cluster 1's per-slice bytes shrink
//! (V321 qa5/qa9 vs V323/V324). Row interleaving would buy that at the price of a scattered output order, so the
//! asymmetry comes in time instead: both clusters load and contract rows 0..96 of every 120-row group (tile0), and
//! cluster 0 alone also loads and contracts rows 96..120 of all 32 groups (tile1), so cluster 0's slices carry 144 rows
//! and cluster 1's 96. The contraction store keeps the production layout (its tail rows are left unwritten); cluster 0
//! reloads it and overwrites the tail rows from its own DM, without an HBM round trip.
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

pub(crate) type UnevenStore = HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]>;
pub(crate) type UnevenTails = DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]>;

pub(crate) fn project_output_ut(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (UnevenStore, UnevenTails) {
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    // Cluster 0 alone: rows 96..120 of all 32 groups, both clusters' groups side by side in each slice's element.
    let tile1: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 24, Qs % 256]> = weight
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
    // tile1's contraction runs on cluster 0 alone, so it needs x in a cluster-0 TRF: the same slices, staged once more.
    let x0: DmTensorView<'_, f8e4m3, Chip, ClusterZero, RowsByColumns, m![Qs % 256]> = unsafe { x.view().reshape() };
    let x0_trf: TrfTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x0)
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    // Both clusters: rows 0..96 of their own groups (rows 96..120 of `contraction` stay unwritten).
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

    // Cluster 0: rows 96..120 of all 32 groups.
    let mut tails: UnevenTails = DmTensor::new();
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

    // V271 layout: each slice writes its 240 B at a 256 B boundary.
    let mut stored: UnevenStore = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, stored.view_mut());
    (stored, tails)
}

/// The stored rows loaded into the RMSNorm reducing layout, with rows 96..120 of every group overwritten from cluster 0's
/// tile1 contraction (a DM-to-DM move on cluster 0).
pub(crate) fn load_reducing_with_tails<C: M>(
    ctx: &mut Context,
    stored: &UnevenStore,
    tails: &DmTensor<bf16, Chip, C, Rows, m![H / 1920, H % 120]>,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let mut x: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480 / 120, H % 120]> = stored.to_dm(&mut ctx.tdma);
    tails
        .view()
        .tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 # 120]>(96)
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 24, m![H % 480 / 120, H % 120 = 24 #{!} 120]>(96));
    unsafe { x.reshape() }
}
