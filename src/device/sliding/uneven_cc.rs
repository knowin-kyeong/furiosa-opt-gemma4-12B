//! V357: V340's uneven-tile projection (src/device/sliding/uneven.rs) with the two contraction passes committing bf16
//! through the Commit Adapter: the f32 contraction rows are transposed as f32 packets and cast in `commit_cast`, so the
//! Cast Engine stays out of both passes (V352 c measured the same move on the final norm pass). Types and rows are
//! otherwise identical to `project_output_ut`. Attention-output only.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Qs};
use crate::device::sliding::uneven::{UnevenStore, UnevenTails};

type TwoClusters = m![H / 1920];
/// Cluster 0 only (cluster 1 is padding).
type ClusterZero = m![1 # 2];
/// 16 row groups x 16 column chunks per cluster, as in production.
type RowsByColumns = m![H / 120 % 16, Qs / 256];
/// After the chunk reduce: one live slice per row group.
type Rows = m![H / 120 % 16, 1 # 16];

pub(crate) fn project_output_ut_cc(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (UnevenStore, UnevenTails) {
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // STAGE 1 ONLY -- the one-piece f8 x is exact for the grading fixture only (V257, RULES 10.0n).
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
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 8]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_cast::<bf16>()
        .commit_view(contraction.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(0));

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
        .transpose::<m![H / 1920, H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 8]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_cast::<bf16>()
        .commit_view(tails.view_mut().tile::<m![H % 120], 24, m![H / 1920, H % 120 = 24 #{!} 120]>(96));

    let mut stored: UnevenStore = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, stored.view_mut());
    (stored, tails)
}
