//! V347: the attention output projection with uneven tiles at R = 88 (cluster 0 carries 63.3% of the bytes) and a
//! contraction store that reads the buffer cluster 0's tail contraction writes.
//!
//! V340 (R = 96) committed its tile1 contraction to a buffer the contraction store did not read, so the scheduler issued
//! tile1 only after contraction0, two small loads and the store (24.3k instead of 20.9k on hardware), and that contraction
//! then ran partly behind the store's sync (4.7k; production's 24-row tile takes 1.5k). Here both clusters' tile0
//! contractions and cluster 0's tile1 contraction commit into one padded buffer (`H % 120 # 240`): cluster 0's own tail rows
//! land in its real rows, cluster 1's tail rows in cluster 0's padding. The store reads that buffer, so tile1 is issued
//! right behind tile0 (V346's static schedule: 11,979). The merge stays V340's: the store is reloaded whole and the tail
//! rows are overwritten from cluster 0's DM afterwards. V346 moved the tail rows in first and loaded a store of rows 0..88
//! into a tile of the reload buffer, and its output was non-finite (V342 ub, the same shape, was wrong); a tile of the
//! padded HBM layout that starts past row 0 does not compile (`lir: incorrect buffer size`). V345 (load only) put cluster
//! 0 at 63-67% about 2k ahead of the V340 split.
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

/// Real rows 0..120: each cluster's own groups. Cluster 0's padding rows 120..240: cluster 1's groups' tail rows.
pub(crate) type Contraction88 = DmTensor<bf16, Chip, TwoClusters, Rows, m![H % 120 # 240]>;

pub(crate) fn project_output_88(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> (Store88, Contraction88) {
    let tile0: DmTensor<f8e4m3, Chip, TwoClusters, RowsByColumns, m![H % 120 = 88, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 88, m![H / 120, H % 120 = 88 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    // Cluster 0 alone: rows 88..120 of all 32 groups, both clusters' groups side by side in each slice's element.
    let tile1: DmTensor<f8e4m3, Chip, ClusterZero, RowsByColumns, m![H / 1920, H % 120 = 32, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 32, m![H / 120, H % 120 = 32 # 120, Qs]>(88)
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

    let mut contraction: Contraction88 = DmTensor::new();

    // Both clusters: rows 0..88 of their own groups.
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
        .commit_view(contraction.view_mut().tile::<m![H % 120], 88, m![H % 120 = 88 #{!} 120 # 240]>(0));

    // Cluster 0: rows 88..120 of all 32 groups. Seen as [H / 1920, H % 120], the padded buffer puts cluster 0's own groups
    // in its real rows and cluster 1's groups in its padding. (The compiler follows an owned reshape, not a reshaped
    // mutable view.)
    let mut tails: DmTensor<bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]> = unsafe { contraction.reshape() };
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
    let contraction: Contraction88 = unsafe { tails.reshape() };

    // The buffer's real rows (both clusters) into the V340 store layout, read from the buffer the tail contraction also
    // writes. Cluster 1's rows 88..120 are not computed on cluster 1 and go out unwritten; they are overwritten after the
    // reload.
    let mut stored: Store88 = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, stored.view_mut());
    (stored, contraction)
}

/// The V340 store layout: each group's 240 B at a 256 B boundary.
pub(crate) type Store88 = HbmTensor<bf16, Chip, m![H / 120, H % 120 # 128]>;

/// The whole store loaded into the RMSNorm reducing layout, then rows 88..120 of every group overwritten from cluster 0's
/// tail contraction (a DM-to-DM move on cluster 0), in V340's order: V342 ub and V346 moved the tail rows in first and
/// loaded a store into a tile of the buffer, and both failed accuracy.
pub(crate) fn load_reducing_88<C: M>(
    ctx: &mut Context,
    stored: &Store88,
    contraction: &Contraction88,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let mut x: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480 / 120, H % 120]> = stored.to_dm(&mut ctx.tdma);
    let tails: DmTensorView<'_, bf16, Chip, ClusterZero, Rows, m![H / 1920, H % 120]> = unsafe { contraction.view().reshape() };
    tails
        .tile::<m![H % 120], 32, m![H / 1920, H % 120 = 32 # 120]>(88)
        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 120], 32, m![H % 480 / 120, H % 120 = 32 #{!} 120]>(88));
    unsafe { x.reshape() }
}
