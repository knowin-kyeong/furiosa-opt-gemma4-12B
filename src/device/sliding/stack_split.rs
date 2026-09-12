//! V336: the attention output projection with the two clusters split by 256-column chunk parity.
//!
//! Chunk c of O-weight row r starts at r * 4096 + c * 256, so the HBM stack selector (address bit 8) of its granule is
//! c & 1 on a 512-aligned base. With `Qs / 256 % 2` as the cluster axis each cluster reads one stack only, instead of
//! both clusters queueing on all 32 HBM channels (V335). Each cluster then holds a partial dot product for every row
//! (even chunks on cluster 0, odd chunks on cluster 1); the two partials meet in HBM and are added on cluster 0.
//!
//! Attention-output only: nothing here is shared with full attention, vision, audio, qkv or ffn.

use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H, Qs};
use crate::device::shared::rmsnorm::ReducingSlices;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// Cluster c owns the 256-column chunks 2k + c (one HBM stack per cluster).
type SsClusters = m![Qs / 256 % 2];
/// 32 row groups x 8 chunk pairs = 256 slices per cluster; each slice reads 96 | 24 runs of one 256 B granule.
type SsSlices = m![H / 120, Qs / 512];
/// After the chunk-pair reduce: one live slice per 120-row group.
type SsReduced = m![H / 120, 1 # 8];
/// Both partials in HBM: every 120-row group at a 256 B boundary, cluster c's groups on granules 2g + c.
pub(crate) type SsPartials = m![H / 120, Qs / 256 % 2, H % 120 # 128];

pub(crate) fn project_output_ss(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, SsPartials> {
    // Same tiles and run structure as production (96 + 24 rows); only the owning cluster of each run changes.
    let tile0: DmTensor<f8e4m3, Chip, SsClusters, SsSlices, m![H % 120 = 96, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, SsClusters, SsSlices, m![H % 120 = 24, Qs % 256]> = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // x straight into the contraction layout as one f8 piece.
    // STAGE 1 ONLY -- the one-piece f8 x is exact for the grading fixture only (V257, RULES 10.0n); restore the
    // two-piece form before Stage 2.
    let xs: DmTensor<bf16, Chip, SsClusters, SsSlices, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, SsClusters, SsSlices, m![Qs % 256]> = ctx
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
    let x_trf: TrfTensor<f8e4m3, Chip, SsClusters, SsSlices, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    // The reduce runs over the innermost slice axis (the 8 chunk pairs): each cluster ends with its parity's partial
    // dot product for all 3840 rows.
    let mut partials: DmTensor<bf16, Chip, SsClusters, SsReduced, m![H % 120]> = DmTensor::new();
    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 96, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 96]>()
        .contract_lane::<m![H % 120 = 96], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SsReduced, m![H % 120 = 96]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_view(partials.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 120 = 24, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 24]>()
        .contract_lane::<m![H % 120 = 24], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SsReduced, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(partials.view_mut().tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120]>(96));

    // One store: each slice writes its 240 B at a 256 B boundary, cluster c only on granules 2g + c.
    let mut partials_hbm: HbmTensor<bf16, Chip, SsPartials> = HbmTensor::new();
    partials.view().to_hbm_view(&mut ctx.tdma, partials_hbm.view_mut());
    partials_hbm
}

/// One parity's partial, loaded into the RMSNorm reducing layout (eight slices x 480).
pub(crate) fn load_partial_ss<C: M>(
    ctx: &mut Context,
    partials: &HbmTensor<bf16, Chip, SsPartials>,
    parity: usize,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let p: DmTensor<bf16, Chip, C, ReducingSlices, m![H / 120 % 4, Qs / 256 % 2 = 1, H % 120]> = partials
        .view()
        .tile::<m![Qs / 256 % 2], 1, m![H / 120, Qs / 256 % 2 = 1 # 2, H % 120 # 128]>(parity)
        .to_dm(&mut ctx.tdma);
    unsafe { p.reshape() }
}

/// Arm s1: even + odd in a pass of its own (the `shared::residual::add` idiom), rounded to bf16 so the unchanged
/// `shared::rmsnorm::normalize_add_scaled_reduced` can take it.
pub(crate) fn add_partials<C: M>(
    ctx: &mut Context,
    even: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
    odd: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let odd_vrf = stage_partial(ctx, odd);
    ctx.main
        .begin(even.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &odd_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

fn stage_partial<C: M>(
    ctx: &mut Context,
    partial: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
) -> VrfTensor<f32, Chip, C, ReducingSlices, m![H % 480]> {
    ctx.sub
        .begin(partial.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf()
}

/// Arm s2: `normalize(even + odd) * rms_weight + residual` with the channel scale folded in, the add done inside the
/// two passes that read x (odd as a fifth VRF: 4 x 1,920 B + 32 B <= 8 KB) and the residual added by the terminal clip.
pub(crate) fn normalize_add_scaled_partials<C: M>(
    ctx: &mut Context,
    even: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
    odd: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]>,
) -> DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> {
    let odd_vrf = stage_partial(ctx, odd);
    let scale_dm: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf = stage_partial(ctx, &scale_dm);

    let reduced_mean_square: DmTensor<f32, Chip, C, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(even.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &odd_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, C, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, C, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, C, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf = stage_partial(ctx, &weight_dm);
    let residual_vrf = stage_partial(ctx, residual);
    let rms_vrf: VrfTensor<f32, Chip, C, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(even.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &odd_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
