//! V352: pass trims in the attention-output post-attention norm tail (attention output only; the shared
//! `rmsnorm::normalize_add_scaled_reduced` is unchanged).
//!
//! V340 on hardware (v342_r0, arm ut#1) after the tail DM-to-DM: mean-square 41.9-43.1k -> rms (+EPS, sqrt) 43.1-43.7k ->
//! residual VRF 43.2-43.8k -> rms VRF 43.8-44.1k -> final pass 44.1-45.1k -> store 45.1-46.0k.
//! - `normalize_add_scaled_reduced_b`: +EPS and sqrt run in the sub-context pass that stages the rms VRF, so the Main rms
//!   pass and its commit disappear (V263 took a tail pass out of the same norm for -1.05%). A sub vector chain may end in
//!   `to_vrf` (`CanApplyToVrf for PositionVectorFinal`); V274's hang was a *Main* vector pass writing the VRF.
//! - `normalize_add_scaled_reduced_c`: the final pass casts f32 -> bf16 in the Commit Adapter (`commit_cast`, legal after
//!   `commit_trim`) instead of the Cast Engine.
//! - `normalize_add_scaled_reduced_bc`: both.

use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};
use crate::device::shared::rmsnorm::ReducingSlices;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// The mean-square pass with the cross-slice sum folded in (V263), as in production.
fn reduced_mean_square_scaled<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    scale_vrf: &VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Production's Main rms pass (+EPS, sqrt) and its plain sub staging.
fn rms_vrf_main<Cluster: M>(
    ctx: &mut Context,
    reduced_mean_square: &DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> {
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };
    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// B: +EPS and sqrt in the sub staging pass itself.
fn rms_vrf_sub<Cluster: M>(
    ctx: &mut Context,
    reduced_mean_square: &DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> {
    let reduced_mean_square: DmTensorView<'_, f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        unsafe { reduced_mean_square.view().reshape() };
    ctx.sub
        .begin(reduced_mean_square)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf()
}

fn stage_row_vrf<Cluster: M>(
    ctx: &mut Context,
    v: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> {
    ctx.sub
        .begin(v.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf()
}

/// Arm `b`.
pub(crate) fn normalize_add_scaled_reduced_b<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let scale_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf = stage_row_vrf(ctx, &scale_dm);
    let reduced_mean_square = reduced_mean_square_scaled(ctx, x, &scale_vrf);
    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf = stage_row_vrf(ctx, &weight_dm);
    let residual_vrf = stage_row_vrf(ctx, residual);
    let rms_vrf = rms_vrf_sub(ctx, &reduced_mean_square);

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Arm `c`.
pub(crate) fn normalize_add_scaled_reduced_c<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let scale_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf = stage_row_vrf(ctx, &scale_dm);
    let reduced_mean_square = reduced_mean_square_scaled(ctx, x, &scale_vrf);
    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf = stage_row_vrf(ctx, &weight_dm);
    let residual_vrf = stage_row_vrf(ctx, residual);
    let rms_vrf = rms_vrf_main(ctx, &reduced_mean_square);

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit_cast::<bf16>()
        .commit()
}

/// Arm `bc`.
pub(crate) fn normalize_add_scaled_reduced_bc<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let scale_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf = stage_row_vrf(ctx, &scale_dm);
    let reduced_mean_square = reduced_mean_square_scaled(ctx, x, &scale_vrf);
    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf = stage_row_vrf(ctx, &weight_dm);
    let residual_vrf = stage_row_vrf(ctx, residual);
    let rms_vrf = rms_vrf_sub(ctx, &reduced_mean_square);

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit_cast::<bf16>()
        .commit()
}
