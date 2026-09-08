
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, Ns};
use crate::device::layout::{Cluster, HeadSlices, Slice};
use crate::{Chip, EPS};

const DS_F32: f32 = Ds::SIZE as f32;

pub(crate) fn normalize_query<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 2], m![Ns % 2, Gs]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_final()
        .commit_trim::<m![Ns % 4, Gs]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn load_norm_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = rms_weight.to_dm(&mut ctx.tdma);

    ctx.sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

fn root_mean_square<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 4], m![Ns % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![1], m![Ns]>()
        .vector_final()
        .commit_trim::<m![Ns]>()
        .commit();

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .to_vrf()
}

fn scale_by_rms_and_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn scale_by_rms<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_key(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_rms_and_weight(ctx, x, &rms_vrf, &weight_vrf)
}

pub(crate) fn normalize_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);

    scale_by_rms(ctx, x, &rms_vrf)
}

/// `normalize_query` for q sitting one head per slice: the Ds reduction never leaves a slice
/// and the two group rows are handled as two scalar packets.
pub(crate) fn normalize_query_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, HeadSlices, m![Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, HeadSlices, m![Gs, 1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, HeadSlices, m![Gs, 1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, HeadSlices>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Gs, 1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// 1 / rms of one head-row per slice, as a scalar packet in the VRF.
fn root_mean_square_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx
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

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn normalize_key_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]> {
    let rms_vrf = root_mean_square_heads(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, HeadSlices>(ctx, rms_weight);

    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_value_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]> {
    let rms_vrf = root_mean_square_heads(ctx, x);

    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
