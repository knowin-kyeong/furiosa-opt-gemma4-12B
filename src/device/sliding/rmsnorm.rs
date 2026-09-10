
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Slice};
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
/// The projection's per-channel weight scale is folded in: the norm sees `x * channel_scale`.
pub(crate) fn normalize_query_heads<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    channel_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, C, S, m![Gs, Ds]> {
    let channel_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { channel_scale.view().reshape() };
    let scale_dm: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, C, S, m![Gs, 1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, C, S, m![Gs, 1 # 8]> = ctx
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

    let weight_vrf = load_norm_weight::<C, S>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, C, S, m![Gs, 1 # 8]> = ctx
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// The projection's per-channel weight scale `[Ns, Ds]` in the head layout, as an f32 VRF row.
fn load_channel_scale_heads<C: M, S: M>(
    ctx: &mut Context,
    channel_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> VrfTensor<f32, Chip, C, S, m![Ds]> {
    let channel_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> = unsafe { channel_scale.view().reshape() };
    let scale_dm: DmTensor<bf16, Chip, C, S, m![Ds]> = channel_scale.to_dm(&mut ctx.tdma);

    ctx.sub
        .begin(scale_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// 1 / rms of one head-row per slice, as a scalar packet in the VRF, of `x * scale_vrf`.
fn root_mean_square_heads<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
) -> VrfTensor<f32, Chip, C, S, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, C, S, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, C, S, m![1 # 8]> = ctx
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

pub(crate) fn normalize_key_heads<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    channel_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, C, S, m![Ds]> {
    let scale_vrf = load_channel_scale_heads::<C, S>(ctx, channel_scale);
    let rms_vrf = root_mean_square_heads::<C, S>(ctx, x, &scale_vrf);
    let weight_vrf = load_norm_weight::<C, S>(ctx, rms_weight);

    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_value_heads<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    channel_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, C, S, m![Ds]> {
    let scale_vrf = load_channel_scale_heads::<C, S>(ctx, channel_scale);
    let rms_vrf = root_mean_square_heads::<C, S>(ctx, x, &scale_vrf);

    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

// -------------------------------------------------------------------------------------------
// V239: the qkv tail in as few passes as the shapes allow.
//
// V218 measured the tail (three head RMSNorms plus RoPE) at 16,012 real cycles for 7,435 static
// and priced one pass at ~600 real: the PE core issues passes in order, and that issue cost, not
// the vector work, is what the tail is made of. The three head norms and the RoPE run seventeen
// passes over one or two rows each, and a pass over four rows costs the same as a pass over one.
// So q's two group rows, k and v share one four-row buffer per slice, `m![Dummy2, Gs, ..]`:
// the three sqrt passes become one, the RoPE's four rotate-half passes become two and its two
// sin passes one. Twelve passes instead of seventeen.
//
// Every tile into that buffer is an extent-1 axis that the mapping elides, which is the one form
// the fetch unit accepts -- a partial axis (`m![Ns / 2 = 2]`) on a freshly allocated tensor is
// rejected as `lower_fetch_unit: There should be not-exactly-matched from_in_slice slots`.
//
// Row (1, 1) is dead. v is normalized into its own tensor because `dma_scatter` takes a whole
// `DmTensor`, not a tile of one, so v cannot ride in the shared buffer and still reach the cache.

/// q's mean square into rows (0, 0) and (0, 1) of the shared buffer.
pub(crate) fn head_mean_square_query<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Gs, Ds]>,
    ms: &mut DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]>,
) {
    ctx.main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(
            ms.view_mut()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Gs, 1 # 8]>(0),
        );
}

/// One head row's mean square into row (1, `offset`) of the shared buffer.
pub(crate) fn head_mean_square_row<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
    offset: usize,
    ms: &mut DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]>,
) {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(
            ms.view_mut()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Gs, 1 # 8]>(1)
                .tile::<m![Gs], 1, m![Dummy2 = 1 #{!} 2, Gs = 1 #{!} 2, 1 # 8]>(offset),
        );
}

/// The one sqrt pass that replaces q's, k's and v's.
pub(crate) fn head_rms_all<C: M, S: M>(
    ctx: &mut Context,
    ms: &DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]>,
) -> DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]> {
    ctx.main
        .begin(ms.view())
        .fetch::<m![Dummy2, Gs], m![1 # 8]>()
        .collect::<m![Dummy2, Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// q's two rows of the shared rms buffer, staged to the VRF.
pub(crate) fn head_rms_vrf_query<C: M, S: M>(
    ctx: &mut Context,
    rms: &DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]>,
) -> VrfTensor<f32, Chip, C, S, m![Gs, 1 # 8]> {
    ctx.sub
        .begin(rms.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Gs, 1 # 8]>(0))
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf()
}

/// One row of the shared rms buffer, staged to the VRF.
pub(crate) fn head_rms_vrf_row<C: M, S: M>(
    ctx: &mut Context,
    rms: &DmTensor<f32, Chip, C, S, m![Dummy2, Gs, 1 # 8]>,
    offset: usize,
) -> VrfTensor<f32, Chip, C, S, m![1 # 8]> {
    ctx.sub
        .begin(
            rms.view()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Gs, 1 # 8]>(1)
                .tile::<m![Gs], 1, m![Dummy2 = 1 # 2, Gs = 1 # 2, 1 # 8]>(offset),
        )
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// q normalized into rows (0, 0) and (0, 1) of the shared RoPE buffer.
pub(crate) fn head_normalize_query<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Gs, Ds]>,
    weight_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, C, S, m![Gs, 1 # 8]>,
    out: &mut DmTensor<bf16, Chip, C, S, m![Dummy2, Gs, Ds]>,
) {
    ctx.main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(
            out.view_mut()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Gs, Ds]>(0),
        );
}

/// k normalized into row (1, `offset`) of the shared RoPE buffer.
pub(crate) fn head_normalize_row<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
    weight_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, C, S, m![1 # 8]>,
    offset: usize,
    out: &mut DmTensor<bf16, Chip, C, S, m![Dummy2, Gs, Ds]>,
) {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(
            out.view_mut()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Gs, Ds]>(1)
                .tile::<m![Gs], 1, m![Dummy2 = 1 #{!} 2, Gs = 1 #{!} 2, Ds]>(offset),
        );
}

/// v normalized (no gamma) into its own tensor, ready for the scatter.
pub(crate) fn head_normalize_value_row<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    scale_vrf: &VrfTensor<f32, Chip, C, S, m![Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, C, S, m![1 # 8]>,
) -> DmTensor<bf16, Chip, C, S, m![Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// The per-channel weight scale in the head layout, for q's two rows.
pub(crate) fn load_channel_scale_query<C: M, S: M>(
    ctx: &mut Context,
    channel_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> VrfTensor<f32, Chip, C, S, m![Gs, Ds]> {
    let channel_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { channel_scale.view().reshape() };
    let scale_dm: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = channel_scale.to_dm(&mut ctx.tdma);
    ctx.sub
        .begin(scale_dm.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

pub(crate) fn load_channel_scale_row<C: M, S: M>(
    ctx: &mut Context,
    channel_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> VrfTensor<f32, Chip, C, S, m![Ds]> {
    load_channel_scale_heads::<C, S>(ctx, channel_scale)
}

pub(crate) fn load_head_norm_weight<C: M, S: M>(
    ctx: &mut Context,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, C, S, m![Ds]> {
    load_norm_weight::<C, S>(ctx, rms_weight)
}
