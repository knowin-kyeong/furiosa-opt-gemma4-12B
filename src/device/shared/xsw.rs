//! V292: the qkv input staged on both clusters and replicated on chip (see xsw.py in the V292 notes).

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::EPS;
use crate::axes::{Dummy2, Dummy256, Dummy8, H, Ns, Qs};
use crate::device::layout::{BothClusters, Replicated};
use crate::{hi_lo_fns, max_square_fns, pow2_scale_fns, stage_packet_fns};

const H_F32: f32 = H::SIZE as f32;
const INVSQRT2: f32 = 0.70710678118f32;

/// Both clusters, as a copy axis of [H] tensors (a real cluster label: a dummy cluster axis serves cluster 0 only, V155).
pub(crate) type XCl = m![Qs / 2048];
/// Eight copies of x's eight 480-element chunks, in group 0 of every 32-slice sub-ring (64 live slices per cluster).
pub(crate) type XBlocks = m![Ns, 1 # 4, H / 480];

pow2_scale_fns!(pow2_scale_blocks, XCl, XBlocks);
stage_packet_fns!(stage_packet_blocks, XCl, XBlocks);
max_square_fns!(max_square_blocks, XCl, XBlocks, H, 480, 30, 60, 120);
hi_lo_fns!(hi_lo_blocks, XCl, XBlocks, H, 480, 15, 30, 60, 120);

/// x from HBM onto both clusters' chunk slices (a replicated load: both copy axes absent from the source).
pub(crate) fn load_blocks(ctx: &mut Context, x: &HbmTensor<bf16, Chip, m![H]>) -> DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]> {
    x.to_dm(&mut ctx.tdma)
}

pub(crate) fn normalize_blocks_f32(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<f32, Chip, XCl, XBlocks, m![H % 480]> {

    let mean_square: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, XCl, XBlocks, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn stage_x_hi_lo_blocks(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, XCl, XBlocks, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, XCl, XBlocks, m![Dummy2, H % 480]> {
    let x: DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]> = ctx
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

    let m_local = max_square_blocks(ctx, &x);
    let m_all: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, _inv_s) = pow2_scale_blocks(ctx, &m_all);
    let s_vrf = stage_packet_blocks(ctx, &s);

    let (x_hi, x_lo) = hi_lo_blocks(ctx, &x, &s_vrf);
    // m1: the two pieces are gathered into one buffer on the slices that already hold them, so
    // the staging costs one DMA command instead of two.
    let mut x2: DmTensor<f8e4m3, Chip, XCl, XBlocks, m![Dummy2, H % 480]> = DmTensor::new();
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
    x2
}

/// One ring-32 all-gather: every slice of a 32-slice sub-ring receives the eight live chunks of both pieces
/// (the 24 padded slots are trimmed), so x lands whole on all 256 slices of both clusters.
pub(crate) fn replicate_blocks(
    ctx: &mut Context,
    x2: &DmTensor<f8e4m3, Chip, XCl, XBlocks, m![Dummy2, H % 480]>,
) -> DmTensor<f8e4m3, Chip, BothClusters, Replicated, m![Dummy2, H]> {
    let x: DmTensor<f8e4m3, Chip, XCl, m![Ns, Dummy256 / 8], m![Dummy2, H]> = ctx
        .main
        .begin(x2.view())
        .fetch::<m![Dummy2], m![H % 480]>()
        .switch::<m![Ns, Dummy256 / 8], m![Dummy2, H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    unsafe { x.reshape() }
}

// V293: the ffn input path.
pub(crate) fn normalize_blocks_f32_fused(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<f32, Chip, XCl, XBlocks, m![H % 480]> {

    let reduced_mean_square: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        // V263: the cross-slice sum joins the mean-square pass (Widen -> InterSliceReduce is a legal
        // transition, furiosa-opt-std-0.6.0 stage/markers.rs:349), saving a pass and a commit. Only
        // Tag/Filter/Output may follow an inter-slice reduce, so the +EPS moves into the sqrt pass below.
        .vector_inter_slice_reduce::<m![Ns, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
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
    let rms: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, XCl, XBlocks, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn stage_x_hi_lo_full_blocks(
    ctx: &mut Context,
    normalized: &DmTensor<f32, Chip, XCl, XBlocks, m![H % 480]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (
    DmTensor<f8e4m3, Chip, XCl, XBlocks, m![Dummy2, H % 480]>,
    DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]>,
    DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]>,
) {
    let x: DmTensor<bf16, Chip, XCl, XBlocks, m![H % 480]> = ctx
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

    let m_local = max_square_blocks(ctx, &x);
    let m_all: DmTensor<f32, Chip, XCl, m![Ns, 1 # 4, Dummy8], m![1 # 8]> = ctx
        .sub
        .begin(m_local.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let m_all: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = unsafe { m_all.reshape() };
    let (s, inv_s) = pow2_scale_blocks(ctx, &m_all);
    let s_vrf = stage_packet_blocks(ctx, &s);
    let inv_s_vrf = stage_packet_blocks(ctx, &inv_s);

    let (x_hi, x_lo) = hi_lo_blocks(ctx, &x, &s_vrf);
    // m1: the two pieces are gathered into one buffer on the slices that already hold them, so
    // the staging costs one DMA command instead of two.
    let mut x2: DmTensor<f8e4m3, Chip, XCl, XBlocks, m![Dummy2, H % 480]> = DmTensor::new();
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

    // The geglu scalars.
    let s_up: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = up_global_scale.to_dm(&mut ctx.tdma);
    let s_gate: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = gate_global_scale.to_dm(&mut ctx.tdma);
    let s_gate_vrf = stage_packet_blocks(ctx, &s_gate);
    let erf_scale: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
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
    let half_inv_s2: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
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
    let half_inv_s2_vrf = stage_packet_blocks(ctx, &half_inv_s2);
    let out_scale: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]> = ctx
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
    (x2, erf_scale, out_scale)
}

/// A per-cluster scalar computed in XBlocks (the same value on all 64 live slices) onto all 256 slices: slot 0 of each
/// 32-slice sub-ring is the source of a ring-32 broadcast (the other live copies are read as padding).
pub(crate) fn broadcast_scalar_blocks(
    ctx: &mut Context,
    v: DmTensor<f32, Chip, XCl, XBlocks, m![1 # 8]>,
) -> DmTensor<f32, Chip, BothClusters, Replicated, m![1 # 8]> {
    let one: DmTensor<f32, Chip, XCl, m![Ns, 1 # 32], m![1 # 8]> = unsafe { v.reshape() };
    let all: DmTensor<f32, Chip, XCl, m![Ns, Dummy256 / 8], m![1 # 8]> = ctx
        .main
        .begin(one.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Ns, Dummy256 / 8], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    unsafe { all.reshape() }
}
