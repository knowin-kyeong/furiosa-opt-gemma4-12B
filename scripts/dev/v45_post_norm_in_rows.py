"""v45.py -- attn_out post-norm in the projection's own row layout.

Drops the intermediate [H] store (64 descriptors, 2,510), the HBM hop and the reload into the
reducing layout (546). The mean square then spans both clusters, so the two per-cluster sums are
exchanged through one f32 scalar in HBM.
"""
import io

P = "src/device/sliding/projection.rs"
O = "src/ops.rs"


def edit(rel, old, new):
    s = io.open(rel, encoding='utf-8', newline='').read()
    nl = '\r\n' if '\r\n' in s else '\n'
    o, n = old.replace('\n', nl), new.replace('\n', nl)
    assert o in s, "NOT FOUND in %s:\n%s" % (rel, old[:200])
    io.open(rel, 'w', encoding='utf-8', newline='').write(s.replace(o, n, 1))


edit(P, "use crate::Chip;", "use crate::{Chip, EPS};\n\n/// H as f32 for the mean-square divide (rmsnorm keeps its own private copy).\nconst H_F32: f32 = H::SIZE as f32;")
edit(P, "use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};",
        "use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};")

edit(P, """    // Each cluster writes its half of the [H] vector to HBM; the caller loads it back in the
    // layout it needs. (Collecting the 32 row groups onto one slice first, to cut the 64
    // store descriptors to 2, costs as much in the switch as it saves: the live slices sit
    // eight apart, so the ring spans all 256 slices, 2,055 cycles for 458 saved on the store.)
    let mut gathered_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    contraction.view().to_hbm_view(&mut ctx.tdma, gathered_hbm.view_mut());""",
"""    // The post-attention RMSNorm runs here, in the projection's own row layout, instead of
    // through HBM. The old path paid a 64-descriptor store (2,510), an HBM hop and a reload into
    // the reducing layout (546) only to change layout; what that bought was a mean square that
    // spans one cluster. Here it spans both, so the two per-cluster sums are exchanged through
    // one f32 scalar in HBM.
    let scale_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = channel_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows, m![H / 4 % 15, H % 4 # 8]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![H / 4 % 15], m![H % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4 % 15], m![H % 4 # 8]>()
        .to_vrf();

    let ms_slice: DmTensor<f32, Chip, TwoClusters, HiddenRows, m![1 # 16]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![H / 4 % 15], m![H % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4 % 15], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 15, 1 # 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 16]>()
        .vector_final()
        .commit_trim::<m![1 # 16]>()
        .commit();

    let ms_cluster: DmTensor<f32, Chip, TwoClusters, m![Dummy256 / 8, 1 # 8], m![1 # 8]> = ctx
        .main
        .begin(ms_slice.view())
        .fetch::<m![1], m![1 # 16]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Dummy256 / 8, 1 # 8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let ms_one: DmTensor<f32, Chip, TwoClusters, Slice, m![1 # 8]> = unsafe { ms_cluster.reshape() };
    let mut ms_hbm: HbmTensor<f32, Chip, m![H / 1920 # 8]> = HbmTensor::new();
    ms_one.view().to_hbm_view(&mut ctx.tdma, ms_hbm.view_mut());
    let ms_both: DmTensor<f32, Chip, TwoClusters, HiddenRows, m![H / 1920 # 8]> = ms_hbm.to_dm(&mut ctx.tdma);

    // Three passes, as in rmsnorm: a reduce may only be followed by the divide and the widen,
    // so the eps add and the square root each need their own.
    let ms_sum: DmTensor<f32, Chip, TwoClusters, HiddenRows, m![1 # 8]> = ctx
        .main
        .begin(ms_both.view())
        .fetch::<m![1], m![H / 1920 # 8]>()
        .collect::<m![1], m![H / 1920 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 1920 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let ms_eps: DmTensor<f32, Chip, TwoClusters, HiddenRows, m![1 # 8]> = ctx
        .main
        .begin(ms_sum.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, TwoClusters, HiddenRows, m![1 # 8]> = ctx
        .main
        .begin(ms_eps.view())
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

    let rms_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let weight_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows, m![H / 4 % 15, H % 4 # 8]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 4 % 15], m![H % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4 % 15], m![H % 4 # 8]>()
        .to_vrf();

    let residual_dm: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = residual_hbm.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, TwoClusters, HiddenRows, m![H / 4 % 15, H % 4 # 8]> = ctx
        .sub
        .begin(residual_dm.view())
        .fetch::<m![H / 4 % 15], m![H % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4 % 15], m![H % 4 # 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, TwoClusters, HiddenRows, m![H % 60]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![H / 4 % 15], m![H % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4 % 15], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 15, 1 # 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_widen_concat::<m![H / 4 % 15], m![H % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());""")

edit(P, """pub(crate) fn project_output(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {""",
"""pub(crate) fn project_output(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {""")

edit(P, """    gathered_hbm
}""", """}""")

edit(O, """    let x_hbm = sliding::projection::project_output(ctx, x, o_weight);
    // Both operands of the post-attention RMSNorm are loaded straight into its reducing layout.
    let x = shared::rmsnorm::load_reducing::<Cluster>(ctx, &x_hbm);
    let residual = shared::rmsnorm::load_reducing::<Cluster>(ctx, residual_hbm);
    // The result is stored straight from the reducing layout (eight descriptors, no switch pass).
    let residual = shared::rmsnorm::normalize_add_scaled_reduced::<Cluster>(ctx, &x, o_weight_scale, post_attn_rms_weight, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());""",
"""    // The post-attention RMSNorm runs inside project_output, in the projection's row layout: the
    // intermediate [H] store, the HBM hop and the reload into the reducing layout are all gone.
    sliding::projection::project_output(ctx, x, o_weight, o_weight_scale, post_attn_rms_weight, residual_hbm);""")

print("V45 applied")
