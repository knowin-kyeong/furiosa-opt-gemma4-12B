
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, E, Gs, Ns};
use crate::device::layout::{Cluster, Slice};

type KvHeadsAcrossSlices = m![1 # 32, Ns];

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);

    let cos: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = cos.to_dm(&mut ctx.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = sin.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8, Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Ns], m![Gs, Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8, Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ns], m![Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .collect::<m![Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![1], m![Ds]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q_cos.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k_cos.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let result_q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = ctx
        .main
        .begin(result_q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .switch::<Slice, m![Ns]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let result_k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = ctx
        .main
        .begin(result_k.view())
        .fetch::<m![1], m![Ds]>()
        .switch::<Slice, m![Ns]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Ns, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    (result_q, result_k)
}

/// `apply_rope` for q and k that already sit one head per slice: no transposes in, no
/// broadcast back out.
pub(crate) fn apply_rope_heads<C: M, S: M>(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    DmTensor<bf16, Chip, C, S, m![Ds]>,
) {
    // The gathered rows land on one cluster; stage them through HBM so both clusters can
    // load them into the head layout (a DM-to-DM DMA cannot change the cluster mapping).
    // V30 gathered straight into the head layout on the premise that a cluster or slice axis
    // absent from the table replicates. It does not (V50): the slices that were not written
    // read uninitialised HBM, and q and k came out non-finite from d = 129 on while v, which
    // takes no RoPE, stayed correct.
    // The two gathered rows share one staging buffer, so the head layout is filled by a single
    // load instead of two. V189 priced each trip through HBM at 2.2k real cycles and the pair of
    // gathers at 1.8k; dropping one load was worth 3.4k warm and 8.5k cold.
    let cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    let mut cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = HbmTensor::new();
    cos_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    sin_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    let cs: DmTensor<bf16, Chip, C, S, m![Dummy2, Ds]> = cs_hbm.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_sin: DmTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // The cos multiply and the sin add share one chain, so the f32 scratch pass is gone.
    let result_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // Same fusion on the key side.
    let result_k: DmTensor<bf16, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    (result_q, result_k)
}

/// V355: `apply_rope_heads` with its bf16 casts in the Commit Adapter (`commit_cast`).
pub(crate) fn apply_rope_heads_cc<C: M, S: M>(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    DmTensor<bf16, Chip, C, S, m![Ds]>,
) {
    // The gathered rows land on one cluster; stage them through HBM so both clusters can
    // load them into the head layout (a DM-to-DM DMA cannot change the cluster mapping).
    // V30 gathered straight into the head layout on the premise that a cluster or slice axis
    // absent from the table replicates. It does not (V50): the slices that were not written
    // read uninitialised HBM, and q and k came out non-finite from d = 129 on while v, which
    // takes no RoPE, stayed correct.
    // The two gathered rows share one staging buffer, so the head layout is filled by a single
    // load instead of two. V189 priced each trip through HBM at 2.2k real cycles and the pair of
    // gathers at 1.8k; dropping one load was worth 3.4k warm and 8.5k cold.
    let cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    let mut cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = HbmTensor::new();
    cos_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    sin_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    let cs: DmTensor<bf16, Chip, C, S, m![Dummy2, Ds]> = cs_hbm.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_sin: DmTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // The cos multiply and the sin add share one chain, so the f32 scratch pass is gone.
    let result_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    let k_sin: DmTensor<f32, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // Same fusion on the key side.
    let result_k: DmTensor<bf16, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    (result_q, result_k)
}

/// V383: `apply_rope_heads_cc` with the two RoPE rows staged by one HBM store (one ExplicitSync).
pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    DmTensor<bf16, Chip, C, S, m![Ds]>,
) {
    // The gathered rows land on one cluster; stage them through HBM so both clusters can
    // load them into the head layout (a DM-to-DM DMA cannot change the cluster mapping).
    // V30 gathered straight into the head layout on the premise that a cluster or slice axis
    // absent from the table replicates. It does not (V50): the slices that were not written
    // read uninitialised HBM, and q and k came out non-finite from d = 129 on while v, which
    // takes no RoPE, stayed correct.
    // The two gathered rows share one staging buffer, so the head layout is filled by a single
    // load instead of two. V189 priced each trip through HBM at 2.2k real cycles and the pair of
    // gathers at 1.8k; dropping one load was worth 3.4k warm and 8.5k cold.
    let cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    // V383: both rows are copied on chip into one cluster-0 staging buffer and stored by ONE HBM store, so a single
    // ExplicitSync (instead of one per row) stands in front of the head-layout reload. The first-launch penalty and
    // the random cross-cluster wait land on those syncs (V371/V371cold/V382 spans).
    let mut cs_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Dummy2, Ds]> = DmTensor::new();
    ctx.main
        .begin(cos_row.view())
        .fetch::<m![Ds / 128], m![Ds % 128]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    ctx.main
        .begin(sin_row.view())
        .fetch::<m![Ds / 128], m![Ds % 128]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    let cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = cs_dm.to_hbm(&mut ctx.tdma);
    let cs: DmTensor<bf16, Chip, C, S, m![Dummy2, Ds]> = cs_hbm.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cs.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_sin: DmTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // The cos multiply and the sin add share one chain, so the f32 scratch pass is gone.
    let result_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    let k_sin: DmTensor<f32, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // Same fusion on the key side.
    let result_k: DmTensor<bf16, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    (result_q, result_k)
}

// ---------------------------------------------------------------------------------------------
// V385: the RoPE rows computed on the head slices (no table gathers, no HBM store, no ExplicitSync, no reload).
// ---------------------------------------------------------------------------------------------
/// V385: `apply_rope_heads_cc` with the cos/sin rows computed on the head slices (see gen_v385.py).
pub(crate) fn apply_rope_heads_cc_oc<C: M, S: M>(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
) -> (
    DmTensor<bf16, Chip, C, S, m![Gs, Ds]>,
    DmTensor<bf16, Chip, C, S, m![Ds]>,
) {
    // pos on every head slice: rope_offset is the table row's byte offset (512 B per row), so read as fixed point
    // with 9 fraction bits it is pos itself.
    let offset: DmTensor<i32, Chip, C, S, m![1 # 2]> = rope_offset.view().pad::<m![1 # 2]>().to_dm(&mut ctx.tdma);
    let pos: DmTensor<f32, Chip, C, S, m![1 # 8]> = ctx
        .main
        .begin(offset.view())
        .fetch::<m![1], m![1 # 2]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(22)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let pos_vrf: VrfTensor<f32, Chip, C, S, m![1 # 8]> = ctx
        .sub
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // w = [0, 1] along Dummy2, both packets made from pos (BitAnd 0 clears it).
    let mut w: DmTensor<f32, Chip, C, S, m![Dummy2, 1 # 8]> = DmTensor::new();
    ctx
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(0));
    ctx
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(1));
    // The angle row, one index bit per pass from the low bit up (the new bit outermost): x_{k+1} = x_k * [1, e_k]
    // with e_k = 10000^(-2^k / 128), computed as w * (e_k - 1) * x_k + x_k with x_k in the VRF and w read replayed
    // over the bits already built. The seed is pos, so the row ends as pos * 10000^(-(d mod 128) / 128).
    let y0: DmTensor<f32, Chip, C, S, m![Dummy2, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2], m![1 # 8]>()
        .collect::<m![Dummy2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.0694279596f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &pos_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &pos_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x1: DmTensor<f32, Chip, C, S, m![Ds % 2, 1 # 8]> = unsafe { y0.reshape() };
    let x1_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 2, 1 # 8]> = ctx
        .sub
        .begin(x1.view())
        .fetch::<m![Ds % 2], m![1 # 8]>()
        .collect::<m![Ds % 2], m![1 # 8]>()
        .to_vrf();
    let y1: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 2, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 2], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.134035677f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x1_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x1_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x2: DmTensor<f32, Chip, C, S, m![Ds % 4, 1 # 8]> = unsafe { y1.reshape() };
    let x2_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 4, 1 # 8]> = ctx
        .sub
        .begin(x2.view())
        .fetch::<m![Ds % 4], m![1 # 8]>()
        .collect::<m![Ds % 4], m![1 # 8]>()
        .to_vrf();
    let y2: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 4, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 4], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.250105798f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x2_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x2_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x3: DmTensor<f32, Chip, C, S, m![Ds % 8, 1 # 8]> = unsafe { y2.reshape() };
    let x3_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 8, 1 # 8]> = ctx
        .sub
        .begin(x3.view())
        .fetch::<m![Ds % 8], m![1 # 8]>()
        .collect::<m![Ds % 8], m![1 # 8]>()
        .to_vrf();
    let y3: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 8, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 8], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.437658668f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x3_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x3_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x4: DmTensor<f32, Chip, C, S, m![Ds % 16, 1 # 8]> = unsafe { y3.reshape() };
    let x4_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 16, 1 # 8]> = ctx
        .sub
        .begin(x4.view())
        .fetch::<m![Ds % 16], m![1 # 8]>()
        .collect::<m![Ds % 16], m![1 # 8]>()
        .to_vrf();
    let y4: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 16, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 16], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 16], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.683772206f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x4_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x4_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x5: DmTensor<f32, Chip, C, S, m![Ds % 32, 1 # 8]> = unsafe { y4.reshape() };
    let x5_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 32, 1 # 8]> = ctx
        .sub
        .begin(x5.view())
        .fetch::<m![Ds % 32], m![1 # 8]>()
        .collect::<m![Ds % 32], m![1 # 8]>()
        .to_vrf();
    let y5: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 32, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 32], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 32], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.899999976f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x5_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x5_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x6: DmTensor<f32, Chip, C, S, m![Ds % 64, 1 # 8]> = unsafe { y5.reshape() };
    let x6_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 64, 1 # 8]> = ctx
        .sub
        .begin(x6.view())
        .fetch::<m![Ds % 64], m![1 # 8]>()
        .collect::<m![Ds % 64], m![1 # 8]>()
        .to_vrf();
    let y6: DmTensor<f32, Chip, C, S, m![Dummy2, Ds % 64, 1 # 8]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![Dummy2, Ds % 64], m![1 # 8]>()
        .collect::<m![Dummy2, Ds % 64], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -0.99000001f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &x6_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &x6_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let x7: DmTensor<f32, Chip, C, S, m![Ds % 128, 1 # 8]> = unsafe { y6.reshape() };
    // cos over both halves of the row and sin negated over the low half (the table's convention), each pass packing
    // its 128 scalars into a dense bf16 half-row with the 4-row transpose.
    let mut cos_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();
    let mut sin_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();
    ctx
        .main
        .begin(x7.view())
        .fetch::<m![Ds % 128], m![1 # 8]>()
        .collect::<m![Ds % 128], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit_view(cos_row.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds % 128]>(0));
    ctx
        .main
        .begin(x7.view())
        .fetch::<m![Ds % 128], m![1 # 8]>()
        .collect::<m![Ds % 128], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit_view(cos_row.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds % 128]>(1));
    ctx
        .main
        .begin(x7.view())
        .fetch::<m![Ds % 128], m![1 # 8]>()
        .collect::<m![Ds % 128], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sin)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit_view(sin_row.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds % 128]>(0));
    ctx
        .main
        .begin(x7.view())
        .fetch::<m![Ds % 128], m![1 # 8]>()
        .collect::<m![Ds % 128], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sin)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit_view(sin_row.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds % 128]>(1));

    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(cos_row.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(sin_row.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_sin: DmTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, C, S, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // The cos multiply and the sin add share one chain, so the f32 scratch pass is gone.
    let result_q: DmTensor<bf16, Chip, C, S, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    let k_sin: DmTensor<f32, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // Same fusion on the key side.
    let result_k: DmTensor<bf16, Chip, C, S, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit_cast::<bf16>()
        .commit();

    (result_q, result_k)
}
