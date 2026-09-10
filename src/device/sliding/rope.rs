
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

/// V232: `apply_rope_heads` without the `rotate_half` staging.
///
/// Four main passes in the production form do nothing but copy each tensor's halves into swapped
/// positions so the sin multiply can read them in place. Streaming the halves and committing the
/// products swapped removes two of those four.
pub(crate) fn apply_rope_heads_swap<C: M, S: M>(
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

    // V232: the two halves of sin separately. `rotate_half` was materialised by two pure-copy
    // main passes per tensor purely so the next pass could read q[i^128] at position i. Streaming
    // q's halves directly and committing them into the swapped position does the same thing in one
    // pass fewer per tensor -- the swap moves onto the 256-element sin vector, which q and k share.
    //
    //   out[i] = q[i] * cos[i] + q[i ^ 128] * sin[i]
    //
    // so streaming q[j] for j < 128 against sin[j + 128] and committing at j + 128 fills the upper
    // half, and streaming q[j] for j >= 128 against sin[j - 128] and committing at j - 128 fills
    // the lower half. sin already carries the sign of the rotation per output position.
    let sin_lo_vrf: VrfTensor<f32, Chip, C, S, m![Ds = 128]> = ctx
        .sub
        .begin(
            cs.view()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1)
                .tile::<m![Ds], 128, m![Dummy2 = 1, Ds = 128 # 256]>(0),
        )
        .fetch::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let sin_hi_vrf: VrfTensor<f32, Chip, C, S, m![Ds = 128]> = ctx
        .sub
        .begin(
            cs.view()
                .tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1)
                .tile::<m![Ds], 128, m![Dummy2 = 1, Ds = 128 # 256]>(128),
        )
        .fetch::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut q_sin: DmTensor<f32, Chip, C, S, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi_vrf)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo_vrf)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0));

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut k_sin: DmTensor<f32, Chip, C, S, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

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
