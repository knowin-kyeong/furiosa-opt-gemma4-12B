
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryRows = m![Qs / 16];

    let x: DmTensorView<'_, bf16, Chip, Cluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, QueryRows, m![Qs % 16, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_dm: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let contraction: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 16]>()
        .contract_lane::<m![Qs % 16], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 16]>()
        .switch::<Slice, m![Qs / 16]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();

    unsafe { output.reshape() }
}

type KvRows = m![Ps / 8];

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, KvRows, m![Ps % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_dm: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let contraction: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    ctx.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 8 # 16]>()
        .switch::<Slice, m![Ps / 8]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 8], m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit()
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, Cluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // Every slice owns 60 hidden rows x one 1024-wide Qs chunk (64 row groups x 4 chunks =
    // 256 slices), so each slice needs only a quarter of x, and the four chunk partials are
    // summed across slices like project_down. The rows come in 5 tiles of 12, all issued up
    // front into distinct buffers, so the lookup and contraction of one tile overlap the
    // loads of the rest. (Live slice counts must be a power of two, and H = 3840 = 2^8 x 15
    // only tiles into 4-row transpose packets at 60 or 120 rows.)
    let tile0: DmTensor<f8e4m3, Chip, Cluster, HiddenRowsByColumns, m![H % 60 = 12, Qs % 1024]> = weight
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Qs]>(12 * 0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, Cluster, HiddenRowsByColumns, m![H % 60 = 12, Qs % 1024]> = weight
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Qs]>(12 * 1)
        .to_dm(&mut ctx.tdma);
    let tile2: DmTensor<f8e4m3, Chip, Cluster, HiddenRowsByColumns, m![H % 60 = 12, Qs % 1024]> = weight
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Qs]>(12 * 2)
        .to_dm(&mut ctx.tdma);
    let tile3: DmTensor<f8e4m3, Chip, Cluster, HiddenRowsByColumns, m![H % 60 = 12, Qs % 1024]> = weight
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Qs]>(12 * 3)
        .to_dm(&mut ctx.tdma);
    let tile4: DmTensor<f8e4m3, Chip, Cluster, HiddenRowsByColumns, m![H % 60 = 12, Qs % 1024]> = weight
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Qs]>(12 * 4)
        .to_dm(&mut ctx.tdma);

    let x: DmTensor<bf16, Chip, Cluster, HiddenRowsByColumns, m![Qs % 1024]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, HiddenRowsByColumns, m![1], m![Qs % 1024]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 16 % 64], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 64], m![Qs % 16]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 60]> = DmTensor::new();

    ctx.main
        .begin(tile0.view())
        .fetch::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60 = 12, Qs / 16 % 64], m![Qs % 16]>()
        .contract_outer::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * 0));
    ctx.main
        .begin(tile1.view())
        .fetch::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60 = 12, Qs / 16 % 64], m![Qs % 16]>()
        .contract_outer::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * 1));
    ctx.main
        .begin(tile2.view())
        .fetch::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60 = 12, Qs / 16 % 64], m![Qs % 16]>()
        .contract_outer::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * 2));
    ctx.main
        .begin(tile3.view())
        .fetch::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60 = 12, Qs / 16 % 64], m![Qs % 16]>()
        .contract_outer::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * 3));
    ctx.main
        .begin(tile4.view())
        .fetch::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 60 = 12, Qs / 16 % 64], m![Qs % 16]>()
        .contract_outer::<m![H % 60 = 12, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * 4));

    // 60 rows do not tile into 8-wide f32 vector packets, so gather the [H] vector onto the
    // Slice layout first and apply the per-channel scale there.
    let contraction: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = contraction.to_dm(&mut ctx.tdma);
    apply_output_channel_scale(ctx, &contraction, weight_scale)
}

type HiddenRows = m![H / 60, 1 # 4];
type HiddenRowsByColumns = m![H / 60, Qs / 1024];

fn apply_output_channel_scale(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // The [H] f32 scale (15 KB) does not fit the 8 KB VRF, so scale in two 1920-wide tiles.
    const TILE: usize = 1920;
    const TILES: usize = H::SIZE / TILE;

    let weight_scale: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = weight_scale.to_dm(&mut ctx.tdma);
    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = DmTensor::new();

    for i in 0..TILES {
        let x_tile = x.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(TILE * i);
        let scale_tile = weight_scale.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(TILE * i);

        let scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H = 1920]> = ctx
            .sub
            .begin(scale_tile)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .to_vrf();

        ctx.main
            .begin(x_tile)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H = 1920 / 4], m![H = 1920 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_concat::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_final()
            .cast::<bf16, m![H = 1920 % 8 # 16]>()
            .commit_trim::<m![H = 1920 % 8]>()
            .commit_view(output.view_mut().tile::<m![H], 1920, m![H = 1920 #{!} 3840]>(TILE * i));
    }

    output
}
