
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 4;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    // The weights are streamed 4 rows at a time so that a pass can start as soon as its
    // tile lands instead of behind a whole-matrix load on the DMA queue. Only the packed f4
    // tiles ever reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of
    // the scale pass.
    let up_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);

    let mut up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();

    for i in 0..PASSES {
        let cur_up = load_up_gate_rows(ctx, up_weight_packed, i);
        project_up_gate_rows(ctx, x_trf, &cur_up, &up_weight_scale, i, &mut up);
        let cur_gate = load_up_gate_rows(ctx, gate_weight_packed, i);
        project_up_gate_rows(ctx, x_trf, &cur_gate, &gate_weight_scale, i, &mut gate);
    }

    (up, gate)
}

/// Loads `ROWS_PER_PASS` packed rows of one up/gate matrix into every slice's row group.
fn load_up_gate_rows(
    ctx: &mut Context,
    packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    pass: usize,
) -> DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60 = 4, H]> {
    packed
        .view()
        .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(4 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Dequantizes `ROWS_PER_PASS` rows of one up/gate matrix and contracts them with `x_trf`.
fn project_up_gate_rows(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    packed: &DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60 = 4, H]>,
    scale: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]>,
    pass: usize,
    out: &mut DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 4, H / 16]> = ctx
        .sub
        .begin(scale.view().tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16]>(4 * pass))
        .fetch::<m![L % 60 = 4], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    let weight: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 4, H]> = ctx
        .main
        .begin(packed.view())
        .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 4, H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 4, H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![L % 60 = 4, H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(weight.view())
        .fetch::<m![L % 60 = 4, H / 16], m![H % 16]>()
        .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 60 = 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 4]>()
        .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![L % 60 = 4 # 16]>()
        .commit_trim::<m![L % 60 = 4]>()
        .commit_view(out.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(4 * pass));
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // The down-projection block scales do not depend on anything computed here, so put
    // their load on the DMA queue before the up/gate work instead of after it.
    let down_weight_scale: DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    let x: DmTensor<bf16, Chip, Cluster, UpGateRows, m![H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let (up, gate) = project_up_and_gate(
        ctx,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    let down = project_down(ctx, &x, down_weight_packed, &down_weight_scale);

    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

pub(crate) fn geglu(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let up_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

pub(crate) type DownRows = m![H / 120, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 120, L / 1920];

pub(crate) fn project_down(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // One relayout from the geglu layout straight to the column-split layout the contraction
    // reads (each slice needs 1920 of L), instead of replicating all of L to every slice first.
    let x: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    const ROWS_PER_SLICE: usize = 120;
    const ROWS_PER_PASS: usize = 4;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    let mut down: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = DmTensor::new();

    // Two tiles are live per iteration so they land in different DM buffers: the second
    // tile's load overlaps the first tile's dequant instead of waiting for its buffer.
    for j in 0..PASSES / 2 {
        let tile_a = load_down_rows(ctx, down_weight_packed, 2 * j);
        let tile_b = load_down_rows(ctx, down_weight_packed, 2 * j + 1);
        project_down_rows(ctx, &x_trf, &tile_a, down_weight_scale, 2 * j, &mut down);
        project_down_rows(ctx, &x_trf, &tile_b, down_weight_scale, 2 * j + 1, &mut down);
    }

    down.to_dm(&mut ctx.tdma)
}

/// Loads four packed rows x each slice's L / 1920 column chunk of the down matrix.
fn load_down_rows(
    ctx: &mut Context,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    pass: usize,
) -> DmTensor<f4e2m1, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L % 1920]> {
    down_weight_packed
        .view()
        .tile::<m![H % 120], 4, m![H / 120, H % 120 = 4 # 120, L]>(4 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Dequantizes four rows' column chunk, contracts with `x_trf`, and sums the eight chunks.
fn project_down_rows(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]>,
    packed: &DmTensor<f4e2m1, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L % 1920]>,
    down_weight_scale: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]>,
    pass: usize,
    down: &mut DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]>,
) {
    let down_weight_scale_vrf: VrfTensor<f32, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 120], 4, m![H % 120 = 4 # 120, L / 16 % 120]>(4 * pass),
            )
            .fetch::<m![H % 120 = 4], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 120 = 4, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

    // f4 -> f8 lookup and f8 -> f32 cast in the fetch stage; no f8 copy is written to DM.
    let down_weight: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L % 1920]> = ctx
        .main
        .begin(packed.view())
        .fetch::<m![H % 120 = 4, L / 32 % 60], m![L % 32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 = 4, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 = 4, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
        .vector_widen_concat::<m![H % 120 = 4, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight.view())
        .fetch::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 120 = 4, L / 32 % 60], m![L % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 4]>()
        .contract_lane::<m![H % 120 = 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![H % 120 = 4 # 16]>()
        .commit_trim::<m![H % 120 = 4]>()
        .commit_view(down.view_mut().tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120]>(4 * pass));
}
