
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

/// Both clusters do real work on the up/gate projections: the L rows are split across the
/// two clusters, then 128 row groups per cluster, and H across two 1920-column halves
/// (512 slices x 60 rows x 1920 columns). The two half partials are summed across slices.
type UpGateClusters = m![L / 7680];
type UpGateRowsSplit = m![L / 60 % 128, 1 # 2];
type UpGateRowsByColumns = m![L / 60 % 128, H / 1920];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    // All 5 tiles per matrix (and their block scales) are issued up front into distinct
    // buffers so the loads stream back to back while each tile is dequantized as it lands.
    // Only the packed f4 tiles reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the
    // fetch stage of the scale pass.
    let up0 = load_up_gate_rows(ctx, up_weight_packed, 0);
    let up_scale0 = load_up_gate_scale(ctx, up_weight_scale, 0);
    let gate0 = load_up_gate_rows(ctx, gate_weight_packed, 0);
    let gate_scale0 = load_up_gate_scale(ctx, gate_weight_scale, 0);
    let up1 = load_up_gate_rows(ctx, up_weight_packed, 1);
    let up_scale1 = load_up_gate_scale(ctx, up_weight_scale, 1);
    let gate1 = load_up_gate_rows(ctx, gate_weight_packed, 1);
    let gate_scale1 = load_up_gate_scale(ctx, gate_weight_scale, 1);
    let up2 = load_up_gate_rows(ctx, up_weight_packed, 2);
    let up_scale2 = load_up_gate_scale(ctx, up_weight_scale, 2);
    let gate2 = load_up_gate_rows(ctx, gate_weight_packed, 2);
    let gate_scale2 = load_up_gate_scale(ctx, gate_weight_scale, 2);
    let up3 = load_up_gate_rows(ctx, up_weight_packed, 3);
    let up_scale3 = load_up_gate_scale(ctx, up_weight_scale, 3);
    let gate3 = load_up_gate_rows(ctx, gate_weight_packed, 3);
    let gate_scale3 = load_up_gate_scale(ctx, gate_weight_scale, 3);
    let up4 = load_up_gate_rows(ctx, up_weight_packed, 4);
    let up_scale4 = load_up_gate_scale(ctx, up_weight_scale, 4);
    let gate4 = load_up_gate_rows(ctx, gate_weight_packed, 4);
    let gate_scale4 = load_up_gate_scale(ctx, gate_weight_scale, 4);

    // Each slice needs only its 1920-wide half of x.
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![H % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .collect::<m![H / 16 % 120], m![H % 16]>()
        .to_trf();

    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = DmTensor::new();
    let w = dequant_up_gate_rows(ctx, &up0, &up_scale0);
    contract_up_gate_rows(ctx, &x_trf, &w, 0, &mut up);
    let w = dequant_up_gate_rows(ctx, &gate0, &gate_scale0);
    contract_up_gate_rows(ctx, &x_trf, &w, 0, &mut gate);
    let w = dequant_up_gate_rows(ctx, &up1, &up_scale1);
    contract_up_gate_rows(ctx, &x_trf, &w, 1, &mut up);
    let w = dequant_up_gate_rows(ctx, &gate1, &gate_scale1);
    contract_up_gate_rows(ctx, &x_trf, &w, 1, &mut gate);
    let w = dequant_up_gate_rows(ctx, &up2, &up_scale2);
    contract_up_gate_rows(ctx, &x_trf, &w, 2, &mut up);
    let w = dequant_up_gate_rows(ctx, &gate2, &gate_scale2);
    contract_up_gate_rows(ctx, &x_trf, &w, 2, &mut gate);
    let w = dequant_up_gate_rows(ctx, &up3, &up_scale3);
    contract_up_gate_rows(ctx, &x_trf, &w, 3, &mut up);
    let w = dequant_up_gate_rows(ctx, &gate3, &gate_scale3);
    contract_up_gate_rows(ctx, &x_trf, &w, 3, &mut gate);
    let w = dequant_up_gate_rows(ctx, &up4, &up_scale4);
    contract_up_gate_rows(ctx, &x_trf, &w, 4, &mut up);
    let w = dequant_up_gate_rows(ctx, &gate4, &gate_scale4);
    contract_up_gate_rows(ctx, &x_trf, &w, 4, &mut gate);

    // Bring both results back to the single-cluster row layout geglu works on, through HBM
    // (a cross-cluster DM-to-DM DMA is rejected by the synchronization checker).
    let mut up_hbm: HbmTensor<bf16, Chip, m![L]> = HbmTensor::new();
    up.view().to_hbm_view(&mut ctx.tdma, up_hbm.view_mut());
    let mut gate_hbm: HbmTensor<bf16, Chip, m![L]> = HbmTensor::new();
    gate.view().to_hbm_view(&mut ctx.tdma, gate_hbm.view_mut());
    let up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = up_hbm.to_dm(&mut ctx.tdma);
    let gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = gate_hbm.to_dm(&mut ctx.tdma);

    (up, gate)
}

/// Loads 12 packed rows x each slice's 1920-column half of one up/gate matrix.
fn load_up_gate_rows(
    ctx: &mut Context,
    packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    pass: usize,
) -> DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H % 1920]> {
    packed
        .view()
        .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Loads 12 rows' block scales for each slice's 1920-column half.
fn load_up_gate_scale(
    ctx: &mut Context,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pass: usize,
) -> DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> {
    scale
        .view()
        .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H / 16]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Dequantizes 12 rows' column half of one up/gate matrix to bf16.
fn dequant_up_gate_rows(
    ctx: &mut Context,
    packed: &DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H % 1920]>,
    scale: &DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H % 1920]> {
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(scale.view())
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
        .to_vrf();

    ctx.main
        .begin(packed.view())
        .fetch::<m![L % 60 = 12, H / 32 % 60], m![H % 32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Contracts 12 dequantized rows with `x_trf` and sums the two column halves into `out`.
fn contract_up_gate_rows(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]>,
    weight: &DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H % 1920]>,
    pass: usize,
    out: &mut DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]>,
) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRowsSplit, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(out.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12 * pass));
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
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
    let (up, gate) = project_up_and_gate(
        ctx,
        x,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    // Stage the geglu output through HBM: a DM-to-DM relayout into the column-split layout
    // costs 11.7k cycles, an HBM round trip a fraction of that (see V7).
    let mut x_hbm: HbmTensor<bf16, Chip, m![L]> = HbmTensor::new();
    x.view().to_hbm_view(&mut ctx.tdma, x_hbm.view_mut());
    let down = project_down(ctx, &x_hbm, down_weight_packed, down_weight_scale);

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

/// Both clusters do real work on the down projection: hidden rows are split across the
/// two clusters, then 32 row groups per cluster, and L across 8 column chunks (512 slices x
/// 60 rows x 1920 columns). The chunk partials are summed across slices within a cluster.
pub(crate) type DownClusters = m![H / 1920];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

pub(crate) fn project_down(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![L]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // All 5 tiles (and their block scales) are issued up front into distinct buffers so the
    // loads stream back to back while each tile is dequantized as it lands.
    let tile0 = load_down_rows(ctx, down_weight_packed, 0);
    let scale0 = load_down_scale(ctx, down_weight_scale, 0);
    let tile1 = load_down_rows(ctx, down_weight_packed, 1);
    let scale1 = load_down_scale(ctx, down_weight_scale, 1);
    let tile2 = load_down_rows(ctx, down_weight_packed, 2);
    let scale2 = load_down_scale(ctx, down_weight_scale, 2);
    let tile3 = load_down_rows(ctx, down_weight_packed, 3);
    let scale3 = load_down_scale(ctx, down_weight_scale, 3);
    let tile4 = load_down_rows(ctx, down_weight_packed, 4);
    let scale4 = load_down_scale(ctx, down_weight_scale, 4);

    // Each slice loads only its 1920-wide chunk of the geglu output from HBM.
    let x: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    let w = dequant_down_rows(ctx, &tile0, &scale0);
    contract_down_rows(ctx, &x_trf, &w, 0, &mut down);
    let w = dequant_down_rows(ctx, &tile1, &scale1);
    contract_down_rows(ctx, &x_trf, &w, 1, &mut down);
    let w = dequant_down_rows(ctx, &tile2, &scale2);
    contract_down_rows(ctx, &x_trf, &w, 2, &mut down);
    let w = dequant_down_rows(ctx, &tile3, &scale3);
    contract_down_rows(ctx, &x_trf, &w, 3, &mut down);
    let w = dequant_down_rows(ctx, &tile4, &scale4);
    contract_down_rows(ctx, &x_trf, &w, 4, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker) onto the Slice layout.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    down_hbm.to_dm(&mut ctx.tdma)
}

/// Loads `ROWS_PER_PASS` packed rows x each slice's L / 1920 column chunk of the down matrix.
fn load_down_rows(
    ctx: &mut Context,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    pass: usize,
) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> {
    down_weight_packed
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Loads `ROWS_PER_PASS` rows' block scales for each slice's L / 1920 column chunk.
fn load_down_scale(
    ctx: &mut Context,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    pass: usize,
) -> DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> {
    down_weight_scale
        .view()
        .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L / 16]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Dequantizes `ROWS_PER_PASS` rows' column chunk of the down matrix to bf16.
fn dequant_down_rows(
    ctx: &mut Context,
    packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]>,
    down_weight_scale: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]>,
) -> DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> {
    let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(down_weight_scale.view())
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

    // f4 -> f8 lookup and f8 -> f32 cast in the fetch stage; no f8 copy is written to DM.
    ctx.main
        .begin(packed.view())
        .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

/// Contracts `ROWS_PER_PASS` dequantized rows with `x_trf` and sums the eight column chunks into `down`.
fn contract_down_rows(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]>,
    down_weight: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]>,
    pass: usize,
    down: &mut DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]>,
) {
    ctx.main
        .begin(down_weight.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * pass));
}
