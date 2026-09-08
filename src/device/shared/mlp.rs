
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRows, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 12;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    // The weights are streamed 12 rows at a time so that a pass can start as soon as its
    // tile lands instead of behind a whole-matrix load on the DMA queue. Only the packed f4
    // tiles ever reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of
    // the scale pass, which works on 1920-column halves so its scale slice fits the 8 KB VRF.
    // Fewer, larger passes cut the per-pass fixed costs (one anonymous 4 KB load each).
    let mut up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();

    // The first two dequant passes need only their tiles, so they run while x is still being
    // replicated; x reaches the TRF (a Sub-context op, ordered after their scale preloads)
    // just before the first contraction.
    let first_up = load_up_gate_rows(ctx, up_weight_packed, 0);
    let first_up_scale = load_up_gate_scale(ctx, up_weight_scale, 0);
    let first_gate = load_up_gate_rows(ctx, gate_weight_packed, 0);
    let first_gate_scale = load_up_gate_scale(ctx, gate_weight_scale, 0);
    let first_up_weight = dequant_up_gate_rows(ctx, &first_up, &first_up_scale);
    let first_gate_weight = dequant_up_gate_rows(ctx, &first_gate, &first_gate_scale);
    let x_trf: TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();
    contract_up_gate_rows(ctx, &x_trf, &first_up_weight, 0, &mut up);
    contract_up_gate_rows(ctx, &x_trf, &first_gate_weight, 0, &mut gate);
    // Two passes (four tiles) are loaded per iteration before any of them is dequantized so
    // that the loads land in distinct buffers and overlap the dequant of the previous tiles.
    const PAIRS: usize = (PASSES - 1) / 2;
    for k in 0..PAIRS {
        let i = 2 * k + 1;
        let i2 = 2 * k + 2;
        let up_a = load_up_gate_rows(ctx, up_weight_packed, i);
        let up_a_scale = load_up_gate_scale(ctx, up_weight_scale, i);
        let gate_a = load_up_gate_rows(ctx, gate_weight_packed, i);
        let gate_a_scale = load_up_gate_scale(ctx, gate_weight_scale, i);
        let up_b = load_up_gate_rows(ctx, up_weight_packed, i2);
        let up_b_scale = load_up_gate_scale(ctx, up_weight_scale, i2);
        let gate_b = load_up_gate_rows(ctx, gate_weight_packed, i2);
        let gate_b_scale = load_up_gate_scale(ctx, gate_weight_scale, i2);
        let w = dequant_up_gate_rows(ctx, &up_a, &up_a_scale);
        contract_up_gate_rows(ctx, &x_trf, &w, i, &mut up);
        let w = dequant_up_gate_rows(ctx, &gate_a, &gate_a_scale);
        contract_up_gate_rows(ctx, &x_trf, &w, i, &mut gate);
        let w = dequant_up_gate_rows(ctx, &up_b, &up_b_scale);
        contract_up_gate_rows(ctx, &x_trf, &w, i2, &mut up);
        let w = dequant_up_gate_rows(ctx, &gate_b, &gate_b_scale);
        contract_up_gate_rows(ctx, &x_trf, &w, i2, &mut gate);
    }

    (up, gate)
}

/// Loads `ROWS_PER_PASS` packed rows of one up/gate matrix into every slice's row group.
fn load_up_gate_rows(
    ctx: &mut Context,
    packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    pass: usize,
) -> DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60 = 12, H]> {
    packed
        .view()
        .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Loads `ROWS_PER_PASS` rows of block scales of one up/gate matrix.
fn load_up_gate_scale(
    ctx: &mut Context,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pass: usize,
) -> DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60 = 12, H / 16]> {
    scale
        .view()
        .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H / 16]>(12 * pass)
        .to_dm(&mut ctx.tdma)
}

/// Dequantizes `ROWS_PER_PASS` rows of one up/gate matrix to bf16, one 1920-column half at a time.
fn dequant_up_gate_rows(
    ctx: &mut Context,
    packed: &DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60 = 12, H]>,
    scale: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60 = 12, H / 16]>,
) -> DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 12, H]> {
    let mut weight: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 12, H]> = DmTensor::new();
    let scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 12, H / 16 = 120]> = ctx
        .sub
        .begin(scale.view().tile::<m![H / 16], 120, m![L % 60 = 12, H / 16 = 120 # 240]>(0))
        .fetch::<m![L % 60 = 12], m![H / 16 = 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 = 120 / 8], m![H / 16 = 120 % 8]>()
        .to_vrf();
    ctx.main
        .begin(packed.view().tile::<m![H], 1920, m![L % 60 = 12, H = 1920 # 3840]>(0))
        .fetch::<m![L % 60 = 12], m![H = 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H = 1920 / 8], m![H = 1920 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H = 1920 / 4], m![H = 1920 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![L % 60 = 12, H = 1920 / 8], m![H = 1920 % 8]>()
        .vector_final()
        .cast::<bf16, m![H = 1920 % 8 # 16]>()
        .commit_trim::<m![H = 1920 % 8]>()
        .commit_view(weight.view_mut().tile::<m![H], 1920, m![L % 60 = 12, H = 1920 #{!} 3840]>(0));
    let scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 12, H / 16 = 120]> = ctx
        .sub
        .begin(scale.view().tile::<m![H / 16], 120, m![L % 60 = 12, H / 16 = 120 # 240]>(120))
        .fetch::<m![L % 60 = 12], m![H / 16 = 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 = 120 / 8], m![H / 16 = 120 % 8]>()
        .to_vrf();
    ctx.main
        .begin(packed.view().tile::<m![H], 1920, m![L % 60 = 12, H = 1920 # 3840]>(1920))
        .fetch::<m![L % 60 = 12], m![H = 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H = 1920 / 8], m![H = 1920 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H = 1920 / 4], m![H = 1920 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![L % 60 = 12, H = 1920 / 8], m![H = 1920 % 8]>()
        .vector_final()
        .cast::<bf16, m![H = 1920 % 8 # 16]>()
        .commit_trim::<m![H = 1920 % 8]>()
        .commit_view(weight.view_mut().tile::<m![H], 1920, m![L % 60 = 12, H = 1920 #{!} 3840]>(1920));

    weight
}

/// Contracts `ROWS_PER_PASS` dequantized rows with `x_trf` into `out`.
fn contract_up_gate_rows(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    weight: &DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 12, H]>,
    pass: usize,
    out: &mut DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![L % 60 = 12, H / 16], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(out.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12 * pass));
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
    let x: DmTensor<bf16, Chip, Cluster, UpGateRows, m![H]> = unsafe { x.reshape() };

    let (up, gate) = project_up_and_gate(
        ctx,
        &x,
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
