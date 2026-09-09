
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::Cluster;
use crate::device::shared::rmsnorm::{self, ReducingSlices};

const INVSQRT2: f32 = 0.70710678118f32;

/// Both clusters do real work on the up/gate projections: the L rows are split across the
/// two clusters, then 128 row groups per cluster, and H across two 1920-column halves
/// (512 slices x 60 rows x 1920 columns). The two half partials are summed across slices.
type UpGateClusters = m![L / 7680];
type UpGateRowsSplit = m![L / 60 % 128, 1 # 2];
type UpGateRowsByColumns = m![L / 60 % 128, H / 1920];
/// Eight row groups per slice after the ring-16 gather ahead of the geglu output store.
type UpGateRowsGathered = m![L / 480 % 16, 1 # 16];

/// The up/gate tile helpers for one tile height: load `$rows` packed rows x each slice's
/// 1920-column half, dequantize them to bf16, and contract them with `x_trf` (summing the two
/// column halves into `out`). Stamped out per tile height: 16 (the scale VRF takes 16 x 120
/// f32 = 7.7 KB) and the 12-row remainder.
macro_rules! up_gate_tile_fns {
    ($load:ident, $dequant:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]> {
            packed
                .view()
                .tile::<m![L % 60], $rows, m![L / 60, L % 60 = $rows # 60, H]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        fn $dequant(
            ctx: &mut Context,
            packed: &DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]>,
            scale_all: &DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]>,
            offset: usize,
        ) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]> {
            let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H / 16 % 120]> = ctx
                .sub
                .begin(scale_all.view().tile::<m![L % 60], $rows, m![L % 60 = $rows # 60, H / 16 % 120]>(offset))
                .fetch::<m![L % 60 = $rows], m![H / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = $rows, H / 128 % 15], m![H / 16 % 8]>()
                .to_vrf();

            ctx.main
                .begin(packed.view())
                .fetch::<m![L % 60 = $rows, H / 32 % 60], m![H % 32]>()
                .fetch_table_lookup::<f8e4m3>()
                .fetch_cast::<f32>()
                .collect::<m![L % 60 = $rows, H / 8 % 240], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![L % 60 = $rows, H / 4 % 480], m![H % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_widen_concat::<m![L % 60 = $rows, H / 8 % 240], m![H % 8]>()
                .vector_final()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
                .commit()
        }

        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]>,
            weight: &DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = $rows, H % 1920]>,
            offset: usize,
            out: &mut DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]>,
        ) {
            ctx.main
                .begin(weight.view())
                .fetch::<m![L % 60 = $rows, H / 16 % 120], m![H % 16]>()
                .collect::<m![L % 60 = $rows, H / 16 % 120], m![H % 16]>()
                .contract_outer::<m![L % 60 = $rows, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![L % 60 = $rows]>()
                .contract_lane::<m![L % 60 = $rows], m![1 # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<UpGateRowsSplit, m![L % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![L % 60 = $rows / 4], m![L % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![L % 60 = $rows % 4]>()
                .commit_view(out.view_mut().tile::<m![L % 60], $rows, m![L % 60 = $rows #{!} 60]>(offset));
        }
    };
}
up_gate_tile_fns!(load_up_gate_rows_16, dequant_up_gate_rows_16, contract_up_gate_rows_16, 16);
up_gate_tile_fns!(load_up_gate_rows_12, dequant_up_gate_rows_12, contract_up_gate_rows_12, 12);

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
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    // All weight tiles are issued up front into distinct buffers so the loads stream back to
    // back while each tile is dequantized as it lands. up/gate: 4 tiles per matrix (16, 16,
    // 16 and 12 rows); down: 5 tiles (16, 16, 16, 8 and 4 rows), the last one small so that
    // little dequant + contract work trails the final weight load. Only the packed f4 tiles
    // reach DM: the f4 -> f8 lookup and f8 -> f32 cast run in the fetch stage of the scale
    // pass.
    let up0 = load_up_gate_rows_16(ctx, up_weight_packed, 0);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate0 = load_up_gate_rows_16(ctx, gate_weight_packed, 0);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let up1 = load_up_gate_rows_16(ctx, up_weight_packed, 16);
    let gate1 = load_up_gate_rows_16(ctx, gate_weight_packed, 16);
    let up2 = load_up_gate_rows_16(ctx, up_weight_packed, 32);
    let gate2 = load_up_gate_rows_16(ctx, gate_weight_packed, 32);
    let up3 = load_up_gate_rows_12(ctx, up_weight_packed, 48);
    let gate3 = load_up_gate_rows_12(ctx, gate_weight_packed, 48);
    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);
    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);
    let down3 = load_down_rows_8(ctx, down_weight_packed, 48);
    let down4 = load_down_rows_4(ctx, down_weight_packed, 56);

    // Each slice needs only its 1920-wide half of x.
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![H % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .collect::<m![H / 16 % 120], m![H % 16]>()
        .to_trf();

    // The down tiles do not depend on the geglu output, so their dequantization is
    // interleaved with the up/gate passes (the up/gate phase is DMA-bound and leaves the
    // vector engine idle); only their contractions wait for x.
    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = DmTensor::new();
    let w = dequant_up_gate_rows_16(ctx, &up0, &up_scale, 0);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 0, &mut up);
    let w = dequant_up_gate_rows_16(ctx, &gate0, &gate_scale, 0);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 0, &mut gate);
    let w = dequant_up_gate_rows_16(ctx, &up1, &up_scale, 16);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 16, &mut up);
    let w = dequant_up_gate_rows_16(ctx, &gate1, &gate_scale, 16);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 16, &mut gate);
    let d0 = dequant_down_rows_16(ctx, &down0, &down_scale, 0);
    let w = dequant_up_gate_rows_16(ctx, &up2, &up_scale, 32);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 32, &mut up);
    let w = dequant_up_gate_rows_16(ctx, &gate2, &gate_scale, 32);
    contract_up_gate_rows_16(ctx, &x_trf, &w, 32, &mut gate);
    let d1 = dequant_down_rows_16(ctx, &down1, &down_scale, 16);
    let w = dequant_up_gate_rows_12(ctx, &up3, &up_scale, 48);
    contract_up_gate_rows_12(ctx, &x_trf, &w, 48, &mut up);
    let w = dequant_up_gate_rows_12(ctx, &gate3, &gate_scale, 48);
    contract_up_gate_rows_12(ctx, &x_trf, &w, 48, &mut gate);
    let d2 = dequant_down_rows_16(ctx, &down2, &down_scale, 32);

    // geglu runs in the up/gate reduce layout (see geglu_split); its output is staged through
    // HBM (see V7). Storing 60 rows from each of 256 slices costs 4.5k cycles of descriptors,
    // so eight row groups are first gathered onto one slice over a ring of 16 (the live slices
    // sit two apart): store 1,870, switch 503. Wider rings save nothing more on the store and
    // cost more in the switch.
    let x = geglu_split(ctx, up, gate, up_global_scale, gate_global_scale);
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsGathered, m![L % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsGathered, m![L / 4 % 15, L / 60 % 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 2 })
        .collect::<m![L / 4 % 15, L / 60 % 8], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();
    let mut x_hbm: HbmTensor<bf16, Chip, m![L]> = HbmTensor::new();
    x.view().to_hbm_view(&mut ctx.tdma, x_hbm.view_mut());

    // Each slice loads only its 1920-wide chunk of the geglu output from HBM.
    let x: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = x_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    contract_down_rows_16(ctx, &x_trf, &d0, 0, &mut down);
    contract_down_rows_16(ctx, &x_trf, &d1, 16, &mut down);
    contract_down_rows_16(ctx, &x_trf, &d2, 32, &mut down);
    let w = dequant_down_rows_8(ctx, &down3, &down_scale, 48);
    contract_down_rows_8(ctx, &x_trf, &w, 48, &mut down);
    let w = dequant_down_rows_4(ctx, &down4, &down_scale, 56);
    contract_down_rows_4(ctx, &x_trf, &w, 56, &mut down);

    // Gather the [H] vector from both clusters through HBM (a cross-cluster DM-to-DM DMA is
    // rejected by the synchronization checker), then load it in the layout the post-FF
    // RMSNorm reduces in (8 slices x 480 elements) and apply the global scale there: 1/8 of
    // the pass and no relayout afterwards.
    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    down.view().to_hbm_view(&mut ctx.tdma, down_hbm.view_mut());
    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

/// A scalar packet (one f32 on every live slice of the reduce layout) staged into the VRF.
fn stage_scalar(
    ctx: &mut Context,
    scalar: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]>,
) -> VrfTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]> {
    ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// geglu on up and gate as the up/gate projections leave them: one row group of 60 per live
/// slice, both clusters. 60 f32 do not fill 8-wide packets, so every 4-element packet is
/// padded to 8 and the vector passes work on the live half (the V18 scale-packet pattern).
///
/// The global scales are folded into the two passes as `s_gate / sqrt(2)` (the erf argument)
/// and `s_up * s_gate / 2` (the output factor): gelu(s_gate * g) = s_gate * g / 2 * (1 + erf(s_gate * g / sqrt(2))).
/// The two scalars are derived in tiny passes whose only inputs are the scale loads, so those
/// loads go at the head of the DMA queue and the geglu can run as soon as the projections end
/// (with the scalars loaded by the geglu itself they sat behind the down tiles, and the
/// geglu behind the down dequantization).
pub(crate) fn geglu_split(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]>,
    gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> {
    let s_up: DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]> = up_global_scale.to_dm(&mut ctx.tdma);
    let s_gate: DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let s_gate_vrf = stage_scalar(ctx, &s_gate);

    let erf_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]> = ctx
        .sub
        .begin(s_gate.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let erf_scale_vrf = stage_scalar(ctx, &erf_scale);

    let out_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![1 # 8]> = ctx
        .sub
        .begin(s_up.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_gate_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let out_scale_vrf = stage_scalar(ctx, &out_scale);

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![L % 60]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 15, 1 # 2], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &erf_scale_vrf)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsSplit, m![L / 4 % 15, L % 4 # 8]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 15, 1 # 2], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &out_scale_vrf)
        .vector_widen_concat::<m![L / 4 % 15], m![L % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit()
}

/// Both clusters do real work on the down projection: hidden rows are split across the
/// two clusters, then 32 row groups per cluster, and L across 8 column chunks (512 slices x
/// 60 rows x 1920 columns). The chunk partials are summed across slices within a cluster.
pub(crate) type DownClusters = m![H / 1920];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

/// The down tile helpers for one tile height (see `up_gate_tile_fns`): 16, 8 and 4 rows. The
/// last tile is the small one so that the dequant + contract left after the final weight load
/// is short.
macro_rules! down_tile_fns {
    ($load:ident, $dequant:ident, $contract:ident, $rows:literal) => {
        fn $load(
            ctx: &mut Context,
            down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
            offset: usize,
        ) -> DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]> {
            down_weight_packed
                .view()
                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)
                .to_dm(&mut ctx.tdma)
        }

        fn $dequant(
            ctx: &mut Context,
            packed: &DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]>,
            scale_all: &DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]>,
            offset: usize,
        ) -> DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]> {
            let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L / 16 % 120]> =
                ctx.sub
                    .begin(scale_all.view().tile::<m![H % 60], $rows, m![H % 60 = $rows # 60, L / 16 % 120]>(offset))
                    .fetch::<m![H % 60 = $rows], m![L / 16 % 120]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H % 60 = $rows, L / 128 % 15], m![L / 16 % 8]>()
                    .to_vrf();

            // f4 -> f8 lookup and f8 -> f32 cast in the fetch stage; no f8 copy is written to DM.
            ctx.main
                .begin(packed.view())
                .fetch::<m![H % 60 = $rows, L / 32 % 60], m![L % 32]>()
                .fetch_table_lookup::<f8e4m3>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = $rows, L / 8 % 240], m![L % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 = $rows, L / 4 % 480], m![L % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
                .vector_widen_concat::<m![H % 60 = $rows, L / 8 % 240], m![L % 8]>()
                .vector_final()
                .cast::<bf16, m![L % 8 # 16]>()
                .commit_trim::<m![L % 8]>()
                .commit()
        }

        fn $contract(
            ctx: &mut Context,
            x_trf: &TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]>,
            down_weight: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = $rows, L % 1920]>,
            offset: usize,
            down: &mut DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]>,
        ) {
            ctx.main
                .begin(down_weight.view())
                .fetch::<m![H % 60 = $rows, L / 16 % 120], m![L % 16]>()
                .collect::<m![H % 60 = $rows, L / 16 % 120], m![L % 16]>()
                .contract_outer::<m![H % 60 = $rows, L / 32 % 60], m![L % 32], _, _, _>(x_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 60 = $rows]>()
                .contract_lane::<m![H % 60 = $rows], m![1 # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<DownRows, m![H % 60 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![H % 60 = $rows / 4], m![H % 60 = $rows % 4 # 16]>()
                .commit_trim::<m![H % 60 = $rows % 4]>()
                .commit_view(down.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));
        }
    };
}
down_tile_fns!(load_down_rows_16, dequant_down_rows_16, contract_down_rows_16, 16);
down_tile_fns!(load_down_rows_8, dequant_down_rows_8, contract_down_rows_8, 8);
down_tile_fns!(load_down_rows_4, dequant_down_rows_4, contract_down_rows_4, 4);
