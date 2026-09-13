"""gen_v390.py <mlp.rs> <rmsnorm.rs> <ops.rs> <mode: arm|submit> -- V390: the ffn down stage split by column halves.

Today `DownClusters = H / 1920` (a cluster owns half the rows) so every down slice consumes all of L -- both clusters'
geglu output -- and the per-cluster 1/s scalar has to cross clusters through HBM: a 32 B store that takes ~6.6k on
hardware plus a second ExplicitSync, both on cluster 1's critical path (V366 spans; V384's aligned store changed
nothing). Here a cluster owns half the columns: `DownClustersC = L / 7680`, `DownRowsByColumnsC = [H / 60, L / 1920 % 4]`
(64 row groups x 4 chunks = 256 slices), `DownRowsC = [H / 60, 1 # 4]`. The weight tiles and scales carry the same bytes
in the new split, the x2 reload brings a cluster its own four chunks, 1/s is the replicated `inv_s_all` reshaped (no
store, no sync, no reload, no switch), pass B reduces over a ring of 4, and the down output is two per-cluster partial
[H] rows (`[L / 7680, H]` bf16, one 15 KB store) that the post-FF norm loads at once and sums in one f32 pass before the
mean-square / rms / final passes.
New functions only: `load_down_rows_*_c`, `contract_down_rows_*_lane_folded_c`, `reduce_down_rows_*_c`,
`stage_geglu_hi_lo_hbm_x2`, `feedforward_fo_t1_c` (mlp.rs), `load_reducing_pair`, `normalize_add_gate_reduced_t1_two`
(rmsnorm.rs); mode=arm adds the ops arm `decoder_feedforward_c`, mode=submit swaps the production body.
"""
import io
import re
import sys

mlp_path, rms_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
assert mode in ("arm", "submit"), mode

m = io.open(mlp_path, encoding="utf-8").read()


def block(text, head, doc=None, end_marker="\n}\n"):
    start = text.index(head)
    if doc is not None:
        d = text.rfind(doc, 0, start)
        assert d != -1 and text[d:start].count("\n") <= 3, (head, doc)
        start = d
    end = text.index(end_marker, start) + len(end_marker)
    return text[start:end]


def retype(s):
    s = re.sub(r"\bDownRowsByColumns\b", "DownRowsByColumnsC", s)
    s = re.sub(r"\bDownClusters\b", "DownClustersC", s)
    s = re.sub(r"\bDownRows\b", "DownRowsC", s)
    return s


types = (
    "// ---------------------------------------------------------------------------------------------\n"
    "// V390: the down stage split by column halves. A cluster owns half the columns of every row, so each down slice\n"
    "// consumes only its own cluster's geglu chunks and the per-cluster 1/s never leaves the chip; the two per-cluster\n"
    "// partial rows are summed by the post-FF norm. Pass B reduces the four chunk slices of a row group (ring 4).\n"
    "// ---------------------------------------------------------------------------------------------\n"
    "pub(crate) type DownClustersC = m![L / 7680];\n"
    "pub(crate) type DownRowsC = m![H / 60, 1 # 4];\n"
    "pub(crate) type DownRowsByColumnsC = m![H / 60, L / 1920 % 4];\n"
    "stage_packet_fns!(stage_packet_down_c, DownClustersC, DownRowsByColumnsC);\n\n"
)

# 1. the tile macros with the new layout types
tile_macro = block(m, "macro_rules! down_tile_fns {", end_marker="\n}\n")
tile_macro_c = retype(tile_macro).replace("macro_rules! down_tile_fns {", "macro_rules! down_tile_fns_c {", 1)
tile_macro_c += "down_tile_fns_c!(load_down_rows_16_c, contract_down_rows_16_c, reduce_down_rows_16_c, 16);\n"
tile_macro_c += "down_tile_fns_c!(load_down_rows_12_c, contract_down_rows_12_c, reduce_down_rows_12_c, 12);\n\n"
lane_macro = block(m, "macro_rules! down_tile_lane_folded_fns {", end_marker="\n}\n")
lane_macro_c = retype(lane_macro).replace("macro_rules! down_tile_lane_folded_fns {", "macro_rules! down_tile_lane_folded_fns_c {", 1)
lane_macro_c += "down_tile_lane_folded_fns_c!(contract_down_rows_16_lane_folded_c, 16);\n"
lane_macro_c += "down_tile_lane_folded_fns_c!(contract_down_rows_12_lane_folded_c, 12);\n\n"

# 2. the geglu staging without the 1/s store
stage = block(m, "fn stage_geglu_hi_lo_hbm_one_store(", doc="/// `stage_geglu_hi_lo_hbm` with both pieces written by one store command.")
old_sig = ") -> (HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]>, HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>) {"
old_tail = (
    "    let inv_s_one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = unsafe { inv_s_all.reshape() };\n"
    "    let mut inv_s_hbm: HbmTensor<f32, Chip, m![L / 7680, 1 # 8]> = HbmTensor::new();\n"
    "    inv_s_one.view().to_hbm_view(&mut ctx.tdma, inv_s_hbm.view_mut());\n"
    "    (x2_hbm, inv_s_hbm)\n"
)
assert stage.count(old_sig) == 1 and stage.count(old_tail) == 1, "stage"
stage_c = stage.replace(old_sig, ") -> (HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]>, DmTensor<f32, Chip, UpGateClusters, m![Dummy256], m![1 # 8]>) {", 1)
stage_c = stage_c.replace(old_tail, "    // V390: 1/s stays on chip -- every slice of both clusters already holds its cluster's value.\n    (x2_hbm, inv_s_all)\n", 1)
stage_c = stage_c.replace("fn stage_geglu_hi_lo_hbm_one_store(", "fn stage_geglu_hi_lo_hbm_x2(", 1)
stage_c = stage_c.replace("/// `stage_geglu_hi_lo_hbm` with both pieces written by one store command.",
                          "/// V390: `stage_geglu_hi_lo_hbm_one_store` returning 1/s on chip instead of storing it.", 1)

# 3. the feedforward body
ff = block(m, "pub(crate) fn feedforward_fo_t1(", doc="/// V366 (T1):")
subs = [
    ("pub(crate) fn feedforward_fo_t1(", "pub(crate) fn feedforward_fo_t1_c("),
    (") -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>, DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>) {",
     ") -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![L / 7680, H % 480]>, DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>) {"),
    ("    let down0 = load_down_rows_16(ctx, down_weight_packed, 0);\n", "    let down0 = load_down_rows_16_c(ctx, down_weight_packed, 0);\n"),
    ("    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =\n",
     "    let down_scale: DmTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![H % 60, L / 16 % 120]> =\n"),
    ("    let down1 = load_down_rows_16(ctx, down_weight_packed, 16);\n", "    let down1 = load_down_rows_16_c(ctx, down_weight_packed, 16);\n"),
    ("    let down2 = load_down_rows_16(ctx, down_weight_packed, 32);\n", "    let down2 = load_down_rows_16_c(ctx, down_weight_packed, 32);\n"),
    ("    let down3 = load_down_rows_12(ctx, down_weight_packed, 48);\n", "    let down3 = load_down_rows_12_c(ctx, down_weight_packed, 48);\n"),
    ("    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm_one_store(ctx, &x);\n    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);\n",
     "    let (x2_hbm, inv_s_all) = stage_geglu_hi_lo_hbm_x2(ctx, &x);\n"
     "    // V390: 1/s is replicated on every slice of its cluster already; the down layout is a relabelling of it.\n"
     "    let inv_s_c: DmTensor<f32, Chip, DownClustersC, DownRowsByColumnsC, m![1 # 8]> = unsafe { inv_s_all.reshape() };\n"
     "    let inv_s_vrf = stage_packet_down_c(ctx, &inv_s_c);\n"),
    ("    let x: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);\n",
     "    let x: DmTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);\n"),
    ("    let x_trf_lane: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![Dummy2], m![L % 1920]> = ctx\n",
     "    let x_trf_lane: TrfTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![Dummy2], m![L % 1920]> = ctx\n"),
    ("    let mut down_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();\n",
     "    let mut down_hbm: HbmTensor<bf16, Chip, m![L / 7680, H]> = HbmTensor::new();\n"),
    ("    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();\n",
     "    let mut down: DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60]> = DmTensor::new();\n"),
    ("contract_down_rows_16_lane_folded(", "contract_down_rows_16_lane_folded_c("),
    ("contract_down_rows_12_lane_folded(", "contract_down_rows_12_lane_folded_c("),
    ("reduce_down_rows_16(", "reduce_down_rows_16_c("),
    ("reduce_down_rows_12(", "reduce_down_rows_12_c("),
    ("    let down = rmsnorm::load_reducing::<Cluster>(ctx, &down_hbm);\n",
     "    // V390: both clusters' partial rows come back in one load; the norm sums them.\n"
     "    let down = rmsnorm::load_reducing_pair::<Cluster>(ctx, &down_hbm);\n"),
]
for old, new in subs:
    assert ff.count(old) >= 1, old
    ff = ff.replace(old, new)
ff = ff.replace("/// V366 (T1):", "/// V390: `feedforward_fo_t1` with the down stage split by column halves. V366 (T1):", 1)

m = m.rstrip("\n") + "\n\n" + types + tile_macro_c + lane_macro_c + stage_c + "\n" + ff
io.open(mlp_path, "w", encoding="utf-8", newline="\n").write(m)

# 4. rmsnorm: the pair load and the two-partial norm
r = io.open(rms_path, encoding="utf-8").read()
norm = block(r, "pub(crate) fn normalize_add_gate_reduced_t1<Cluster: M>(")
old_head = (
    "pub(crate) fn normalize_add_gate_reduced_t1<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,\n"
)
assert norm.count(old_head) == 1, "norm head"
new_head = (
    "pub(crate) fn normalize_add_gate_reduced_t1_two<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    xp: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![L / 7680, H % 480]>,\n"
)
norm_c = norm.replace(old_head, new_head, 1)
old_first = (
    "    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx\n"
    "        .main\n"
    "        .begin(x.view())\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
)
assert norm_c.count(old_first) == 1, "ms pass"
new_first = (
    "    // V390: the two per-cluster partial rows are summed once in f32; the mean-square and final passes read the sum.\n"
    "    let x1_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .sub\n"
    "        .begin(xp.view().tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H % 480]>(1))\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .to_vrf();\n"
    "    let x: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .main\n"
    "        .begin(xp.view().tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H % 480]>(0))\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::AddF, &x1_vrf)\n"
    "        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_final()\n"
    "        .commit_trim::<m![H % 8]>()\n"
    "        .commit();\n"
    "\n"
    "    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx\n"
    "        .main\n"
    "        .begin(x.view())\n"
    "        .fetch::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
)
norm_c = norm_c.replace(old_first, new_first, 1)
old_final = (
    "    ctx.main\n"
    "        .begin(x.view())\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)\n"
)
assert norm_c.count(old_final) == 1, "final pass"
new_final = (
    "    ctx.main\n"
    "        .begin(x.view())\n"
    "        .fetch::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)\n"
)
norm_c = norm_c.replace(old_final, new_final, 1)
pair_load = (
    "/// V390: both clusters' partial down rows (`[L / 7680, H]`) loaded into the reducing layout by one command.\n"
    "pub(crate) fn load_reducing_pair<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &HbmTensor<bf16, Chip, m![L / 7680, H]>,\n"
    ") -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![L / 7680, H % 480]> {\n"
    "    x.to_dm(&mut ctx.tdma)\n"
    "}\n\n"
    "/// V390: `normalize_add_gate_reduced_t1` over the sum of two per-cluster partial rows.\n"
)
imp = "use crate::axes::{"
assert r.count(imp) == 1, "axes import"
line_end = r.index("};", r.index(imp))
imports = r[r.index(imp):line_end]
if re.search(r"\bL\b", imports) is None:
    r = r[:line_end] + ", L" + r[line_end:] if not imports.rstrip().endswith(",") else r[:line_end] + " L" + r[line_end:]
r = r.rstrip("\n") + "\n\n" + pair_load + norm_c
io.open(rms_path, "w", encoding="utf-8", newline="\n").write(r)

# 5. ops
o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn decoder_feedforward(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_call = "    let (x, g) = shared::mlp::feedforward_fo_t1(\n"
old_norm = "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1::<Cluster>(ctx, &x, &g, post_ff_rms_weight, &residual, layer_scalar);\n"
assert kfn.count(old_call) == 1 and kfn.count(old_norm) == 1, "ffn body"
body = kfn.replace(old_call, "    // V390: the down stage is split by column halves; the two partial rows are summed by the post-FF norm.\n    let (x, g) = shared::mlp::feedforward_fo_t1_c(\n", 1)
body = body.replace(old_norm, "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1_two::<Cluster>(ctx, &x, &g, post_ff_rms_weight, &residual, layer_scalar);\n", 1)
if mode == "arm":
    arm = body.replace("pub fn decoder_feedforward(", "pub fn decoder_feedforward_c(", 1)
    o = o[:ke] + "\n/// V390 arm c: production ffn with the down stage split by column halves.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V390", mode, "ok")
