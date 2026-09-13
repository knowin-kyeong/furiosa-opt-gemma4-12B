"""gen_v390c.py <mlp.rs> <rmsnorm.rs> <ops.rs> <mode: arm|submit> -- V390c (arm c4): the column-halves ffn with the down
partial store reduced to cluster 1's half.

Run AFTER gen_v390.py and gen_v390b.py (c2) on the same files. `cluster_tile` cannot lower over a derived cluster axis
(`L / 7680`: "cannot find tag L_7680"), so the down layout's cluster axis is the base axis `Gs` (size 2) instead: the HBM
weight, scale and x2 views are relabelled `[H, Gs, L % 7680]`, `[H, Gs, L / 16 % 480]` and `[Gs, L / 1920 % 4, Dummy2,
L % 1920]` (same wire order) before the loads, 1/s and the down output use `Gs` clusters, and the store writes cluster 1's
partial only (`cluster_tile::<m![Gs], 1, m![Gs = 1 # 2]>(1)` into tile 1 of `[Gs, H]`), 7.7 KB / 64 segments like
production's; cluster 0's partial moves DM-to-DM as in c2 and the c2 norm sums them.
Adds `down_tile_fns_g!` / `down_tile_lane_folded_fns_g!` instances, `feedforward_fo_t1_c4` (mlp.rs),
`load_reducing_half_g` (rmsnorm.rs); mode=arm adds the ops arm `decoder_feedforward_c4`.
"""
import io
import re
import sys

mlp_path, rms_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
assert mode in ("arm", "submit"), mode


def block(text, head, doc=None, end_marker="\n}\n"):
    start = text.index(head)
    if doc is not None:
        d = text.rfind(doc, 0, start)
        assert d != -1 and text[d:start].count("\n") <= 3, (head, doc)
        start = d
    end = text.index(end_marker, start) + len(end_marker)
    return text[start:end]


m = io.open(mlp_path, encoding="utf-8").read()
imp = "use crate::axes::{"
line_end = m.index("};", m.index(imp))
imports = m[m.index(imp):line_end]
if re.search(r"\bGs\b", imports) is None:
    m = m[:line_end] + ", Gs" + m[line_end:]

types = (
    "// ---------------------------------------------------------------------------------------------\n"
    "// V390c: the column-halves down layout with `Gs` as the cluster axis, so that a cluster tile (which only lowers over a\n"
    "// base axis) can store cluster 1's partial alone. The HBM views are relabelled to the same wire order before loading.\n"
    "// ---------------------------------------------------------------------------------------------\n"
    "pub(crate) type DownClustersG = m![Gs];\n"
    "stage_packet_fns!(stage_packet_down_g, DownClustersG, DownRowsByColumnsC);\n\n"
)

# tile macro with Gs clusters and relabelled HBM loads
tile_c = block(m, "macro_rules! down_tile_fns_c {")
old_load = (
    "            packed\n"
    "                .view()\n"
    "                .tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, L]>(offset)\n"
    "                .to_dm(&mut ctx.tdma)\n"
)
assert tile_c.count(old_load) == 1, "tile load"
new_load = (
    "            let pv: HbmTensorView<'_, f4e2m1, Chip, m![H, Gs, L % 7680]> = unsafe { packed.view().reshape() };\n"
    "            pv.tile::<m![H % 60], $rows, m![H / 60, H % 60 = $rows # 60, Gs, L % 7680]>(offset)\n"
    "                .to_dm(&mut ctx.tdma)\n"
)
tile_g = tile_c.replace(old_load, new_load, 1).replace("macro_rules! down_tile_fns_c {", "macro_rules! down_tile_fns_g {", 1)
tile_g = re.sub(r"\bDownClustersC\b", "DownClustersG", tile_g)
tile_g += "down_tile_fns_g!(load_down_rows_16_g, contract_down_rows_16_g, reduce_down_rows_16_g, 16);\n"
tile_g += "down_tile_fns_g!(load_down_rows_12_g, contract_down_rows_12_g, reduce_down_rows_12_g, 12);\n\n"
lane_c = block(m, "macro_rules! down_tile_lane_folded_fns_c {")
lane_g = re.sub(r"\bDownClustersC\b", "DownClustersG", lane_c).replace("macro_rules! down_tile_lane_folded_fns_c {", "macro_rules! down_tile_lane_folded_fns_g {", 1)
lane_g += "down_tile_lane_folded_fns_g!(contract_down_rows_16_lane_folded_g, 16);\n"
lane_g += "down_tile_lane_folded_fns_g!(contract_down_rows_12_lane_folded_g, 12);\n\n"

ff = block(m, "pub(crate) fn feedforward_fo_t1_c2(")
subs = [
    ("pub(crate) fn feedforward_fo_t1_c2(", "pub(crate) fn feedforward_fo_t1_c4("),
    ("load_down_rows_16_c(", "load_down_rows_16_g("),
    ("load_down_rows_12_c(", "load_down_rows_12_g("),
    ("    let down_scale: DmTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![H % 60, L / 16 % 120]> =\n        down_weight_scale.to_dm(&mut ctx.tdma);\n",
     "    let scale_view: HbmTensorView<'_, f8e4m3, Chip, m![H, Gs, L / 16 % 480]> = unsafe { down_weight_scale.view().reshape() };\n"
     "    let down_scale: DmTensor<f8e4m3, Chip, DownClustersG, DownRowsByColumnsC, m![H % 60, L / 16 % 120]> =\n        scale_view.to_dm(&mut ctx.tdma);\n"),
    ("    let inv_s_c: DmTensor<f32, Chip, DownClustersC, DownRowsByColumnsC, m![1 # 8]> = unsafe { inv_s_all.reshape() };\n    let inv_s_vrf = stage_packet_down_c(ctx, &inv_s_c);\n",
     "    let inv_s_c: DmTensor<f32, Chip, DownClustersG, DownRowsByColumnsC, m![1 # 8]> = unsafe { inv_s_all.reshape() };\n    let inv_s_vrf = stage_packet_down_g(ctx, &inv_s_c);\n"),
    ("    let x: DmTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![Dummy2, L % 1920]> = x2_hbm.to_dm(&mut ctx.tdma);\n",
     "    let x2_view: HbmTensorView<'_, f8e4m3, Chip, m![Gs, L / 1920 % 4, Dummy2, L % 1920]> = unsafe { x2_hbm.view().reshape() };\n"
     "    let x: DmTensor<f8e4m3, Chip, DownClustersG, DownRowsByColumnsC, m![Dummy2, L % 1920]> = x2_view.to_dm(&mut ctx.tdma);\n"),
    ("    let x_trf_lane: TrfTensor<f8e4m3, Chip, DownClustersC, DownRowsByColumnsC, m![Dummy2], m![L % 1920]> = ctx\n",
     "    let x_trf_lane: TrfTensor<f8e4m3, Chip, DownClustersG, DownRowsByColumnsC, m![Dummy2], m![L % 1920]> = ctx\n"),
    ("    let mut down_hbm: HbmTensor<bf16, Chip, m![L / 7680, H]> = HbmTensor::new();\n",
     "    let mut down_hbm: HbmTensor<bf16, Chip, m![Gs, H]> = HbmTensor::new();\n"),
    ("    let mut down: DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60]> = DmTensor::new();\n",
     "    let mut down: DmTensor<bf16, Chip, DownClustersG, DownRowsC, m![H % 60]> = DmTensor::new();\n"),
    ("contract_down_rows_16_lane_folded_c(", "contract_down_rows_16_lane_folded_g("),
    ("contract_down_rows_12_lane_folded_c(", "contract_down_rows_12_lane_folded_g("),
    ("reduce_down_rows_16_c(", "reduce_down_rows_16_g("),
    ("reduce_down_rows_12_c(", "reduce_down_rows_12_g("),
    ("    down.view().to_hbm_view(\n        &mut ctx.tdma,\n        down_hbm.view_mut(),\n    );\n",
     "    // V390c: only cluster 1's partial goes through HBM (a cluster tile over the base axis Gs).\n"
     "    down.view()\n"
     "        .cluster_tile::<m![Gs], 1, m![Gs = 1 # 2]>(1)\n"
     "        .to_hbm_view(&mut ctx.tdma, down_hbm.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H]>(1));\n"),
    ("    let down1 = rmsnorm::load_reducing_half::<Cluster>(ctx, &down_hbm);\n",
     "    let down1 = rmsnorm::load_reducing_half_g::<Cluster>(ctx, &down_hbm);\n"),
]
for old, new in subs:
    assert ff.count(old) >= 1, old
    ff = ff.replace(old, new)
m = m.rstrip("\n") + "\n\n" + types + tile_g + lane_g + "/// V390c: `feedforward_fo_t1_c2` on `Gs` clusters with cluster 1's partial stored alone.\n" + ff
io.open(mlp_path, "w", encoding="utf-8", newline="\n").write(m)

r = io.open(rms_path, encoding="utf-8").read()
imp = "use crate::axes::{"
line_end = r.index("};", r.index(imp))
imports = r[r.index(imp):line_end]
if re.search(r"\bGs\b", imports) is None:
    r = r[:line_end] + ", Gs" + r[line_end:]
half_g = (
    "/// V390c: cluster 1's partial down row (`[Gs, H]` tile 1) loaded into the reducing layout.\n"
    "pub(crate) fn load_reducing_half_g<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &HbmTensor<bf16, Chip, m![Gs, H]>,\n"
    ") -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {\n"
    "    x.view().tile::<m![Gs], 1, m![Gs = 1 # 2, H]>(1).to_dm(&mut ctx.tdma)\n"
    "}\n"
)
r = r.rstrip("\n") + "\n\n" + half_g
io.open(rms_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn decoder_feedforward(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_call = "    let (x, g) = shared::mlp::feedforward_fo_t1(\n"
old_norm = "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1::<Cluster>(ctx, &x, &g, post_ff_rms_weight, &residual, layer_scalar);\n"
assert kfn.count(old_call) == 1 and kfn.count(old_norm) == 1, "ffn body"
body = kfn.replace(old_call, "    // V390c: column-halves down stage on Gs clusters; cluster 1's partial alone goes through HBM.\n    let (x, x1, g) = shared::mlp::feedforward_fo_t1_c4(\n", 1)
body = body.replace(old_norm, "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1_c2::<Cluster>(ctx, &x, &x1, &g, post_ff_rms_weight, &residual, layer_scalar);\n", 1)
if mode == "arm":
    arm = body.replace("pub fn decoder_feedforward(", "pub fn decoder_feedforward_c4(", 1)
    o = o[:ke] + "\n/// V390c arm c4: column-halves ffn with cluster 1's partial stored alone.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V390c c4", mode, "ok")
