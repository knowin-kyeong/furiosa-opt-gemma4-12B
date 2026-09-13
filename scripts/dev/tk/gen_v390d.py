"""gen_v390d.py <mlp.rs> <rmsnorm.rs> <ops.rs> <mode: arm|submit> -- V390d (arm c5): the column-halves ffn with the
partial store split so the last fence is short.

Run AFTER gen_v390.py and gen_v390b.py (c2). In c2 the two partial rows go to HBM in one 15 KB store after the last
down tile, and the sync behind it waits ~4.3k (production: 7.7 KB, 1.4k). Here the rows of tiles 0..2 (48 of every 60)
live in their own DM tensor and go to their own HBM tensor as soon as tile 2's pass B is done -- under tile 3's
contraction -- and only tile 3's 12 rows per group (3 KB) are stored at the end. The post-FF norm's reload of cluster 1's
partial becomes two tile loads into one reducing-layout tensor (rows 0..47 and 48..59 of every group), and cluster 0's
partial moves DM-to-DM from both tensors.
Adds `reduce_down_rows_16_c48` / `reduce_down_rows_12_c12` (pass B into the split tensors), `feedforward_fo_t1_c5`
(mlp.rs), `load_reducing_half_split` (rmsnorm.rs); mode=arm adds the ops arm `decoder_feedforward_c5`.
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

# pass B variants writing into a 48-row tensor (tiles 0..2) and a 12-row tensor (tile 3)
tile_c = block(m, "macro_rules! down_tile_fns_c {")
red_start = tile_c.index("        /// Pass B: block scales, column reduction and the cross-slice sum of the column chunks.\n")
red_end = tile_c.index("\n        }\n", red_start) + len("\n        }\n")
reduce_fn = tile_c[red_start:red_end]
old_out = "            out: &mut DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60]>,\n"
old_commit = "                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} 60]>(offset));\n"
assert reduce_fn.count(old_out) == 1 and reduce_fn.count(old_commit) == 1, "reduce fn"
split_macro = (
    "/// V390d: pass B writing into a partial-row tensor of `$total` rows per group (tiles 0..2 -> 48, tile 3 -> 12).\n"
    "macro_rules! down_reduce_split_fns {\n"
    "    ($reduce:ident, $rows:literal, $total:literal) => {\n"
    + reduce_fn.replace(old_out, "            out: &mut DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60 = $total]>,\n", 1)
                .replace(old_commit, "                .commit_view(out.view_mut().tile::<m![H % 60], $rows, m![H % 60 = $rows #{!} $total]>(offset));\n", 1)
    + "    };\n}\n"
    "down_reduce_split_fns!(reduce_down_rows_16_c48, 16, 48);\n"
    "down_reduce_split_fns!(reduce_down_rows_12_c12, 12, 12);\n\n"
)

ff = block(m, "pub(crate) fn feedforward_fo_t1_c2(")
subs = [
    ("pub(crate) fn feedforward_fo_t1_c2(", "pub(crate) fn feedforward_fo_t1_c5("),
    ("    let mut down_hbm: HbmTensor<bf16, Chip, m![L / 7680, H]> = HbmTensor::new();\n"
     "    let mut down: DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60]> = DmTensor::new();\n",
     "    // V390d: rows 0..47 of every group (tiles 0..2) and rows 48..59 (tile 3) are separate tensors, stored separately.\n"
     "    let mut down_a_hbm: HbmTensor<bf16, Chip, m![L / 7680, H / 60, H % 60 = 48]> = HbmTensor::new();\n"
     "    let mut down_b_hbm: HbmTensor<bf16, Chip, m![L / 7680, H / 60, H % 60 = 12]> = HbmTensor::new();\n"
     "    let mut down_a: DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60 = 48]> = DmTensor::new();\n"
     "    let mut down_b: DmTensor<bf16, Chip, DownClustersC, DownRowsC, m![H % 60 = 12]> = DmTensor::new();\n"),
    ("    reduce_down_rows_16_c(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down);\n", "    reduce_down_rows_16_c48(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down_a);\n"),
    ("    reduce_down_rows_16_c(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down);\n", "    reduce_down_rows_16_c48(ctx, &p, &down_scale, &inv_s_vrf, 16, &mut down_a);\n"),
    ("    reduce_down_rows_16_c(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down);\n",
     "    reduce_down_rows_16_c48(ctx, &p, &down_scale, &inv_s_vrf, 32, &mut down_a);\n"
     "    down_a.view().to_hbm_view(&mut ctx.tdma, down_a_hbm.view_mut());\n"),
    ("    reduce_down_rows_12_c(ctx, &p, &down_scale, &inv_s_vrf, 48, &mut down);\n",
     "    reduce_down_rows_12_c12(ctx, &p, &down_scale, &inv_s_vrf, 0, &mut down_b);\n"
     "    down_b.view().to_hbm_view(&mut ctx.tdma, down_b_hbm.view_mut());\n"),
    ("    down.view().to_hbm_view(\n        &mut ctx.tdma,\n        down_hbm.view_mut(),\n    );\n", ""),
    ("    let mut down0: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = DmTensor::new();\n"
     "    down.view().to_dm_view(&mut ctx.tdma, down0.view_mut());\n"
     "    let down1 = rmsnorm::load_reducing_half::<Cluster>(ctx, &down_hbm);\n",
     "    let mut down0: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480 / 60, H % 60]> = DmTensor::new();\n"
     "    down_a.view().to_dm_view(&mut ctx.tdma, down0.view_mut().tile::<m![H % 60], 48, m![H % 480 / 60, H % 60 = 48 #{!} 60]>(0));\n"
     "    down_b.view().to_dm_view(&mut ctx.tdma, down0.view_mut().tile::<m![H % 60], 12, m![H % 480 / 60, H % 60 = 12 #{!} 60]>(48));\n"
     "    let down0: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = unsafe { down0.reshape() };\n"
     "    let down1 = rmsnorm::load_reducing_half_split::<Cluster>(ctx, &down_a_hbm, &down_b_hbm);\n"),
]
for old, new in subs:
    assert ff.count(old) == 1, old
    ff = ff.replace(old, new, 1)
m = m.rstrip("\n") + "\n\n" + split_macro + "/// V390d: `feedforward_fo_t1_c2` with the partial store split (tiles 0..2 early, tile 3 last).\n" + ff
io.open(mlp_path, "w", encoding="utf-8", newline="\n").write(m)

r = io.open(rms_path, encoding="utf-8").read()
half_split = (
    "/// V390d: cluster 1's partial down row from the split stores (rows 0..47 and 48..59 of every group), assembled in\n"
    "/// the reducing layout by two tile loads.\n"
    "pub(crate) fn load_reducing_half_split<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    a: &HbmTensor<bf16, Chip, m![L / 7680, H / 60, H % 60 = 48]>,\n"
    "    b: &HbmTensor<bf16, Chip, m![L / 7680, H / 60, H % 60 = 12]>,\n"
    ") -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {\n"
    "    let mut x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480 / 60, H % 60]> = DmTensor::new();\n"
    "    a.view()\n"
    "        .tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H / 60, H % 60 = 48]>(1)\n"
    "        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 60], 48, m![H % 480 / 60, H % 60 = 48 #{!} 60]>(0));\n"
    "    b.view()\n"
    "        .tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H / 60, H % 60 = 12]>(1)\n"
    "        .to_dm_view(&mut ctx.tdma, x.view_mut().tile::<m![H % 60], 12, m![H % 480 / 60, H % 60 = 12 #{!} 60]>(48));\n"
    "    unsafe { x.reshape() }\n"
    "}\n"
)
r = r.rstrip("\n") + "\n\n" + half_split
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
body = kfn.replace(old_call, "    // V390d: column-halves down stage, the partial store split so only tile 3's rows are stored last.\n    let (x, x1, g) = shared::mlp::feedforward_fo_t1_c5(\n", 1)
body = body.replace(old_norm, "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1_c2::<Cluster>(ctx, &x, &x1, &g, post_ff_rms_weight, &residual, layer_scalar);\n", 1)
if mode == "arm":
    arm = body.replace("pub fn decoder_feedforward(", "pub fn decoder_feedforward_c5(", 1)
    o = o[:ke] + "\n/// V390d arm c5: column-halves ffn with the partial store split.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V390d c5", mode, "ok")
