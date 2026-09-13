"""gen_v383.py <rope.rs> <ops.rs> <mode: arm|submit> -- V383: RoPE rows staged by ONE HBM store (one ExplicitSync).

Copies `apply_rope_heads_cc` to `apply_rope_heads_cc_1s` and replaces only its staging block: the two gathered rows are
copied on chip into one cluster-0 buffer `[Dummy2, Ds]` (two Main passes) and stored once, instead of one store (and one
ExplicitSync) per row. mode=arm adds an ops arm `sliding_project_qkv_1s` (production body, RoPE call swapped);
mode=submit swaps the call inside the production `sliding_project_qkv` body.
"""
import io
import sys

rope_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("arm", "submit"), mode

r = io.open(rope_path, encoding="utf-8").read()
start = r.index("pub(crate) fn apply_rope_heads_cc<C: M, S: M>(")
doc = r.rfind("\n/// V355", 0, start)
end = r.index("\n}\n", start) + 3
fn = r[start:end]
old = (
    "    let mut cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = HbmTensor::new();\n"
    "    cos_row\n"
    "        .view()\n"
    "        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));\n"
    "    sin_row\n"
    "        .view()\n"
    "        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));\n"
    "    let cs: DmTensor<bf16, Chip, C, S, m![Dummy2, Ds]> = cs_hbm.to_dm(&mut ctx.tdma);\n"
)
assert fn.count(old) == 1, "staging block"
new = (
    "    // V383: both rows are copied on chip into one cluster-0 staging buffer and stored by ONE HBM store, so a single\n"
    "    // ExplicitSync (instead of one per row) stands in front of the head-layout reload. The first-launch penalty and\n"
    "    // the random cross-cluster wait land on those syncs (V371/V371cold/V382 spans).\n"
    "    let mut cs_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Dummy2, Ds]> = DmTensor::new();\n"
    "    ctx.main\n"
    "        .begin(cos_row.view())\n"
    "        .fetch::<m![Ds / 128], m![Ds % 128]>()\n"
    "        .collect::<m![Ds / 16], m![Ds % 16]>()\n"
    "        .commit_trim::<m![Ds % 16]>()\n"
    "        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));\n"
    "    ctx.main\n"
    "        .begin(sin_row.view())\n"
    "        .fetch::<m![Ds / 128], m![Ds % 128]>()\n"
    "        .collect::<m![Ds / 16], m![Ds % 16]>()\n"
    "        .commit_trim::<m![Ds % 16]>()\n"
    "        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));\n"
    "    let cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = cs_dm.to_hbm(&mut ctx.tdma);\n"
    "    let cs: DmTensor<bf16, Chip, C, S, m![Dummy2, Ds]> = cs_hbm.to_dm(&mut ctx.tdma);\n"
)
copy = fn.replace(old, new, 1).replace(
    "pub(crate) fn apply_rope_heads_cc<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(", 1)
copy = "/// V383: `apply_rope_heads_cc` with the two RoPE rows staged by one HBM store (one ExplicitSync).\n" + copy
r = r[:end] + "\n" + copy + r[end:]
io.open(rope_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
call = "sliding::rope::apply_rope_heads_cc::<"
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_project_qkv(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
assert kfn.count(call) == 1, "rope call"
note = ("    // V383: the RoPE rows are staged by one HBM store and one ExplicitSync (`apply_rope_heads_cc_1s`).\n")
if mode == "arm":
    arm = kfn.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_1s(", 1)
    arm = arm.replace("    let (q, k) = " + call, note + "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<", 1)
    assert arm.count("apply_rope_heads_cc_1s::<") == 1, "arm call"
    o = o[:ke] + "\n/// V383 arm 1s: production qkv with the RoPE rows staged by one store.\n" + attr + arm + o[ke:]
else:
    body = kfn.replace("    let (q, k) = " + call, note + "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<", 1)
    assert body.count("apply_rope_heads_cc_1s::<") == 1, "submit call"
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V383", mode, "ok:", rope_path, ops_path)
