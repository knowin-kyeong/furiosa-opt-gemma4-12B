"""gen_v389b.py <rope.rs> <ops.rs> <mode: arm|submit> -- V389b: the RoPE staging behind BOTH the K and V projections.

V389 (gen_v389.py) hangs the staging copy passes on the raw V projection output; the scheduler then loads V before K and
the table gathers still sit in the DMA FIFO ahead of the last (K) load. Here the zero packet is made from k_raw and then
added to one made from v_raw, so the copy passes -- and the gathers placed just before them -- come after whichever
projection the scheduler finishes last: gathers, store, sync and reload all in the tail.
Adds `apply_rope_heads_cc_lkv` (V383's `apply_rope_heads_cc_1s` with `dep_k`, `dep_v` row parameters); mode=arm adds
the ops arm `sliding_project_qkv_lkv`, mode=submit swaps the production call.
"""
import io
import sys

rope_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("arm", "submit"), mode

r = io.open(rope_path, encoding="utf-8").read()
start = r.index("pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(")
end = r.index("\n}\n", start) + 3
fn = r[start:end]


def copy_pass(row, tile):
    return (
        "    ctx.main\n"
        "        .begin(" + row + ".view())\n"
        "        .fetch::<m![Ds / 128], m![Ds % 128]>()\n"
        "        .collect::<m![Ds / 16], m![Ds % 16]>()\n"
        "        .commit_trim::<m![Ds % 16]>()\n"
        "        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(" + str(tile) + "));\n"
    )


def dep_pass(row, tile):
    return (
        "    ctx.main\n"
        "        .begin(" + row + ".view())\n"
        "        .fetch::<m![Ds / 16], m![Ds % 16]>()\n"
        "        .fetch_cast::<f32>()\n"
        "        .collect::<m![Ds / 8], m![Ds % 8]>()\n"
        "        .vector_init()\n"
        "        .vector_intra_slice_tag(TagMode::Zero)\n"
        "        .vector_logic(LogicBinaryOpF32::BitOr, &dz_vrf)\n"
        "        .vector_final()\n"
        "        .commit_trim::<m![Ds % 8]>()\n"
        "        .commit_cast::<bf16>()\n"
        "        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(" + str(tile) + "));\n"
    )


def zero_pass(src, name, extra):
    # the eight lanes are live (`Ds = 8`), so an fp op goes through narrow_split / widen_concat, never a trim
    return (
        "    let " + name + ": DmTensor<f32, Chip, C, S, m![Ds = 8]> = ctx\n"
        "        .main\n"
        "        .begin(" + src + ".view().tile::<m![Ds], 8, m![Ds = 8 # 256]>(0))\n"
        "        .fetch::<m![1], m![Ds = 8]>()\n"
        "        .fetch_cast::<f32>()\n"
        "        .collect::<m![1], m![Ds = 8]>()\n"
        "        .vector_init()\n"
        "        .vector_intra_slice_tag(TagMode::Zero)\n"
        "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n"
        + extra +
        "        .vector_final()\n"
        "        .commit_trim::<m![Ds = 8]>()\n"
        "        .commit();\n"
        "    let " + name + ": DmTensor<f32, Chip, C, S, m![1 # 8]> = unsafe { " + name + ".reshape() };\n"
        "    let " + name + "_vrf: VrfTensor<f32, Chip, C, S, m![1 # 8]> = ctx\n"
        "        .sub\n"
        "        .begin(" + name + ".view())\n"
        "        .fetch::<m![1], m![1 # 8]>()\n"
        "        .collect::<m![1], m![1 # 8]>()\n"
        "        .to_vrf();\n"
    )


old = copy_pass("cos_row", 0) + copy_pass("sin_row", 1)
assert fn.count(old) == 1, "copy passes"
new = (
    "    // V389b: the copy passes depend on both raw projection outputs (a zero packet from dep_k, added to a zero packet\n"
    "    // from dep_v, OR-ed into the rows: an identity), so they -- and the table gathers placed just before them -- come\n"
    "    // after whichever weight load the scheduler streams last: gathers, store, sync (a DMA fence plus a barrier on\n"
    "    // hardware) and reload all run in the tail under the last contraction and the head norms.\n"
    + zero_pass("dep_k", "kz", "")
    + zero_pass("dep_v", "dz",
                "        .vector_narrow_split::<m![Ds = 8 / 4], m![Ds = 8 % 4]>()\n"
                "        .vector_fp_binary(FpBinaryOp::AddF, &kz_vrf)\n"
                "        .vector_widen_concat::<m![1], m![Ds = 8]>()\n")
    + dep_pass("cos_row", 0) + dep_pass("sin_row", 1)
)
old_sig = "    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n"
assert fn.count(old_sig) == 1, "signature"
copy = fn.replace(old, new, 1).replace(
    old_sig,
    "    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    dep_k: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n"
    "    dep_v: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n", 1
).replace("pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_lkv<C: M, S: M>(", 1)
copy = "/// V389b: `apply_rope_heads_cc_1s` with the staging copy passes dependent on both raw K and V projection outputs.\n" + copy
r = r[:end] + "\n" + copy + r[end:]
io.open(rope_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_project_qkv(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_kv = "    let (k, v) = sliding::projection::project_key_value_one(ctx, &x, &k_weight, &v_weight);\n"
old_kn = "    let k = sliding::rmsnorm::normalize_key_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &k,\n"
old_vn = "    let v = sliding::rmsnorm::normalize_value_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &v,\n"
old_call = (
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        rope_offset,\n"
)
for s in (old_kv, old_kn, old_vn, old_call):
    assert kfn.count(s) == 1, s
body = kfn.replace(old_kv, "    let (k_raw, v_raw) = sliding::projection::project_key_value_one(ctx, &x, &k_weight, &v_weight);\n", 1)
body = body.replace(old_kn, "    let k = sliding::rmsnorm::normalize_key_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &k_raw,\n", 1)
body = body.replace(old_vn, "    let v = sliding::rmsnorm::normalize_value_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &v_raw,\n", 1)
body = body.replace(old_call,
    "    // V389b: the RoPE staging (gathers, store, sync, reload) is scheduled behind both projections (`apply_rope_heads_cc_lkv`).\n"
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_lkv::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        &k_raw,\n        &v_raw,\n        rope_offset,\n", 1)
if mode == "arm":
    arm = body.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_lkv(", 1)
    o = o[:ke] + "\n/// V389b arm lkv: production qkv with the RoPE staging behind both the K and V projections.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V389b", mode, "ok")
