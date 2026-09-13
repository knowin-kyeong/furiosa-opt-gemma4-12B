"""gen_v389.py <rope.rs> <ops.rs> <mode: arm|submit> -- V389: the RoPE staging (gathers, store, sync, reload) behind the V projection.

Reading of V383/V387: an ExplicitSync behaves as a DMA fence plus a cross-cluster barrier -- it ends only when the
weight load issued before it has completed on both clusters (V383 1s#1: sync 9.4k-28.7k = the Q load's end on cluster
1), and the reload behind it holds the next big load in the in-order FIFO (~5k idle). V387 moved the staging behind the
K projection, but the scheduler then issued the Q load between the store and the sync, so the fence still hit a big load.
Here the two copy passes depend on the raw V projection output (a zero packet made from it, OR-ed in), which is ready only
after the last weight load has completed: the gathers, the store, the fence and the reload all fall into the tail, where
the DMA engine is idle under the V contraction and the head norms, and cluster 0's wait at the barrier is cluster 1's
lateness that the end-of-kernel barrier would absorb anyway.
Adds `apply_rope_heads_cc_lv` (V383's `apply_rope_heads_cc_1s` with a `dep` row parameter); mode=arm adds the ops arm
`sliding_project_qkv_lv`, mode=submit swaps the production call.
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


old = copy_pass("cos_row", 0) + copy_pass("sin_row", 1)
assert fn.count(old) == 1, "copy passes"
new = (
    "    // V389: the copy passes take a data dependency on `dep` (the raw V projection output, ready only after the last\n"
    "    // weight load): a zero packet made from its first eight elements, staged as a scalar VRF on the head layout (head 0\n"
    "    // is cluster 0 / slice 0, the copy passes' slice) and OR-ed into the rows (an identity). The gathers, the store,\n"
    "    // the sync (a DMA fence plus a barrier on hardware) and the reload then run in the tail, under the V contraction\n"
    "    // and the head norms, instead of holding the K or V weight load in the in-order DMA FIFO (V383, V387).\n"
    "    let dz: DmTensor<f32, Chip, C, S, m![Ds = 8]> = ctx\n"
    "        .main\n"
    "        .begin(dep.view().tile::<m![Ds], 8, m![Ds = 8 # 256]>(0))\n"
    "        .fetch::<m![1], m![Ds = 8]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![1], m![Ds = 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n"
    "        .vector_final()\n"
    "        .commit_trim::<m![Ds = 8]>()\n"
    "        .commit();\n"
    "    let dz: DmTensor<f32, Chip, C, S, m![1 # 8]> = unsafe { dz.reshape() };\n"
    "    let dz_vrf: VrfTensor<f32, Chip, C, S, m![1 # 8]> = ctx\n"
    "        .sub\n"
    "        .begin(dz.view())\n"
    "        .fetch::<m![1], m![1 # 8]>()\n"
    "        .collect::<m![1], m![1 # 8]>()\n"
    "        .to_vrf();\n"
    + dep_pass("cos_row", 0) + dep_pass("sin_row", 1)
)
old_sig = "    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n"
assert fn.count(old_sig) == 1, "signature"
copy = fn.replace(old, new, 1).replace(
    old_sig, "    k: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    dep: &DmTensor<bf16, Chip, C, S, m![Ds]>,\n    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n", 1
).replace("pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_lv<C: M, S: M>(", 1)
copy = "/// V389: `apply_rope_heads_cc_1s` with the staging copy passes dependent on `dep` (the raw V projection output).\n" + copy
r = r[:end] + "\n" + copy + r[end:]
io.open(rope_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_project_qkv(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_kv = "    let (k, v) = sliding::projection::project_key_value_one(ctx, &x, &k_weight, &v_weight);\n"
old_vn = "    let v = sliding::rmsnorm::normalize_value_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &v,\n"
old_call = (
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        rope_offset,\n"
)
for s in (old_kv, old_vn, old_call):
    assert kfn.count(s) == 1, s
body = kfn.replace(old_kv, "    let (k, v_raw) = sliding::projection::project_key_value_one(ctx, &x, &k_weight, &v_weight);\n", 1)
body = body.replace(old_vn, "    let v = sliding::rmsnorm::normalize_value_heads_cc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n        ctx,\n        &v_raw,\n", 1)
body = body.replace(old_call,
    "    // V389: the RoPE staging (gathers, store, sync, reload) is scheduled behind the V projection (`apply_rope_heads_cc_lv`).\n"
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_lv::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        &v_raw,\n        rope_offset,\n", 1)
if mode == "arm":
    arm = body.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_lv(", 1)
    o = o[:ke] + "\n/// V389 arm lv: production qkv with the RoPE staging behind the V projection.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V389", mode, "ok")
