"""gen_v387.py <rope.rs> <ops.rs> <mode: arm|submit> -- V387: the RoPE staging (store, sync, reload) moved behind the K projection.

V383's remaining RoPE sync costs the qkv kernel the K weight load's issue: the DMA FIFO is in order, the cs reload sits in
it right ahead of the K load, the reload cannot start until the ExplicitSync completes (cluster 1 arrives 10-20k late),
so the K load idles behind it (r40 1s#1: sync 9.4k-28.7k, reload 28.7k-30.3k, K load from 30.3k while the Q load ended
at 25.4k). Here the two copy passes that build the staging buffer take a data dependency on the K projection output k
(a zero packet made from k, staged as a scalar VRF on cluster 0 / slice 0, OR-ed into the rows), so the store, the sync
and the reload are scheduled after the K contraction -- behind the K and V load issues -- and the sync's wait overlaps
the V weight stream instead of holding the FIFO. The process-first launch's first-sync penalty moves there as well.
Adds `apply_rope_heads_cc_lt` (V383's `apply_rope_heads_cc_1s` with the dependency); mode=arm adds the ops arm
`sliding_project_qkv_lt`, mode=submit swaps the production call.
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
        "        .vector_logic(LogicBinaryOpF32::BitOr, &kz_vrf)\n"
        "        .vector_final()\n"
        "        .commit_trim::<m![Ds % 8]>()\n"
        "        .commit_cast::<bf16>()\n"
        "        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(" + str(tile) + "));\n"
    )


old = copy_pass("cos_row", 0) + copy_pass("sin_row", 1)
assert fn.count(old) == 1, "copy passes"
new = (
    "    // V387: the copy passes take a data dependency on k -- a zero packet made from k's first eight elements, staged as\n"
    "    // a scalar VRF on cluster 0 / slice 0 and OR-ed into the rows (an identity) -- so the store, the ExplicitSync and\n"
    "    // the reload are scheduled after the K contraction: behind the K and V weight load issues in the DMA FIFO, where\n"
    "    // the sync's wait overlaps the V weight stream instead of holding the K load (V383: reload ahead of the K load).\n"
    "    let kz: DmTensor<f32, Chip, C, S, m![Ds = 8]> = ctx\n"
    "        .main\n"
    "        .begin(k.view().tile::<m![Ds], 8, m![Ds = 8 # 256]>(0))\n"
    "        .fetch::<m![1], m![Ds = 8]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![1], m![Ds = 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n"
    "        .vector_final()\n"
    "        .commit_trim::<m![Ds = 8]>()\n"
    "        .commit();\n"
    "    let kz: DmTensor<f32, Chip, C, S, m![1 # 8]> = unsafe { kz.reshape() };\n"
    "    // The register is staged on the head layout (head 0 lives on cluster 0 / slice 0, the copy passes' slice); a\n"
    "    // cluster_tile over `Ns / 4` does not lower (visa: cannot find tag Ns_4).\n"
    "    let kz_vrf: VrfTensor<f32, Chip, C, S, m![1 # 8]> = ctx\n"
    "        .sub\n"
    "        .begin(kz.view())\n"
    "        .fetch::<m![1], m![1 # 8]>()\n"
    "        .collect::<m![1], m![1 # 8]>()\n"
    "        .to_vrf();\n"
    + dep_pass("cos_row", 0) + dep_pass("sin_row", 1)
)
copy = fn.replace(old, new, 1).replace(
    "pub(crate) fn apply_rope_heads_cc_1s<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_lt<C: M, S: M>(", 1)
copy = "/// V387: `apply_rope_heads_cc_1s` with the staging copy passes dependent on k (store, sync and reload after the K projection).\n" + copy
r = r[:end] + "\n" + copy + r[end:]
io.open(rope_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_project_qkv(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_call = "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
assert kfn.count(old_call) == 1, "rope call"
note = "    // V387: the RoPE staging store, sync and reload are scheduled after the K projection (`apply_rope_heads_cc_lt`).\n"
new_call = note + "    let (q, k) = sliding::rope::apply_rope_heads_cc_lt::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
if mode == "arm":
    arm = kfn.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_lt(", 1).replace(old_call, new_call, 1)
    o = o[:ke] + "\n/// V387 arm lt: production qkv with the RoPE staging behind the K projection.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + kfn.replace(old_call, new_call, 1) + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V387", mode, "ok")
