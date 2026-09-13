"""gen_v390b.py <mlp.rs> <rmsnorm.rs> <ops.rs> <mode: arm|submit> [variant: c2|c3] -- V390b: V390 with a cheaper merge.

Run AFTER gen_v390.py on the same files. V390 removed the 1/s store and the second geglu sync (-5k on the geglu syncs in
the V390 spans) but stored both clusters' partial rows (15 KB, 128 segments) and reloaded both, so the down-store sync
grew (+2.9k) and the norm tail too (+1.2k). Here:
  * c2: the norm loads only cluster 1's partial from HBM (a 7.7 KB tile) and takes cluster 0's partial by a DM-to-DM move
    into the reducing layout (no HBM hop for it); the sum is fused into the mean-square pass (AddF before the g multiply)
    and into the final pass (x1 added first, the residual joins through the clip stage);
  * c3: c2 plus the store itself reduced to cluster 1's half through a `cluster_tile` view (may not lower).
Adds `feedforward_fo_t1_<v>` and `normalize_add_gate_reduced_t1_<v>`; mode=arm adds the ops arm `decoder_feedforward_<v>`.
"""
import io
import sys

mlp_path, rms_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
variant = sys.argv[5] if len(sys.argv) > 5 else "c2"
assert mode in ("arm", "submit") and variant in ("c2", "c3"), (mode, variant)


def block(text, head, doc=None):
    start = text.index(head)
    if doc is not None:
        d = text.rfind(doc, 0, start)
        assert d != -1 and text[d:start].count("\n") <= 3, (head, doc)
        start = d
    end = text.index("\n}\n", start) + 3
    return text[start:end]


m = io.open(mlp_path, encoding="utf-8").read()
ff = block(m, "pub(crate) fn feedforward_fo_t1_c(")
old_ret = ") -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![L / 7680, H % 480]>, DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>) {"
new_ret = ") -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>, DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>, DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]>) {"
old_store = (
    "    down.view().to_hbm_view(\n"
    "        &mut ctx.tdma,\n"
    "        down_hbm.view_mut(),\n"
    "    );\n"
    "    // V390: both clusters' partial rows come back in one load; the norm sums them.\n"
    "    let down = rmsnorm::load_reducing_pair::<Cluster>(ctx, &down_hbm);\n"
)
assert ff.count(old_ret) == 1 and ff.count(old_store) == 1, "fo_t1_c shape"
if variant == "c2":
    store = (
        "    down.view().to_hbm_view(\n"
        "        &mut ctx.tdma,\n"
        "        down_hbm.view_mut(),\n"
        "    );\n"
    )
else:
    store = (
        "    // c3: only cluster 1's partial goes through HBM.\n"
        "    down.view()\n"
        "        .cluster_tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2]>(1)\n"
        "        .to_hbm_view(&mut ctx.tdma, down_hbm.view_mut().tile::<m![L / 7680], 1, m![L / 7680 = 1 #{!} 2, H]>(1));\n"
    )
new_store = store + (
    "    // V390b: cluster 0's partial moves DM-to-DM into the reducing layout; only cluster 1's comes back from HBM.\n"
    "    let mut down0: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = DmTensor::new();\n"
    "    down.view().to_dm_view(&mut ctx.tdma, down0.view_mut());\n"
    "    let down1 = rmsnorm::load_reducing_half::<Cluster>(ctx, &down_hbm);\n"
)
ff2 = ff.replace(old_ret, new_ret, 1).replace(old_store, new_store, 1)
ff2 = ff2.replace("    (down, g)\n}\n", "    (down0, down1, g)\n}\n", 1)
assert ff2.count("(down0, down1, g)") == 1, "return"
ff2 = ff2.replace("pub(crate) fn feedforward_fo_t1_c(", "pub(crate) fn feedforward_fo_t1_%s(" % variant, 1)
m = m.rstrip("\n") + "\n\n/// V390b `%s`: `feedforward_fo_t1_c` with cluster 0's partial kept on chip.\n" % variant + ff2
io.open(mlp_path, "w", encoding="utf-8", newline="\n").write(m)

r = io.open(rms_path, encoding="utf-8").read()
norm = block(r, "pub(crate) fn normalize_add_gate_reduced_t1<Cluster: M>(")
old_head = (
    "pub(crate) fn normalize_add_gate_reduced_t1<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,\n"
)
new_head = (
    "pub(crate) fn normalize_add_gate_reduced_t1_" + variant + "<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,\n"
    "    x1: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,\n"
)
assert norm.count(old_head) == 1, "norm head"
n2 = norm.replace(old_head, new_head, 1)
old_ms = (
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &g_vrf)\n"
    "        .vector_stash()\n"
)
assert n2.count(old_ms) == 1, "ms pass"
new_ms = (
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::AddF, &x1_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &g_vrf)\n"
    "        .vector_stash()\n"
)
n2 = n2.replace(old_ms, new_ms, 1)
old_gv = "    let g_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx\n"
assert n2.count(old_gv) == 1, "g_vrf"
n2 = n2.replace(old_gv,
    "    // V390b: the second partial rides in a register; both passes add it first.\n"
    "    let x1_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .sub\n"
    "        .begin(x1.view())\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .to_vrf();\n" + old_gv, 1)
old_final = (
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &gate_vrf)\n"
    "        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_final()\n"
)
assert n2.count(old_final) == 1, "final pass"
new_final = (
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::AddF, &x1_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &gate_vrf)\n"
    "        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_clip(ClipBinaryOpF32::Add, &residual_gate_vrf)\n"
    "        .vector_final()\n"
)
n2 = n2.replace(old_final, new_final, 1)
# the residual must be gated too: (x/rms*w + residual) * gate = x/rms*w*gate + residual*gate -> stage residual*gate
old_res = (
    "    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .sub\n"
    "        .begin(residual.view())\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .to_vrf();\n"
)
assert n2.count(old_res) == 1, "residual vrf"
new_res = (
    "    // V390b: the residual is pre-gated in one pass so the final pass can add x1 with its fp adder and take the\n"
    "    // residual through the clip stage: (x0 + x1) / rms * w * gate + residual * gate.\n"
    "    let residual_gate: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .main\n"
    "        .begin(residual.view())\n"
    "        .fetch::<m![H / 16 % 30], m![H % 16]>()\n"
    "        .fetch_cast::<f32>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_init()\n"
    "        .vector_intra_slice_tag(TagMode::Zero)\n"
    "        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()\n"
    "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_vrf)\n"
    "        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .vector_final()\n"
    "        .commit_trim::<m![H % 8]>()\n"
    "        .commit();\n"
    "    let residual_gate_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx\n"
    "        .sub\n"
    "        .begin(residual_gate.view())\n"
    "        .fetch::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .collect::<m![H / 8 % 60], m![H % 8]>()\n"
    "        .to_vrf();\n"
)
n2 = n2.replace(old_res, new_res, 1)
# gate_vrf must be staged before residual_gate: move the gate block above by relying on source order in the function --
# the original defines gate_vrf after residual_vrf, so swap: cut the gate block and insert it before the residual block.
gate_block_start = n2.index("    let gate_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = layer_scalar.to_dm(&mut ctx.tdma);\n")
gate_block_end = n2.index("        .to_vrf();\n", n2.index("    let gate_vrf:", gate_block_start)) + len("        .to_vrf();\n")
gate_block = n2[gate_block_start:gate_block_end]
n2 = n2[:gate_block_start] + n2[gate_block_end:]
res_start = n2.index("    // V390b: the residual is pre-gated")
n2 = n2[:res_start] + gate_block + "\n" + n2[res_start:]
half_load = (
    "/// V390b: cluster 1's partial down row (`[L / 7680, H]` tile 1) loaded into the reducing layout.\n"
    "pub(crate) fn load_reducing_half<Cluster: M>(\n"
    "    ctx: &mut Context,\n"
    "    x: &HbmTensor<bf16, Chip, m![L / 7680, H]>,\n"
    ") -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {\n"
    "    x.view().tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H]>(1).to_dm(&mut ctx.tdma)\n"
    "}\n\n"
)
if "pub(crate) fn load_reducing_half<" not in r:
    r = r.rstrip("\n") + "\n\n" + half_load
r = r.rstrip("\n") + "\n\n/// V390b `%s`: `normalize_add_gate_reduced_t1` over the sum of two partial rows, the second in a register.\n" % variant + n2
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
body = kfn.replace(old_call, "    // V390b: the down stage is split by column halves; cluster 0's partial stays on chip, cluster 1's comes back from HBM.\n    let (x, x1, g) = shared::mlp::feedforward_fo_t1_%s(\n" % variant, 1)
body = body.replace(old_norm, "    let residual = shared::rmsnorm::normalize_add_gate_reduced_t1_%s::<Cluster>(ctx, &x, &x1, &g, post_ff_rms_weight, &residual, layer_scalar);\n" % variant, 1)
if mode == "arm":
    arm = body.replace("pub fn decoder_feedforward(", "pub fn decoder_feedforward_%s(" % variant, 1)
    o = o[:ke] + "\n/// V390b arm %s: column-halves ffn with cluster 0's partial kept on chip.\n" % variant + attr + arm + o[ke:]
else:
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V390b", variant, mode, "ok")
