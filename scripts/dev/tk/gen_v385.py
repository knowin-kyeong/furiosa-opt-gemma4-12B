"""gen_v385.py <rope.rs> <ops.rs> <mode: arm|submit> [variants...] -- V385: the RoPE cos/sin rows computed on the head slices.

Production (V383) gathers the two table rows onto cluster 0, copies them into one buffer, stores it to HBM, syncs and
reloads it into the head layout: two gathers and a store in the DMA FIFO ahead of the Q weight load, one ExplicitSync
(where the process-first launch pays +3.5-4.3k, V381/V382) and a reload. Here every head slice builds the rows itself,
with nothing but plain passes (the tag unit's AxisToggle is not lowered in 0.6.0):
  * pos = rope_offset / 512, read as fixed point with 9 fraction bits (`vector_fxp_to_fp(22)`), one tiny replicated load;
  * w = [0, 1] along Dummy2 (two tiny passes over pos: BitAnd 0, and BitAnd 0 + 1);
  * the angle row grows one index bit per pass from the low bit up, the new bit outermost: with the row x_k over
    d mod 2^k in the VRF (staged replayed over Dummy2 so no register wraps) and w read replayed over those bits,
    x_{k+1} = w * (e_k - 1) * x_k + x_k = x_k * [1, e_k] where e_k = 10000^(-2^k / 128); the seed is pos, so after seven
    passes the row holds pos * 10000^(-(d mod 128)/128) -- the RoPE angles -- as `[Ds % 128, 1 # 8]`;
  * the angles are reduced to [-pi, pi] (n = round(a / 2pi) through fp_to_fxp / fxp_to_fp, r = a - 2pi n);
  * Cos (twice, for both halves of the row) and Sin (negated into the low half, plain into the high half), packed to
    dense bf16 rows by the 4-row transpose (the pass-B pattern of the ffn down tiles), so the head-row VRF staging is
    the production code.
Variants: `oc` (the real thing), `od` (diagnostic: the reduced angle scaled by 1/1024 in place of cos/sin -- finite
iff the ramp and the reduction are), `oe` (diagnostic: the unreduced angle scaled by 1/1024 -- finite iff the ramp is).
Each variant adds `apply_rope_heads_cc_<v>` (the rest of `apply_rope_heads_cc` verbatim) and, in mode=arm, the ops arm
`sliding_project_qkv_<v>`; mode=submit swaps the production body's call to `apply_rope_heads_cc_oc` (cos/sin become
unused parameters).
"""
import io
import math
import struct
import sys

rope_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
variants = sys.argv[4:] or ["oc"]
assert mode in ("arm", "submit"), mode
assert all(v in ("oc", "od", "oe") for v in variants), variants


def f32(x):
    return "%.9gf32" % struct.unpack("f", struct.pack("f", x))[0]


c = math.log(10000.0) / 128.0
EM1 = [f32(math.exp(-c * (1 << k)) - 1.0) for k in range(7)]  # e_k - 1
INV_2PI, NEG_2PI, SCALE = f32(1.0 / (2.0 * math.pi)), f32(-2.0 * math.pi), f32(1.0 / 1024.0)
TRIM = "        .vector_narrow_trim::<m![1 # 4]>()\n"
PAD = "        .vector_widen_pad::<m![1 # 8]>()\n"


def scalar_pass(src, fetch_time, body, out_ty, tail, pre_tag=""):
    head = "    let %s = ctx\n" % out_ty if out_ty else "    ctx\n"
    return (head + "        .main\n        .begin(" + src + ")\n"
            "        .fetch::<" + fetch_time + ", m![1 # 8]>()\n        .collect::<" + fetch_time + ", m![1 # 8]>()\n"
            "        .vector_init()\n        .vector_intra_slice_tag(TagMode::Zero)\n"
            + pre_tag + body + "        .vector_final()\n" + tail)


def rope_block(variant):
    blk = []
    blk.append(
        "    // pos on every head slice: rope_offset is the table row's byte offset (512 B per row), so read as fixed point\n"
        "    // with 9 fraction bits it is pos itself.\n"
        "    let offset: DmTensor<i32, Chip, C, S, m![1 # 2]> = rope_offset.view().pad::<m![1 # 2]>().to_dm(&mut ctx.tdma);\n"
        "    let pos: DmTensor<f32, Chip, C, S, m![1 # 8]> = ctx\n"
        "        .main\n"
        "        .begin(offset.view())\n"
        "        .fetch::<m![1], m![1 # 2]>()\n"
        "        .collect::<m![1], m![1 # 8]>()\n"
        "        .vector_init()\n"
        "        .vector_intra_slice_tag(TagMode::Zero)\n"
        "        .vector_fxp_to_fp(22)\n"
        "        .vector_final()\n"
        "        .commit_trim::<m![1 # 8]>()\n"
        "        .commit();\n"
        "    let pos_vrf: VrfTensor<f32, Chip, C, S, m![1 # 8]> = ctx\n"
        "        .sub\n"
        "        .begin(pos.view())\n"
        "        .fetch::<m![1], m![1 # 8]>()\n"
        "        .collect::<m![1], m![1 # 8]>()\n"
        "        .to_vrf();\n"
        "    // w = [0, 1] along Dummy2, both packets made from pos (BitAnd 0 clears it).\n"
        "    let mut w: DmTensor<f32, Chip, C, S, m![Dummy2, 1 # 8]> = DmTensor::new();\n"
    )
    blk.append(scalar_pass("pos.view()", "m![1]", "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n", "",
                           "        .commit_trim::<m![1 # 8]>()\n"
                           "        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(0));\n"))
    blk.append(scalar_pass("pos.view()", "m![1]",
                           "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n" + TRIM +
                           "        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)\n" + PAD, "",
                           "        .commit_trim::<m![1 # 8]>()\n"
                           "        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(1));\n"))
    blk.append(
        "    // The angle row, one index bit per pass from the low bit up (the new bit outermost): x_{k+1} = x_k * [1, e_k]\n"
        "    // with e_k = 10000^(-2^k / 128), computed as w * (e_k - 1) * x_k + x_k with x_k in the VRF (staged replayed\n"
        "    // over Dummy2, so the register covers the whole stream) and w read replayed over the bits already built. The\n"
        "    // seed is pos, so the row ends as pos * 10000^(-(d mod 128) / 128).\n"
    )
    for k in range(7):
        if k == 0:
            vrf, fetch_time, out_map, next_map = "pos_vrf", "m![Dummy2]", "m![Dummy2, 1 # 8]", "m![Ds % 2, 1 # 8]"
        else:
            n = 1 << k
            blk.append(
                "    let x%d_vrf: VrfTensor<f32, Chip, C, S, m![Dummy2, Ds %% %d, 1 # 8]> = ctx\n"
                "        .sub\n"
                "        .begin(x%d.view())\n"
                "        .fetch::<m![Dummy2, Ds %% %d], m![1 # 8]>()\n"
                "        .collect::<m![Dummy2, Ds %% %d], m![1 # 8]>()\n"
                "        .to_vrf();\n" % (k, n, k, n, n))
            vrf, fetch_time = "x%d_vrf" % k, "m![Dummy2, Ds %% %d]" % n
            out_map, next_map = "m![Dummy2, Ds %% %d, 1 # 8]" % n, "m![Ds %% %d, 1 # 8]" % (n << 1)
        body = (TRIM +
                "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % EM1[k] +
                "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &%s)\n" % vrf +
                "        .vector_fp_binary(FpBinaryOp::AddF, &%s)\n" % vrf + PAD)
        blk.append(scalar_pass("w.view()", fetch_time, body, "y%d: DmTensor<f32, Chip, C, S, %s>" % (k, out_map),
                               "        .commit_trim::<m![1 # 8]>()\n        .commit();\n"))
        blk.append("    let x%d: DmTensor<f32, Chip, C, S, %s> = unsafe { y%d.reshape() };\n" % (k + 1, next_map, k))
    if variant in ("oc", "od"):
        blk.append(
            "    // Reduce the angles to [-pi, pi]: n = round(a / 2pi) by way of the fixed-point conversions, r = a - 2pi n.\n"
        )
        blk.append(scalar_pass("x7.view()", "m![Ds % 128]",
                               TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % INV_2PI + PAD +
                               "        .vector_fp_to_fxp(31)\n",
                               "n: DmTensor<i32, Chip, C, S, m![Ds % 128, 1 # 8]>",
                               "        .commit_trim::<m![1 # 8]>()\n        .commit();\n"))
        blk.append(
            "    let a_vrf: VrfTensor<f32, Chip, C, S, m![Ds % 128, 1 # 8]> = ctx\n"
            "        .sub\n"
            "        .begin(x7.view())\n"
            "        .fetch::<m![Ds % 128], m![1 # 8]>()\n"
            "        .collect::<m![Ds % 128], m![1 # 8]>()\n"
            "        .to_vrf();\n"
        )
        blk.append(scalar_pass("n.view()", "m![Ds % 128]",
                               TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % NEG_2PI +
                               "        .vector_fp_binary(FpBinaryOp::AddF, &a_vrf)\n" + PAD,
                               "r: DmTensor<f32, Chip, C, S, m![Ds % 128, 1 # 8]>",
                               "        .commit_trim::<m![1 # 8]>()\n        .commit();\n",
                               pre_tag="        .vector_fxp_to_fp(31)\n"))
        src = "r.view()"
    else:
        src = "x7.view()"
    pack = ("        .cast::<bf16, m![1 # 16]>()\n"
            "        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()\n"
            "        .commit_trim::<m![Ds % 4]>()\n")
    if variant == "oc":
        cos_op, sin_op = "        .vector_fp_unary(FpUnaryOp::Cos)\n", "        .vector_fp_unary(FpUnaryOp::Sin)\n"
    else:
        cos_op = sin_op = "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), %s)\n" % SCALE
    blk.append(
        "    // cos over both halves of the row and sin negated over the low half (the table's convention), each pass\n"
        "    // packing its 128 scalars into a dense bf16 half-row with the 4-row transpose.\n"
        "    let mut cos_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();\n"
        "    let mut sin_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();\n"
    )
    for name, half, body in (
        ("cos_row", 0, cos_op),
        ("cos_row", 1, cos_op),
        ("sin_row", 0, sin_op + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -1.0f32)\n"),
        ("sin_row", 1, sin_op),
    ):
        blk.append(scalar_pass(src, "m![Ds % 128]", TRIM + body + PAD, "",
                               pack + "        .commit_view(%s.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds %% 128]>(%d));\n" % (name, half)))
    blk.append(
        "\n    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx\n"
        "        .sub\n"
        "        .begin(cos_row.view())\n"
        "        .fetch::<m![Ds / 16], m![Ds % 16]>()\n"
        "        .fetch_cast::<f32>()\n"
        "        .collect::<m![Ds / 8], m![Ds % 8]>()\n"
        "        .to_vrf();\n\n"
        "    let sin_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx\n"
        "        .sub\n"
        "        .begin(sin_row.view())\n"
        "        .fetch::<m![Ds / 16], m![Ds % 16]>()\n"
        "        .fetch_cast::<f32>()\n"
        "        .collect::<m![Ds / 8], m![Ds % 8]>()\n"
        "        .to_vrf();\n\n"
    )
    return "".join(blk)


r = io.open(rope_path, encoding="utf-8").read()
start = r.index("pub(crate) fn apply_rope_heads_cc<C: M, S: M>(")
end = r.index("\n}\n", start) + 3
fn = r[start:end]
old_sig = (
    "    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n"
    "    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,\n"
    "    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,\n"
)
assert fn.count(old_sig) == 1, "signature"
a = fn.index("    // The gathered rows land on one cluster; stage them through HBM")
b = fn.index("    let first_half_q = ")
assert a < b and fn[a:b].count(".to_vrf();") == 2, "staging block"
header = (
    "// ---------------------------------------------------------------------------------------------\n"
    "// V385: the RoPE rows computed on the head slices (no table gathers, no HBM store, no ExplicitSync, no reload).\n"
    "// ---------------------------------------------------------------------------------------------\n"
)
copies = []
for v in variants:
    copy = fn[:a].replace(old_sig, "    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n", 1) + rope_block(v) + fn[b:]
    copy = copy.replace("pub(crate) fn apply_rope_heads_cc<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_%s<C: M, S: M>(" % v, 1)
    copies.append("/// V385 `%s`: `apply_rope_heads_cc` with the cos/sin rows computed on the head slices (see gen_v385.py).\n%s" % (v, copy))
r = r.rstrip("\n") + "\n\n" + header + "\n".join(copies)
io.open(rope_path, "w", encoding="utf-8", newline="\n").write(r)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_project_qkv(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
old_call = (
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_1s::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        rope_offset,\n        cos,\n        sin,\n    );\n"
)
assert kfn.count(old_call) == 1, "rope call"


def new_call(v):
    return (
        "    // V385: the RoPE rows are computed on the head slices; the cos/sin tables are not read.\n"
        "    let (q, k) = sliding::rope::apply_rope_heads_cc_%s::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
        "        ctx,\n        &q,\n        &k,\n        rope_offset,\n    );\n"
        "    let _ = (cos, sin);\n" % v
    )


if mode == "arm":
    arms = "".join(
        "\n/// V385 arm %s: production qkv with the RoPE rows computed on chip.\n%s%s" % (
            v, attr, kfn.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_%s(" % v, 1).replace(old_call, new_call(v), 1))
        for v in variants)
    o = o[:ke] + arms + o[ke:]
else:
    o = o[:ks] + kfn.replace(old_call, new_call("oc"), 1) + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V385", mode, variants, "ok; e_k - 1 =", EM1)
