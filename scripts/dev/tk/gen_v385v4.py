"""gen_v385v4.py <rope.rs> <ops.rs> <mode: arm|submit> [variants...] -- V385 v4: on-chip RoPE rows with a shorter chain.

v3 (gen_v385.py) is correct on Arena but its longer dependency chain (a 7-pass serial ramp, then a two-pass range
reduction) raised the RoPE chain's priority in the static scheduler: the rope_offset load moved to the front of the DMA
FIFO and the weight loads slipped (static 42.5k vs v2's 38.7k). v4 keeps the same arithmetic with a shallower graph:
  * two ramps built in parallel -- bits 0..2 seeded by pos (`[Ds % 8, 1 # 8]`) and bits 3..6 seeded by 1
    (`[Ds / 8 % 16, 1 # 8]`) -- combined by one pass (the low ramp read replayed over the high bits, the high ramp in the
    VRF replayed over the low bits): a = pos * 10000^(-(d mod 128)/128) as `[Ds % 128, 1 # 8]`;
  * one pass makes n = round(a / 2pi) (`fp_to_fxp(31)`), and each trig pass reduces on its own:
    fxp_to_fp(31), r = a - 2pi n (a in the VRF), Cos / Sin; the low half of sin is sin(2pi n - a) = -sin(r);
  * the rows are packed to dense bf16 by the 4-row transpose as before.
Main passes: w 2, pos 1, ramps 3 + 4, combine 1, n 1, trig 4 = 16 (v3 16, but the longest chain is 12 steps instead
of 18); sub passes 10. Variants: `oc` (real), `od` (diagnostic: r / 1024 in place of cos/sin).
"""
import io
import math
import struct
import sys

rope_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
variants = sys.argv[4:] or ["oc"]
assert mode in ("arm", "submit"), mode
assert all(v in ("oc", "od") for v in variants), variants


def f32(x):
    return "%.9gf32" % struct.unpack("f", struct.pack("f", x))[0]


c = math.log(10000.0) / 128.0
EM1 = [f32(math.exp(-c * (1 << k)) - 1.0) for k in range(7)]  # e_k - 1
INV_2PI, NEG_2PI, POS_2PI, SCALE = f32(1.0 / (2.0 * math.pi)), f32(-2.0 * math.pi), f32(2.0 * math.pi), f32(1.0 / 1024.0)
TRIM = "        .vector_narrow_trim::<m![1 # 4]>()\n"
PAD = "        .vector_widen_pad::<m![1 # 8]>()\n"


def main_pass(src, fetch_time, body, out_ty, tail, pre_tag=""):
    head = "    let %s = ctx\n" % out_ty if out_ty else "    ctx\n"
    return (head + "        .main\n        .begin(" + src + ")\n"
            "        .fetch::<" + fetch_time + ", m![1 # 8]>()\n        .collect::<" + fetch_time + ", m![1 # 8]>()\n"
            "        .vector_init()\n        .vector_intra_slice_tag(TagMode::Zero)\n"
            + pre_tag + body + "        .vector_final()\n" + tail)


def sub_vrf(name, src, fetch_time, ty):
    return ("    let %s: VrfTensor<f32, Chip, C, S, %s> = ctx\n        .sub\n        .begin(%s)\n"
            "        .fetch::<%s, m![1 # 8]>()\n        .collect::<%s, m![1 # 8]>()\n        .to_vrf();\n"
            % (name, ty, src, fetch_time, fetch_time))


COMMIT = "        .commit_trim::<m![1 # 8]>()\n        .commit();\n"


def rope_block(variant):
    blk = [
        "    // pos on every head slice: rope_offset is the table row's byte offset (512 B per row), so read as fixed point\n"
        "    // with 9 fraction bits it is pos itself.\n"
        "    let offset: DmTensor<i32, Chip, C, S, m![1 # 2]> = rope_offset.view().pad::<m![1 # 2]>().to_dm(&mut ctx.tdma);\n",
        main_pass("offset.view()", "m![1]", "", "pos: DmTensor<f32, Chip, C, S, m![1 # 8]>", COMMIT,
                  pre_tag="        .vector_fxp_to_fp(22)\n").replace("        .fetch::<m![1], m![1 # 8]>()\n", "        .fetch::<m![1], m![1 # 2]>()\n", 1),
        sub_vrf("pos_vrf", "pos.view()", "m![1]", "m![1 # 8]"),
        "    // w = [0, 1] along Dummy2, both packets made from pos (BitAnd 0 clears it).\n"
        "    let mut w: DmTensor<f32, Chip, C, S, m![Dummy2, 1 # 8]> = DmTensor::new();\n",
        main_pass("pos.view()", "m![1]", "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n", "",
                  "        .commit_trim::<m![1 # 8]>()\n"
                  "        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(0));\n"),
        main_pass("pos.view()", "m![1]",
                  "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n" + TRIM +
                  "        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)\n" + PAD, "",
                  "        .commit_trim::<m![1 # 8]>()\n"
                  "        .commit_view(w.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 8]>(1));\n"),
        "    // Two ramps grown one index bit per pass (the new bit outermost), x_{k+1} = x_k * [1, e_k] computed as\n"
        "    // w * (e_k - 1) * x_k + x_k with x_k in the VRF (staged replayed over Dummy2) and w read replayed over the bits\n"
        "    // already built, e_k = 10000^(-2^k / 128): bits 0..2 seeded by pos and bits 3..6 seeded by 1, then combined.\n",
    ]
    # low ramp: bits 0..2, seed pos; layouts [Ds % 2^(k+1), 1 # 8]
    for k in range(3):
        if k == 0:
            vrf, fetch_time, out_map, next_map = "pos_vrf", "m![Dummy2]", "m![Dummy2, 1 # 8]", "m![Ds % 2, 1 # 8]"
        else:
            n = 1 << k
            blk.append(sub_vrf("xl%d_vrf" % k, "xl%d.view()" % k, "m![Dummy2, Ds %% %d]" % n, "m![Dummy2, Ds %% %d, 1 # 8]" % n))
            vrf, fetch_time = "xl%d_vrf" % k, "m![Dummy2, Ds %% %d]" % n
            out_map, next_map = "m![Dummy2, Ds %% %d, 1 # 8]" % n, "m![Ds %% %d, 1 # 8]" % (n << 1)
        body = (TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % EM1[k]
                + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &%s)\n" % vrf
                + "        .vector_fp_binary(FpBinaryOp::AddF, &%s)\n" % vrf + PAD)
        blk.append(main_pass("w.view()", fetch_time, body, "yl%d: DmTensor<f32, Chip, C, S, %s>" % (k, out_map), COMMIT))
        blk.append("    let xl%d: DmTensor<f32, Chip, C, S, %s> = unsafe { yl%d.reshape() };\n" % (k + 1, next_map, k))
    # high ramp: bits 3..6, seed 1; layouts [Ds / 8 % 2^(k-2), 1 # 8]
    for k in range(3, 7):
        m = k - 3
        if m == 0:
            fetch_time, out_map, next_map = "m![Dummy2]", "m![Dummy2, 1 # 8]", "m![Ds / 8 % 2, 1 # 8]"
            body = (TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % EM1[k]
                    + "        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)\n" + PAD)
        else:
            n = 1 << m
            blk.append(sub_vrf("xh%d_vrf" % m, "xh%d.view()" % m, "m![Dummy2, Ds / 8 %% %d]" % n, "m![Dummy2, Ds / 8 %% %d, 1 # 8]" % n))
            fetch_time = "m![Dummy2, Ds / 8 %% %d]" % n
            out_map, next_map = "m![Dummy2, Ds / 8 %% %d, 1 # 8]" % n, "m![Ds / 8 %% %d, 1 # 8]" % (n << 1)
            body = (TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % EM1[k]
                    + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &xh%d_vrf)\n" % m
                    + "        .vector_fp_binary(FpBinaryOp::AddF, &xh%d_vrf)\n" % m + PAD)
        blk.append(main_pass("w.view()", fetch_time, body, "yh%d: DmTensor<f32, Chip, C, S, %s>" % (m, out_map), COMMIT))
        blk.append("    let xh%d: DmTensor<f32, Chip, C, S, %s> = unsafe { yh%d.reshape() };\n" % (m + 1, next_map, m))
    # combine: a[hi, lo] = xl[lo] * xh[hi]
    blk.append(
        "    // a = xl[d mod 8] * xh[d / 8]: the low ramp read replayed over the high bits, the high ramp staged replayed\n"
        "    // over the low bits so the register covers the whole stream.\n"
    )
    blk.append(sub_vrf("xh_rep_vrf", "xh4.view()", "m![Ds / 8 % 16, Ds % 8]", "m![Ds / 8 % 16, Ds % 8, 1 # 8]"))
    blk.append(main_pass("xl3.view()", "m![Ds / 8 % 16, Ds % 8]",
                         TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &xh_rep_vrf)\n" + PAD,
                         "ya: DmTensor<f32, Chip, C, S, m![Ds / 8 % 16, Ds % 8, 1 # 8]>", COMMIT))
    blk.append("    let a: DmTensor<f32, Chip, C, S, m![Ds % 128, 1 # 8]> = unsafe { ya.reshape() };\n")
    # n = round(a / 2pi)
    blk.append(
        "    // n = round(a / 2pi) by way of the fixed-point conversion; each trig pass reduces on its own: r = a - 2pi n.\n"
    )
    blk.append(main_pass("a.view()", "m![Ds % 128]",
                         TRIM + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % INV_2PI + PAD
                         + "        .vector_fp_to_fxp(31)\n",
                         "n: DmTensor<i32, Chip, C, S, m![Ds % 128, 1 # 8]>", COMMIT))
    blk.append(sub_vrf("a_vrf", "a.view()", "m![Ds % 128]", "m![Ds % 128, 1 # 8]"))
    pack = ("        .cast::<bf16, m![1 # 16]>()\n"
            "        .transpose::<m![Ds / 4 % 32], m![Ds % 4 # 16]>()\n"
            "        .commit_trim::<m![Ds % 4]>()\n")
    reduce_pos = ("        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % NEG_2PI
                  + "        .vector_fp_binary(FpBinaryOp::AddF, &a_vrf)\n")
    reduce_neg = ("        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), %s)\n" % POS_2PI
                  + "        .vector_fp_binary(FpBinaryOp::SubF, &a_vrf)\n")
    if variant == "oc":
        cos_op, sin_op = "        .vector_fp_unary(FpUnaryOp::Cos)\n", "        .vector_fp_unary(FpUnaryOp::Sin)\n"
    else:
        cos_op = sin_op = "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), %s)\n" % SCALE
    blk.append(
        "    // cos over both halves of the row and sin negated over the low half (the table's convention: the low half is\n"
        "    // sin(2pi n - a) = -sin(r)), each pass packing its 128 scalars into a dense bf16 half-row with the 4-row transpose.\n"
        "    let mut cos_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();\n"
        "    let mut sin_row: DmTensor<bf16, Chip, C, S, m![Ds]> = DmTensor::new();\n"
    )
    for name, half, body in (
        ("cos_row", 0, reduce_pos + cos_op),
        ("cos_row", 1, reduce_pos + cos_op),
        ("sin_row", 0, reduce_neg + sin_op),
        ("sin_row", 1, reduce_pos + sin_op),
    ):
        blk.append(main_pass("n.view()", "m![Ds % 128]", TRIM + body + PAD, "",
                             pack + "        .commit_view(%s.view_mut().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 #{!} 2, Ds %% 128]>(%d));\n" % (name, half),
                             pre_tag="        .vector_fxp_to_fp(31)\n"))
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
    copies.append("/// V385 `%s`: `apply_rope_heads_cc` with the cos/sin rows computed on the head slices (see gen_v385v4.py).\n%s" % (v, copy))
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
print("V385 v4", mode, variants, "ok")
