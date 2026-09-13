"""gen_v385.py <rope.rs> <ops.rs> <mode: arm|submit> -- V385: the RoPE cos/sin rows computed on the head slices.

Production (V383) gathers the two table rows onto cluster 0, copies them into one buffer, stores it to HBM, syncs and
reloads it into the head layout: two gathers and a store in the DMA FIFO ahead of the Q weight load, one ExplicitSync
(the first sync of the kernel is where the process-first launch pays +3.5-4.3k, V381/V382) and a reload. Here every
head slice builds the rows itself:
  * pos = rope_offset / 512 read as fixed point with 9 fraction bits (`vector_fxp_to_fp(22)`), one tiny replicated load;
  * theta_(d mod 128) = 10000^(-(d mod 128)/128) as a product over the index bits of d: the row is laid out by index
    bits (`RopeBits`, eight size-2 axes), and each pass multiplies in e_j = 10000^(-2^j/128) where the tag unit's
    AxisToggle on bit j's axis is set (seven passes, the first also zeroes the fresh buffer);
  * angle = pos * theta, then Cos and Sin (the low half of sin negated, AxisToggle on bit 7) committed as bf16 like the
    table so the head-row VRF staging is unchanged.
Adds `apply_rope_heads_cc_oc` (the rest of `apply_rope_heads_cc` verbatim); mode=arm adds the ops arm
`sliding_project_qkv_oc`, mode=submit swaps the call in the production body (cos/sin become unused parameters).
"""
import io
import math
import struct
import sys

rope_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("arm", "submit"), mode


def f32(x):
    return "%.9gf32" % struct.unpack("f", struct.pack("f", x))[0]


c = math.log(10000.0) / 128.0
E = [f32(math.exp(-c * (1 << j))) for j in range(7)]
TIME, PACKET = "m![Rb7, Rb6, Rb5, Rb4, Rb3]", "m![Rb2, Rb1, Rb0]"
TIME2, PACKET2 = "m![Rb7, Rb6, Rb5, Rb4, Rb3, Rb2]", "m![Rb1, Rb0]"
SET = "TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One])"
CLR = "TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::Zero])"


def pass_(src, dst, tag, body, cast=False):
    out = "    let %s: DmTensor<%s, Chip, C, S, RopeBits> = ctx\n" % (dst, "bf16" if cast else "f32")
    out += "        .main\n        .begin(%s.view())\n" % src
    out += "        .fetch::<%s, %s>()\n        .collect::<%s, %s>()\n" % (TIME, PACKET, TIME, PACKET)
    out += "        .vector_init()\n        .vector_intra_slice_tag(%s)\n" % tag
    out += body
    out += "        .vector_widen_concat::<%s, %s>()\n        .vector_final()\n" % (TIME, PACKET)
    out += "        .commit_trim::<%s>()\n" % PACKET
    if cast:
        out += "        .commit_cast::<bf16>()\n"
    out += "        .commit();\n"
    return out


split = "        .vector_narrow_split::<%s, %s>()\n" % (TIME2, PACKET2)
passes = ""
# bit 0: zero the fresh buffer, then 1 -> e_0 where the bit is set
passes += pass_("ramp", "r0", "TagMode::AxisToggle { axis: Rb0::NAME }",
                "        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)\n" + split +
                "        .vector_fp_binary(FpBinaryOp::AddF, Branched::imm(%s, %s).imm(%s, 1.0f32))\n" % (SET, E[0], CLR))
for j in range(1, 7):
    body = split + "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Branched::imm(%s, %s).imm(%s, 1.0f32))\n" % (SET, E[j], CLR)
    if j == 6:
        body += "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &pos_vrf)\n"
    passes += pass_("r%d" % (j - 1), "angle" if j == 6 else "r%d" % j, "TagMode::AxisToggle { axis: Rb%d::NAME }" % j, body)
passes += pass_("angle", "cos_bits", "TagMode::Zero", split + "        .vector_fp_unary(FpUnaryOp::Cos)\n", cast=True)
passes += pass_("angle", "sin_bits", "TagMode::AxisToggle { axis: Rb7::NAME }",
                split + "        .vector_fp_unary(FpUnaryOp::Sin)\n" +
                "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Branched::imm(%s, 1.0f32).imm(%s, -1.0f32))\n" % (SET, CLR),
                cast=True)

header = (
    "// ---------------------------------------------------------------------------------------------\n"
    "// V385: the RoPE rows computed on the head slices (no table gathers, no HBM store, no ExplicitSync, no reload).\n"
    "// ---------------------------------------------------------------------------------------------\n"
    "axes![Rb0 = 2, Rb1 = 2, Rb2 = 2, Rb3 = 2, Rb4 = 2, Rb5 = 2, Rb6 = 2, Rb7 = 2];\n"
    "/// A head row's 256 elements by index bits, bit 0 innermost: d = 128 Rb7 + 64 Rb6 + ... + Rb0.\n"
    "type RopeBits = m![Rb7, Rb6, Rb5, Rb4, Rb3, Rb2, Rb1, Rb0];\n\n"
)
new_block = (
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
    "    // theta_(d mod 128) = 10000^(-(d mod 128)/128) as a product over the index bits of d: e_j = 10000^(-2^j/128)\n"
    "    // is multiplied in where bit j is set (AxisToggle on that bit's axis, one bit per pass). The first pass zeroes\n"
    "    // the fresh buffer with BitAnd so nothing depends on what it held. The last one multiplies in pos.\n"
    "    let ramp: DmTensor<f32, Chip, C, S, RopeBits> = DmTensor::new();\n"
    + passes +
    "    let cos_row: DmTensor<bf16, Chip, C, S, m![Ds]> = unsafe { cos_bits.reshape() };\n"
    "    let sin_row: DmTensor<bf16, Chip, C, S, m![Ds]> = unsafe { sin_bits.reshape() };\n\n"
    "    let cos_vrf: VrfTensor<f32, Chip, C, S, m![Ds]> = ctx\n"
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
copy = fn[:a].replace(old_sig, "    rope_offset: &HbmTensor<i32, Chip, m![1]>,\n", 1) + new_block + fn[b:]
copy = copy.replace("pub(crate) fn apply_rope_heads_cc<C: M, S: M>(", "pub(crate) fn apply_rope_heads_cc_oc<C: M, S: M>(", 1)
r = r.rstrip("\n") + "\n\n" + header + "/// V385: `apply_rope_heads_cc` with the cos/sin rows computed on the head slices (see the block comment above).\n" + copy
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
new_call = (
    "    // V385: the RoPE rows are computed on the head slices; the cos/sin tables are not read.\n"
    "    let (q, k) = sliding::rope::apply_rope_heads_cc_oc::<layout::HeadClusters, layout::HeadSlicesPerCluster>(\n"
    "        ctx,\n        &q,\n        &k,\n        rope_offset,\n    );\n"
    "    let _ = (cos, sin);\n"
)
if mode == "arm":
    arm = kfn.replace("pub fn sliding_project_qkv(", "pub fn sliding_project_qkv_oc(", 1).replace(old_call, new_call, 1)
    o = o[:ke] + "\n/// V385 arm oc: production qkv with the RoPE rows computed on chip.\n" + attr + arm + o[ke:]
else:
    o = o[:ks] + kfn.replace(old_call, new_call, 1) + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V385", mode, "ok; e_j =", E)
