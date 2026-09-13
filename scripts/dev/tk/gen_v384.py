"""gen_v384.py <mlp.rs> <ops.rs> <mode: arm|submit> -- V384: the ffn geglu 1/s store as one aligned 256 B block per cluster.

Today `stage_geglu_hi_lo_hbm_one_store` stores 1/s from one slice per cluster as a 32 B write into a shared 256 B HBM
line (`[L / 7680, 1 # 8]`): on hardware that partial-line write takes ~6k cycles (V382/V366 spans: the DmaStore between
the two geglu syncs, static 755), and the second ExplicitSync waits for it. V271 showed the same read-modify-write
penalty on the attn store. Here the eight consecutive slices that already hold the replicated value (`[Dummy256]`) write
one full 256 B block per cluster (`[L / 7680, Ns, 1 # 8]`); the down side loads the block and reads its first packet.
New functions only: `stage_geglu_hi_lo_hbm_one_store_ia`, `broadcast_inv_s_down_ia`, `feedforward_fo_t1_ia`.
mode=arm adds the ops arm `decoder_feedforward_ia`; mode=submit swaps the call in the production body.
"""
import io
import sys

mlp_path, ops_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("arm", "submit"), mode

m = io.open(mlp_path, encoding="utf-8").read()


def cut(text, head, doc):
    start = text.index(head)
    d = text.rfind(doc, 0, start)
    assert d != -1 and text[d:start].count("\n") <= 3, (head, doc)
    end = text.index("\n}\n", start) + 3
    return d, end


# 1. the staging function with the aligned 1/s store
a, b = cut(m, "fn stage_geglu_hi_lo_hbm_one_store(", "/// `stage_geglu_hi_lo_hbm` with both pieces written by one store command.")
fn = m[a:b]
old_sig = ") -> (HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]>, HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>) {"
assert fn.count(old_sig) == 1, "signature"
old_tail = (
    "    let inv_s_one: DmTensor<f32, Chip, UpGateClusters, m![1 # 256], m![1 # 8]> = unsafe { inv_s_all.reshape() };\n"
    "    let mut inv_s_hbm: HbmTensor<f32, Chip, m![L / 7680, 1 # 8]> = HbmTensor::new();\n"
    "    inv_s_one.view().to_hbm_view(&mut ctx.tdma, inv_s_hbm.view_mut());\n"
)
assert fn.count(old_tail) == 1, "inv_s tail"
new_tail = (
    "    // V384: 1/s is replicated on every slice; eight consecutive slices write it as one full 256 B HBM block per\n"
    "    // cluster instead of one slice writing 32 B into a line both clusters share (a read-modify-write partial write,\n"
    "    // ~6k real cycles under the second geglu sync).\n"
    "    let inv_s_eight: DmTensor<f32, Chip, UpGateClusters, m![1 # 32, Ns], m![1 # 8]> = unsafe { inv_s_all.reshape() };\n"
    "    let mut inv_s_hbm: HbmTensor<f32, Chip, m![L / 7680, Ns, 1 # 8]> = HbmTensor::new();\n"
    "    inv_s_eight.view().to_hbm_view(&mut ctx.tdma, inv_s_hbm.view_mut());\n"
)
stage = fn.replace(old_sig, ") -> (HbmTensor<f8e4m3, Chip, m![L / 1920, Dummy2, L % 1920]>, HbmTensor<f32, Chip, m![L / 7680, Ns, 1 # 8]>) {", 1)
stage = stage.replace(old_tail, new_tail, 1)
stage = stage.replace("fn stage_geglu_hi_lo_hbm_one_store(", "fn stage_geglu_hi_lo_hbm_one_store_ia(", 1)
stage = stage.replace("/// `stage_geglu_hi_lo_hbm` with both pieces written by one store command.",
                      "/// V384: `stage_geglu_hi_lo_hbm_one_store` with 1/s stored as one aligned 256 B block per cluster.", 1)

# 2. the down-side broadcast reading the first packet of the block
a2, b2 = cut(m, "fn broadcast_inv_s_down(", "/// 1/s_c of the geglu pieces onto every down slice")
bc = m[a2:b2]
old_load = (
    "    v: &HbmTensor<f32, Chip, m![L / 7680, 1 # 8]>,\n"
    ") -> VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> {\n"
    "    let two: DmTensor<f32, Chip, DownClusters, m![1 # 128, L / 7680], m![1 # 8]> = v.to_dm(&mut ctx.tdma);\n"
    "    let all: DmTensor<f32, Chip, DownClusters, m![Dummy256 / 8, L / 7680, Dummy8 / 2], m![1 # 8]> = ctx\n"
    "        .main\n"
    "        .begin(two.view())\n"
)
assert bc.count(old_load) == 1, "broadcast load"
new_load = (
    "    v: &HbmTensor<f32, Chip, m![L / 7680, Ns, 1 # 8]>,\n"
    ") -> VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> {\n"
    "    let two: DmTensor<f32, Chip, DownClusters, m![1 # 128, L / 7680], m![Ns, 1 # 8]> = v.to_dm(&mut ctx.tdma);\n"
    "    let all: DmTensor<f32, Chip, DownClusters, m![Dummy256 / 8, L / 7680, Dummy8 / 2], m![1 # 8]> = ctx\n"
    "        .main\n"
    "        .begin(two.view().tile::<m![Ns], 1, m![Ns = 1 # 8, 1 # 8]>(0))\n"
)
bc = bc.replace(old_load, new_load, 1).replace("fn broadcast_inv_s_down(", "fn broadcast_inv_s_down_ia(", 1)
bc = bc.replace("/// 1/s_c of the geglu pieces onto every down slice", "/// V384: `broadcast_inv_s_down` reading the first packet of each cluster's 256 B block", 1)

# 3. the feedforward body calling both
a3, b3 = cut(m, "pub(crate) fn feedforward_fo_t1(", "/// V366 (T1):")
ff = m[a3:b3]
for old, new in (
    ("    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm_one_store(ctx, &x);\n",
     "    let (x2_hbm, inv_s_hbm) = stage_geglu_hi_lo_hbm_one_store_ia(ctx, &x);\n"),
    ("    let inv_s_vrf = broadcast_inv_s_down(ctx, &inv_s_hbm);\n", "    let inv_s_vrf = broadcast_inv_s_down_ia(ctx, &inv_s_hbm);\n"),
):
    assert ff.count(old) == 1, old
    ff = ff.replace(old, new, 1)
ff = ff.replace("pub(crate) fn feedforward_fo_t1(", "pub(crate) fn feedforward_fo_t1_ia(", 1)
ff = ff.replace("/// V366 (T1):", "/// V384: `feedforward_fo_t1` with the geglu 1/s stored as one aligned block per cluster. V366 (T1):", 1)

imp = "use crate::axes::{C, Dummy2, Dummy256, Dummy8, H, L};"
assert m.count(imp) == 1, "axes import"
m = m.replace(imp, "use crate::axes::{C, Dummy2, Dummy256, Dummy8, H, L, Ns};", 1)
m = m.rstrip("\n") + "\n\n" + stage + "\n" + bc + "\n" + ff
io.open(mlp_path, "w", encoding="utf-8", newline="\n").write(m)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn decoder_feedforward(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
call = "    let (x, g) = shared::mlp::feedforward_fo_t1(\n"
assert kfn.count(call) == 1, "ffn call"
note = "    // V384: the geglu 1/s is stored as one aligned 256 B block per cluster (no partial-line write under the sync).\n"
if mode == "arm":
    arm = kfn.replace("pub fn decoder_feedforward(", "pub fn decoder_feedforward_ia(", 1)
    arm = arm.replace(call, note + "    let (x, g) = shared::mlp::feedforward_fo_t1_ia(\n", 1)
    o = o[:ke] + "\n/// V384 arm ia: production ffn with the geglu 1/s stored as one aligned block per cluster.\n" + attr + arm + o[ke:]
else:
    body = kfn.replace(call, note + "    let (x, g) = shared::mlp::feedforward_fo_t1_ia(\n", 1)
    o = o[:ks] + body + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V384", mode, "ok")
