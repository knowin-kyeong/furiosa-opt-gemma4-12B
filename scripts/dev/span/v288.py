"""v288.py -- qkv RoPE: one HBM store for the cos and sin rows instead of two (V273 tree + v287.py), as _rs copies.

In 0.6.0 every DmaStore is followed by an ExplicitSync, 2.4-17k cycles on hardware. apply_rope_heads gathers the
cos row and the sin row onto one slice and stores each into its tile of the staging HBM buffer -- two stores, two
syncs, back to back. Copy both rows into the two tiles of one DM buffer with two small Main passes (the x_hi/x_lo
idiom of shared/mlp.rs) and store that buffer once; the single head-layout load that follows is unchanged.
Adds apply_rope_heads_rs, sliding_project_qkv_rs (V273 + this) and sliding_project_qkv_hr (V287 h2 + this)."""
import io


def rd(p):
    return io.open(p, encoding='utf-8').read()


def wr(p, s):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)


def fn_text(s, sig):
    a = s.index(sig)
    b = s.index('\n}\n', a) + 3
    return s[a:b]


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:80], t.count(old), n)
    return t.replace(old, new)


p = 'src/device/sliding/rope.rs'
s = rd(p)
f = fn_text(s, 'pub(crate) fn apply_rope_heads<')
f = rep(f, 'pub(crate) fn apply_rope_heads<', 'pub(crate) fn apply_rope_heads_rs<')
old = """    let mut cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = HbmTensor::new();
    cos_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    sin_row
        .view()
        .to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
"""
new = """    // V288: copy both gathered rows into one DM buffer and store it once -- every DMA store is followed by an
    // ExplicitSync (2.4-17k cycles on hardware), and these two stores ran back to back.
    let mut cs_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Dummy2, Ds]> = DmTensor::new();
    ctx.main
        .begin(cos_row.view())
        .fetch::<m![Ds / 128], m![Ds % 128]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    ctx.main
        .begin(sin_row.view())
        .fetch::<m![Ds / 128], m![Ds % 128]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(cs_dm.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    let mut cs_hbm: HbmTensor<bf16, Chip, m![Dummy2, Ds]> = HbmTensor::new();
    cs_dm.view().to_hbm_view(&mut ctx.tdma, cs_hbm.view_mut());
"""
f = rep(f, old, new)
s = s.rstrip('\n') + '\n\n/// V288 harness copy: one store for both RoPE rows.\n' + f
wr(p, s)

p = 'src/ops.rs'
s = rd(p)
for base, suffix in (('sliding_project_qkv', 'rs'), ('sliding_project_qkv_h2', 'hr')):
    sig = '#[device(chip = 1)]\npub fn %s(' % base
    k = fn_text(s, sig)
    k = rep(k, 'pub fn %s(' % base, 'pub fn sliding_project_qkv_%s(' % suffix)
    k = rep(k, 'sliding::rope::apply_rope_heads::<', 'sliding::rope::apply_rope_heads_rs::<')
    b = s.index(sig)
    e = s.index('\n}\n', b) + 3
    s = s[:e] + '\n/// V288 harness kernel (%s).\n' % suffix + k + s[e:]
wr(p, s)
print('V288 _rs copies added')
