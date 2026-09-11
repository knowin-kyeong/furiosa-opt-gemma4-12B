"""ds.py [suffix] -- attn_out with a dummy early HBM store (V273 tree): does the first ExplicitSync of a launch carry a
one-off cost?

Span timelines: the first mid-kernel sync of a launch is the long one (attn 10-15k, qkv x2 9-16k, ffn x2 6-18k) and
later syncs are mostly short (qkv rope 2.4-3.2k, several ffn 0.4-2.5k), and base-kernel long syncs are released by no
observable DMA or TU completion. If the first sync of a launch pays a one-off setup, a tiny store at the start of the
kernel -- its sync landing while the 18.5k O-weight tile load runs and nothing waits -- would leave the contraction
store's sync short. The dummy copies post_attn_rms_weight (a read-only input; copying residual_hbm, which the kernel also writes, crashes the driver) to a scratch HBM buffer.
Without a suffix the kernel is edited in place (static probe); with one, a copy `sliding_attention_output_<suffix>`."""
import io
import sys

suffix = sys.argv[1] if len(sys.argv) > 1 else ''
S = ('_' + suffix) if suffix else ''


def rd(p):
    return io.open(p, encoding='utf-8').read()


def wr(p, s):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:80], t.count(old), n)
    return t.replace(old, new)


p = 'src/ops.rs'
s = rd(p)
sig = '#[device(chip = 1)]\npub fn sliding_attention_output('
a = s.index(sig)
b = s.index('\n}\n', a) + 3
k = s[a:b]
old = '    let x: HbmTensorView<\'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };\n'
new = ('    // DS: a tiny early store so the launch\'s first ExplicitSync happens while the O-weight load runs.\n'
       '    let early = shared::rmsnorm::load_reducing::<Cluster>(ctx, post_attn_rms_weight);\n'
       '    let mut early_hbm: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();\n'
       '    early.view().to_hbm_view(&mut ctx.tdma, early_hbm.view_mut());\n') + old
k = rep(k, old, new)
if S:
    k = rep(k, 'pub fn sliding_attention_output(', 'pub fn sliding_attention_output%s(' % S)
    s = s[:b] + '\n/// DS harness kernel.\n' + k + s[b:]
else:
    s = s[:a] + k + s[b:]
wr(p, s)
print('DS applied' + ((' as sliding_attention_output' + S) if S else ' in place'))
