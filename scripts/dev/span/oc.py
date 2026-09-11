"""oc.py [suffix] -- attn_out on ONE cluster with no mid-kernel HBM store (V273 tree).

Hardware spans (V280): in 0.6.0 every DmaStore is followed by an ExplicitSync, and the one after the contraction store
costs 4-16k cycles on hardware (static 600) -- about a quarter of the kernel -- before the x reload may run. The store
exists only to move the two clusters' halves of [H] back onto one cluster, and a cross-cluster DM->DM DMA is rejected
by the synchronization checker. So run the contraction on one cluster (16 row groups of 240 rows x 16 column chunks =
256 slices; the same weight bytes, twice the rows per slice) and relay its output into the reducing layout with a
same-cluster DM->DM `to_dm`, which inserts no synchronization.
Without a suffix `project_output` and `sliding_attention_output` are edited in place (static probe); with one,
`project_output_<suffix>` and `sliding_attention_output_<suffix>` are added (harness)."""
import io
import sys

suffix = sys.argv[1] if len(sys.argv) > 1 else ''
S = ('_' + suffix) if suffix else ''


def span(src, sig):
    a = src.index(sig)
    return a, src.index('\n}\n', a) + 3


def rep(body, old, new, what):
    assert body.count(old) == 1, (what, body.count(old), old[:90])
    return body.replace(old, new, 1)


def contraction(rows, off):
    return '''    ctx.main
        .begin(tile{t}.view())
        .fetch::<m![H % 240 = {r}, Qs / 64 % 4], m![Qs % 64]>()
        .collect::<m![H % 240 = {r}, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
        .contract_outer::<m![H % 240 = {r}, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 240 = {r}]>()
        .contract_lane::<m![H % 240 = {r}], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<OneClusterRows, m![H % 240 = {r}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 240 = {r} / 4], m![H % 240 = {r} % 4 # 16]>()
        .commit_trim::<m![H % 240 = {r} % 4]>()
        .commit_view(contraction.view_mut().tile::<m![H % 240], {r}, m![H % 240 = {r} #{{!}} 240]>({o}));
'''.format(t=0 if off == 0 else 1, r=rows, o=off)


R0, R1 = 192, 48
body = '''pub(crate) fn project_output{S}(
    ctx: &mut Context,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::shared::rmsnorm::ReducingSlices, m![H % 480]> {{
    // OC: one cluster, so the contraction output reaches the RMSNorm's reducing layout through a same-cluster
    // DM->DM relay instead of an HBM store + ExplicitSync + reload (the sync alone is 4-16k cycles on hardware).
    // 256 slices = 16 row groups of 240 rows x 16 column chunks of 256; the rows come in two tiles, {R0} then {R1}.
    let tile0: DmTensor<f8e4m3, Chip, Cluster, OneClusterRowsByColumns, m![H % 240 = {R0}, Qs % 256]> = weight
        .view()
        .tile::<m![H % 240], {R0}, m![H / 240, H % 240 = {R0} # 240, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let tile1: DmTensor<f8e4m3, Chip, Cluster, OneClusterRowsByColumns, m![H % 240 = {R1}, Qs % 256]> = weight
        .view()
        .tile::<m![H % 240], {R1}, m![H / 240, H % 240 = {R1} # 240, Qs]>({R0})
        .to_dm(&mut ctx.tdma);

    // V257 (STAGE 1 ONLY, RULES 10.0n): one f8 piece of x, s = 16.
    let xs: DmTensor<bf16, Chip, Cluster, OneClusterRowsByColumns, m![Qs % 256]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<f8e4m3, Chip, Cluster, OneClusterRowsByColumns, m![Qs % 256]> = ctx
        .main
        .begin(xs.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, Cluster, OneClusterRowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let mut contraction: DmTensor<bf16, Chip, Cluster, OneClusterRows, m![H % 240]> = DmTensor::new();
{C0}{C1}
    contraction.to_dm(&mut ctx.tdma)
}}
'''.format(S=S, R0=R0, R1=R1, C0=contraction(R0, 0), C1=contraction(R1, R0))

p = 'src/device/sliding/projection.rs'
s = io.open(p, encoding='utf-8').read()
types = '''
/// OC: one cluster, 16 row groups of 240 rows (x 16 column chunks for the weight) = 256 slices.
type OneClusterRows = m![H / 240, 1 # 16];
type OneClusterRowsByColumns = m![H / 240, Qs / 256];
'''
if S:
    s = s.rstrip('\n') + '\n\n/// OC harness copy.\n' + body + ('' if 'type OneClusterRows =' in s else types)
else:
    a, b = span(s, 'pub(crate) fn project_output(')
    s = s[:a] + body + s[b:]
    if 'type OneClusterRows =' not in s:
        s = s.rstrip('\n') + '\n' + types
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'src/ops.rs'
s = io.open(p, encoding='utf-8').read()
a, b = span(s, '#[device(chip = 1)]\npub fn sliding_attention_output(')
k = s[a:b]
k = rep(k, 'let x_hbm = sliding::projection::project_output(ctx, x, o_weight);',
        'let x = sliding::projection::project_output%s(ctx, x, o_weight);' % S, 'call')
k = rep(k, '    let x = shared::rmsnorm::load_reducing_aligned::<Cluster>(ctx, &x_hbm);\n', '', 'reload')
if S:
    k = rep(k, 'pub fn sliding_attention_output(', 'pub fn sliding_attention_output%s(' % S, 'kname')
    s = s[:b] + '\n/// OC harness kernel.\n' + k + s[b:]
else:
    s = s[:a] + k + s[b:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('OC applied' + ((' as sliding_attention_output' + S) if S else ' in place'))
