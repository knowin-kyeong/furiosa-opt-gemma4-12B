import io


def rw(p, f):
    s = io.open(p, encoding='utf-8').read()
    s = f(s)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)


def probe(name, rows, slices_ty, slices_doc):
    return '''
/// V286 probe: the query weight at {rows} rows per live slice ({doc}).
pub(crate) fn {name}(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) {{
    let probe: DmTensor<f8e4m3, Chip, QueryClusters, {sl}, m![Qs % {rows}, H]> = weight.to_dm(&mut ctx.tdma);
    let _keep: TrfTensor<f8e4m3, Chip, QueryClusters, {sl}, m![1], m![Qs % {rows} = 1, H]> = ctx
        .sub
        .begin(probe.view().tile::<m![Qs % {rows}], 1, m![Qs % {rows} = 1 # {rows}, H]>(0))
        .fetch::<m![Qs % {rows} = 1, H / 32], m![H % 32]>()
        .collect::<m![Qs % {rows} = 1, H / 32], m![H % 32]>()
        .to_trf();
}}
'''.format(name=name, rows=rows, sl=slices_ty, doc=slices_doc)


HEAD = '''
// ---------------------------------------------------------------------------------------------
// V286: qkv's weight loads stream at ~494 B/cycle on hardware (two-point fit of its Q and V load spans, no per-slice
// cost) against 645 for ffn's up/gate and 680 for attn's O-weight. Time the same 15.73 MB of query weight in three
// slice layouts with the span harness; each probe is kept alive by staging one row to TRF (the V208 idiom).
// ---------------------------------------------------------------------------------------------
'''
PROBES = HEAD + probe('probe_q_rows8', 8, 'QueryRows', '256 slices per cluster, the production layout') \
    + probe('probe_q_rows16', 16, 'm![Qs / 16 % 128, 1 # 2]', '128 live slices per cluster') \
    + probe('probe_q_rows32', 32, 'm![Qs / 32 % 64, 1 # 4]', '64 live slices per cluster')
rw('src/device/sliding/projection.rs', lambda s: s.rstrip('\n') + '\n' + PROBES)


def ops(s):
    old = '    let q_weight = sliding::projection::load_query_weight(ctx, q_weight);\n'
    a = s.index('pub fn sliding_project_qkv(')
    i = s.index(old, a)
    new = ('    sliding::projection::probe_q_rows8(ctx, q_weight);\n'
           '    sliding::projection::probe_q_rows16(ctx, q_weight);\n'
           '    sliding::projection::probe_q_rows32(ctx, q_weight);\n') + old
    return s[:i] + new + s[i + len(old):]


rw('src/ops.rs', ops)
print('V286 probes added')
