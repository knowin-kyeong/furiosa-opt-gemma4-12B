"""v287.py -- qkv weights on half the slices (16 Q rows / 8 KV rows per live slice), V273 tree, as _h2 copies.

V286 timed the query weight's hardware load at 612-630 B/cycle at 16-32 rows per live slice against 505-520 at the
production 8 rows per slice. qkv's critical path is its DMA FIFO, so load the Q/K/V weights at twice the rows per
live slice (live slices at even indices, so every DMN still gets traffic) and let the contractions run twice the
rows per slice. x stays replicated onto every slice; the projections reshape it onto the live half. The head gathers
become Broadcast1 { slice1: 32, slice0: 2 } over the same 64-slice rings and land every head on the slice it lands on
today, so the RMSNorm / RoPE / store / scatter path is unchanged."""
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


p = 'src/device/sliding/projection.rs'
s = rd(p)

q = fn_text(s, 'pub(crate) fn project_query(')
q = rep(q, 'pub(crate) fn project_query(', 'pub(crate) fn project_query_h2(')
q = rep(q, 'weight_f8: &QueryWeight,', 'weight_f8: &QueryWeightH2,')
q = rep(q, 'QueryClusters, QueryRows,', 'QueryClusters, QueryRowsH2,', 3)
q = rep(q, 'm![Qs % 8]> = ctx', 'm![Qs % 16]> = ctx')
q = rep(q, '.fetch::<m![Qs % 8, H / 64, Dummy2], m![H % 64]>()',
        '.fetch::<m![Qs % 16, H / 64, Dummy2], m![H % 64]>()')
q = rep(q, '.collect::<m![Qs % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()',
        '.collect::<m![Qs % 16, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()')
q = rep(q, '.contract_outer::<m![Qs % 8, H / 64, Dummy2], m![H % 64], _, _, _>(&x_trf)',
        '.contract_outer::<m![Qs % 16, H / 64, Dummy2], m![H % 64], _, _, _>(&x_trf)')
q = rep(q, '.contract_time::<m![Qs % 8]>()', '.contract_time::<m![Qs % 16]>()')
q = rep(q, '.contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)',
        '.contract_lane::<m![Qs % 16], m![1 # 8]>(LaneMode::Interleaved)')
q = rep(q, '.transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()', '.transpose::<m![Qs / 4 % 4], m![Qs % 4 # 16]>()')
q = rep(q, 'HeadClusters, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =',
        'HeadClusters, m![Ns % 4, Gs, Ds / 16, 1 # 2], m![Ds % 16]> =')
old_gather = """    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
"""
new_gather = """    // V287: 32 live slices x 16 rows per head, padding innermost; the ring is still 64 slices and every head lands on
    // the slice HeadSlicesPerCluster puts it on.
    let gathered: DmTensor<bf16, Chip, HeadClusters, m![Ns % 4, 1 # 32, 1 # 2], m![Gs, Ds]> = ctx
        .main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 16]>()
        .switch::<m![Ns % 4, 1 # 32, 1 # 2], m![Gs, Ds / 16]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();
    unsafe { gathered.reshape() }
}
"""
q = rep(q, old_gather, new_gather)

k1 = fn_text(s, 'fn project_one_kv_matrix(')
k1 = rep(k1, 'fn project_one_kv_matrix(', 'fn project_one_kv_matrix_h2(')
k1 = rep(k1, 'x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![Dummy2, H]>,',
         'x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRowsH2, m![1], m![Dummy2, H]>,')
k1 = rep(k1, 'weight_f8: &KvWeight,', 'weight_f8: &KvWeightH2,')
k1 = rep(k1, 'DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx',
         'DmTensor<bf16, Chip, KvClusters, KvRowsH2, m![Ps % 8]> = ctx')
k1 = rep(k1, '.fetch::<m![Ps % 4, H / 64, Dummy2], m![H % 64]>()',
         '.fetch::<m![Ps % 8, H / 64, Dummy2], m![H % 64]>()')
k1 = rep(k1, '.collect::<m![Ps % 4, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()',
         '.collect::<m![Ps % 8, H / 64, Dummy2, H / 32 % 2], m![H % 32]>()')
k1 = rep(k1, '.contract_outer::<m![Ps % 4, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)',
         '.contract_outer::<m![Ps % 8, H / 64, Dummy2], m![H % 64], _, _, _>(x_trf)')
k1 = rep(k1, '.contract_time::<m![Ps % 4]>()', '.contract_time::<m![Ps % 8]>()')
k1 = rep(k1, '.contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)',
         '.contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)')
k1 = rep(k1, '.transpose::<m![1], m![Ps % 4 # 16]>()', '.transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()')
k1 = rep(k1, 'HeadClusters, m![Ns % 4, Ds / 4], m![Ds % 4]> =',
         'HeadClusters, m![Ns % 4, Ds / 8, 1 # 2], m![Ds % 8]> =')
old_kg = """    ctx.main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<HeadSlicesPerCluster, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}
"""
new_kg = """    let gathered: DmTensor<bf16, Chip, HeadClusters, m![Ns % 4, 1 # 32, 1 # 2], m![Ds]> = ctx
        .main
        .begin(scaled)
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<m![Ns % 4, 1 # 32, 1 # 2], m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    unsafe { gathered.reshape() }
}
"""
k1 = rep(k1, old_kg, new_kg)

kv = fn_text(s, 'pub(crate) fn project_key_value(')
kv = rep(kv, 'pub(crate) fn project_key_value(', 'pub(crate) fn project_key_value_h2(')
kv = rep(kv, 'k_weight: &KvWeight,', 'k_weight: &KvWeightH2,')
kv = rep(kv, 'v_weight: &KvWeight,', 'v_weight: &KvWeightH2,')
kv = rep(kv, 'KvClusters, KvRows,', 'KvClusters, KvRowsH2,', 2)
kv = rep(kv, 'project_one_kv_matrix(ctx,', 'project_one_kv_matrix_h2(ctx,', 2)

head = """
// ---------------------------------------------------------------------------------------------
// V287: the Q/K/V weights on half the slices. V286 timed the query weight's hardware load at 612-630 B/cycle at
// 16-32 rows per live slice against 505-520 at 8, and qkv's critical path is its DMA FIFO. Live slices sit at even
// indices so every DMN still takes traffic; x stays replicated on every slice and is reshaped onto the live half.
// ---------------------------------------------------------------------------------------------
type QueryRowsH2 = m![Qs / 16 % 128, 1 # 2];
pub(crate) type QueryWeightH2 = DmTensor<f8e4m3, Chip, QueryClusters, QueryRowsH2, m![Qs % 16, H]>;

pub(crate) fn load_query_weight_h2(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QueryWeightH2 {
    weight.to_dm(&mut ctx.tdma)
}

type KvRowsH2 = m![Ps / 8 % 128, 1 # 2];
pub(crate) type KvWeightH2 = DmTensor<f8e4m3, Chip, KvClusters, KvRowsH2, m![Ps % 8, H]>;

pub(crate) fn load_kv_weight_h2(ctx: &mut Context, weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>) -> KvWeightH2 {
    weight.to_dm(&mut ctx.tdma)
}
"""
s = s.rstrip('\n') + '\n' + head + '\n' + q + '\n' + k1 + '\n' + kv
wr(p, s)

p = 'src/ops.rs'
s = rd(p)
sig = '#[device(chip = 1)]\npub fn sliding_project_qkv('
k = fn_text(s, sig)
k = rep(k, 'pub fn sliding_project_qkv(', 'pub fn sliding_project_qkv_h2(')
k = rep(k, 'sliding::projection::load_query_weight(ctx, q_weight)', 'sliding::projection::load_query_weight_h2(ctx, q_weight)')
k = rep(k, 'sliding::projection::load_kv_weight(ctx,', 'sliding::projection::load_kv_weight_h2(ctx,', 2)
k = rep(k, 'sliding::projection::project_query(ctx,', 'sliding::projection::project_query_h2(ctx,')
k = rep(k, 'sliding::projection::project_key_value(ctx,', 'sliding::projection::project_key_value_h2(ctx,')
b = s.index(sig)
e = s.index('\n}\n', b) + 3
s = s[:e] + '\n/// V287 harness kernel: Q/K/V weights on half the slices.\n' + k + s[e:]
wr(p, s)
print('V287 _h2 copies added')
