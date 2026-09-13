"""mk_cold_harness.py <src harness> <dst> <arm:ops_fn>... -- official-order cold-launch census harness (V381).

Starts from the V280 three-kernel span harness (origin/V280_hw_span_dump:tests/test_kernels.rs). The grader runs each
kernel once, qkv first and attention second, so the attention score is the first run of the attention program in a
process that has already run qkv. This harness reproduces that condition for many arms at once:
  * the ffn plan, its dispatch and its shim are dropped (the fixture guard is disarmed, as in mk_qkv_harness.py);
  * qkv runs once (production) to warm the process;
  * the attention shim runs each arm exactly once -- production "" plus every <tag> -- in an order rotated by
    wall-clock minute, so across jobs every arm takes every position after the qkv warm-up.
Every attention launch is therefore program-cold. paired_tails.py reads the logs unchanged (one launch per arm per job).
"""
import io
import re
import sys

src, dst, arms = sys.argv[1], sys.argv[2], [a.split(':') for a in sys.argv[3:]]
t = io.open(src, encoding='utf-8').read()

old = '    Plan { name: "decoder_feedforward", atol: 0.01, rtol: RTOL, order: FFN_SWEEP },\n'
assert t.count(old) == 1, old
t = t.replace(old, "", 1)
old = '        "decoder_feedforward" => decoder_feedforward(ctx, fixture, bench, plan).await,\n'
assert t.count(old) == 1, old
t = t.replace(old, "", 1)
a = t.index("async fn decoder_feedforward(")
b = t.index("\n}\n", a) + 3
t = t[:a] + t[b:]

g = "orphans.is_empty(),"
assert t.count(g) == 1, "fixture guard"
t = t.replace(g, "orphans.is_empty() || !orphans.is_empty(),", 1)

t, n = re.subn(r'const QKV_SWEEP: &\[&str\] = &\[""; \d+\];', 'const QKV_SWEEP: &[&str] = &[""; 1];', t)
assert n == 1, "QKV_SWEEP"

tags = [""] + [tag for tag, _ in arms]
body = "const ATTN_SWEEP: &[&str] = &[%s];" % ", ".join('"%s"' % x for x in tags)
t, n = re.subn(r'const ATTN_SWEEP: &\[&str\] = &\[""; \d+\];', body, t)
assert n == 1, "ATTN_SWEEP"

shim = t.index("async fn sliding_attention_output(")
loop = '    for (i, variant) in plan.order.iter().enumerate() {\n'
i = t.index(loop, shim)
rot = (
    '    // V381: one launch per arm, order rotated by wall-clock minute -- every arm is program-cold and, across jobs,\n'
    '    // takes every position after the qkv warm-up launch (the grader runs attention right after qkv).\n'
    '    let rot = (std::time::SystemTime::now()\n'
    '        .duration_since(std::time::UNIX_EPOCH)\n'
    '        .map(|d| d.as_secs())\n'
    '        .unwrap_or(0)\n'
    '        / 60) as usize\n'
    '        % plan.order.len();\n'
    '    let order: Vec<&\'static str> = (0..plan.order.len()).map(|j| plan.order[(j + rot) % plan.order.len()]).collect();\n'
    '    println!("    sweep rotation {rot}: {order:?}");\n'
    '    for (i, variant) in order.iter().enumerate() {\n'
)
t = t[:i] + rot + t[i + len(loop):]
old = "        if plan.order[..i].iter().all(|seen| seen != variant) {\n"
j = t.index(old, shim)
t = t[:j] + "        if order[..i].iter().all(|seen| seen != variant) {\n" + t[j + len(old):]

mark = '            other => panic!("no variant `{other}` for sliding_attention_output"),'
start = t.index('            "" => {', shim)
end = t.index('\n            }\n', start) + len('\n            }\n')
tmpl = t[start:end]
assert tmpl.count('ops::sliding_attention_output,') == 1, "attention template"
k = t.index(mark, shim)
clones = "".join(
    tmpl.replace('            "" => {', '            "%s" => {' % tag, 1).replace('ops::sliding_attention_output,', 'ops::%s,' % fn, 1)
    for tag, fn in arms
)
t = t[:k] + clones + t[k:]
io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
print("cold harness:", dst, "qkv warm-up x1, attention arms", tags)
