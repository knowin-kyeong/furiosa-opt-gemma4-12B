"""mk_qkv_harness.py <src harness> <dst> <arm:ops_fn>... -- a qkv-only span harness with a minute-rotated sweep.

Starts from the V280 span harness (20709e8), drops the ffn and attn plans and shims (and disarms the fixture guard, as
/root/tk/one_only.py does), rotates the qkv sweep by wall-clock minute, and clones the "" arm of the qkv shim into one
arm per <tag>:<ops fn>. The sweep is every arm (production last in the base order) repeated 4 times, rotated each row.
"""
import io
import re
import sys

src, dst, arms = sys.argv[1], sys.argv[2], [a.split(':') for a in sys.argv[3:]]
t = io.open(src, encoding='utf-8').read()

for k, atol, sweep in (("decoder_feedforward", "0.01", "FFN_SWEEP"), ("sliding_attention_output", "0.05", "ATTN_SWEEP")):
    old = '    Plan { name: "%s", atol: %s, rtol: RTOL, order: %s },\n' % (k, atol, sweep)
    assert t.count(old) == 1, old
    t = t.replace(old, "", 1)
    old = '        "%s" => %s(ctx, fixture, bench, plan).await,\n' % (k, k)
    assert t.count(old) == 1, old
    t = t.replace(old, "", 1)
    a = t.index("async fn %s(" % k)
    b = t.index("\n}\n", a) + 3
    t = t[:a] + t[b:]

g = "orphans.is_empty(),"
assert t.count(g) == 1, "fixture guard"
t = t.replace(g, "orphans.is_empty() || !orphans.is_empty(),", 1)

tags = [tag for tag, _ in arms] + [""]
rows = []
for r in range(4):
    row = tags[r % len(tags):] + tags[:r % len(tags)]
    rows.append("    " + ", ".join('"%s"' % x for x in row) + ",")
body = "const QKV_SWEEP: &[&str] = &[\n%s\n];" % "\n".join(rows)
t, n = re.subn(r'const QKV_SWEEP: &\[&str\] = &\[""; \d+\];', body, t)
assert n == 1, "QKV_SWEEP"

shim = t.index("async fn sliding_project_qkv(")
loop = '    for (i, variant) in plan.order.iter().enumerate() {\n'
i = t.index(loop, shim)
rot = (
    '    // Sweep rotated by wall-clock minute: `rngd rerun` repeats of one binary vary which variant launches first\n'
    '    // (the process-cold launch, and the clean writer of shared DM/TRF/VRF state -- RULES 10.0p / 10.0t).\n'
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

mark = '            other => panic!("no variant `{other}` for sliding_project_qkv"),'
start = t.index('            "" => {', shim)
end = t.index('\n            }\n', start) + len('\n            }\n')
tmpl = t[start:end]
assert tmpl.count('ops::sliding_project_qkv,') == 1
k = t.index(mark, shim)
clones = "".join(
    tmpl.replace('            "" => {', '            "%s" => {' % tag, 1).replace('ops::sliding_project_qkv,', 'ops::%s,' % fn, 1)
    for tag, fn in arms
)
t = t[:k] + clones + t[k:]
io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
print("qkv harness:", dst, "arms", tags)
