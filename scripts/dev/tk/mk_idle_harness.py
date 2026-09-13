"""mk_idle_harness.py <src harness> <dst> <gap_ms,...> -- qkv idle-gap cold probe harness (V382).

Starts from the V280 three-kernel span harness (origin/V280_hw_span_dump:tests/test_kernels.rs). The grader's qkv score
is the first launch of the process, and that launch alone carries a +3.5k..+4.3k penalty (V371/V371cold spans). This
harness asks whether the penalty comes back after the device idles:
  * the ffn and attention plans, dispatch arms and shims are dropped (the fixture guard is disarmed, as in
    mk_qkv_harness.py);
  * production qkv runs once per entry of the gap list, sleeping gap_ms before each launch (the first entry should be
    0 so launch #0 keeps the process-first condition), and prints "gap_ms=N" before every launch;
  * the spans of every launch are dumped (V280 dumps only the first two per key).
"""
import io
import re
import sys

src, dst, gaps = sys.argv[1], sys.argv[2], [int(g) for g in sys.argv[3].split(',')]
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

body = ('const QKV_SWEEP: &[&str] = &[""; %d];\n'
        '/// V382: device idle before each qkv launch, in milliseconds (entry 0 keeps the process-first launch).\n'
        'const QKV_GAPS_MS: [u64; %d] = [%s];') % (len(gaps), len(gaps), ", ".join(str(x) for x in gaps))
t, n = re.subn(r'const QKV_SWEEP: &\[&str\] = &\[""; \d+\];', body, t)
assert n == 1, "QKV_SWEEP"

shim = t.index("async fn sliding_project_qkv(")
loop = '    for (i, variant) in plan.order.iter().enumerate() {\n'
i = t.index(loop, shim) + len(loop)
sleep = (
    '        // V382: idle the device before this launch; entry 0 is 0 so the first launch stays process-first.\n'
    '        let gap = QKV_GAPS_MS[i % QKV_GAPS_MS.len()];\n'
    '        if gap > 0 {\n'
    '            tokio::time::sleep(std::time::Duration::from_millis(gap)).await;\n'
    '        }\n'
    '        println!("    gap_ms={gap}");\n'
)
t = t[:i] + sleep + t[i:]

old = "                if seen < 2 {\n"
assert t.count(old) == 1, "span dump limit"
t = t.replace(old, "                if seen < %d {\n" % len(gaps), 1)

io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
print("idle harness:", dst, "qkv launches", len(gaps), "gaps_ms", gaps)
