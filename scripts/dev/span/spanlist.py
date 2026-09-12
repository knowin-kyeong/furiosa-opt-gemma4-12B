"""spanlist.py <log> <label...> (stdin to the pod's python3) -- ordered cluster-0 span timeline of the given launch labels
(e.g. "sliding_attention_output#0" "sliding_attention_output ut#0"): start, end, duration, name, relative to the window start."""
import sys

log, labels = sys.argv[1], sys.argv[2:]
spans = {}
for l in open(log, errors='replace'):
    if l.startswith('SPAN\t'):
        r = l.rstrip('\n').split('\t')
        if r[1] in labels:
            spans.setdefault(r[1], []).append((int(r[2]), int(r[3]), r[5] if len(r) > 5 else r[-1]))
for lab in labels:
    sp = sorted(spans.get(lab, []))
    if not sp:
        print('== %s: no spans' % lab)
        continue
    t0 = sp[0][0]
    t1 = max(e for _, e, _ in sp)
    print('== %s  window %d' % (lab, t1 - t0))
    for b, e, n in sp:
        if n == 'Task':
            continue
        print('  %6d %6d %6d  %s' % (b - t0, e - t0, e - b, n))
