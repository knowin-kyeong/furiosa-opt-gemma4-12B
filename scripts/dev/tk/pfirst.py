"""pfirst.py <kernel> <TAG>... -- the process-first launch of each job (the grader's qkv condition) per arm.

For every /root/tk/<TAG>_r*.log the first launch printed is process-cold. Groups those launches by arm and prints
their distribution, the same job's later launches of that arm (median = warm reference) and the per-job penalty
(first launch - that job's warm median of the same arm).
"""
import glob
import re
import statistics as st
import sys

kernel, tags = sys.argv[1], sys.argv[2:]
cyc_re = re.compile(r"^\s+%s(?: (\S+))? cycles=(\d+)\s*$" % re.escape(kernel))


def pct(v, q):
    v = sorted(v)
    return v[int(round(q * (len(v) - 1)))]


first, warm, pen = {}, {}, {}
for tag in tags:
    for f in glob.glob("/root/tk/%s_r*.log" % tag):
        seq = []
        for line in open(f, errors="replace"):
            m = cyc_re.match(line)
            if m:
                seq.append((m.group(1) or "prod", int(m.group(2))))
        if len(seq) < 2:
            continue
        arm, c = seq[0]
        rest = [x for a, x in seq[1:] if a == arm]
        first.setdefault(arm, []).append(c)
        for a, x in seq:
            later = [y for b, y in seq if b == a][1:]
            if later:
                warm.setdefault(a, []).append(st.median(later))
        if len(rest) >= 2:
            pen.setdefault(arm, []).append(c - st.median(rest[1:]))

print("%s process-first launches from %s" % (kernel, " ".join(tags)))
for arm in sorted(first):
    v = first[arm]
    print("  %-5s n=%2d  first: min %6d p10 %6d p25 %6d med %6.0f mean %6.0f max %6d | job warm med %6.0f | penalty med %+6.0f mean %+6.0f (n %d)" % (
        arm, len(v), min(v), pct(v, .1), pct(v, .25), st.median(v), st.mean(v), max(v),
        st.median(warm.get(arm, [0])), st.median(pen.get(arm, [0])), st.mean(pen.get(arm, [0])), len(pen.get(arm, []))))
    print("        sorted first launches: %s" % sorted(v))
