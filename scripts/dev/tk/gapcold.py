"""gapcold.py <TAG> [kernel] -- V382: does device idle bring back the first-launch penalty, and where does it land?

Reads /root/tk/<TAG>_r*.log from the idle-gap harness (a "gap_ms=N" line before every launch, spans of every launch).
Per job the warm reference is the median of the launches that followed another launch immediately (gap 0, index >= 1).
Prints, per launch index, the gap and the distribution of (launch - job reference). Then, for launch 0 and every
launch after a gap, matches its spans to the immediately following launch by (type, rank of start within type) and
prints the spans whose duration changed by >= 250 cycles or that are cross-cluster (Cluster) spans.
"""
import glob
import re
import statistics as st
import sys

tag = sys.argv[1]
kernel = sys.argv[2] if len(sys.argv) > 2 else "sliding_project_qkv"
files = sorted(glob.glob("/root/tk/%s_r*.log" % tag), key=lambda f: int(re.search(r"_r(\d+)\.log$", f).group(1)))
gap_re = re.compile(r"^\s+gap_ms=(\d+)")
cyc_re = re.compile(r"^\s+%s(?: \S+)? cycles=(\d+)" % re.escape(kernel))

by_idx, pairs = {}, {}
jobs = fails = 0
for f in files:
    gaps, cyc, spans = [], [], {}
    for line in open(f, errors="replace"):
        m = gap_re.match(line)
        if m:
            gaps.append(int(m.group(1)))
            continue
        m = cyc_re.match(line)
        if m:
            cyc.append(int(m.group(1)))
            continue
        if line.startswith("SPAN\t"):
            p = line.rstrip("\n").split("\t")
            name, k = p[1].rsplit("#", 1)
            if name.split(" ")[0] == kernel:
                spans.setdefault(int(k), []).append((int(p[2]), int(p[3]), int(p[4]), p[5]))
        if "-> FAIL" in line:
            fails += 1
    if len(cyc) < 3 or len(cyc) != len(gaps):
        continue
    jobs += 1
    ref = st.median([c for i, (g, c) in enumerate(zip(gaps, cyc)) if i >= 1 and g == 0])
    for i, (g, c) in enumerate(zip(gaps, cyc)):
        by_idx.setdefault(i, (g, []))[1].append(c - ref)
    for i, g in enumerate(gaps):
        if (i == 0 or g > 0) and i + 1 < len(gaps) and i in spans and i + 1 in spans and len(spans[i]) == len(spans[i + 1]):
            a, b = {}, {}
            for src, dst in ((spans[i], a), (spans[i + 1], b)):
                for s in sorted(src):
                    dst.setdefault(s[3], []).append(s)
            for t in b:
                if t == "Task" or len(a.get(t, [])) != len(b[t]):
                    continue
                for r, (x, y) in enumerate(zip(a[t], b[t])):
                    pairs.setdefault((i, g), {}).setdefault((t, r), []).append((x, y))

print("%s: %d jobs, FAIL lines %d" % (tag, jobs, fails))
print("\nlaunch - job warm reference (median of gap-0 launches after the first)")
print("  %3s %7s %8s %8s %8s %8s %s" % ("idx", "gap_ms", "median", "mean", "min", "max", ">+2k"))
for i in sorted(by_idx):
    g, d = by_idx[i]
    print("  %3d %7d %+8.0f %+8.0f %+8d %+8d %d/%d" % (i, g, st.median(d), st.mean(d), min(d), max(d), sum(x > 2000 for x in d), len(d)))

for (i, g), rows in sorted(pairs.items()):
    print("\n[launch %d, gap %d ms] vs launch %d: spans with |d change| >= 250 or Cluster" % (i, g, i + 1))
    print("  %-10s %4s %8s %7s %8s %9s %3s" % ("type", "rank", "st_next", "d_next", "d_change", "end_shift", "n"))
    for (t, r), v in sorted(rows.items(), key=lambda kv: st.median([y[0] for _, y in kv[1]])):
        dch = st.median([x[2] - y[2] for x, y in v])
        if abs(dch) < 250 and t != "Cluster":
            continue
        print("  %-10s %4d %8.0f %7.0f %+8.0f %+9.0f %3d" % (
            t.split("::")[-1], r, st.median([y[0] for _, y in v]), st.median([y[2] for _, y in v]), dch,
            st.median([x[1] - y[1] for x, y in v]), len(v)))
