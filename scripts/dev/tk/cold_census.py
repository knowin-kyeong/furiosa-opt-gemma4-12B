"""cold_census.py <TAG> <kernel> [base_arm] -- per-arm program-cold distribution with rotation positions (V381).

Reads /root/tk/<TAG>_r*.log. In each job the first launch of every arm is taken (program-cold; in the V381 harness
every arm runs exactly once). Prints pooled stats per arm, the arm x position table (position = index among the
kernel's launches in that job, 0 = first launch after the warm-up kernel), a position effect normalised by each
arm's pooled median, job-paired differences against the base arm, and the median span-group sums per arm.
"""
import glob
import re
import statistics as st
import sys

tag, kernel = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else ""
files = sorted(glob.glob("/root/tk/%s_r*.log" % tag), key=lambda f: int(re.search(r"_r(\d+)\.log$", f).group(1)))
cyc_re = re.compile(r"^\s+((?:sliding|decoder|global)_\w+)(?: (\S+))? cycles=(\d+)\s*$")


def pct(v, q):
    v = sorted(v)
    return v[int(round(q * (len(v) - 1)))]


jobs = []
fails = 0
for f in files:
    launches, groups = [], {}
    for line in open(f, errors="replace"):
        m = cyc_re.match(line)
        if m:
            launches.append((m.group(1), m.group(2) or "", int(m.group(3))))
        elif line.startswith("GROUP\t"):
            p = line.rstrip("\n").split("\t")
            name, k = p[1].rsplit("#", 1)
            kn, _, arm = name.partition(" ")
            groups.setdefault((kn, arm, int(k)), {})[p[-1]] = int(p[3].split("=")[1])
        if "-> FAIL" in line:
            fails += 1
    if launches:
        jobs.append((f, launches, groups))

print("%s %s: %d jobs, FAIL lines %d" % (tag, kernel, len(jobs), fails))
first = {}   # arm -> list of (job index, position, cycles)
for j, (f, launches, groups) in enumerate(jobs):
    seen, pos = set(), 0
    for kn, arm, c in launches:
        if kn != kernel:
            continue
        if arm not in seen:
            seen.add(arm)
            first.setdefault(arm, []).append((j, pos, c))
        pos += 1
arms = sorted(first, key=lambda a: (a != base, a))
npos = max(p for a in arms for _, p, _ in first[a]) + 1

print("\npooled first launch per arm")
print("  %-6s %3s %8s %8s %8s %8s %8s %8s" % ("arm", "n", "min", "p10", "p25", "median", "mean", "max"))
med = {}
for a in arms:
    v = [c for _, _, c in first[a]]
    med[a] = st.median(v)
    print("  %-6s %3d %8d %8d %8d %8.0f %8.0f %8d" % (a or "prod", len(v), min(v), pct(v, 0.10), pct(v, 0.25), med[a], st.mean(v), max(v)))

print("\nmedian by position (n)")
print("  %-6s " % "arm" + " ".join("%12s" % ("pos%d" % p) for p in range(npos)))
for a in arms:
    row = []
    for p in range(npos):
        v = [c for _, q, c in first[a] if q == p]
        row.append("%7.0f (%2d)" % (st.median(v), len(v)) if v else "%12s" % "-")
    print("  %-6s " % (a or "prod") + " ".join(row))
print("  %-6s " % "ratio" + " ".join(
    "%12.4f" % st.median([c / med[a] for a in arms for _, q, c in first[a] if q == p]) for p in range(npos)))

print("\npos0 launches per arm (sorted)")
for a in arms:
    print("  %-6s %s" % (a or "prod", sorted(c for _, q, c in first[a] if q == 0)))

if base in first:
    b = {j: c for j, _, c in first[base]}
    print("\njob-paired vs %s" % (base or "prod"))
    for a in arms:
        if a == base:
            continue
        d = [c - b[j] for j, _, c in first[a] if j in b]
        if not d:
            continue
        print("  %-6s n=%2d median diff %+7.0f (%+.2f%%) wins %d/%d  mean %+7.0f" % (
            a or "prod", len(d), st.median(d), 100 * st.median(d) / med[base], sum(x < 0 for x in d), len(d), st.mean(d)))

print("\nmedian span-group sums of the first launch (cluster 0)")
types = ["Task", "DMA", "Renegade::TuExec", "Cluster", "Renegade::StoTrf", "Renegade::StoVrf"]
print("  %-6s " % "arm" + " ".join("%10s" % t.split("::")[-1] for t in types))
for a in arms:
    row = []
    for t in types:
        v = [g[t] for (_, launches, gr) in jobs for (kn, arm, k), g in gr.items() if kn == kernel and arm == a and k == 0 and t in g]
        row.append("%10.0f" % st.median(v) if v else "%10s" % "-")
    print("  %-6s " % (a or "prod") + " ".join(row))
