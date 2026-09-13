"""spandiff.py <TAG> <kernel> <arm> -- where does the first launch of a program lose time? (cold #0 vs warm #1)

Reads /root/tk/<TAG>_r*.log (harnesses that print SPAN lines for the first two launches of each arm). Per job the
spans of launch #0 and #1 of <arm> are matched by (type, rank of start time within the type); jobs whose two
launches have different span counts are skipped. Jobs are split into "pfirst" (the arm's #0 was the first launch
of the process) and "pwarm" (another launch ran before it). For each matched span the table gives the median warm
start/duration, the median duration change and the median change of its end time (the cumulative shift), so the
first row whose end shift jumps is where the cold launch starts losing.
"""
import glob
import re
import statistics as st
import sys

tag, kernel, arm = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""
arm = "" if arm in ("prod", '""') else arm
files = sorted(glob.glob("/root/tk/%s_r*.log" % tag), key=lambda f: int(re.search(r"_r(\d+)\.log$", f).group(1)))
cyc_re = re.compile(r"^\s+((?:sliding|decoder|global)_\w+)(?: (\S+))? cycles=(\d+)\s*$")

rows = {"pfirst": {}, "pwarm": {}}
tot = {"pfirst": [], "pwarm": []}
skipped = 0
for f in files:
    launches, spans = [], {}
    for line in open(f, errors="replace"):
        m = cyc_re.match(line)
        if m:
            launches.append((m.group(1), m.group(2) or ""))
        elif line.startswith("SPAN\t"):
            p = line.rstrip("\n").split("\t")
            name, k = p[1].rsplit("#", 1)
            kn, _, a = name.partition(" ")
            if kn == kernel and a == arm:
                spans.setdefault(int(k), []).append((int(p[2]), int(p[3]), int(p[4]), p[5]))
    if 0 not in spans or 1 not in spans or not launches:
        continue
    if len(spans[0]) != len(spans[1]):
        skipped += 1
        continue
    cls = "pfirst" if launches[0] == (kernel, arm) else "pwarm"
    s0 = {}
    s1 = {}
    for k, dst in ((0, s0), (1, s1)):
        for s in sorted(spans[k]):
            dst.setdefault(s[3], []).append(s)
    task0 = [s for s in spans[0] if s[3] == "Task"]
    task1 = [s for s in spans[1] if s[3] == "Task"]
    if task0 and task1:
        tot[cls].append(task0[0][1] - task1[0][1])
    for t in s1:
        if t == "Task" or len(s0.get(t, [])) != len(s1[t]):
            continue
        for r, (a, b) in enumerate(zip(s0[t], s1[t])):
            rows[cls].setdefault((t, r), []).append((a, b))

print("%s %s arm=%s  skipped (span count differs) %d" % (tag, kernel, arm or "prod", skipped))
for cls in ("pfirst", "pwarm"):
    if not tot[cls]:
        continue
    print("\n[%s] jobs %d  cold - warm total: median %+.0f  mean %+.0f" % (cls, len(tot[cls]), st.median(tot[cls]), st.mean(tot[cls])))
    print("  %-10s %4s %8s %7s %8s %8s %8s %3s" % ("type", "rank", "st_warm", "d_warm", "d_cold-w", "end_shift", "st_shift", "n"))
    order = sorted(rows[cls].items(), key=lambda kv: st.median([b[0] for _, b in kv[1]]))
    for (t, r), pairs in order:
        print("  %-10s %4d %8.0f %7.0f %+8.0f %+9.0f %+8.0f %3d" % (
            t.split("::")[-1], r,
            st.median([b[0] for _, b in pairs]),
            st.median([b[2] for _, b in pairs]),
            st.median([a[2] - b[2] for a, b in pairs]),
            st.median([a[1] - b[1] for a, b in pairs]),
            st.median([a[0] - b[0] for a, b in pairs]),
            len(pairs)))
