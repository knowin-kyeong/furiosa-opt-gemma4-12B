"""paired_medians.py <tag> <kernel> <base_arm> <arm...> -- per-job paired comparison of harness arms.

Run on the pod with stdin: `ssh <pod> 'python3 - v350 decoder_feedforward prod f1d fo' < scripts/dev/paired_medians.py`.
Reads /root/tk/<tag>_r*.log (the first Arena job is <tag>_r0.log, `rerun_n.sh` writes _r1.._rN). Harness lines look like
"    <kernel>[ <arm>] cycles=N"; the unsuffixed kernel line is the arm "prod". For every job that has all arms it takes
each arm's median launch, then prints the paired differences against the base arm: sign count, mean, median.
(The older pairstats.py only parsed attention logs; this one takes the kernel name as an argument.)
"""
import glob
import re
import statistics as st
import sys

tag, kernel, base, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
all_arms = [base] + [a for a in arms if a != base]
pattern = re.compile(r"\s+%s(?: (\w+))? cycles=(\d+)" % re.escape(kernel))
files = sorted(glob.glob("/root/tk/%s_r*.log" % tag), key=lambda p: int(re.search(r"_r(\d+)\.log$", p).group(1)))

rows = []
for path in files:
    samples = {a: [] for a in all_arms}
    for line in open(path, errors="replace"):
        m = pattern.match(line)
        if m:
            arm = m.group(1) or "prod"
            if arm in samples:
                samples[arm].append(int(m.group(2)))
    if all(samples[a] for a in all_arms):
        rows.append((path.split("/")[-1], {a: st.median(samples[a]) for a in all_arms}))

print("files", len(files), "jobs with all arms", len(rows))
for name, med in rows:
    print("%-14s" % name, " ".join("%s=%d" % (a, med[a]) for a in all_arms))
for arm in all_arms[1:]:
    diffs = [med[arm] - med[base] for _, med in rows]
    if not diffs:
        continue
    mean = sum(diffs) / len(diffs)
    ref = sum(med[base] for _, med in rows) / len(rows)
    negative = sum(1 for d in diffs if d < 0)
    print("%-5s - %-5s: n=%d negative=%d mean=%+.0f (%+.2f%%) median=%+.0f" % (
        arm, base, len(diffs), negative, mean, 100 * mean / ref, st.median(diffs)))
