"""paired_tails.py <tag> <kernel> <base_arm> <arm...> -- tail-aware comparison of harness arms (LB 8 plan, metric C1).

Run on the pod with stdin: `ssh <pod> 'python3 - V370 sliding_attention_output prod tx dr' < scripts/dev/paired_tails.py`.
Reads /root/tk/<tag>_r*.log like paired_medians.py (lines "    <kernel>[ <arm>] cycles=N"; the unsuffixed kernel line is
the arm "prod"). The leaderboard keeps the best of many cold draws, so a change is judged on the lower tail as well as the
median:
  * pooled warm launches per arm: n, min, p05, p10, p25, median;
  * per job: paired differences of the median, the minimum and the arm's first launch in that job;
  * pooled p10 and p05 differences against the base arm with a job-bootstrap 90% interval (jobs resampled with
    replacement, 2,000 draws).
"""
import glob
import random
import re
import statistics as st
import sys

tag, kernel, base, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
all_arms = [base] + [a for a in arms if a != base]
pattern = re.compile(r"\s+%s(?: (\w+))? cycles=(\d+)" % re.escape(kernel))
files = sorted(glob.glob("/root/tk/%s_r*.log" % tag), key=lambda p: int(re.search(r"_r(\d+)\.log$", p).group(1)))


def quantile(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


jobs = []
for path in files:
    samples = {a: [] for a in all_arms}
    for line in open(path, errors="replace"):
        m = pattern.match(line)
        if m and (m.group(1) or "prod") in samples:
            samples[m.group(1) or "prod"].append(int(m.group(2)))
    if all(samples[a] for a in all_arms):
        jobs.append(samples)

print("files", len(files), "jobs with all arms", len(jobs))
if not jobs:
    sys.exit(0)

print("pooled launches:")
for arm in all_arms:
    v = [c for job in jobs for c in job[arm]]
    print("  %-6s n=%4d min %7d p05 %7d p10 %7d p25 %7d med %7d" % (
        arm, len(v), min(v), quantile(v, 0.05), quantile(v, 0.10), quantile(v, 0.25), st.median(v)))


def pooled_diff(sample_jobs, arm, q):
    a = [c for job in sample_jobs for c in job[arm]]
    b = [c for job in sample_jobs for c in job[base]]
    return quantile(a, q) - quantile(b, q)


random.seed(20260913)
for arm in all_arms[1:]:
    ref = st.median([c for job in jobs for c in job[base]])
    rows = {
        "median": [st.median(job[arm]) - st.median(job[base]) for job in jobs],
        "min": [min(job[arm]) - min(job[base]) for job in jobs],
        "first": [job[arm][0] - job[base][0] for job in jobs],
    }
    print("%s - %s (n jobs %d):" % (arm, base, len(jobs)))
    for name, diffs in rows.items():
        negative = sum(1 for d in diffs if d < 0)
        mean = sum(diffs) / len(diffs)
        print("  per-job %-6s negative %2d/%d mean %+7.0f (%+.2f%%) median %+7.0f" % (
            name, negative, len(diffs), mean, 100 * mean / ref, st.median(diffs)))
    for q in (0.10, 0.05):
        point = pooled_diff(jobs, arm, q)
        boot = sorted(pooled_diff([random.choice(jobs) for _ in jobs], arm, q) for _ in range(2000))
        print("  pooled p%02d diff %+7.0f (%+.2f%%)  90%% job-bootstrap [%+.0f, %+.0f]" % (
            round(q * 100), point, 100 * point / ref, boot[100], boot[1899]))
