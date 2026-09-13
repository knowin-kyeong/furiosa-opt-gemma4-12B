"""v383_verdict.py [first_index last_index] -- the pre-registered V383 verdict (RESULTS row V383), run by chain_0913n.sh.

Reads /root/tk/V383_r*.log (qkv harness, arms prod and 1s, 4 launches each, minute rotation). Prints the metrics and a
last line "VERDICT WIN" or "VERDICT LOSE".
  gate  every job PASS (>= 6 PASS lines, no FAIL), >= 24 jobs, and >= 8 jobs where 1s is the process-first launch
        (in those jobs 1s cannot read state left by production, so its PASS is genuine);
  warm  per job, median(1s) - median(prod) over the launches after the job's first launch: median and wins;
  cold  median of each arm's process-first launches (the grader's qkv launch is the first of its process);
  WIN   gate and ((warm <= -0.3% and wins >= 19/32 of jobs and cold diff <= +500)
                  or (cold diff <= -2,000 and warm <= +0.3%)).
"""
import glob
import re
import statistics as st
import sys

TAG, K, ARM = "V383", "sliding_project_qkv", "1s"
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10 ** 6)
pat = re.compile(r"^\s+%s(?: (\w+))? cycles=(\d+)" % K)
jobs, bad, arm_first = [], 0, 0
for f in sorted(glob.glob("/root/tk/%s_r*.log" % TAG), key=lambda p: int(re.search(r"_r(\d+)\.log$", p).group(1))):
    idx = int(re.search(r"_r(\d+)\.log$", f).group(1))
    if not lo <= idx <= hi:
        continue
    seq, npass, nfail = [], 0, 0
    for line in open(f, errors="replace"):
        m = pat.match(line)
        if m:
            seq.append((m.group(1) or "prod", int(m.group(2))))
        if "-> PASS" in line:
            npass += 1
        if "FAIL" in line:
            nfail += 1
    if not seq:
        continue
    if npass < 6 or nfail:
        bad += 1
    if not (any(a == "prod" for a, _ in seq[1:]) and any(a == ARM for a, _ in seq[1:])):
        continue
    jobs.append(seq)
    arm_first += seq[0][0] == ARM

n = len(jobs)
if n == 0:
    print("no jobs")
    print("VERDICT LOSE")
    sys.exit(0)
warm_prod = st.median([c for s in jobs for a, c in s[1:] if a == "prod"])
diffs = []
for s in jobs:
    diffs.append(st.median([c for a, c in s[1:] if a == ARM]) - st.median([c for a, c in s[1:] if a == "prod"]))
dw, wins = st.median(diffs), sum(x < 0 for x in diffs)
pf = {a: [s[0][1] for s in jobs if s[0][0] == a] for a in ("prod", ARM)}
cold = st.median(pf[ARM]) - st.median(pf["prod"]) if pf[ARM] and pf["prod"] else None
print("jobs %d  bad logs %d  %s-first jobs %d" % (n, bad, ARM, arm_first))
print("warm: prod median %.0f  per-job median diff %+.0f (%+.2f%%)  wins %d/%d" % (warm_prod, dw, 100 * dw / warm_prod, wins, n))
for a in ("prod", ARM):
    v = sorted(pf[a])
    print("cold %-4s n=%2d median %s  sorted %s" % (a, len(v), "%.0f" % st.median(v) if v else "-", v))
print("cold diff (%s - prod process-first medians): %s" % (ARM, "%+.0f" % cold if cold is not None else "-"))
gate = n >= 24 and bad == 0 and arm_first >= 8
need = -(-19 * n // 32)
win_a = cold is not None and dw <= -0.003 * warm_prod and wins >= need and cold <= 500
win_b = cold is not None and cold <= -2000 and dw <= 0.003 * warm_prod
print("gate %s  rule a %s  rule b %s" % (gate, win_a, win_b))
print("VERDICT", "WIN" if gate and (win_a or win_b) else "LOSE")
