#!/bin/bash
# pairjobs.sh <repo> <tag> <kernel> <base_variant> <test_variant> <N> -- pod: N paired Arena jobs.
#
# Runs the release build already in <repo>. The sweep harness launches both variants of <kernel> in rotated
# order inside each job, so every job gives one within-job difference (RULES 10.0l: official draws cannot
# compare code). Prints one line per job, then n / negatives / mean / two-sided sign-test p.
# Use "" for the default variant. Effects under 2% need 8 launches per variant (V263).
#
# Read the pass/fail columns before the timing: a variant generator that silently drops a computation
# looks faster (V259, V260). Never believe a job with FAIL lines.
. /root/env.sh
REPO=${1:?usage: pairjobs.sh repo tag kernel base_variant test_variant N}
TAG=${2:?}; K=${3:?}; B=${4-}; T=${5:?}; N=${6:-8}
med() {
  printf "%s\n" "$1" | awk -v k="$K" -v v="$2" '
    $1 == k && ((v == "" && $2 ~ /^n=/) || $2 == v) {
      for (i = 2; i <= NF; i++) if ($i ~ /^median=/) { sub("median=", "", $i); print $i; exit }
    }'
}
diffs=""
for i in $(seq 1 "$N"); do
  out=$(REPO="$REPO" ARENA_WAIT=1500 bash /root/auto/arena.sh "${TAG}_$i" 2>&1)
  mb=$(med "$out" "$B"); mt=$(med "$out" "$T")
  np=$(printf "%s\n" "$out" | grep -c -- "-> PASS"); nf=$(printf "%s\n" "$out" | grep -c FAIL)
  if [ -n "$mb" ] && [ -n "$mt" ]; then d=$(( mt - mb )); diffs="$diffs $d"; else d="?"; fi
  echo "r$i $K ${B:-base}=$mb $T=$mt d=$d pass=$np fail=$nf"
done
python3 -c '
import math, sys
d = [int(x) for x in sys.argv[1:]]
n = len(d)
if n:
    neg = sum(1 for x in d if x < 0)
    k = min(neg, n - neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    print("n=%d negatives=%d mean=%+.0f sign-test p=%.4f" % (n, neg, sum(d) / n, p))
' $diffs
