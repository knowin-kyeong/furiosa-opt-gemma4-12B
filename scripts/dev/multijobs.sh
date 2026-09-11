#!/bin/bash
# multijobs.sh <repo> <tag> <kernel> <base_variant> <variant...> -- pod: N Arena jobs, several variants against one base.
#
# The multi-variant form of pairjobs.sh. The sweep harness launches every listed variant of <kernel> in rotated
# order inside each job, so each job gives one within-job difference per variant (RULES 10.0l: official draws
# cannot compare code). Prints one line per job, then n / negatives / mean / two-sided sign-test p per variant.
# N comes from the environment (default 16). Use "" for the default base variant; test variants must be named.
#
# Read the pass/fail columns before the timing: a generator that silently drops a computation looks faster.
. /root/env.sh
REPO=${1:?usage: multijobs.sh repo tag kernel base_variant variant...}; TAG=${2:?}; K=${3:?}; B=${4-}
shift 4
VARS=("$@"); N=${N:-16}
[ ${#VARS[@]} -gt 0 ] || { echo "multijobs.sh: no test variants" >&2; exit 2; }
med() {
  printf "%s\n" "$1" | awk -v k="$K" -v v="$2" '
    $1 == k && ((v == "" && $2 ~ /^n=/) || $2 == v) {
      for (i = 2; i <= NF; i++) if ($i ~ /^median=/) { sub("median=", "", $i); print $i; exit }
    }'
}
declare -A DIFFS
for i in $(seq 1 "$N"); do
  out=$(REPO="$REPO" ARENA_WAIT=1500 bash /root/auto/arena.sh "${TAG}_$i" 2>&1)
  mb=$(med "$out" "$B")
  np=$(printf "%s\n" "$out" | grep -c -- "-> PASS"); nf=$(printf "%s\n" "$out" | grep -c FAIL)
  line="r$i $K ${B:-base}=$mb"
  for v in "${VARS[@]}"; do
    mt=$(med "$out" "$v")
    if [ -n "$mb" ] && [ -n "$mt" ]; then d=$(( mt - mb )); DIFFS[$v]="${DIFFS[$v]} $d"; else d="?"; fi
    line="$line | $v=$mt d=$d"
  done
  echo "$line | pass=$np fail=$nf"
done
for v in "${VARS[@]}"; do
  printf "%s: " "$v"
  python3 -c '
import math, sys
d = [int(x) for x in sys.argv[1:]]
n = len(d)
if n:
    neg = sum(1 for x in d if x < 0)
    k = min(neg, n - neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    print("n=%d negatives=%d mean=%+.0f sign-test p=%.4f" % (n, neg, sum(d) / n, p))
else:
    print("no data")
' ${DIFFS[$v]}
done
