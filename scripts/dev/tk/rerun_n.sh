# rerun_n.sh <job_id> <tag> <N> <kernel_regex> [first_index] -- N more samples of an uploaded Arena job via `rngd rerun`
# (no upload), retrying while the 2-active-job quota is full. Logs go to /root/tk/<tag>_r<i>.log for i = first..first+N-1.
set -u
SRC=${1:?job id}; TAG=${2:?tag}; N=${3:-8}; KRE=${4:-.}; FIRST=${5:-1}
S=/root/tk/${TAG}_rerun_summary.txt
[ "$FIRST" = 1 ] && : > "$S"
for i in $(seq "$FIRST" $((FIRST + N - 1))); do
  L=/root/tk/${TAG}_r$i.log
  ARENA_WAIT=1500 bash /root/tk/arena_retry.sh /root/auto/arena_rerun.sh "$SRC" "${TAG}_r$i" > "$L" 2>&1
  {
    echo "rerun $i ($(grep -m1 -oE 'waiting on job [0-9]+' "$L")): pass=$(grep -c -- '-> PASS' "$L") fail=$(grep -c FAIL "$L")"
    grep -E "median=" "$L" | grep -E "$KRE" | head -6
  } | tee -a "$S"
done
echo DONE >> "$S"
