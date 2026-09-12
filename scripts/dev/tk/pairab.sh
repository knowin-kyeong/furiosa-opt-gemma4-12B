#!/bin/bash
# pairab.sh <labA> <labB> <tag> <N> -- paired Arena samples of two builds of the same harness (A: no forced order,
# B: forced order). Submits each build once (/root/tk/<tag>A_r0.log, <tag>B_r0.log), then runs N rounds in which the two
# jobs are rerun at the same time (the account runs 2 jobs at once), so every A/B pair shares a time window.
set -u
LA=$1; LB=$2; TAG=$3; N=${4:-7}
S=/root/tk/${TAG}_pairs.txt
: > "$S"
REPO="/root/$LA" bash /root/tk/arena_retry.sh /root/auto/arena.sh "${TAG}A_r0" > "/root/tk/${TAG}A_r0.log" 2>&1 &
pa=$!
sleep 20
REPO="/root/$LB" bash /root/tk/arena_retry.sh /root/auto/arena.sh "${TAG}B_r0" > "/root/tk/${TAG}B_r0.log" 2>&1 &
pb=$!
wait $pa $pb
JA=$(grep -m1 -oE "submitted job [0-9]+" "/root/tk/${TAG}A_r0.log" | grep -oE "[0-9]+$")
JB=$(grep -m1 -oE "submitted job [0-9]+" "/root/tk/${TAG}B_r0.log" | grep -oE "[0-9]+$")
echo "A job $JA pass=$(grep -c -- '-> PASS' /root/tk/${TAG}A_r0.log) | B job $JB pass=$(grep -c -- '-> PASS' /root/tk/${TAG}B_r0.log) fail=$(grep -c FAIL /root/tk/${TAG}B_r0.log)" | tee -a "$S"
[ -n "$JA" ] && [ -n "$JB" ] || { echo SUBMIT_FAILED | tee -a "$S"; exit 1; }
for i in $(seq 1 "$N"); do
  ARENA_WAIT=1500 bash /root/tk/arena_retry.sh /root/auto/arena_rerun.sh "$JA" "${TAG}A_r$i" > "/root/tk/${TAG}A_r$i.log" 2>&1 &
  pa=$!
  sleep 20
  ARENA_WAIT=1500 bash /root/tk/arena_retry.sh /root/auto/arena_rerun.sh "$JB" "${TAG}B_r$i" > "/root/tk/${TAG}B_r$i.log" 2>&1 &
  pb=$!
  wait $pa $pb
  a=$(grep -m1 -E "median=" "/root/tk/${TAG}A_r$i.log" | grep -oE "median=[0-9]+")
  b=$(grep -m1 -E "median=" "/root/tk/${TAG}B_r$i.log" | grep -oE "median=[0-9]+")
  echo "round $i A $a B $b" | tee -a "$S"
done
echo PAIRS_DONE | tee -a "$S"
