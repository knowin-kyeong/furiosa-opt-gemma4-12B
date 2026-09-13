#!/bin/bash
# chainv2_0913n.sh -- takes over from chain_0913n.sh (started 01:38 UTC) once that chain reaches its extra reruns.
# `rngd rerun`s finish in ~11 s, so rapid reruns share a wall-clock minute and skew the minute rotation (which arm is the
# process-first launch). From here every V383 rerun is spaced 50 s apart, and the pre-registered verdict is re-taken at
# 64 and 96 jobs. Same switch rule (first WIN -> V383_submit subverify -> draw target) and log as chain_0913n.sh.
set -u
. /root/env.sh
LOG=/root/tk/chain_0913n.log
GH=https://github.com/knowin-kyeong/furiosa-opt-gemma4-12B.git
say() { echo "$(date -u +%FT%T) $*" >> "$LOG"; }
say "chain v2 armed (waits for the old chain to reach its extra reruns)"

# the old chain is left alone through its 32-job verdict and any subverify/switch; stop it once it starts rapid reruns
while pgrep -f "^bash /root/tk/chain_0913n.sh" > /dev/null && ! pgrep -f "rerun_n.sh [0-9]+ V383 32 " > /dev/null; do sleep 10; done
pkill -f "^bash /root/tk/chain_0913n.sh"
pkill -f "rerun_n.sh [0-9]+ V383 32 "
while pgrep -f "arena_retry.sh|arena_rerun.sh" > /dev/null; do sleep 10; done
say "chain v2 takes over ($(ls /root/tk/V383_r*.log 2>/dev/null | wc -l) V383 logs)"
grep -q V383_DONE /root/tk/V383_chain.out 2>/dev/null || { say "V383 chain did not finish; v2 exits, draws stay as they are"; exit 0; }

switched=0
[ "$(cat /root/tk/draw_target 2>/dev/null)" = V383_submit ] && switched=1
decide() {  # $1 = label, $2 $3 = rerun index range
  python3 /root/tk/v383_verdict.py "$2" "$3" > "/root/tk/V383_verdict_$1.txt" 2>&1
  python3 /root/tk/paired_tails.py V383 sliding_project_qkv prod 1s >> "/root/tk/V383_verdict_$1.txt" 2>&1
  python3 /root/tk/pfirst.py sliding_project_qkv V383 >> "/root/tk/V383_verdict_$1.txt" 2>&1
  local v; v=$(awk '/^VERDICT/{print $2}' "/root/tk/V383_verdict_$1.txt")
  say "V383 verdict ($1): ${v:-none} | $(grep -E '^jobs|^warm|^cold diff' "/root/tk/V383_verdict_$1.txt" | tr '\n' ' ')"
  [ "$v" = WIN ] && [ "$switched" = 0 ] || return 0
  cd /root/lab7 || return 0
  git fetch -q "$GH" +tmp_V383s:tmp_V383s || { say "V383s fetch failed"; return 0; }
  bash /root/tk/subverify.sh lab7 V383_submit V383s > /root/tk/V383s_subverify.out 2>&1
  local p1 f1 p2 f2
  p1=$(grep -c -- "-> PASS" /root/tk/V383s_verify1.log 2>/dev/null); f1=$(grep -c FAIL /root/tk/V383s_verify1.log 2>/dev/null)
  p2=$(grep -c -- "-> PASS" /root/tk/V383s_verify2.log 2>/dev/null); f2=$(grep -c FAIL /root/tk/V383s_verify2.log 2>/dev/null)
  say "V383_submit subverify: verify1 pass=$p1 fail=$f1, verify2 pass=$p2 fail=$f2"
  if [ "${f1:-1}" = 0 ] && [ "${f2:-1}" = 0 ] && [ "${p1:-0}" -ge 25 ] && [ "${p2:-0}" -ge 25 ]; then
    git -C /root/draw_src fetch -q "$GH" +V383_submit:V383_submit || { say "draw_src fetch failed"; return 0; }
    echo V383_submit > /root/tk/draw_target
    pkill -f "^bash /root/drawchain_follow.sh" && say "stopped the V378 follow loop"
    switched=1
    say "DRAW TARGET -> V383_submit (drawkeeper starts it after the running batch)"
  else
    say "V383_submit verification failed; draws stay on V378_submit"
  fi
}
[ -f /root/tk/V383_verdict_32.txt ] || decide 32 0 31

JOB=$(grep -m1 -oE "^JOB=[0-9]+" /root/tk/V383_chain.out | cut -d= -f2)
[ -n "$JOB" ] || { say "no V383 job id; v2 exits"; exit 0; }
next=$(( $(ls /root/tk/V383_r*.log | sed -E 's/.*_r([0-9]+)\.log$/\1/' | sort -n | tail -1) + 1 ))
for target in 64 96; do
  while [ "$next" -lt "$target" ]; do
    bash /root/tk/rerun_n.sh "$JOB" V383 1 sliding_project_qkv "$next" > /dev/null 2>&1
    next=$((next + 1))
    sleep 50
  done
  say "V383 spaced reruns up to r$((target - 1)) done"
  decide "$target" 0 $((target - 1))
done
say "chain v2 done"
