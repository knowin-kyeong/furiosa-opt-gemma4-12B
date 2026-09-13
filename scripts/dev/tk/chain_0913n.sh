#!/bin/bash
# chain_0913n.sh -- unattended chain while the user is away (2026-09-13 ~02:00 -> ~14:00 UTC). Log /root/tk/chain_0913n.log.
#  1. V382 (idle-gap probe) analysis once its chain ends -> /root/tk/V382_analysis.txt
#  2. V383 (RoPE rows staged by one store) on lab6: dumps, build, accuracy gate, 32 jobs -> pre-registered verdict
#  3. 32 more reruns of the V383 job (process-first samples) -> verdict over all 64 jobs
#  4. the first WIN (32 or 64 jobs): V383_submit subverify on lab7 (Arena all PASS twice) -> draw target V383_submit
#     (/root/tk/draw_target; drawkeeper.sh starts the new target after the running batch; the V378 follow loop is
#     stopped by PID so it cannot start another V378 batch)
# Draws themselves are kept alive by /root/tk/drawkeeper.sh (separate process).
set -u
. /root/env.sh
LOG=/root/tk/chain_0913n.log
GH=https://github.com/knowin-kyeong/furiosa-opt-gemma4-12B.git
say() { echo "$(date -u +%FT%T) $*" >> "$LOG"; }
say "chain start"

# 1. V382 analysis
while pgrep -f "vchain.sh lab5 V382" > /dev/null || pgrep -f "rerun_n.sh [0-9]+ V382 " > /dev/null; do sleep 60; done
python3 /root/tk/gapcold.py V382 > /root/tk/V382_analysis.txt 2>&1
say "V382 done: $(grep -c . /root/tk/V382_analysis.txt) analysis lines ($(tail -1 /root/tk/V382_chain.out))"

# 2. V383 experiment
cd /root/lab6 || exit 1
git fetch -q "$GH" +tmp_V383:tmp_V383 || { say "V383 fetch failed"; exit 1; }
NRERUN=31 bash /root/tk/vchain.sh lab6 V383_qkv_rope_one_store V383 sliding_project_qkv 1s > /root/tk/V383_chain.out 2>&1
say "V383 chain: $(tail -1 /root/tk/V383_chain.out)"
grep -q V383_DONE /root/tk/V383_chain.out || { say "V383 chain did not finish (see V383_chain.out); draws stay on V378_submit"; exit 0; }

switched=0
decide() {  # $1 = label, $2 $3 = rerun index range
  python3 /root/tk/v383_verdict.py "$2" "$3" > "/root/tk/V383_verdict_$1.txt" 2>&1
  python3 /root/tk/paired_tails.py V383 sliding_project_qkv prod 1s >> "/root/tk/V383_verdict_$1.txt" 2>&1
  python3 /root/tk/pfirst.py sliding_project_qkv V383 >> "/root/tk/V383_verdict_$1.txt" 2>&1
  local v; v=$(awk '/^VERDICT/{print $2}' "/root/tk/V383_verdict_$1.txt")
  say "V383 verdict ($1): ${v:-none} | $(grep -E '^warm|^cold diff' "/root/tk/V383_verdict_$1.txt" | tr '\n' ' ')"
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
decide 32 0 31

# 3. more process-first samples
JOB=$(grep -m1 -oE "^JOB=[0-9]+" /root/tk/V383_chain.out | cut -d= -f2)
if [ -n "$JOB" ]; then
  bash /root/tk/rerun_n.sh "$JOB" V383 32 sliding_project_qkv 32 > /dev/null 2>&1
  say "V383 extra reruns done ($(ls /root/tk/V383_r*.log | wc -l) logs)"
  decide 64 0 63
fi
say "chain done"
