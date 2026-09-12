#!/bin/bash
# drawloop.sh <branch> <N> <gap_seconds> -- pod /root/lab3: N serialised leaderboard draws of <branch>, spaced by
# <gap_seconds> so they sample different machine states (a cross-cluster ExplicitSync costs a random 0.4-15k cycles
# on hardware, so the score of identical code is a lottery). Do not touch /root/lab3 while this runs.
. /root/env.sh
BR=${1:?branch}; N=${2:-12}; GAP=${3:-480}
L=/root/drawloop_$BR.log
cd ${DRAW_SRC:-/root/lab3} || exit 1
git checkout -q "$BR" && [ -z "$(git status --short | grep -v '^??')" ] || { echo "source not clean on $BR" >> $L; exit 2; }
busy() { timeout 60 /root/.cargo/bin/moa-submitter status 2>/dev/null | tail -3 | grep -qE "building|queued|running|pending|evaluating"; }
echo "start $(date -u +%T) $BR $(git log --oneline -1 | cut -c1-12) N=$N gap=$GAP" >> $L
for i in $(seq 1 "$N"); do
  for w in $(seq 1 160); do busy || break; sleep 15; done
  echo "=== draw $i $(date -u +%T)" >> $L
  timeout 900 /root/.cargo/bin/moa-submitter submit --source ${DRAW_SRC:-/root/lab3} 2>&1 | grep -i "log \|error" >> $L
  sleep "$GAP"
done
for w in $(seq 1 160); do busy || break; sleep 15; done
timeout 60 /root/.cargo/bin/moa-submitter status 2>&1 | tail -$((N + 2)) >> $L
echo "done $(date -u +%T)" >> $L
