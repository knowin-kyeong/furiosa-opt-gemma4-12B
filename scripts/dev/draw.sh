#!/bin/bash
# draw.sh <branch> <N> [srcdir] -- pod: check <branch> out in <srcdir> (default /root/lab3, the submission clone)
# and make N serialised leaderboard submissions from it. Verify the branch on Arena first. There is no stated
# submission cap (2026-09-11), but a draw cannot compare code (RULES 10.0l); it only harvests the distribution.
# Check nothing else out in <srcdir> meanwhile: "moa-submitter submit --source" uploads the working tree.
. /root/env.sh
BR=${1:?usage: draw.sh branch N [srcdir]}; N=${2:-5}; SRC=${3:-/root/lab3}
cd "$SRC" || exit 1
git fetch -q origin && git checkout -q -B "$BR" "origin/$BR" || exit 2
echo "HEAD=$(git log --oneline -1)"
busy() { timeout 60 /root/.cargo/bin/moa-submitter status 2>/dev/null | tail -3 | grep -qE "building|queued|running|pending|evaluating"; }
for i in $(seq 1 "$N"); do
  for w in $(seq 1 160); do busy || break; sleep 15; done
  echo "=== draw $i $(date -u +%T) ==="
  timeout 900 /root/.cargo/bin/moa-submitter submit --source "$SRC" 2>&1 | tail -3
done
for w in $(seq 1 160); do busy || break; sleep 15; done
echo "=== status ==="
timeout 60 /root/.cargo/bin/moa-submitter status 2>&1 | tail -10
