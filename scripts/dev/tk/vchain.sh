#!/bin/bash
# vchain.sh LAB BRANCH TAG KERNEL ARM... -- pod: check out tmp_TAG (pushed from the laptop) in /root/LAB as BRANCH,
# compile + static dump of production and every arm kernel, build the release test binary, first Arena job with an
# accuracy gate, then NRERUN (default 15) reruns. DUMP_ONLY=1 stops after the dumps.
set -u
. /root/env.sh
LAB=$1; BR=$2; TAG=$3; K=$4; shift 4; ARMS=("$@")
cd "/root/$LAB" || exit 1
git reset -q --hard
git checkout -q -B "$BR" "tmp_$TAG" || { echo CHECKOUT_FAILED; exit 1; }
git log --oneline -1
for a in "" "${ARMS[@]}"; do
  fn=$K${a:+_$a}; t=${a:-prod}
  cargo furiosa-opt compile "ops::$fn" --exact --dump-schedule "/root/tk/S${TAG}_$t.json" > "/root/tk/${TAG}_dump_$t.log" 2>&1
  rc=$?
  echo "dump $t exit=$rc $(python3 /root/tk/schedorder.py "/root/tk/S${TAG}_$t.json" 0 2>/dev/null | sed -n 2p)"
  [ $rc -ne 0 ] && grep -E "caused by|^error" "/root/tk/${TAG}_dump_$t.log" | head -6
done
[ "${DUMP_ONLY:-0}" = 1 ] && { echo DUMPS_DONE; exit 0; }
cargo furiosa-opt test --release --test test_kernels --no-run > "/root/tk/${TAG}_build.log" 2>&1
B=$?; echo "BUILD_EXIT=$B"
[ "$B" -eq 0 ] || { grep -E "caused by|^error" -A 6 "/root/tk/${TAG}_build.log" | head -60; exit 1; }
REPO="/root/$LAB" bash /root/tk/arena_retry.sh /root/auto/arena.sh "${TAG}_r0" > "/root/tk/${TAG}_r0.log" 2>&1
JOB=$(grep -m1 -oE "submitted job [0-9]+" "/root/tk/${TAG}_r0.log" | grep -oE "[0-9]+$")
echo "JOB=$JOB pass=$(grep -c -- "-> PASS" "/root/tk/${TAG}_r0.log") fail=$(grep -c FAIL "/root/tk/${TAG}_r0.log")"
[ -n "$JOB" ] || { tail -30 "/root/tk/${TAG}_r0.log"; exit 1; }
grep -E "sweep rotation|-> (PASS|FAIL)|FAIL --" "/root/tk/${TAG}_r0.log"
grep -q FAIL "/root/tk/${TAG}_r0.log" && { echo ACCURACY_FAIL; exit 1; }
bash /root/tk/rerun_n.sh "$JOB" "$TAG" "${NRERUN:-15}" "$K" > /dev/null 2>&1
echo "${TAG}_DONE"
