#!/bin/bash
# subverify.sh LAB BRANCH TAG -- submission-branch check on /root/LAB: check out tmp_TAG as BRANCH (stock
# tests/test_kernels.rs, all kernels), build the release test binary, one Arena job and one `rngd rerun` of it
# (retrying while the 2-active-job quota is full). A submission branch is adopted only with every test PASS in both.
set -u
. /root/env.sh
LAB=$1; BR=$2; TAG=$3
cd "/root/$LAB" || exit 1
git reset -q --hard
git checkout -q -B "$BR" "tmp_$TAG" || { echo CHECKOUT_FAILED; exit 1; }
git log --oneline -1
cargo furiosa-opt test --release --test test_kernels --no-run > "/root/tk/${TAG}_build.log" 2>&1
B=$?; echo "BUILD_EXIT=$B"; grep -E "Finished|^error" "/root/tk/${TAG}_build.log" | tail -3
[ "$B" -eq 0 ] || exit 1
REPO="/root/$LAB" bash /root/tk/arena_retry.sh /root/auto/arena.sh "${TAG}_verify1" > "/root/tk/${TAG}_verify1.log" 2>&1
JOB=$(grep -m1 -oE "submitted job [0-9]+" "/root/tk/${TAG}_verify1.log" | grep -oE "[0-9]+$")
echo "JOB=$JOB verify1 pass=$(grep -c -- "-> PASS" "/root/tk/${TAG}_verify1.log") fail=$(grep -c FAIL "/root/tk/${TAG}_verify1.log")"
[ -n "$JOB" ] || { tail -20 "/root/tk/${TAG}_verify1.log"; exit 1; }
ARENA_WAIT=1500 bash /root/tk/arena_retry.sh /root/auto/arena_rerun.sh "$JOB" "${TAG}_verify2" > "/root/tk/${TAG}_verify2.log" 2>&1
echo "verify2 pass=$(grep -c -- "-> PASS" "/root/tk/${TAG}_verify2.log") fail=$(grep -c FAIL "/root/tk/${TAG}_verify2.log")"
grep -E "^ops::|cycles=" "/root/tk/${TAG}_verify1.log" | head -8
echo VERIFY_DONE
