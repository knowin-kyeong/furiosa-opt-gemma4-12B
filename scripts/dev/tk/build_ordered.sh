#!/bin/bash
# build_ordered.sh <lab> <branch> <tmp_ref> <order_file> <ref_lab> -- build the harness test binary in /root/<lab> with
# SCHEDULER_MANUAL_ORDERING_PATH=<order_file>. The kernel cache is not keyed by the ordering file, so the ffn kernel's
# cached artifacts are deleted first; the rebuilt .bin is compared with the same file in /root/<ref_lab> (built without
# the order) to show the order took effect.
set -u
. /root/env.sh
LAB=$1; BR=$2; REF=$3; ORD=$4; REFLAB=$5
K=decoder_feedforward
cd "/root/$LAB" || exit 1
git reset -q --hard
git fetch -q https://github.com/knowin-kyeong/furiosa-opt-gemma4-12B.git "+$REF:$REF" || exit 1
git checkout -q -B "$BR" "$REF" || exit 1
git log --oneline -1
find target/furiosa-opt -name "*ops::${K}.*" -delete 2>/dev/null
SCHEDULER_MANUAL_ORDERING_PATH="$ORD" cargo furiosa-opt test --release --test test_kernels --no-run > "/root/tk/${BR}_build.log" 2>&1
echo "BUILD_EXIT=$?"
tail -n 2 "/root/tk/${BR}_build.log"
b=$(find target/furiosa-opt -name "*ops::${K}.bin" | head -1)
r=$(find "/root/$REFLAB/target/furiosa-opt" -name "*ops::${K}.bin" | head -1)
echo "ordered $(md5sum "$b" | cut -c1-12)  reference $(md5sum "$r" | cut -c1-12)"
