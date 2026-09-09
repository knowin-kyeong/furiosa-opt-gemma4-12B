#!/bin/bash
# dump.sh TAG — compile the three Stage-1 kernels with --dump-schedule into
# target/schedules/TAG_<kernel>.json (driver-owned copy of scripts/dev/dump_schedules.sh, so
# that branches which predate scripts/dev/ can be measured too). Exit 1 if any dump is missing.
. /root/env.sh 2>/dev/null || true
TAG=${1:?tag}
cd /root/furiosa-opt-gemma4-12B || exit 1
mkdir -p target/schedules
rc=0
for k in sliding_project_qkv sliding_attention_output decoder_feedforward; do
    echo "########## $k ##########"
    SECONDS=0
    rm -f "target/schedules/${TAG}_${k}.json"
    cargo furiosa-opt compile "ops::$k" --exact --dump-schedule "target/schedules/${TAG}_${k}.json" 2>&1 | tail -60
    [ -f "target/schedules/${TAG}_${k}.json" ] || rc=1
    echo "compile wall ${SECONDS}s"
done
exit $rc
