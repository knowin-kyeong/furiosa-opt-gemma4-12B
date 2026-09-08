#!/bin/bash
# Dump schedules for the three Stage-1 kernels. Usage: dump_schedules.sh <tag>
set -uo pipefail
. /root/env.sh 2>/dev/null || true
TAG="${1:-V0}"
cd "$(dirname "$0")/../.."
mkdir -p target/schedules
for k in sliding_project_qkv sliding_attention_output decoder_feedforward; do
  echo "########## $k ##########"
  SECONDS=0
  cargo furiosa-opt compile "ops::$k" --exact --dump-schedule "target/schedules/${TAG}_${k}.json" 2>&1 | tail -15
  echo "compile wall ${SECONDS}s"
  ls -la "target/schedules/${TAG}_${k}.json" 2>/dev/null
done
echo "########## MAKESPAN ##########"
python3 "$(dirname "$0")/makespan.py" "$TAG"
echo "########## DUMP DONE ##########"
