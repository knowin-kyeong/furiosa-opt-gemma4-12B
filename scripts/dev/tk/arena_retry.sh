#!/bin/bash
# arena_retry.sh <script> <args...> -- run /root/auto/arena.sh or /root/auto/arena_rerun.sh, retrying every 60 s while
# Arena refuses the job for the per-account quota ("quota exceeded: you already have 2 active jobs"). Prints the
# final attempt's output and returns its exit code.
set -u
out=""
rc=1
for i in $(seq 1 180); do
  out=$(bash "$@" 2>&1)
  rc=$?
  if printf "%s\n" "$out" | grep -q "quota exceeded"; then
    sleep 60
    continue
  fi
  printf "%s\n" "$out"
  exit $rc
done
printf "%s\n" "$out"
exit 1
