#!/bin/bash
# drawchain_follow.sh <branch> <first_batch> <n_batches> -- wait until the running draw chain/loop exits, then keep
# drawing <branch> in 12-draw batches (240 s gap) from /root/draw_src. Each finished log is moved to batch<b>.log
# before the next loop starts, so earlier batch logs are never overwritten. Do not touch /root/draw_src meanwhile.
BR=${1:?branch}; B0=${2:?first batch index}; N=${3:-6}
L=/root/drawloop_$BR.log
while ps -eo args | grep -qE "^bash /root/(drawchain_v313submit|drawloop2)\.sh"; do sleep 30; done
for b in $(seq "$B0" $((B0 + N - 1))); do
  [ -f "$L" ] && mv "$L" "/root/drawloop_$BR.batch$b.log"
  DRAW_SRC=/root/draw_src bash /root/drawloop2.sh "$BR" 12 240
done
echo "follow done $(date -u +%T)"
