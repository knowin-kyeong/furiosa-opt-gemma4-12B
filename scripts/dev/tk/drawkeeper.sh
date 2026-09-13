#!/bin/bash
# drawkeeper.sh <deadline_epoch> -- keep leaderboard draws going until the deadline (user away, 2026-09-13).
# Starts one 12-draw batch (60 s gap, /root/draw_src) of the branch named in /root/tk/draw_target whenever no
# drawchain_follow.sh / drawloop2.sh is alive, so it never overlaps the running V378 follow loop and picks up a new
# target (written by chain_0913n.sh) after the running batch. Batch numbers per branch start at 101 (the follow loops
# used 1..8), scores go to /root/tk/<branch>_scores.txt after every batch. Log /root/tk/drawkeeper.log.
DL=${1:?deadline epoch}
while [ "$(date +%s)" -lt "$DL" ]; do
  if ! ps -eo args | grep -qE "^bash /root/(drawchain_follow|drawloop2)\.sh"; then
    BR=$(cat /root/tk/draw_target 2>/dev/null || echo V378_submit)
    F=/root/tk/draw_batch_$BR
    B=$(( $(cat "$F" 2>/dev/null || echo 100) + 1 )); echo "$B" > "$F"
    L=/root/drawloop_$BR.log
    [ -f "$L" ] && mv "$L" "/root/drawloop_$BR.batch$B.log"
    echo "$(date -u +%FT%T) batch $B of $BR" >> /root/tk/drawkeeper.log
    DRAW_SRC=/root/draw_src bash /root/drawloop2.sh "$BR" 12 60
    bash /root/tk/drawscores.sh "$BR" > /dev/null 2>&1
    echo "$(date -u +%FT%T) batch $B of $BR done, $(wc -l < "/root/tk/${BR}_scores.txt" 2>/dev/null) scores" >> /root/tk/drawkeeper.log
  fi
  sleep 60
done
echo "$(date -u +%FT%T) deadline reached" >> /root/tk/drawkeeper.log
