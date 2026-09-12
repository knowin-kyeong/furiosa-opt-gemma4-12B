#!/bin/bash
# drawswitch_v367.sh -- wait until the running drawloop2 batch (V360_submit) exits, archive its log, then keep drawing
# V367_submit in 12-draw batches (drawchain_follow.sh, /root/draw_src). Launch only after the V360 follow loop has been
# stopped by PID (so it cannot start another V360 batch) and V367_submit has been fetched into /root/draw_src.
while ps -eo args | grep -qE "^bash /root/drawloop2\.sh"; do sleep 30; done
[ -f /root/drawloop_V360_submit.log ] && mv /root/drawloop_V360_submit.log /root/drawloop_V360_submit.batch_last.log
exec bash /root/drawchain_follow.sh V367_submit 1 8
