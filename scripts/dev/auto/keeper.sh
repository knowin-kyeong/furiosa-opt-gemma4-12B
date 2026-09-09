#!/bin/bash
# Restarts driver.sh if it dies, until the deadline in /root/auto/deadline passes or
# /root/auto/STOP exists. Launch detached:
#   cd /root/auto && setsid nohup bash keeper.sh > keeper.out 2>&1 < /dev/null &
set -u
STATE=/root/auto
cd "$STATE" || exit 1
while [ ! -f "$STATE/STOP" ] && [ "$(date +%s)" -lt "$(cat "$STATE/deadline" 2>/dev/null || echo 0)" ]; do
    bash "$STATE/driver.sh" >> "$STATE/driver.out" 2>&1
    sleep 60
done
echo "$(date -u +%FT%TZ) keeper exit" >> "$STATE/driver.log"
