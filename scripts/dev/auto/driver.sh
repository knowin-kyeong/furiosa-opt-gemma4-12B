#!/bin/bash
# Unattended experiment chain for the RunPod box (RULES.md §11).
#
# Reads auto/queue.txt from the `auto_results` branch (one entry per line: `BRANCH` or
# `BRANCH@COMMIT`), and for every entry that has no final result yet:
#   1. checks out the ref (detached) in /root/furiosa-opt-gemma4-12B,
#   2. dumps the three Stage-1 schedules (makespan),
#   3. builds the full test binary (the crate-wide compile gate),
#   4. submits to Arena with the driver's own arena.sh when the pod is logged in
#      (not the branch's scripts/rngd_test.sh: its --timeout default exceeds the server maximum)
#      (otherwise the entry stays `pending` and is retried every poll),
#   5. writes auto/results/<id>.json + regenerates auto/BOARD.md, commits and pushes
#      the `auto_results` branch (deploy key, see RULES.md §11).
# One failure never stops the loop. The loop ends at the deadline in /root/auto/deadline.
set -u
. /root/env.sh 2>/dev/null || true

REPO=/root/furiosa-opt-gemma4-12B
STATE=/root/auto
WT=$STATE/wt
RB=auto_results
LOG=$STATE/driver.log
POLL_SECONDS=${AUTO_POLL_SECONDS:-600}
V0="116583 194020 1693200"   # V0_baseline makespans (qkv attn_out ffn), for the board's geomean

mkdir -p "$STATE/logs"
exec 9>"$STATE/driver.lock"
flock -n 9 || { echo "driver already running"; exit 0; }

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

deadline() { cat "$STATE/deadline" 2>/dev/null || echo 0; }
past_deadline() { [ "$(date +%s)" -ge "$(deadline)" ]; }

arena_ok() { timeout 60 furiosa-arena list >/dev/null 2>&1; }
# Arena submission is switched off by `touch /root/auto/arena_off`. Entries then keep
# `arena=pending`, which is exactly the state the driver retries once submission is switched back
# on (rm the file): a screening-only pass costs no Arena time and loses no work.
arena_enabled() { [ ! -f "$STATE/arena_off" ]; }

ensure_worktree() {
    cd "$REPO" || return 1
    git fetch -q origin "+refs/heads/*:refs/remotes/origin/*" 2>>"$LOG" || log "fetch failed"
    if ! git -C "$WT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        rm -rf "$WT"
        git worktree prune 2>/dev/null
        if git show-ref --verify --quiet "refs/heads/$RB"; then
            git worktree add -f "$WT" "$RB" >>"$LOG" 2>&1
        else
            git worktree add -f --track -b "$RB" "$WT" "origin/$RB" >>"$LOG" 2>&1
        fi
    fi
    [ -d "$WT/auto" ] || { log "worktree has no auto/ dir"; return 1; }
    # The worktree is never merged with origin (a rebase can stall on identity or conflicts and
    # would silently stop the chain): results accumulate on the local branch, and the queue is
    # always read straight from origin (see read_queue).
    git -C "$WT" config user.name auto-driver
    git -C "$WT" config user.email auto-driver@runpod
    git -C "$WT" rebase --abort >/dev/null 2>&1 || true
    mkdir -p "$WT/auto/results" "$WT/auto/logs"
}

# The queue as last pushed to origin (people append lines there); falls back to the local copy.
read_queue() {
    git -C "$REPO" show "origin/$RB:auto/queue.txt" > "$STATE/queue.txt.new" 2>/dev/null && mv "$STATE/queue.txt.new" "$STATE/queue.txt"
    [ -f "$STATE/queue.txt" ] || cp "$WT/auto/queue.txt" "$STATE/queue.txt" 2>/dev/null
    grep -v '^[[:space:]]*#' "$STATE/queue.txt" 2>/dev/null | sed 's/[[:space:]]*$//' | grep -v '^$'
}

publish() {
    git -C "$WT" add -A auto >>"$LOG" 2>&1
    git -C "$WT" commit -q -m "auto: $1" >>"$LOG" 2>&1 || true
    git -C "$WT" push -q origin "$RB" >>"$LOG" 2>&1 || log "push failed for $1 (kept in $WT; needs a deploy key, RULES §11)"
}

# id = entry with characters outside [A-Za-z0-9_] replaced (Arena job names, file names)
entry_id() { printf '%s' "$1" | sed 's/@/_at_/g; s/[^A-Za-z0-9_]/_/g'; }

status_of() {  # prints none|pending|done
    local f="$WT/auto/results/$1.json"
    [ -f "$f" ] || { echo none; return; }
    python3 - "$f" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("none"); sys.exit()
print("pending" if d.get("arena") == "pending" else "done")
EOF
}

run_entry() {
    local entry=$1 id=$2 branch ref dump_rc build_rc arena rc
    branch=${entry%%@*}
    if [ "$branch" = "$entry" ]; then ref="origin/$branch"; else ref=${entry#*@}; fi
    log "=== $id ($entry) start"
    cd "$REPO" || return
    git fetch -q origin "+refs/heads/*:refs/remotes/origin/*" 2>>"$LOG"
    if ! git checkout -q -f --detach "$ref" 2>>"$LOG"; then
        log "checkout of $ref failed"
        python3 "$STATE/record.py" "$WT" "$id" "$entry" "checkout-failed" "n/a" "n/a" "$STATE/logs" $V0 >>"$LOG" 2>&1
        publish "$id (checkout failed)"
        return
    fi
    git clean -fdq 2>>"$LOG"
    chmod +x scripts/dev/*.sh scripts/*.sh 2>/dev/null
    local commit; commit=$(git rev-parse --short HEAD)
    echo "$commit" > "$STATE/logs/$id.commit"

    # 1. schedule dump (makespan)
    # (own copy of scripts/dev/dump_schedules.sh: old branches such as V0_baseline lack it)
    timeout 2400 bash "$STATE/dump.sh" "$id" > "$STATE/logs/$id.dump.log" 2>&1; dump_rc=$?
    python3 "$STATE/summarize.py" "$id" > "$STATE/logs/$id.makespan.json" 2>>"$STATE/logs/$id.dump.log"
    log "$id dump rc=$dump_rc: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:(v or {}).get('makespan') for k,v in d.items()})" "$STATE/logs/$id.makespan.json" 2>/dev/null)"

    # 2. full-crate build of the test binary (the submission artifact)
    timeout 3600 cargo furiosa-opt test --release --test test_kernels --no-run > "$STATE/logs/$id.build.log" 2>&1; build_rc=$?
    log "$id build rc=$build_rc"

    # 3. host CPU emulator accuracy run (advisory; see auto/BOARD.md header)
    cpu=skip
    if [ "$build_rc" -eq 0 ] && [ "${AUTO_CPU_TEST:-1}" = "1" ]; then
        timeout 3600 bash "$STATE/cpu_test.sh" > "$STATE/logs/$id.cpu.log" 2>&1
        crc=$?
        if grep -q "all 3 tests passed\|3 passed" "$STATE/logs/$id.cpu.log" 2>/dev/null; then cpu=pass
        elif grep -q "panicked at" "$STATE/logs/$id.cpu.log" 2>/dev/null; then cpu=panic
        elif [ "$crc" -eq 124 ]; then cpu=timeout
        else cpu=fail; fi
        # the qkv verdicts alone, which the emulator does evaluate faithfully on the baseline
        qkvv=$(grep -o "sliding_project_qkv [qkv] .*-> [A-Z]*" "$STATE/logs/$id.cpu.log" 2>/dev/null | grep -o "[A-Z]*$" | tr "
" "/" )
        [ -n "$qkvv" ] && cpu="$cpu(qkv ${qkvv%/})"
        log "$id cpu=$cpu"
    fi

    # 4. Arena (only when logged in and the binary exists)
    arena=pending
    if [ "$build_rc" -ne 0 ]; then
        arena="n/a"
    elif ! arena_enabled; then
        log "$id arena skipped (arena_off); stays pending"
    elif arena_ok; then
        timeout 3600 bash "$STATE/arena.sh" "$id" > "$STATE/logs/$id.arena.log" 2>&1; rc=$?
        if [ "$rc" -eq 0 ]; then arena=pass; else arena=fail; fi
        log "$id arena rc=$rc -> $arena"
    else
        log "$id arena pending (not logged in)"
    fi

    python3 "$STATE/record.py" "$WT" "$id" "$entry" "$dump_rc" "$build_rc" "$arena" "$STATE/logs" $V0 "$cpu" >>"$LOG" 2>&1
    publish "$id dump=$dump_rc build=$build_rc cpu=$cpu arena=$arena"
    log "=== $id end"
}

log "driver start (deadline $(date -u -d @"$(deadline)" +%FT%TZ 2>/dev/null))"
while ! past_deadline; do
    if ! ensure_worktree; then log "worktree unavailable, retrying in $POLL_SECONDS s"; sleep "$POLL_SECONDS"; continue; fi
    mapfile -t QUEUE < <(read_queue)
    did=0
    for entry in "${QUEUE[@]}"; do
        id=$(entry_id "$entry")
        st=$(status_of "$id")
        case "$st" in
            done) continue ;;
            pending)
                arena_enabled || continue
                arena_ok || continue ;;
        esac
        run_entry "$entry" "$id"
        did=1
        past_deadline && break
        # re-read the queue between entries so appended lines are seen promptly
        break
    done
    if [ "$did" -eq 0 ]; then
        log "idle: queue done ($(printf '%s\n' "${QUEUE[@]}" | wc -l) entries); arena $(arena_ok && echo logged-in || echo not-logged-in); sleeping $POLL_SECONDS s"
        sleep "$POLL_SECONDS"
    fi
done
log "deadline reached, driver exit"
