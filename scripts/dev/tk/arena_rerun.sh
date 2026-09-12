#!/bin/bash
# arena_rerun.sh <job_id> <name> -- queue an already-uploaded Arena submission again (`rngd rerun`, no re-upload) and
# wait for it like arena.sh: prints "==> job <new id> <state> (exit N); log follows" and the job log. User rule
# (2026-09-12): Arena storage is full, so repeats of one binary are reruns of its first job, not new submits.
set -u
. /root/env.sh 2>/dev/null || true
SRC=${1:?usage: arena_rerun.sh JOB_ID NAME}
NAME=${2:-rerun}
WAIT=${ARENA_WAIT:-1800}
POLL=${ARENA_POLL:-5}

echo "==> rerun of job $SRC ($NAME)"
out=$(timeout 120 rngd rerun "$SRC" 2>&1)
rc=$?
echo "$out"
[ "$rc" -eq 0 ] || { echo "arena_rerun.sh: rerun failed (exit $rc)" >&2; exit 1; }
job=$(printf '%s\n' "$out" | grep -oE '(job|id)[^0-9]{0,6}[0-9]+' | grep -oE '[0-9]+' | grep -v "^$SRC\$" | tail -1)
[ -n "$job" ] || job=$(printf '%s\n' "$out" | grep -oE '\b[0-9]{4,}\b' | grep -v "^$SRC\$" | tail -1)
[ -n "$job" ] || { echo "arena_rerun.sh: no new job id in the output above" >&2; exit 1; }

json_field() { sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" <<<"$1" | head -1; }
is_terminal() { case "$1" in succeeded|failed|completed|cancelled|canceled) return 0 ;; *) return 1 ;; esac; }

echo "==> waiting on job $job (poll ${POLL}s, giving up after ${WAIT}s in the queue)"
deadline=$(( SECONDS + WAIT ))
state=""
status_output=""
while [ "$SECONDS" -lt "$deadline" ]; do
    status_output=$(timeout 60 rngd status "$job" 2>&1 || true)
    state=$(json_field "$status_output" status | tr '[:upper:]' '[:lower:]')
    is_terminal "$state" && break
    sleep "$POLL"
done
if ! is_terminal "$state"; then
    echo "arena_rerun.sh: job $job still '${state:-unknown}' after ${WAIT}s; cancel with: rngd cancel $job" >&2
    exit 1
fi
code=$(json_field "$status_output" exit_code)
echo "==> job $job $state (exit ${code:-?}); log follows"
timeout 120 rngd logs "$job" || true
[ "${code:-1}" = "0" ]
