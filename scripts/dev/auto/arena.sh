#!/bin/bash
# arena.sh ID — submit the checked-out build to Arena as one job and wait for its verdict.
#
# Driver-owned copy of the submit half of scripts/rngd_test.sh (like dump.sh), for two reasons
# found on 2026-09-09 by the first real submission:
#   * the controller rejects `--timeout` above its maximum (70 s), and every branch in the queue
#     carries an upstream rngd_test.sh whose default is 1800 — every entry would be rejected;
#   * rngd_test.sh runs `submit_output=$(rngd submit ...)` under `set -e`, so a rejected submit
#     kills the script *before* the echo and the error never reaches the log (the failure looks
#     like an empty log).
# The device-side execution limit and the wall-clock wait for a queued job are separate budgets
# here: ARENA_TIMEOUT is what the controller enforces on the run, ARENA_WAIT is how long we are
# willing to sit in its queue.
set -u
. /root/env.sh 2>/dev/null || true

ID=${1:?usage: arena.sh ID}
REPO=${REPO:-/root/furiosa-opt-gemma4-12B}
JOB_TIMEOUT=${ARENA_TIMEOUT:-70}
WAIT=${ARENA_WAIT:-1800}
POLL=${ARENA_POLL:-5}

cd "$REPO" || exit 1

BINARY=$(find target/release/deps -maxdepth 1 -type f -name 'test_kernels-*' ! -name '*.d' -perm -u+x 2>/dev/null | xargs -r ls -t | head -1)
FIXTURE=ref/fixtures.safetensors
ENTRY=scripts/rngd/remote_entrypoint.sh
for f in "$BINARY" "$FIXTURE" "$ENTRY"; do
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "arena.sh: missing submission artifact (binary='$BINARY' fixture='$FIXTURE' entrypoint='$ENTRY')" >&2
        exit 2
    fi
done

staging=$(mktemp -d) || exit 2
trap 'rm -rf "$staging"' EXIT
cp "$ENTRY" "$staging/remote_entrypoint.sh"
cp "$BINARY" "$staging/test_runtime"
cp "$FIXTURE" "$staging/fixtures.safetensors"
chmod +x "$staging/remote_entrypoint.sh" "$staging/test_runtime"

echo "==> submitting $ID (binary $BINARY, timeout ${JOB_TIMEOUT}s)"
submit_output=$(rngd submit \
    "$staging/remote_entrypoint.sh" \
    "$staging/test_runtime" \
    "$staging/fixtures.safetensors" \
    --name "$ID" \
    --entrypoint remote_entrypoint.sh \
    --timeout "$JOB_TIMEOUT" 2>&1)
submit_rc=$?
echo "$submit_output"
if [ "$submit_rc" -ne 0 ]; then
    echo "arena.sh: submit failed (exit $submit_rc)" >&2
    exit 1
fi

job=$(printf '%s\n' "$submit_output" | sed -n 's/.*submitted job \([0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$job" ]; then
    echo "arena.sh: no job id in the submit output above" >&2
    exit 1
fi

json_field() { sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" <<<"$1" | head -1; }
is_terminal() {
    case "$1" in
        succeeded|failed|completed|cancelled|canceled) return 0 ;;
        *) return 1 ;;
    esac
}

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
    echo "arena.sh: job $job still '${state:-unknown}' after ${WAIT}s; cancel with: rngd cancel $job" >&2
    exit 1
fi

code=$(json_field "$status_output" exit_code)
echo "==> job $job $state (exit ${code:-?}); log follows"
timeout 120 rngd logs "$job" || true

[ "${code:-1}" = "0" ]
