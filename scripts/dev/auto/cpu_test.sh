#!/bin/bash
# cpu_test.sh — run the three kernel tests on the host CPU emulator and report the verdict.
#
# `cargo furiosa-opt test` compiles with `--cfg backend="npu"` and needs a real chip; plain cargo
# builds the same crate against the CPU buffer emulator ("For the host CPU, use cargo itself"),
# which checks numerics with no Arena round trip. That is the gate the 2026-09-09 lineage lacked:
# V15..V41 passed every makespan screen while producing NaN on hardware.
#
# The CPU build uses its own target dir so the two cfgs do not evict each other's artifacts.
set -u
. /root/env.sh 2>/dev/null || true

REPO=${REPO:-/root/furiosa-opt-gemma4-12B}
TIMEOUT=${CPU_TEST_TIMEOUT:-5400}
cd "$REPO" || exit 2

[ -f ref/fixtures.safetensors ] || { echo "cpu_test.sh: ref/fixtures.safetensors missing" >&2; exit 2; }

CARGO_TARGET_DIR="${CPU_TARGET_DIR:-$REPO/target_cpu}" \
    timeout "$TIMEOUT" cargo test --release --test test_kernels -- --nocapture
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "cpu_test.sh: timed out after ${TIMEOUT}s" >&2
fi
exit "$rc"
