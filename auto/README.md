# auto/ — unattended experiment chain (branch `auto_results`)

This branch carries no kernel code of its own: it is the message board between the RunPod
driver and the humans/sessions that read it. See `RULES.md` §11 for the protocol.

| file | who writes it | what |
|---|---|---|
| `auto/queue.txt` | people / sessions | work list, one `BRANCH` or `BRANCH@COMMIT` per line |
| `auto/BOARD.md` | driver | one row per processed entry (makespan × 3, geomean vs V0, build/Arena status) |
| `auto/results/<id>.json` | driver | machine-readable record of one entry |
| `auto/logs/<id>.*` | driver | tail of the dump / build / Arena logs |

The driver lives in `scripts/dev/auto/` (copied to `/root/auto/` on the pod):
`keeper.sh` (restarts the driver until `/root/auto/deadline`), `driver.sh` (the loop),
`dump.sh` (schedule dump), `arena.sh` (one Arena job: stage → submit → wait → log),
`summarize.py` (schedule JSON → makespan JSON), `record.py` (result JSON + BOARD.md).

`dump.sh` and `arena.sh` are the driver's *own* copies of what the branch's `scripts/` would
otherwise provide, because the queue spans branches whose scripts differ: `V0_baseline` has no
`scripts/dev/` at all, and every branch's `scripts/rngd_test.sh` submits with a `--timeout`
above the controller's maximum (70 s), which is rejected — silently, since that script loses
the error to `set -e` before it echoes the submit output.

Results are committed on the pod's local `auto_results` worktree (`/root/auto/wt`). Pushing
needs a deploy key on the pod (RULES §11.2); without one, fetch them with
`scp -r -P 41008 root@<pod>:/root/auto/wt/auto ./auto` and commit from a machine that can push.
