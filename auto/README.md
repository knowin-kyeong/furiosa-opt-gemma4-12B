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
`summarize.py` (schedule JSON → makespan JSON), `record.py` (result JSON + BOARD.md).

Results are committed on the pod's local `auto_results` worktree (`/root/auto/wt`). Pushing
needs a deploy key on the pod (RULES §11.2); without one, fetch them with
`scp -r -P 41008 root@<pod>:/root/auto/wt/auto ./auto` and commit from a machine that can push.
