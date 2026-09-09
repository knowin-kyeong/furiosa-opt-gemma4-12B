"""summarize.py TAG -> JSON with makespan and per-context busy cycles of the three Stage-1 kernels.

Reads target/schedules/<TAG>_<kernel>.json (as written by scripts/dev/dump_schedules.sh);
a kernel whose dump is missing or unreadable is reported as null (compile failure).
"""
import collections
import json
import sys

tag = sys.argv[1]
kernels = ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
out = {}
for k in kernels:
    try:
        d = json.load(open(f"target/schedules/{tag}_{k}.json"))
        ins = d.get("instructions", [])
        mk = max((i["lifetime"]["end"] for i in ins), default=0)
        busy = collections.Counter()
        for i in ins:
            dur = i["lifetime"]["end"] - i["lifetime"]["begin"]
            for c in i.get("contexts") or ["?"]:
                busy[c] += dur
        out[k] = {"makespan": mk, "instructions": len(ins), "busy": dict(busy)}
    except Exception as e:  # noqa: BLE001
        out[k] = None
        print(f"{k}: {e}", file=sys.stderr)
print(json.dumps(out))
