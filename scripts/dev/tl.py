"""tl.py <schedule.json> [min_dur] -- compact timeline: every instruction sorted by begin, with contexts and source."""
import json
import re
import sys

ins = json.load(open(sys.argv[1]))["instructions"]
min_dur = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def src(i):
    m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
    return m.group(1).replace("src/device/", "") if m else "?"


mk = max(i["lifetime"]["end"] for i in ins)
print("makespan", mk, "instructions", len(ins))
for i in sorted(ins, key=lambda i: (i["lifetime"]["begin"], i["lifetime"]["end"])):
    b, e = i["lifetime"]["begin"], i["lifetime"]["end"]
    if e - b < min_dur:
        continue
    ctx = ",".join(c.replace("Context", "").replace("Engine", "") for c in (i.get("contexts") or []))
    print("%-14s %6d-%-6d %5d  %-22s %s" % (i.get("tpe"), b, e, e - b, ctx[:22], src(i)))
