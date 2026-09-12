"""schedorder.py <schedule.json> [max_rows] (stdin to the pod's python3) -- static instructions in begin order: index, begin,
end, duration, resource/context, source line, and the first words of the description (to see where each DMA command sits)."""
import json
import re
import sys

d = json.load(open(sys.argv[1]))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ins = d["instructions"]
print("keys of first instruction:", sorted(ins[0].keys()))


def src(i):
    m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
    return m.group(1).replace("src/device/", "") if m else "-"


def kind(i):
    for key in ("context", "resource", "kind", "type", "engine"):
        if key in i and i[key] not in (None, ""):
            v = i[key]
            return v if isinstance(v, str) else json.dumps(v)[:40]
    return (i.get("description") or "").split("\n")[0][:40]


rows = sorted(ins, key=lambda i: (i["lifetime"]["begin"], i["lifetime"]["end"]))
print("makespan", max(i["lifetime"]["end"] for i in ins), "instructions", len(ins))
for i in rows[:limit]:
    lt = i["lifetime"]
    first = (i.get("description") or "").replace("\n", " ")[:70]
    print("%4s %7d %7d %6d  %-28s %-34s %s" % (i.get("index"), lt["begin"], lt["end"], lt["end"] - lt["begin"], str(kind(i))[:28], src(i), first))
