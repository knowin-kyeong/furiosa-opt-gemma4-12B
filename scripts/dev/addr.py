"""addr.py <schedule.json> -- DM (Sram) tensors sorted by address: per-slice offset/size, tensor lifetime, and the
lifetimes of their writers/readers, to see which freed region a later tensor reuses (DramReuse)."""
import json
import re
import sys

d = json.load(open(sys.argv[1]))
ins = d["instructions"]
users = {}
for i in ins:
    for t in i.get("input_tensors") or []:
        users.setdefault(t, []).append(("r", i))
    for t in i.get("output_tensors") or []:
        users.setdefault(t, []).append(("w", i))


def src(i):
    m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
    return m.group(1).replace("src/device/", "") if m else str(i.get("index"))


rows = [t for t in d["tensors"] if t.get("buffer_type") == "Sram"]
for t in sorted(rows, key=lambda t: (t["address"], t["lifetime"]["begin"])):
    tid = int(t["index"])
    lt = t["lifetime"]
    us = users.get(tid, [])
    us = ["%s:%s@%d-%d" % (m, src(i), i["lifetime"]["begin"], i["lifetime"]["end"]) for m, i in us if i["lifetime"]["end"] > 0]
    print("addr=%-9d size=%-9d per_slice=%-6d life=%d-%d blk=%s-%s T%-3d %s" % (
        t["address"], t["size"], t["size"] // 512, lt["begin"], lt["end"], lt.get("block_begin"), lt.get("block_end"),
        tid, t["name"][:34]))
    for u in us[:6]:
        print("        " + u)
