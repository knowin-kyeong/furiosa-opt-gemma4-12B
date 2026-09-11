"""schedcmp.py <base.json> <variant.json> [tail_n] -- static schedule comparison for one kernel.

Prints makespan, DMA busy/idle, every DMA/Core instruction of both schedules side by side (by source line),
and the last tail_n instructions of the variant.
"""
import json
import re
import sys


def load(path):
    d = json.load(open(path))
    return d["instructions"]


def src(i):
    m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
    return m.group(1).replace("src/device/", "") if m else ("?" + str(i.get("index")))


def summary(ins):
    mk = max(i["lifetime"]["end"] for i in ins)
    dma = [i for i in ins if "DmaEngine" in (i.get("contexts") or [])]
    busy = sum(i["lifetime"]["end"] - i["lifetime"]["begin"] for i in dma)
    return mk, busy, len(dma)


def row(i):
    life = i["lifetime"]
    return "%9s %7d-%-7d dur=%5d %s" % (i.get("tpe"), life["begin"], life["end"], life["end"] - life["begin"], src(i))


base, var = load(sys.argv[1]), load(sys.argv[2])
tail_n = int(sys.argv[3]) if len(sys.argv) > 3 else 14
mb, bb, nb = summary(base)
mv, bv, nv = summary(var)
print("makespan base=%d variant=%d delta=%+d | DMA busy %d -> %d | DMA commands %d -> %d" % (mb, mv, mv - mb, bb, bv, nb, nv))
for name, ins in (("base", base), ("variant", var)):
    print("--- %s: DMA and Core instructions" % name)
    for i in sorted(ins, key=lambda i: i["lifetime"]["begin"]):
        if str(i.get("tpe", "")).startswith("Dma") or i.get("tpe") == "Core" and i["lifetime"]["end"] > i["lifetime"]["begin"]:
            print("   " + row(i))
print("--- variant tail")
for i in sorted(var, key=lambda i: i["lifetime"]["end"])[-tail_n:]:
    print("   " + row(i))
