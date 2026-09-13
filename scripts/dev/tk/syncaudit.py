"""syncaudit.py <schedule.json>... -- every ExplicitSync in a static schedule, with the HBM stores in front of it and the
loads that read those HBM tensors afterwards.

A cross-cluster ExplicitSync takes host-side wait on hardware (first-launch penalty, idle stalls: V381/V382), and V383
won -4% by removing one. Instruction `input_tensors` / `output_tensors` are ids into the top-level `tensors` list; views
(tiles, reshapes) are zero-length `Noop` instructions in between, so a store is linked to its readers by following
Noop edges forward from the stored tensor. For each sync: its static begin, the stores that end at or before it (most
recent first), each store's tensor name/buffer/size, and the loads that read the stored tensor (source line, static
begin, the sync-to-reader gap).
"""
import json
import re
import sys
from collections import defaultdict


def life(i):
    l = i.get("lifetime") or {}
    return l.get("begin", 0), l.get("end", 0)


def src(i):
    m = re.search(r"--> src/(?:device/)?([^\s:]+:\d+)", i.get("description") or "")
    return m.group(1) if m else "-"


for path in sys.argv[1:]:
    d = json.load(open(path))
    ins = d["instructions"]
    tens = {str(t.get("index")): t for t in d.get("tensors", [])}
    alias = defaultdict(set)
    for i in ins:
        if i.get("contexts") == ["Noop"]:
            for a in i.get("input_tensors") or []:
                for b in i.get("output_tensors") or []:
                    alias[str(a)].add(str(b))

    def reach(ids):
        seen, todo = set(), [str(x) for x in ids]
        while todo:
            t = todo.pop()
            if t in seen:
                continue
            seen.add(t)
            todo.extend(alias[t])
        return seen

    stores = [i for i in ins if i.get("tpe") == "DmaStore"]
    loads = [i for i in ins if i.get("tpe") in ("DmaLoad", "DmaGather")]
    syncs = sorted((i for i in ins if str(i.get("index")).startswith("ExplicitSync")), key=life)
    print("=== %s: instructions %d, makespan %d, DmaStore %d, DmaLoad/Gather %d, ExplicitSync %d" % (
        path.split("/")[-1], len(ins), max(life(i)[1] for i in ins), len(stores), len(loads), len(syncs)))
    for s in syncs:
        b, e = life(s)
        print("  %s  static %d..%d" % (s["index"], b, e))
        before = sorted((st for st in stores if life(st)[1] <= b), key=lambda st: -life(st)[1])[:2]
        for st in before:
            sb, se = life(st)
            outs = st.get("output_tensors") or []
            t = tens.get(str(outs[0]), {}) if outs else {}
            r = reach(outs)
            readers = [ld for ld in loads if life(ld)[0] >= se and r & {str(x) for x in ld.get("input_tensors") or []}]
            print("    store #%s %d..%d %s  -> %s (%s, %s B)" % (st["index"], sb, se, src(st), t.get("name", "?")[:60], t.get("buffer_type"), t.get("size")))
            for ld in readers[:4]:
                print("        reader #%s %s static %d (gap from sync end %+d)" % (ld["index"], src(ld), life(ld)[0], life(ld)[0] - e))
            if not readers:
                print("        no reader (output tensor or end of kernel)")
