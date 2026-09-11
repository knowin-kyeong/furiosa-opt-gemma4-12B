"""census.py <tag> [kernel...] -- does a schedule dump trip the book DMA rules? (2026-09-11 E0)

  (i)  tensor-unit instructions carrying the DmaEngine context: the book 64-accesses-per-bank rule makes the
       compiler schedule an offending Main/Sub pass as if it occupied DMA;
  (ii) source lines issuing more than one DMA command: a split at a DMN boundary, or just a function that is
       called twice (K/V loads, norm weights), which is all that V261 has;
plus every switch, the eight longest instructions and the schedule tail. On V261 both rules count zero, and the
store rows it prints showed that HBM store cost follows descriptor count, not 256 B alignment.
Reads target/schedules/<tag>_<kernel>.json (dump_schedules.sh).
"""
import collections
import json
import re
import sys

tag = sys.argv[1]
kernels = sys.argv[2:] or ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]


def src(i):
    m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
    return m.group(1).replace("src/device/", "") if m else "?"


def row(i):
    life = i["lifetime"]
    util = (i.get("util") or {}).get("total_util", 0) or 0
    return "%9s %24s %7d-%-7d dur=%5d util=%.3f %s" % (
        i.get("tpe"), ",".join(i.get("contexts") or []), life["begin"], life["end"],
        life["end"] - life["begin"], util, src(i))


for k in kernels:
    try:
        d = json.load(open(f"target/schedules/{tag}_{k}.json"))
    except Exception as e:
        print(f"{k}: {e}")
        continue
    ins = d["instructions"]
    mk = max(i["lifetime"]["end"] for i in ins)
    print("\n===== %s  makespan=%d  n=%d" % (k, mk, len(ins)))
    odd = [i for i in ins if "DmaEngine" in (i.get("contexts") or []) and not str(i.get("tpe", "")).startswith("Dma")]
    print("  (i) non-DMA instructions carrying the DmaEngine context: %d" % len(odd))
    for i in odd[:6]:
        print("      " + row(i))
    per = collections.Counter(src(i) for i in ins if "DmaEngine" in (i.get("contexts") or []))
    print("  (ii) DMA source lines issuing >1 command:", {s: n for s, n in per.items() if n > 1} or "none")
    for i in ins:
        if ".switch::" in (i.get("description") or ""):
            print("  switch  " + row(i))
    print("  longest 8:")
    for i in sorted(ins, key=lambda i: i["lifetime"]["begin"] - i["lifetime"]["end"])[:8]:
        print("    " + row(i))
    print("  tail (last 10 by end):")
    for i in sorted(ins, key=lambda i: i["lifetime"]["end"])[-10:]:
        print("    " + row(i))
