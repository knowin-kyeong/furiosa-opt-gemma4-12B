"""gaps.py <tag> [kernel...] -- DMA engine idle intervals in a schedule dump, largest first.

Merges every DmaEngine instruction lifetime and prints the 12 largest holes with the source line of the
command that ends each one. It found both 2026-09-11 schedule holes (V253 attn_out; V260 ffn, a 3,245-cycle
wait that the split down store filled). Reads target/schedules/<tag>_<kernel>.json (dump_schedules.sh).
"""
import json
import re
import sys

tag = sys.argv[1]
kernels = sys.argv[2:] or ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
for k in kernels:
    try:
        d = json.load(open(f"target/schedules/{tag}_{k}.json"))
    except Exception as e:
        print(f"{k}: {e}")
        continue
    ins = d["instructions"]
    mk = max(i["lifetime"]["end"] for i in ins)
    dma = [i for i in ins if "DmaEngine" in (i.get("contexts") or [])]
    iv = sorted((i["lifetime"]["begin"], i["lifetime"]["end"], n, i) for n, i in enumerate(dma))
    gaps, cur = [], 0
    for b, e, _, i in iv:
        if b > cur:
            gaps.append((cur, b, b - cur, i))
        cur = max(cur, e)
    if cur < mk:
        gaps.append((cur, mk, mk - cur, None))
    busy = sum(e - b for b, e, _, _ in iv)
    print(f"===== {k} makespan={mk} dma_busy={busy} ({100 * busy / mk:.1f}%) idle={mk - busy} n_dma={len(dma)}")
    for b, e, g, i in sorted(gaps, key=lambda x: -x[2])[:12]:
        desc = (i.get("description") or "").replace("\n", " ") if i is not None else "TAIL"
        m = re.search(r"--> (src/[^\s:]+:\d+)", desc)
        print(f"   idle {g:7d}  [{b}-{e}]  next: {m.group(1) if m else desc[:60]}")
