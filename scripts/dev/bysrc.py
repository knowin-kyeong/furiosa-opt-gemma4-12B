import json, sys, re, collections
tag = sys.argv[1]
for k in ["sliding_project_qkv","sliding_attention_output","decoder_feedforward"]:
    d = json.load(open(f"target/schedules/{tag}_{k}.json"))
    ins = d["instructions"]; mk = max(i["lifetime"]["end"] for i in ins)
    agg = collections.defaultdict(lambda: [0,0])
    for i in ins:
        dur = i["lifetime"]["end"]-i["lifetime"]["begin"]
        m = re.search(r"--> (src/[^\s:]+:\d+)", i.get("description") or "")
        key = (i.get("tpe","?"), m.group(1) if m else "?")
        agg[key][0] += dur; agg[key][1] += 1
    print(f"===== {k} makespan={mk} =====")
    print(f"{'total':>9s} {'n':>4s} {'avg':>8s} {'%mk':>6s}  tpe            source")
    for (tpe,src),(tot,n) in sorted(agg.items(), key=lambda kv:-kv[1][0])[:22]:
        print(f"{tot:9d} {n:4d} {tot//n:8d} {100*tot/mk:6.1f}  {tpe:14s} {src}")
    # DMA util summary
    dma = [i for i in ins if "DmaEngine" in (i.get("contexts") or [])]
    utils = [(i.get("util") or {}).get("total_util") for i in dma]
    utils = [u for u in utils if isinstance(u,(int,float))]
    if utils: print(f"  DMA nodes={len(dma)} mean total_util={sum(utils)/len(utils):.3f} min={min(utils):.3f}")
    print()
