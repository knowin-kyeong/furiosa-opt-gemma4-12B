import json, sys, collections
tag = sys.argv[1]
kernels = sys.argv[2:] or ["sliding_project_qkv","sliding_attention_output","decoder_feedforward"]
for k in kernels:
    p = f"target/schedules/{tag}_{k}.json"
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"{k}: cannot load ({e})"); continue
    ins = d.get("instructions", [])
    mk = max((i["lifetime"]["end"] for i in ins), default=0)
    busy = collections.Counter(); n = collections.Counter()
    for i in ins:
        dur = i["lifetime"]["end"] - i["lifetime"]["begin"]
        for c in (i.get("contexts") or ["?"]):
            busy[c] += dur; n[c] += 1
    print(f"{k}: makespan={mk} instructions={len(ins)} tensors={len(d.get('tensors',[]))}")
    for c, b in busy.most_common():
        print(f"    {c:16s} busy={b:>9d} ({100*b/mk:5.1f}% of makespan, may overlap) n={n[c]}")
    top = sorted(ins, key=lambda i: i["lifetime"]["end"]-i["lifetime"]["begin"], reverse=True)[:8]
    for i in top:
        dur = i["lifetime"]["end"]-i["lifetime"]["begin"]
        desc = (i.get("description") or "").replace("\n"," ")[:120]
        print(f"    top: {dur:>8d} [{i['lifetime']['begin']}-{i['lifetime']['end']}] {i.get('tpe','?'):14s} {i.get('contexts')} {desc}")
