import json, sys
kernel = sys.argv[1]; tags = sys.argv[2:]
for tag in tags:
    try: d = json.load(open(f"target/schedules/{tag}_{kernel}.json"))
    except Exception as e: print(tag, e); continue
    for i in d["instructions"]:
        if "DmaEngine" in (i.get("contexts") or []):
            dur = i["lifetime"]["end"] - i["lifetime"]["begin"]
            if dur > 1500:
                desc = (i.get("description") or "").split("\n")[0]
                print(f"{tag} {i.get('tpe'):8s} {dur:7d} util={i.get('util')} {desc}")
