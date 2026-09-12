#!/bin/bash
# screen_ffn_down.sh -- static makespan of forced orders that queue later ffn down tile loads before the geglu x2 store
# (V360_submit ffn in /root/lab, no Arena). Prints the base DMA/sync order with output tensor ids so the T names can be
# checked, then each order's makespan and its DMA/sync order from the down0 load on.
set -u
. /root/env.sh
cd /root/lab || exit 1
git reset -q --hard
git checkout -q V360_submit || exit 1
K=decoder_feedforward
D=/root/tk/ords_ffn
mkdir -p $D
printf "T75 -> T212\n" > $D/d1_before_store.txt
printf "T75 -> T212\nT79 -> T212\n" > $D/d12_before_store.txt
printf "T75 -> T212\nT79 -> T212\nT83 -> T212\n" > $D/d123_before_store.txt
clear_cache() { find target/furiosa-opt -name "*ops::${K}.*" -delete 2>/dev/null; }
show() {
  python3 - "$1" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ins = sorted(d["instructions"], key=lambda x: x["lifetime"]["begin"])
print("makespan", max(i["lifetime"]["end"] for i in ins))
for i in ins:
    t = i["tpe"]; b, e = i["lifetime"]["begin"], i["lifetime"]["end"]
    s = str(i["description"]).splitlines()
    if t in ("DmaLoad", "DmaStore", "ExplicitSync") and e - b >= 400 and b > 60000:
        print("  ", i["index"], t, b, e, "out", i["output_tensors"], (s[-1].strip() if s else "")[:50])
EOF
}
clear_cache
cargo furiosa-opt compile "ops::$K" --exact --dump-schedule $D/base.json > $D/base.log 2>&1 || { echo BASE_FAIL; tail -5 $D/base.log; exit 1; }
echo "== base"; show $D/base.json
for ord in $D/d1_before_store.txt $D/d12_before_store.txt $D/d123_before_store.txt; do
  name=$(basename "$ord" .txt)
  clear_cache
  if SCHEDULER_MANUAL_ORDERING_PATH="$ord" cargo furiosa-opt compile "ops::$K" --exact --dump-schedule $D/$name.json > $D/$name.log 2>&1; then
    echo "== $name"; show $D/$name.json
  else
    echo "== $name COMPILE_FAIL: $(grep -m1 -iE 'no producer|error' $D/$name.log | cut -c1-200)"
  fi
done
clear_cache
echo SCREEN_FFN_DOWN_DONE
