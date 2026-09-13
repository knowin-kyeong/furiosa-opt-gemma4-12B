#!/bin/bash
# drawscores.sh <branch> -- collect score and per-kernel cycles of every <branch> draw into /root/tk/<branch>_scores.txt
# (one line per moa-submitter log id: id score qkv attn ffn; ids already collected are skipped).
. /root/env.sh
BR=${1:?branch}; OUT=/root/tk/${BR}_scores.txt
for f in /root/drawloop_${BR}*.log; do
  [ -f "$f" ] || continue
  for id in $(grep -oE "moa-submitter log [0-9a-f]+" "$f" | awk '{print $3}'); do
    grep -q "^$id " "$OUT" 2>/dev/null && continue
    o=$(timeout 90 /root/.cargo/bin/moa-submitter log "$id" 2>/dev/null)
    s=$(printf "%s\n" "$o" | sed -n "s/^score: //p")
    q=$(printf "%s\n" "$o" | sed -n "s/^ops::sliding_project_qkv: \([0-9]*\).*/\1/p")
    a=$(printf "%s\n" "$o" | sed -n "s/^ops::sliding_attention_output: \([0-9]*\).*/\1/p")
    d=$(printf "%s\n" "$o" | sed -n "s/^ops::decoder_feedforward: \([0-9]*\).*/\1/p")
    [ -n "$s" ] && echo "$id $s $q $a $d" >> "$OUT"
  done
done
