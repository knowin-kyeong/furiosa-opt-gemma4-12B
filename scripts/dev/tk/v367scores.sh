#!/bin/bash
# v367scores.sh -- collect score and per-kernel cycles of every V367_submit draw into /root/tk/v367_scores.txt
# (one line per moa-submitter log id: id score qkv attn ffn; already collected ids are skipped).
. /root/env.sh
for f in /root/drawloop_V367_submit*.log; do
  [ -f "$f" ] || continue
  for id in $(grep -oE "moa-submitter log [0-9a-f]+" "$f" | awk '{print $3}'); do
    grep -q "^$id " /root/tk/v367_scores.txt 2>/dev/null && continue
    o=$(timeout 90 /root/.cargo/bin/moa-submitter log "$id" 2>/dev/null)
    s=$(printf "%s\n" "$o" | sed -n "s/^score: //p")
    q=$(printf "%s\n" "$o" | sed -n "s/^ops::sliding_project_qkv: \([0-9]*\).*/\1/p")
    a=$(printf "%s\n" "$o" | sed -n "s/^ops::sliding_attention_output: \([0-9]*\).*/\1/p")
    d=$(printf "%s\n" "$o" | sed -n "s/^ops::decoder_feedforward: \([0-9]*\).*/\1/p")
    [ -n "$s" ] && echo "$id $s $q $a $d" >> /root/tk/v367_scores.txt
  done
done
