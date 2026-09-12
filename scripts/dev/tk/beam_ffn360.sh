#!/bin/bash
# beam_ffn360.sh -- beam-search trace of the V360_submit ffn (/root/lab): the kept path's node names, so forced orders
# (SCHEDULER_MANUAL_ORDERING_PATH "T<a> -> T<b>") can name the down tile loads and the geglu stores.
set -u
. /root/env.sh
cd /root/lab || exit 1
git reset -q --hard
git checkout -q V360_submit || exit 1
K=decoder_feedforward
find target/furiosa-opt -name "*ops::${K}.*" -delete 2>/dev/null
BEAM_SEARCH_TRACE_DUMP_PATH=/root/tk/beam360_ffn cargo furiosa-opt compile "ops::$K" --exact --dump-schedule /root/tk/S360_ffn.json > /root/tk/beam360_ffn_compile.log 2>&1 || { echo COMPILE_FAIL; tail -5 /root/tk/beam360_ffn_compile.log; exit 1; }
python3 /root/tk/beampath.py /root/tk/beam360_ffn > /root/tk/beam360_ffn_path.txt
head -1 /root/tk/beam360_ffn_path.txt
grep -nE "Dma\.|Sync" /root/tk/beam360_ffn_path.txt
echo BEAM_FFN360_DONE
