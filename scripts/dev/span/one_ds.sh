# one_ds.sh <order_file> <tag> <pairs> -- attn_out with the dummy early store (ds.py), plain order (X) vs forced order (Y).
# Same flow as one.sh: the tests reference only sliding_attention_output, so only it compiles; X is built without and Y
# with SCHEDULER_MANUAL_ORDERING_PATH, and Arena jobs alternate X/Y, Y/X, ...
set -u
ORD=${1:?order file}; TAG=${2:?tag}; PAIRS=${3:-6}
K=sliding_attention_output
. /root/env.sh
cd /root/lab
git checkout -q 20709e8 -- src/ops.rs tests/test_kernels.rs src/device/sliding/projection.rs src/device/sliding/rope.rs
python3 /root/tk/one_only.py "$K" || exit 1
python3 /root/tk/ds.py || exit 1
C=target/furiosa-opt/kernel/furiosa-opt-gemma4
cargo furiosa-opt test --release --test test_kernels --no-run > /root/tk/${TAG}X_build.log 2>&1
grep -E "Compiling \[|Finished|^error" /root/tk/${TAG}X_build.log | tail -4
BIN=$(find target/release/deps -maxdepth 1 -type f -name "test_kernels-*" ! -name "*.d" -perm -u+x | xargs -r ls -t | head -1)
cp "$BIN" /root/tk/${TAG}_binX
ls -la "$C/furiosa_opt_gemma4::ops::$K.bin"
touch src/ops.rs
SCHEDULER_MANUAL_ORDERING_PATH="$ORD" cargo furiosa-opt test --release --test test_kernels --no-run > /root/tk/${TAG}Y_build.log 2>&1
grep -E "Compiling \[|Finished|^error" /root/tk/${TAG}Y_build.log | tail -4
BIN2=$(find target/release/deps -maxdepth 1 -type f -name "test_kernels-*" ! -name "*.d" -perm -u+x | xargs -r ls -t | head -1)
cp "$BIN2" /root/tk/${TAG}_binY
ls -la "$C/furiosa_opt_gemma4::ops::$K.bin"
if cmp -s /root/tk/${TAG}_binX /root/tk/${TAG}_binY; then echo "BINARIES IDENTICAL - abort"; exit 1; fi
: > /root/tk/${TAG}_summary.txt
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then ORDER="X Y"; else ORDER="Y X"; fi
  for v in $ORDER; do
    cp /root/tk/${TAG}_bin$v "$BIN2"; touch "$BIN2"
    REPO=/root/lab ARENA_WAIT=1500 bash /root/auto/arena.sh ${TAG}${v}_$i > /root/tk/${TAG}${v}_$i.log 2>&1
    med=$(awk -v k="$K" '$1==k && $2 ~ /^n=/ {for(j=2;j<=NF;j++) if ($j ~ /^median=/) {sub("median=","",$j); print $j}}' /root/tk/${TAG}${v}_$i.log | head -1)
    np=$(grep -c -- "-> PASS" /root/tk/${TAG}${v}_$i.log); nf=$(grep -c FAIL /root/tk/${TAG}${v}_$i.log)
    echo "pair $i $v median=$med pass=$np fail=$nf" | tee -a /root/tk/${TAG}_summary.txt
  done
done
echo DONE >> /root/tk/${TAG}_summary.txt
