# one.sh <kernel> <order_file> <tag> <pairs> -- single-kernel harness A/B of a forced scheduler order (pod, /root/lab).
# Builds the V280 span harness with only <kernel> referenced by the tests (so only it compiles), once plain (X)
# and once under SCHEDULER_MANUAL_ORDERING_PATH=<order_file> (Y), then alternates Arena jobs X/Y, Y/X, ...
set -u
K=${1:?kernel}; ORD=${2:?order file}; TAG=${3:?tag}; PAIRS=${4:-6}
. /root/env.sh
cd /root/lab
git checkout -q 20709e8 -- src/ops.rs tests/test_kernels.rs
cat > /root/tk/one_only.py <<'PY'
import io, re, sys
keep = sys.argv[1]
p = "tests/test_kernels.rs"
t = io.open(p, encoding="utf-8").read()
info = {"decoder_feedforward": ("0.01", "FFN_SWEEP"), "sliding_project_qkv": ("0.04", "QKV_SWEEP"), "sliding_attention_output": ("0.05", "ATTN_SWEEP")}
for k, (atol, sweep) in info.items():
    if k == keep:
        continue
    old = '    Plan { name: "%s", atol: %s, rtol: RTOL, order: %s },\n' % (k, atol, sweep)
    assert t.count(old) == 1, old
    t = t.replace(old, "", 1)
    old = '        "%s" => %s(ctx, fixture, bench, plan).await,\n' % (k, k)
    assert t.count(old) == 1, old
    t = t.replace(old, "", 1)
    a = t.index("async fn %s(" % k)
    b = t.index("\n}\n", a) + 3
    t = t[:a] + t[b:]
g = "orphans.is_empty(),"
assert t.count(g) == 1, "fixture guard"
t = t.replace(g, "orphans.is_empty() || !orphans.is_empty(),", 1)
sweep = info[keep][1]
t, n = re.subn(r'const %s: &\[&str\] = (?:&\[""; \d+\]|&\[\n(?:    [^\n]*\n)+\]);' % sweep, 'const %s: &[&str] = &[""; 16];' % sweep, t)
assert n == 1, sweep
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("only", keep)
PY
python3 /root/tk/one_only.py "$K" || exit 1
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
