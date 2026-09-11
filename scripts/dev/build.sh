#!/bin/bash
# build.sh <branch> <tag> [repo] -- pod: check out origin/<branch> in <repo> (default /root/lab2) and build the
# release test binary. /root/auto/arena.sh submits target/release/deps/test_kernels-*, so a plain
# "cargo furiosa-opt test --no-run" (debug) leaves Arena measuring an old release binary (V262).
# Never copy target/ between clones (rustc ICE "uninterned StableCrateId"); a new clone builds from empty.
. /root/env.sh
BR=${1:?usage: build.sh branch tag [repo]}; TAG=${2:?}; REPO=${3:-/root/lab2}
cd "$REPO" || exit 1
git fetch -q origin
git checkout -q -B "$BR" "origin/$BR" || exit 2
echo "HEAD=$(git log --oneline -1)"
S=$(date +%s)
timeout 3600 cargo furiosa-opt test --release --test test_kernels --no-run --message-format=json > "/root/build_$TAG.json" 2> "/root/build_$TAG.err"
rc=$?
echo "build exit=$rc elapsed=$(( $(date +%s) - S ))s"
if [ $rc -ne 0 ]; then
  python3 -c '
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("reason") == "compiler-message" and d["message"].get("level") == "error":
        print(d["message"].get("rendered", "")[:3000])
' "/root/build_$TAG.json"
  tail -40 "/root/build_$TAG.err"
  exit 1
fi
