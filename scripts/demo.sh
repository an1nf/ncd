#!/usr/bin/env bash
# Synthetic corpus: codec benchmark, leave-one-out 1-NN, NCD matrix figures.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-demo-out}
NCD=target/release/ncd
mkdir -p "$OUT"

cargo build --release -p ncd-cli
rm -rf "$OUT/synth"
"$NCD" synth "$OUT/synth" --families 6 --per-family 20 --benign 30

echo "== bench concat =="
"$NCD" bench "$OUT/synth" --codecs zstd:3,zstd:19,xz:9,bzip2:9,deflate:9 | tee "$OUT/bench_concat.md"
echo "== bench conditional =="
"$NCD" bench "$OUT/synth" --codecs zstd:3,zstd:19 -m conditional | tee "$OUT/bench_conditional.md"

echo "== leave-one-out 1-NN =="
"$NCD" eval "$OUT/synth" -k 1 -c zstd:19 -m conditional --json "$OUT/eval_loo.json"

echo "== matrix + figures =="
"$NCD" matrix "$OUT/synth" -o "$OUT/synth_ncd.npy" -c zstd:19 -m conditional
PYTHONPATH=python python3 -m ncdeval.plot "$OUT/synth_ncd.npy" --title "Synthetic corpus" --out-prefix "$OUT/synthetic"
echo "outputs in $OUT/"
