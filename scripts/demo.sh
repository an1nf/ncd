#!/usr/bin/env bash
# Reproducible demo on the synthetic corpus: builds the CLI, generates the
# corpus, benchmarks compressors, runs leave-one-out k-NN and renders the
# figures used in the README (docs/img). Runtime: a few minutes on 2 cores.
#
# The synthetic corpus validates the pipeline; it is NOT evidence about real
# malware. See README "Running on real data".
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-demo-out}
NCD=target/release/ncd
mkdir -p "$OUT" docs/img

cargo build --release -p ncd-cli
rm -rf "$OUT/synth"
"$NCD" synth "$OUT/synth" --families 6 --per-family 20 --benign 30

echo "== compressor benchmark (concat) =="
"$NCD" bench "$OUT/synth" --codecs zstd:3,zstd:19,xz:9,bzip2:9,deflate:9 | tee "$OUT/bench_concat.md"
echo "== compressor benchmark (conditional) =="
"$NCD" bench "$OUT/synth" --codecs zstd:3,zstd:19 -m conditional | tee "$OUT/bench_conditional.md"

echo "== leave-one-out 1-NN =="
"$NCD" eval "$OUT/synth" -k 1 -c zstd:19 -m conditional --json "$OUT/eval_loo.json"

echo "== matrix + figures =="
"$NCD" matrix "$OUT/synth" -o "$OUT/synth_ncd.npy" -c zstd:19 -m conditional
python3 python/plot_matrix.py "$OUT/synth_ncd.npy" --title "Synthetic corpus" --out-prefix docs/img/synthetic
echo "done: $OUT/, docs/img/"
