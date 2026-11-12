#!/usr/bin/env bash
# CIC-IDS2017 flow windows (TrafficLabelling/*.csv).
#   A: anomaly detection, benign Monday as reference (with and without destination port)
#   B: attack classification, random split (optimistic)
#   LM=1: byte LM trained on benign Monday windows, used as the NCD compressor
#
#   CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/traffic.sh
#
# DATA=$HOME/ncd-data/cic  RES=results/cic  WINDOW=64  MAX_PER_LABEL=3000  CODEC=zstd:3
# MAX_REFS=1000  DAYS="Monday ... Friday" (full week ~8 GB RAM)  SKIP_A=0
# LM_SIZE=b149k|b892k|b2.8m|b6.5m  LM_WINDOW=12  LM_BYTES=20e6  LM_BLOCK=2048  LM_MAX_REFS=50  DEVICE=
# A window holds W x 79 bytes; LM NCD needs 2 windows + 1 byte in LM_BLOCK.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python

CSV_DIR=${CSV_DIR:?"set CSV_DIR=/path/to/TrafficLabelling"}
DATA=${DATA:-$HOME/ncd-data/cic}
RES=${RES:-results/cic}
WINDOW=${WINDOW:-64}
MAXL=${MAX_PER_LABEL:-3000}
CODEC=${CODEC:-zstd:3}
MAX_REFS=${MAX_REFS:-1000}
DAYS=${DAYS:-"Monday Tuesday Wednesday Thursday Friday"}
DEV=(); [ -n "${DEVICE:-}" ] && DEV=(--device "$DEVICE")

csvs=()
for d in $DAYS; do for f in "$CSV_DIR"/"$d"*.csv; do [ -f "$f" ] && csvs+=("$f"); done; done
[ ${#csvs[@]} -gt 0 ] || { echo "no CSV for $DAYS in $CSV_DIR"; exit 1; }

windows() {  # windows OUT W SPLIT [flows args]
  local out=$1 w=$2 split=$3; shift 3
  [ -f "$out/windows.csv" ] || python3 -m ncdeval.flows "${csvs[@]}" --out "$out" --window "$w" \
      --order rows --split "$split" --train-files Monday --max-per-label "$MAXL" "$@"
}

evaluate() {  # evaluate FLOWS RES [codec max_refs]
  [ $# -gt 2 ] || python3 -m ncdeval.baseline_flows "$1" --out-dir "$2"
  python3 -m ncdeval.eval_flows "$1" --codec "${3:-$CODEC}" --max-refs "${4:-$MAX_REFS}" --out-dir "$2"
  python3 -m ncdeval.compare "$2" --out "$2/summary.md"
}

if [ "${SKIP_A:-0}" != 1 ]; then
  windows "$DATA/flows$WINDOW" "$WINDOW" file
  evaluate "$DATA/flows$WINDOW" "$RES/flows$WINDOW"
  windows "$DATA/flows${WINDOW}_noport" "$WINDOW" file --drop-port
  evaluate "$DATA/flows${WINDOW}_noport" "$RES/flows${WINDOW}_noport"
fi

if [ "${LM:-0}" = 1 ]; then
  LW=${LM_WINDOW:-12}
  flows=$DATA/flows$LW
  model=models/flows$LW/${LM_SIZE:-b2.8m}.pt
  windows "$flows" "$LW" file
  if [ ! -f "$model" ]; then
    case ${LM_SIZE:-b2.8m} in
      b149k) a=(2 64 4 3e-3) ;; b892k) a=(4 128 4 2e-3) ;; b2.8m) a=(6 192 6 1.5e-3) ;; b6.5m) a=(8 256 8 1e-3) ;;
      *) echo "LM_SIZE: b149k, b892k, b2.8m, b6.5m"; exit 1 ;;
    esac
    python3 -m ncdeval.lm.train --corpus flows --data "$flows" --out "$model" \
        --n-layer "${a[0]}" --n-embd "${a[1]}" --n-head "${a[2]}" --lr "${a[3]}" --block "${LM_BLOCK:-2048}" \
        --batch 8 --train-bytes "${LM_BYTES:-20e6}" ${DEV[@]+"${DEV[@]}"}
  fi
  export LMCODEC_DEVICE=${DEVICE:-}
  evaluate "$flows" "$RES/flows$LW"
  evaluate "$flows" "$RES/flows$LW" "lm:$model" "${LM_MAX_REFS:-50}"
fi

windows "$DATA/flows${WINDOW}r" "$WINDOW" random --max-per-label 1000
evaluate "$DATA/flows${WINDOW}r" "$RES/cls$WINDOW"
echo "summaries: $RES/*/summary.md"
