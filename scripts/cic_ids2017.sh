#!/usr/bin/env bash
# CIC-IDS2017 (TrafficLabelling/*.csv de GeneratedLabelledFlows.zip): protocolo A (anomalías,
# lunes como referencia), su variante sin puerto de destino, protocolo B (clasificación) y,
# con LM=1, un modelo de lenguaje por bytes entrenado solo con ventanas benignas del lunes.
#
#   CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/cic_ids2017.sh
#
# Variables: DATA=$HOME/ncd-data/cic  RES=results/cic  WINDOW=64  MAX_PER_LABEL=3000
#   CODEC=zstd:3  MAX_REFS=1000  DAYS="Monday Tuesday Wednesday Thursday Friday" (CSV a usar;
#   la semana completa necesita unos 8 GB de RAM)  ABLATION=0 (1: ventanas de 32 y 128)
#   LM=0  LM_SIZE=b2.8m  LM_WINDOW=12  LM_BYTES=20e6  LM_BLOCK=2048  LM_MAX_REFS=50  DEVICE=
#   (el modelo usa ventanas de LM_WINDOW flujos: 12 x 79 bytes caben dos veces en 2048)
#   SKIP_A=1 salta el protocolo A con ventanas de WINDOW (si ya está hecho)
set -euo pipefail
cd "$(dirname "$0")/.."

CSV_DIR=${CSV_DIR:?"pon CSV_DIR=/ruta/a/TrafficLabelling"}
DATA=${DATA:-$HOME/ncd-data/cic}
RES=${RES:-results/cic}
WINDOW=${WINDOW:-64}
MAXL=${MAX_PER_LABEL:-3000}
CODEC=${CODEC:-zstd:3}
MAX_REFS=${MAX_REFS:-1000}
DAYS=${DAYS:-"Monday Tuesday Wednesday Thursday Friday"}
LM=${LM:-0}
DEV=(); [ -n "${DEVICE:-}" ] && DEV=(--device "$DEVICE")

csvs=()
for d in $DAYS; do for f in "$CSV_DIR"/"$d"*.csv; do [ -f "$f" ] && csvs+=("$f"); done; done
[ ${#csvs[@]} -gt 0 ] || { echo "no hay CSV de $DAYS en $CSV_DIR"; exit 1; }
printf 'CSV: %s\n' "${csvs[@]}"

protocol_a() {  # protocol_a NAME WINDOW [extra args de prepare_flows]
  local name=$1 w=$2; shift 2
  local flows=$DATA/$name res=$RES/$name
  [ -f "$flows/windows.csv" ] || python python/prepare_flows.py "${csvs[@]}" --out "$flows" --window "$w" \
      --order rows --split file --train-files Monday --max-per-label "$MAXL" "$@"
  python python/baseline_flows.py "$flows" --out-dir "$res"
  python python/eval_flows_ncd.py "$flows" --codec "$CODEC" --max-refs "$MAX_REFS" --out-dir "$res"
}

if [ "${SKIP_A:-0}" != 1 ]; then
  echo "== protocolo A: lunes benigno como referencia"
  protocol_a flows$WINDOW "$WINDOW"
  echo "== protocolo A sin puerto de destino"
  protocol_a flows${WINDOW}_noport "$WINDOW" --drop-port
fi

if [ "$LM" = 1 ]; then
  # Una ventana de W flujos ocupa W x (características + 1) bytes: 64 x 79 = 5056 con las 78 de
  # CIC-IDS2017. Para la NCD el contexto del modelo debe caber dos ventanas y un byte inicial,
  # así que el modelo usa ventanas más cortas (LM_WINDOW) y se compara con zstd sobre las mismas.
  LW=${LM_WINDOW:-12}
  echo "== modelo de lenguaje con ventanas benignas del lunes ($LW flujos por ventana)"
  flows=$DATA/flows$LW
  res=$RES/flows$LW
  lmdir=$DATA/lm_flows$LW
  model=models/flows$LW/${LM_SIZE:-b2.8m}.pt
  [ -f "$flows/windows.csv" ] || python python/prepare_flows.py "${csvs[@]}" --out "$flows" --window "$LW" \
      --order rows --split file --train-files Monday --max-per-label "$MAXL"
  python python/baseline_flows.py "$flows" --out-dir "$res"
  python python/eval_flows_ncd.py "$flows" --codec "$CODEC" --max-refs "$MAX_REFS" --out-dir "$res"
  [ -f "$lmdir/train.bin" ] || python python/make_lm_corpus.py --flows "$flows" --out "$lmdir"
  if [ ! -f "${model%.pt}.json" ]; then
    case ${LM_SIZE:-b2.8m} in
      b149k) a=(2 64 4 3e-3) ;; b892k) a=(4 128 4 2e-3) ;; b2.8m) a=(6 192 6 1.5e-3) ;; b6.5m) a=(8 256 8 1e-3) ;;
      *) echo "LM_SIZE: b149k, b892k, b2.8m o b6.5m"; exit 1 ;;
    esac
    python python/train_byte_lm.py --corpus file --data "$lmdir/train.bin" --valid "$lmdir/valid.bin" --out "$model" \
        --n-layer "${a[0]}" --n-embd "${a[1]}" --n-head "${a[2]}" --lr "${a[3]}" --block "${LM_BLOCK:-2048}" \
        --batch 8 --train-bytes "${LM_BYTES:-20e6}" ${DEV[@]+"${DEV[@]}"}
  fi
  export LMCODEC_DEVICE=${DEVICE:-}
  python python/eval_flows_surprise.py "$flows" --model "$model" --out-dir "$res" ${DEV[@]+"${DEV[@]}"}
  maxobj=$(python -c "import pathlib,sys; print(max(p.stat().st_size for p in pathlib.Path(sys.argv[1]).rglob('*.bin')))" "$flows")
  if [ $((2 * maxobj + 1)) -le "${LM_BLOCK:-2048}" ]; then
    # Coste: ventanas de prueba x referencias pasadas del modelo; limita las referencias.
    python python/eval_flows_ncd.py "$flows" --codec "lm:$model" --max-refs "${LM_MAX_REFS:-50}" --out-dir "$res"
  else
    echo "aviso: ventanas de hasta $maxobj bytes; la NCD con el modelo necesita LM_BLOCK >= $((2 * maxobj + 1)). Se omite."
  fi
  python python/compare.py "$res" --out "$res/summary.md"
fi
for d in "$RES/flows$WINDOW" "$RES/flows${WINDOW}_noport"; do
  [ -d "$d" ] && python python/compare.py "$d" --out "$d/summary.md"
done

echo "== protocolo B: clasificación con partición aleatoria (optimista)"
flows=$DATA/flows${WINDOW}r
[ -f "$flows/windows.csv" ] || python python/prepare_flows.py "${csvs[@]}" --out "$flows" --window "$WINDOW" \
    --order rows --split random --max-per-label 1000
python python/baseline_flows.py "$flows" --out-dir "$RES/cls$WINDOW"
python python/eval_flows_ncd.py "$flows" --codec "$CODEC" --max-refs "$MAX_REFS" --out-dir "$RES/cls$WINDOW"
python python/compare.py "$RES/cls$WINDOW" --out "$RES/cls$WINDOW/summary.md"

if [ "${ABLATION:-0}" = 1 ]; then
  for w in 32 128; do
    echo "== protocolo A con ventanas de $w flujos"
    protocol_a flows$w "$w"
    python python/compare.py "$RES/flows$w" --out "$RES/flows$w/summary.md"
  done
fi
echo "resúmenes: $RES/*/summary.md"
