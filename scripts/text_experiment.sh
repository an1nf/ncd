#!/usr/bin/env bash
# Experimento de texto completo: tarea AG News, conversión de GPT-2, entrenamiento de las familias
# por bytes, bits por carácter y NCD para cada compresor, análisis y figuras.
# Cada paso se salta si su salida ya existe, así que se puede interrumpir y reanudar.
#
#   ./scripts/download_text_data.sh && ./scripts/text_experiment.sh
#
# Variables (con sus valores por defecto):
#   DATA=data/text  MODELS=models  RES=results/text  DEVICE=  (cpu | cuda; vacío = cuda si hay)
#   FAMILIES="enwik8 agnews agpair" (vacío: ninguna)   SIZES="b037k b149k b892k b2.8m"   TRAIN_BYTES=8e6
#   CLASSIC="zstd:1 zstd:3 zstd:9 zstd:19 xz:1 xz:6 bzip2:9 deflate:9"
#   GPT2=1   MATCH=1   HF_MODELS=""   (p. ej. "EleutherAI/pythia-70m EleutherAI/pythia-160m")
#   EXCLUDE="agpair/b037k"   (familia/tamaño que no se entrenan; el experimento publicado no lo tiene)
#   Tamaños disponibles: b037k b149k b892k b2.8m b6.5m b38m (los dos últimos, mejor con GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/text}
MODELS=${MODELS:-models}
RES=${RES:-results/text}
DEVICE=${DEVICE:-}
FAMILIES=${FAMILIES-"enwik8 agnews agpair"}
SIZES=${SIZES:-"b037k b149k b892k b2.8m"}
TRAIN_BYTES=${TRAIN_BYTES:-8e6}
CLASSIC=${CLASSIC:-"zstd:1 zstd:3 zstd:9 zstd:19 xz:1 xz:6 bzip2:9 deflate:9"}
GPT2=${GPT2:-1}
MATCH=${MATCH:-1}
HF_MODELS=${HF_MODELS:-}
EXCLUDE=${EXCLUDE-"agpair/b037k"}
TASK=$DATA/text_task
DEV=(); [ -n "$DEVICE" ] && DEV=(--device "$DEVICE")

size_args() {  # capas, ancho, cabezas, lr
  case $1 in
    b037k) echo "1 32 2 3e-3" ;;
    b149k) echo "2 64 4 3e-3" ;;
    b892k) echo "4 128 4 2e-3" ;;
    b2.8m) echo "6 192 6 1.5e-3" ;;
    b6.5m) echo "8 256 8 1e-3" ;;
    b38m)  echo "12 512 8 6e-4" ;;
    *) echo "tamaño desconocido: $1" >&2; exit 1 ;;
  esac
}

echo "== 1. tarea"
[ -f "$TASK/task.jsonl" ] || python python/prep_text_task.py --ag-dir "$DATA/ag" --enwik8 "$DATA/enwik8.gz" --out "$TASK"
[ -f "$TASK/lm_train_paired.txt" ] || python python/make_paired_corpus.py "$TASK/lm_train.txt" --out "$TASK/lm_train_paired.txt"

echo "== 2. GPT-2"
if [ "$GPT2" = 1 ] && [ ! -f "$MODELS/gpt2-124m.pt" ] && [ ! -f "$RES/gpt2-124m.json" ]; then
  python python/convert_gpt2_onnx.py "$DATA/gpt2-lm-head-10.onnx" --tokenizer-dir "$DATA/tok" --out "$MODELS/gpt2-124m.pt"
fi

echo "== 3. modelos por bytes"
for fam in $FAMILIES; do
  for sz in $SIZES; do
    case " $EXCLUDE " in *" $fam/$sz "*) continue ;; esac
    out=$MODELS/$fam/$sz.pt
    [ -f "${out%.pt}.json" ] && continue
    read -r L D H LR <<< "$(size_args "$sz")"
    case $fam in
      enwik8) src=(--data "$DATA/enwik8.gz") ;;
      agnews) src=(--corpus file --data "$TASK/lm_train.txt" --valid "$TASK/lm_valid.txt") ;;
      agpair) src=(--corpus file --data "$TASK/lm_train_paired.txt" --valid "$TASK/lm_valid.txt") ;;
      *) echo "familia desconocida: $fam"; exit 1 ;;
    esac
    python python/train_byte_lm.py "${src[@]}" --out "$out" --n-layer "$L" --n-embd "$D" --n-head "$H" \
        --lr "$LR" --train-bytes "$TRAIN_BYTES" ${DEV[@]+"${DEV[@]}"}
  done
done

echo "== 4. compresores"
read -r -a specs <<< "$CLASSIC"
[ "$GPT2" = 1 ] && specs+=("lm:gpt2-124m=$MODELS/gpt2-124m.pt")
for m in $HF_MODELS; do specs+=("lm:${m##*/}=hf:$m"); done
for fam in $FAMILIES; do
  for sz in $SIZES; do
    case " $EXCLUDE " in *" $fam/$sz "*) continue ;; esac
    specs+=("lm:$fam-$sz=$MODELS/$fam/$sz.pt")
    [ "$MATCH" = 1 ] && specs+=("lm+match:$fam-$sz+match=$MODELS/$fam/$sz.pt")
  done
done
# Uno por proceso: si uno falla, los demás ya quedaron guardados.
for s in "${specs[@]}"; do
  python python/run_text_experiment.py --skip-existing --task-dir "$TASK" --out "$RES" \
      --tokenizer-dir "$DATA/tok" ${DEV[@]+"${DEV[@]}"} "$s"
done

echo "== 5. análisis"
python python/baseline_text.py --task-dir "$TASK" --out "$RES/analysis"
python python/analyze_text.py --task-dir "$TASK" --results "$RES" --out "$RES/analysis"
python python/plot_text.py "$RES/analysis/text_summary.json" --out-dir "$RES/figs"
echo "tabla: $RES/analysis/text_summary.md   figuras: $RES/figs/"
