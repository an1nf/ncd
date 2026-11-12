#!/usr/bin/env bash
# Datos del experimento de texto, con verificación SHA-256:
#   AG News (train.csv, test.csv), enwik8.gz, GPT-2 124M del ONNX Model Zoo y su tokenizador.
#
#   ./scripts/download_text_data.sh [DIR]      # por defecto data/text
#   NO_GPT2=1 ./scripts/download_text_data.sh  # sin los 665 MB de GPT-2
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-data/text}
mkdir -p "$DIR/ag" "$DIR/tok"

fetch() {  # fetch OUT SHA256 URL [URL...]
  local out=$1 sha=$2; shift 2
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --quiet 2>/dev/null; then
    echo "ok   $out"; return
  fi
  for url in "$@"; do
    echo "get  $url"
    if curl -fsSL --retry 3 -o "$out.part" "$url" && echo "$sha  $out.part" | sha256sum -c --quiet; then
      mv "$out.part" "$out"; return
    fi
    echo "     falló o no coincide el hash; pruebo otra fuente"
  done
  rm -f "$out.part"; echo "ERROR: no pude bajar $out"; return 1
}

AG=https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv
fetch "$DIR/ag/train.csv" 76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e "$AG/train.csv"
fetch "$DIR/ag/test.csv" 521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699 "$AG/test.csv"

# enwik8: la copia comprimida de GitHub; si no, el zip oficial de Matt Mahoney (se recomprime a .gz).
ENWIK8_SHA=2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8  # sha256 del archivo descomprimido
if [ -f "$DIR/enwik8.gz" ] && [ "$(gzip -dc "$DIR/enwik8.gz" | sha256sum | cut -d' ' -f1)" = "$ENWIK8_SHA" ]; then
  echo "ok   $DIR/enwik8.gz"
else
  if ! fetch "$DIR/enwik8.gz" d52ea2118b505f79007e084b2f49acef9fd0e45fb481f84b2cab0a7191a7ef60 \
      https://raw.githubusercontent.com/lucidrains/x-transformers/main/data/enwik8.gz; then :; fi
  if [ "$(gzip -dc "$DIR/enwik8.gz" 2>/dev/null | sha256sum | cut -d' ' -f1)" != "$ENWIK8_SHA" ]; then
    curl -fsSL -o "$DIR/enwik8.zip" https://mattmahoney.net/dc/enwik8.zip
    unzip -p "$DIR/enwik8.zip" enwik8 | gzip -9 > "$DIR/enwik8.gz" && rm "$DIR/enwik8.zip"
    [ "$(gzip -dc "$DIR/enwik8.gz" | sha256sum | cut -d' ' -f1)" = "$ENWIK8_SHA" ] || { echo "enwik8 no coincide"; exit 1; }
  fi
fi

if [ "${NO_GPT2:-0}" != 1 ]; then
  OAI=https://openaipublic.blob.core.windows.net/gpt-2/models/124M
  MIRROR=https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2
  fetch "$DIR/tok/encoder.json" 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 \
      "$OAI/encoder.json" "$MIRROR/encoder.json"
  fetch "$DIR/tok/vocab.bpe" 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 \
      "$OAI/vocab.bpe" "$MIRROR/vocab.bpe"
  fetch "$DIR/gpt2-lm-head-10.onnx" 12fbb1ec2d2d70c8ebd21a290a348a4109447b98582af64c6f93b6526d5c8f35 \
      https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
fi
echo "datos en $DIR"
