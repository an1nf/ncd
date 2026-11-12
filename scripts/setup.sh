#!/usr/bin/env bash
# Entorno completo: binario Rust, venv de Python, bindings y pruebas.
#
#   TORCH=cpu  ./scripts/setup.sh   # PyTorch solo CPU (por defecto)
#   TORCH=cuda ./scripts/setup.sh   # PyTorch con CUDA (wheel de PyPI; GPU Turing, sm_75, o posterior)
#   TORCH=cu126 ./scripts/setup.sh  # CUDA 12.6: también GPU Pascal y Volta (GTX 10xx, Titan X/Xp, V100)
#   TORCH=none ./scripts/setup.sh   # sin modelos de lenguaje
#   EMBER=1    ./scripts/setup.sh   # además, .venv-ember para las características EMBER
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-.venv}
TORCH=${TORCH:-cpu}
PY=${PYTHON:-python3}

command -v cargo >/dev/null || { echo "falta Rust: https://rustup.rs"; exit 1; }
cargo build --release

[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install -q -U pip
pip install -q -r python/requirements.txt maturin pyzipper
case "$TORCH" in
  cpu)  pip install -q torch --index-url https://download.pytorch.org/whl/cpu ;;
  cuda) pip install -q torch ;;
  cu126) pip install -q torch --index-url https://download.pytorch.org/whl/cu126 ;;
  none) ;;
  *) echo "TORCH debe ser cpu, cuda, cu126 o none"; exit 1 ;;
esac
if [ "$TORCH" != none ]; then
  pip install -q -r python/requirements-lm.txt transformers tokenizers
fi
maturin develop --release -m crates/ncd-py/Cargo.toml

cargo test --release
python python/smoke_test.py
if [ "$TORCH" != none ]; then python python/test_lmcodec.py; fi

if [ "${EMBER:-0}" = 1 ]; then
  # thrember fija dependencias que chocan con las del proyecto: va en su propio entorno.
  [ -d .venv-ember ] || "$PY" -m venv .venv-ember
  .venv-ember/bin/pip install -q -U pip
  .venv-ember/bin/pip install -q "thrember @ git+https://github.com/FutureComputing4AI/EMBER2024.git" "signify<0.8" \
      "oscrypto @ git+https://github.com/wbond/oscrypto.git@76bfdcb7285d856c44f8d0fbafe0ecca1c6d2f8b" pandas
fi

echo
echo "listo. Activa el entorno con: . $VENV/bin/activate"
echo "binario: target/release/ncd"
