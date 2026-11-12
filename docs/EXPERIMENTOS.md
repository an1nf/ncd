# Experimentos

Los scripts de `scripts/` saltan los pasos cuya salida ya existe; se pueden interrumpir y relanzar.
Los datos van fuera del repositorio (`~/ncd-data`); `results/` y `models/` están en `.gitignore`.

| experimento | orden | datos | tiempo (2 núcleos) |
|---|---|---|---|
| sintético | `./scripts/demo.sh` | ninguno | 3 min |
| PE benignos | `SOURCE=pypi ./scripts/malware.sh` | PyPI, 165 MB | 5 min |
| malware | `SOURCE=bazaar ./scripts/malware.sh` | MalwareBazaar | 10 a 30 min |
| tráfico | `CSV_DIR=... ./scripts/traffic.sh` | CIC-IDS2017, 1 GB | 15 min a 1 h |

## Instalación

```bash
cargo build --release
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # cu126 para Pascal/Volta
pip install -e 'python[lm,fetch]' maturin
maturin develop --release -m crates/ncd-py/Cargo.toml
export PATH="$PWD/target/release:$PATH"
cargo test --release && python python/tests/smoke.py && python python/tests/test_lm.py
```

EMBER v3 (línea base de malware) fija dependencias incompatibles; va en su propio entorno, que
`malware.sh` detecta:

```bash
python -m venv .venv-ember
.venv-ember/bin/pip install pandas "signify<0.8" \
    "thrember @ git+https://github.com/FutureComputing4AI/EMBER2024.git" \
    "oscrypto @ git+https://github.com/wbond/oscrypto.git@76bfdcb7285d856c44f8d0fbafe0ecca1c6d2f8b"
```

## Familias de PE

Particiones (`ncdeval.split`): `random` estratificada; `temporal` con un corte global; `temporal-family`
con el 30 % más reciente de cada familia. MalwareBazaar devuelve las muestras más recientes por
firma, así que cada familia cubre otra ventana de tiempo y el corte global queda desbalanceado:
el resultado principal es `temporal-family`.

```bash
SOURCE=pypi ./scripts/malware.sh                               # ensayo sin malware
ABUSECH_AUTH_KEY=... SOURCE=bazaar ./scripts/malware.sh        # en una VM aislada
FAMILIES="AgentTesla Formbook" PER_FAMILY=60 SOURCE=bazaar ./scripts/malware.sh
```

Las muestras se guardan sin extensión y de solo lectura, nunca se ejecutan, y el script rechaza
`DATA` dentro del repositorio. La Auth-Key es gratuita en auth.abuse.ch y no debe quedar en logs
ni en el historial de la shell.

Evasión por adición: `DONOR_DIR` con código benigno mide relleno genérico; con otra familia del
corpus mide imitación.

```bash
DONOR_DIR=~/ncd-data/pypi/samples SOURCE=bazaar ./scripts/malware.sh
python -m ncdeval.evasion ~/ncd-data/bazaar/corpus_random --split ~/ncd-data/bazaar/splits_random.csv \
    --donor ~/ncd-data/bazaar/corpus_random/quasarrat --fractions 0 0.25 0.5 1 \
    --codec zstd:9 --method conditional --out results/bazaar/mimicry.json
```

BODMAS: características y metadatos públicos, binarios a petición de los autores.

```bash
python -m ncdeval.split --meta bodmas_metadata.csv --id-col sha --label-col family --time-col timestamp \
    --protocol temporal-family --min-per-family 20 --max-per-family 60 \
    --samples-dir ~/ncd-data/bodmas/samples --link-to ~/ncd-data/bodmas/corpus --out ~/ncd-data/bodmas/splits.csv
ncd eval ~/ncd-data/bodmas/corpus --split-file ~/ncd-data/bodmas/splits.csv -k 1 -c zstd:9 -m conditional \
    --json results/bodmas/ncd.json
python -m ncdeval.baseline_malware --npz bodmas.npz --index bodmas_metadata.csv \
    --split ~/ncd-data/bodmas/splits.csv --out-dir results/bodmas
```

## Tráfico

`GeneratedLabelledFlows.zip` de https://www.unb.ca/cic/datasets/ids-2017.html. Las CSV de
`MachineLearningCSV.zip` no sirven: no tienen `Source IP` ni `Timestamp`.

```bash
CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/traffic.sh
DAYS="Monday Wednesday Friday" CSV_DIR=... ./scripts/traffic.sh      # menos RAM
LM=1 DEVICE=cuda LM_SIZE=b2.8m CSV_DIR=... ./scripts/traffic.sh      # modelo por bytes
```

| carpeta | protocolo |
|---|---|
| `results/cic/flows64` | A: anomalías, lunes benigno como referencia |
| `results/cic/flows64_noport` | A sin puerto de destino (atajo conocido) |
| `results/cic/flows12` | A con ventanas de 12 flujos, zstd frente al modelo de lenguaje |
| `results/cic/cls64` | B: clasificación, partición aleatoria (optimista) |

Una ventana de W flujos ocupa W × 79 bytes. La NCD con modelo necesita dos ventanas y un byte en
el contexto: 12 flujos caben en 2048. El ancho de ventana es el parámetro de mayor efecto
(`python -m ncdeval.ablation` sobre tráfico fabricado).

## Modelos de lenguaje

`ncdeval.lm.train` entrena transformers por bytes sobre enwik8, un archivo o las ventanas
benignas de un directorio de flujos. Cualquier módulo acepta `--codec lm:model.pt` o
`lm:hf:EleutherAI/pythia-70m`; `LMCODEC_DEVICE` elige el dispositivo.

El estudio de texto (AG News, 35 compresores, Pythia 70M a 1B) está en la etiqueta
`text-experiment`: `git checkout text-experiment` y `docs/COMPRESION_LM.md`.

## Coste

Una compresión por par; pares = prueba × entrenamiento.

| objeto | códec | ms por par |
|---|---|---:|
| ventana de 64 flujos | zstd:3 concat | 0.03 |
| | zstd:9 conditional | 0.13 |
| sección de código PE | zstd:3 concat | 1.6 |
| | zstd:9 | 6.5 |
| | zstd:19 conditional | 70 |

8 familias × 40 muestras son unos 21 500 pares: 35 s con zstd:3, 25 min con zstd:19.

## Errores comunes

| síntoma | causa |
|---|---|
| `no kernel image is available` | wheel de torch sin la arquitectura de la GPU; usar el índice `cu126` |
| `... exceed the N-token context` | objetos más largos que el contexto; ventanas más cortas o `--block` mayor |
| `LibraryNotFoundError: libcrypto` | oscrypto de PyPI con OpenSSL ≥ 3.0.10; usar el commit indicado |
| `cannot import name 'SignedPEFile'` | `signify>=0.8`; instalar `signify<0.8` |
| `compression type 99` | zip AES sin `pyzipper` |
| `split file: 0 train / 0 test` | los nombres de archivo no coinciden con la columna de id |
| `warning: ... exceed the ... window` | DEFLATE o bzip2; usar zstd o xz |
| `no CSV for ... in ...` | ruta con espacios o barra final; entrecomillar `CSV_DIR` |
