# Pasos: instalación, datos, entrenamiento y experimentos

Guía operativa de todo el proyecto. Cada bloque se puede copiar tal cual. Los scripts de `scripts/` se saltan los pasos cuya salida ya existe, así que se pueden interrumpir y volver a lanzar.

| bloque | script | datos externos | tiempo en 2 núcleos | notas |
|---|---|---|---|---|
| instalación y pruebas | `scripts/setup.sh` | PyPI, crates.io | 5 a 10 min | |
| experimento de texto | `scripts/download_text_data.sh`, `scripts/text_experiment.sh` | AG News, enwik8, GPT-2 (≈ 700 MB) | ≈ 2 h en CPU | GPU opcional |
| modelos de HuggingFace | `HF_MODELS=... scripts/text_experiment.sh` | HuggingFace | minutos con GPU, horas en CPU | GPU recomendada |
| familias de PE, ensayo | `SOURCE=pypi scripts/malware_families.sh` | PyPI (≈ 165 MB) | ≈ 5 min | sin riesgo |
| familias de malware | `SOURCE=bazaar scripts/malware_families.sh` | MalwareBazaar | 10 a 30 min | Auth-Key y VM aislada |
| tráfico | `scripts/cic_ids2017.sh` | CIC-IDS2017 (≈ 1 GB de CSV) | 15 min a 1 h | 8 GB de RAM para la semana completa |

Los datos van fuera del repositorio (por defecto `~/ncd-data`) o en `data/`, `models/` y `results/`, que `.gitignore` excluye.

## 1. Instalación

Requisitos: Rust estable (https://rustup.rs), Python 3.10 o posterior, `curl`, `unzip` y git.

```bash
unzip ncd.zip && cd ncd
TORCH=cpu ./scripts/setup.sh        # TORCH=cuda con GPU NVIDIA Turing o posterior; TORCH=cu126 para Pascal/Volta
                                    # (GTX 10xx, Titan X/Xp, V100); TORCH=none sin modelos de lenguaje
. .venv/bin/activate
export PATH="$PWD/target/release:$PATH"
```

`setup.sh` compila el binario `ncd`, crea `.venv`, instala las dependencias y los bindings `ncdthreat`, y corre las pruebas:

- `cargo test`: el núcleo en Rust;
- `python/smoke_test.py`: todos los scripts con datos fabricados;
- `python/test_lmcodec.py`: el codificador aritmético, el códec con modelo, los estimadores y el adaptador de HuggingFace, con modelos diminutos aleatorios.

Para las características EMBER de la línea base de malware hace falta un segundo entorno, porque `thrember` fija versiones que chocan con las del proyecto:

```bash
EMBER=1 TORCH=cpu ./scripts/setup.sh      # crea también .venv-ember
```

Con Nix, `nix develop` da Rust y Python; luego `pip install -r python/requirements.txt -r python/requirements-lm.txt` dentro del shell.

## 2. Experimento de texto: modelos de lenguaje como compresores

Resultados y método en [COMPRESION_LM.md](COMPRESION_LM.md).

### 2.1 Reproducir lo que ya está

```bash
./scripts/download_text_data.sh                 # data/text, verifica SHA-256
./scripts/text_experiment.sh                    # tarea, GPT-2, 11 modelos por bytes, 31 compresores, análisis
```

Salidas:

- `results/text/analysis/text_summary.md`: la tabla completa, las correlaciones y las comparaciones pareadas;
- `results/text/figs/`: las tres figuras;
- `models/`: los modelos entrenados.

Con la misma semilla, la partición y los compresores clásicos dan exactamente los números del documento. Los modelos entrenados pueden variar en las últimas cifras según el hardware.

Para probar el flujo en un par de minutos:

```bash
FAMILIES=agnews SIZES=b037k TRAIN_BYTES=3e5 CLASSIC="zstd:3 xz:6" GPT2=0 \
    RES=results/text_quick MODELS=models_quick ./scripts/text_experiment.sh
```

### 2.2 Modelos más grandes o más entrenados (GPU)

La pregunta abierta es si un modelo por bytes que sí aprende a copiar (NCD(x,x) bajo sin el modelo de coincidencias) se comporta como GPT-2. Eso requiere más datos y más tamaño:

```bash
DEVICE=cuda SIZES="b2.8m b6.5m b38m" TRAIN_BYTES=200e6 FAMILIES="agnews agpair" \
    MODELS=models_big RES=results/text_big ./scripts/text_experiment.sh
```

Lo primero que hay que mirar es la columna `NCD(x,x)` de `text_summary.md`. Si baja de 0.9 hacia 0.1, el modelo empezó a usar el contexto. Entrenar un modelo suelto:

```bash
python python/train_byte_lm.py --corpus file --data data/text/text_task/lm_train.txt \
    --valid data/text/text_task/lm_valid.txt --out models/agnews/b38m.pt \
    --n-layer 12 --n-embd 512 --n-head 8 --lr 6e-4 --train-bytes 200e6 --device cuda
```

### 2.3 Familia preentrenada de varios tamaños (paso siguiente 1)

Requiere acceso a huggingface.co. El adaptador `hf:` usa el tokenizador de cada modelo y corre en float32. Con `LMCODEC_HF_DTYPE=bfloat16` usa bf16 en GPU.

```bash
pip install transformers tokenizers          # ya instalados por setup.sh si TORCH != none
HF_MODELS="EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b" \
    FAMILIES="" DEVICE=cuda ./scripts/text_experiment.sh
# o la familia GPT-2:
HF_MODELS="openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large openai-community/gpt2-xl" \
    FAMILIES="" GPT2=0 DEVICE=cuda RES=results/text_gpt2 ./scripts/text_experiment.sh
```

`FAMILIES=""` reutiliza los resultados ya calculados en `RES` y solo añade los nuevos modelos. `analyze_text.py` agrupa los modelos de HuggingFace por el prefijo del nombre (`pythia`, `gpt2`) y calcula las correlaciones dentro de esa familia. Lo que hay que mirar:

1. si, dentro de la familia, la exactitud sube cuando bajan los bits por carácter (Spearman negativo con `bpc_doc` y `bpc_stream`);
2. si `class d'` sigue prediciendo mejor que los bits por carácter.

Coste orientativo de la matriz de 300 × 64 en CPU: unos 22 minutos para 124M parámetros, y el adaptador de HuggingFace cuesta el doble porque no reutiliza la caché. Por encima de 400M parámetros, usa GPU.

### 2.4 Usar un modelo como compresor en otros scripts

```bash
python python/eval_flows_ncd.py flows/ --codec lm:models/flows64/b2.8m.pt --max-refs 200
LMCODEC_DEVICE=cuda python python/eval_flows_ncd.py flows/ --codec lm:models/flows64/b2.8m.pt   # en GPU
```

```python
import ncd_backend as nt                    # misma interfaz que ncdthreat
D = nt.cross(queries, refs, codec="lm:models/agnews/b2.8m.pt", method="concat")
```

Límite: un modelo solo ve su contexto (512 bytes en los modelos por bytes de este experimento). Para la NCD, el contexto debe caber dos objetos y un símbolo inicial. Sirve para textos cortos y ventanas de flujos, no para binarios de cientos de KB.

## 3. Familias de malware (PE)

### 3.1 Ensayo con binarios benignos de PyPI (sin riesgo)

```bash
SOURCE=pypi EMBER=auto ./scripts/malware_families.sh
```

Descarga 125 DLL y PYD de 8 proyectos y hace particiones temporal y aleatoria. Corre tres configuraciones de NCD: zstd:3 concat sobre el código, zstd:9 condicional sobre el código y zstd:9 condicional sobre el archivo completo. Si existe `.venv-ember`, añade EMBER con GBDT y k-NN. Resultado esperado en la partición temporal: 0.800 de exactitud para zstd:3 sobre el código y para zstd:9 sobre el archivo completo, y 0.700 para zstd:9 sobre el código, como en el README. Está verificado en este entorno.

### 3.2 MalwareBazaar

Seguridad, antes de nada:

- Trabaja en una VM sin carpetas ni portapapeles compartidos, y quita la red al terminar de descargar.
- El script se niega a guardar muestras de `bazaar` dentro del repositorio.
- Las muestras se guardan sin extensión, de solo lectura, y el proyecto nunca las ejecuta.

```bash
export ABUSECH_AUTH_KEY=...                  # gratis en https://auth.abuse.ch/
SOURCE=bazaar DATA=~/ncd-data/bazaar ./scripts/malware_families.sh
# otras familias o tamaños:
FAMILIES="AgentTesla Formbook RemcosRAT AsyncRAT" PER_FAMILY=60 SOURCE=bazaar ./scripts/malware_families.sh
```

Verifica el nombre exacto de cada firma en bazaar.abuse.ch (búsqueda `signature:`). Si la descarga se corta por el límite diario de la API, vuelve a lanzar: no repite lo ya descargado.

Qué mirar en `results/bazaar/{temporal,random}/summary.md`:

- **Temporal por familia frente a aleatoria.** La caída de la partición aleatoria a `temporal-family` mide cuánto se degrada cada método con muestras nuevas. Es el resultado principal. La partición `temporal` global se incluye para comparar, pero con MalwareBazaar queda desbalanceada (ver DATOS_REALES).
- **Vista `code` frente a `whole`.** Muchas de estas familias son .NET, y su `.text` contiene IL y metadatos.
- **Errores concentrados entre pares de familias** en la matriz de confusión del JSON. Suelen indicar un empaquetador compartido: la NCD agrupa por empaquetador.
- **`fell back` alto** en la salida de `ncd eval`: archivos que no son PE válidos.

### 3.3 Evasión con código benigno añadido (pendiente del README)

```bash
BENIGN_DIR=~/ncd-data/pypi/samples SOURCE=bazaar ./scripts/malware_families.sh
# o directamente:
python python/evasion_append.py ~/ncd-data/bazaar/corpus_temporal --split ~/ncd-data/bazaar/splits_temporal.csv \
    --benign ~/ncd-data/pypi/samples --fractions 0 0.1 0.25 0.5 1 2 --codec zstd:9 --method conditional \
    --out results/bazaar/evasion_temporal.json
```

El script añade a cada muestra de prueba código benigno con una fracción f de su tamaño y la vuelve a clasificar. Con f = 0 debe salir la misma exactitud que `ncd eval`. Como prueba de funcionamiento con PyPI (zstd:3, relleno tomado del propio corpus), añadir la mitad del tamaño bajó la exactitud de 0.80 a 0.48. No es una medida de evasión, porque el relleno incluía archivos de las mismas familias. Los modelos de HuggingFace (`lm:hf:...`) son para texto UTF-8, no para bytes binarios.

Imitación de otra familia (más dura que el relleno benigno):

```bash
python python/evasion_append.py ~/ncd-data/bazaar/corpus_random --split ~/ncd-data/bazaar/splits_random.csv \
    --benign ~/ncd-data/bazaar/corpus_random/quasarrat --fractions 0 0.1 0.25 0.5 1 \
    --codec zstd:9 --method conditional --out results/bazaar/mimicry_quasarrat.json
```

### 3.4 BODMAS

En https://whyisyoung.github.io/BODMAS/ las características (`bodmas.npz`) y los metadatos se descargan de Google Drive. Los binarios de malware se piden por correo a los autores; los benignos no se distribuyen. Con los binarios en `~/ncd-data/bodmas/samples/<sha256>`:

```bash
head -1 bodmas_metadata.csv                  # comprueba los nombres de columna
python python/make_split.py --meta bodmas_metadata.csv --id-col sha --label-col family --time-col timestamp \
    --protocol temporal --min-per-family 20 --max-per-family 60 \
    --samples-dir ~/ncd-data/bodmas/samples --link-to ~/ncd-data/bodmas/corpus --out ~/ncd-data/bodmas/splits.csv
ncd eval ~/ncd-data/bodmas/corpus --split-file ~/ncd-data/bodmas/splits.csv -k 1 -c zstd:9 -m conditional \
    --json results/bodmas/ncd_zstd9_cond_code.json
python python/baseline_malware.py --npz bodmas.npz --index bodmas_metadata.csv --split ~/ncd-data/bodmas/splits.csv \
    --k 1 --out-dir results/bodmas
python python/compare.py results/bodmas --out results/bodmas/summary.md
```

`--max-per-family` acota el coste cuadrático.

## 4. Tráfico: CIC-IDS2017

### 4.1 Datos

Descarga `GeneratedLabelledFlows.zip` desde la página oficial, https://www.unb.ca/cic/datasets/ids-2017.html (enlace "Download this dataset"), o desde http://cicresearch.ca/CICDataset/CIC-IDS-2017/. `MachineLearningCSV.zip` no sirve: le faltan `Source IP` y `Timestamp`.

```bash
mkdir -p ~/ncd-data/cic && cd ~/ncd-data/cic && unzip GeneratedLabelledFlows.zip && ls TrafficLabelling/
```

### 4.2 Protocolos A y B

```bash
CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/cic_ids2017.sh
# con menos RAM, menos días:
DAYS="Monday Wednesday Friday" CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/cic_ids2017.sh
# ablación del tamaño de ventana:
ABLATION=1 CSV_DIR=... ./scripts/cic_ids2017.sh
```

Salidas en `results/cic/`:

- `flows64/` y `flows64_noport/`: protocolo A, AUC de anomalías con el lunes como referencia, con y sin el puerto de destino;
- `cls64/`: protocolo B, clasificación con partición aleatoria, que es optimista;
- `roc.png` en cada carpeta.

Si el AUC cae mucho sin el puerto, el modelo usaba ese atajo.

### 4.3 Modelo de lenguaje sobre tráfico benigno (pasos siguientes 2 y 3)

```bash
LM=1 DEVICE=cuda LM_SIZE=b2.8m LM_BYTES=20e6 CSV_DIR=~/ncd-data/cic/TrafficLabelling ./scripts/cic_ids2017.sh
```

Pasos que hace:

1. `make_lm_corpus.py` junta las ventanas benignas de entrenamiento del lunes.
2. `train_byte_lm.py` entrena con un contexto de 2048 bytes, porque una ventana de 64 flujos ocupa unos 700 bytes.
3. Calcula dos puntuaciones:
   - `eval_flows_surprise.py`: bits por byte de cada ventana, sin referencias y con coste lineal; además, una versión de dos colas (distancia a la mediana benigna en unidades de MAD), porque las inundaciones y los barridos pueden ser *más* predecibles que el tráfico normal;
   - `eval_flows_ncd.py --codec lm:...`: la NCD con el modelo, comparable con zstd.

En datos fabricados, la sorpresa de una cola dio un AUC de 0.01 (invertida) y la de dos colas 0.99. Hay que reportar las dos.

## 5. Orden recomendado de los pasos siguientes

1. [ ] `SOURCE=pypi ./scripts/malware_families.sh`, para comprobar la instalación con datos reales sin riesgo.
2. [ ] MalwareBazaar en una VM, con particiones temporal y aleatoria. Es el resultado principal pendiente del proyecto.
3. [ ] CIC-IDS2017, protocolos A y B, con y sin puerto de destino.
4. [ ] Evasión con `BENIGN_DIR`, para cuantificar el límite que el README da como no medido.
5. [ ] Familia Pythia o GPT-2 con GPU (§2.3): cierra la pregunta abierta del experimento de texto.
6. [ ] Modelo de lenguaje sobre tráfico benigno (§4.3), con sorpresa y NCD, frente a zstd:3.
7. [ ] Coste cuadrático: medoides por familia o un prefiltro LSH antes de la NCD (no implementado).
8. [ ] Fuzzing de la extracción de secciones (`cargo fuzz`, no configurado aún).

Para publicar, por cada conjunto: la tabla de `compare.py`, el tamaño de prueba, el tipo de partición, el tiempo y cualquier aviso de ventana o de `fell back` (§5 de DATOS_REALES).

## 6. Problemas comunes

| síntoma | causa y solución |
|---|---|
| `ValueError: libcublasLt... not found` al importar torch | wheel de PyPI con CUDA sin sus dependencias; usa `TORCH=cpu` (índice CPU de PyTorch) |
| `GPT-2 checkpoints need --tokenizer-dir` | pasa `--tokenizer-dir data/text/tok` o deja `encoder.json` y `vocab.bpe` junto al `.pt` |
| `... exceed the 512-token context` | objetos más largos que el contexto del modelo; córtalos o entrena con `--block` mayor |
| `warning: N pairs exceed the ... context` | el prefijo se recortó; igual que la ventana de DEFLATE, la NCD pierde información |
| `OSError: ... huggingface.co` | sin acceso a HuggingFace; habilítalo o usa los modelos por bytes y GPT-2 de ONNX |
| exactitudes distintas en 1 o 2 puntos con otro script de k-NN | empates en la distancia mínima; `analyze_text.py` los cuenta como desempate al azar |
| `no kernel image is available for execution on the device` | la GPU es más antigua que las que cubre el wheel de PyPI; `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126` |
| `No matching distribution found for ember2024` | el paquete se llama `thrember`; vuelve a correr `EMBER=1 ./scripts/setup.sh` con la versión actual del script |
| `TimeoutError` en `fetch_pypi_pe.py` | conexión lenta; vuelve a lanzar: reintenta y guarda las wheels ya bajadas en `wheels/` |
| `CUDA out of memory` | baja `batch_tokens` (`LMCodec(..., batch_tokens=1024)`) o usa `LMCODEC_HF_DTYPE=bfloat16` |
| errores de EMBER, oscrypto o signify | ver la tabla de §6 en [DATOS_REALES.md](DATOS_REALES.md) |
