# Ejecución con datos reales

Guía para pasar del corpus sintético a binarios y tráfico reales. Los tiempos son medidos en 2 núcleos; con más núcleos escalan casi linealmente.

## 0. Preparación

```bash
cargo build --release
python -m venv .venv && . .venv/bin/activate
pip install -r python/requirements.txt maturin pyzipper
maturin develop --release -m crates/ncd-py/Cargo.toml
export PATH="$PWD/target/release:$PATH"

cargo test --release            # 25 pruebas
python python/smoke_test.py     # pipeline completo con datos generados
```

Las características EMBER (línea base de malware) usan `thrember`, que fija dependencias pesadas. Conviene un entorno aparte:

```bash
python -m venv .venv-ember && . .venv-ember/bin/activate
pip install "EMBER2024 @ git+https://github.com/FutureComputing4AI/EMBER2024.git" "signify<0.8" \
    "oscrypto @ git+https://github.com/wbond/oscrypto.git@76bfdcb7285d856c44f8d0fbafe0ecca1c6d2f8b"
```

`signify>=0.8` eliminó `SignedPEFile`, que thrember importa. La versión de oscrypto en PyPI no reconoce OpenSSL 3.0.10 o posterior y falla con `LibraryNotFoundError`.

Los datos van en `data/`, que `.gitignore` excluye junto con cualquier binario.

## 1. Ensayo con binarios reales benignos

Antes de tocar malware, el mismo protocolo se puede correr con DLL y PYD de paquetes de PyPI para Windows. La familia es el proyecto y la fecha es la de publicación, así que hay variantes en el tiempo igual que en una familia de malware. No requiere registro ni una VM.

```bash
python python/fetch_pypi_pe.py --until 2026-10-31 --out data/pypi        # ~165 MB de wheels, ~1.5 min
python python/make_split.py --meta data/pypi/metadata.csv --id-col sha256 --label-col family \
    --time-col first_seen --protocol temporal --test-frac 0.3 --min-per-family 5 \
    --samples-dir data/pypi/samples --link-to data/pypi/corpus --out data/pypi/splits.csv
ncd eval data/pypi/corpus --split-file data/pypi/splits.csv -k 1 -c zstd:3 -m concat \
    --json results/pypi/ncd_zstd3_concat.json
ncd eval data/pypi/corpus --split-file data/pypi/splits.csv -k 1 -c zstd:9 -m conditional -v whole \
    --json results/pypi/ncd_zstd9_cond_whole.json
.venv-ember/bin/python python/ember_features.py data/pypi/samples --out data/pypi/ember
python python/baseline_malware.py --features data/pypi/ember.npy --index data/pypi/ember.index.csv \
    --split data/pypi/splits.csv --k 1 --out-dir results/pypi
python python/compare.py results/pypi --out results/pypi/summary.md
ncd matrix data/pypi/corpus -o results/pypi/ncd.npy -c zstd:3
python python/plot_matrix.py results/pypi/ncd.npy --title "PyPI PE"
```

Qué revisar en la salida de `ncd eval`:

- `formats {"PE": 125}; 0 fell back to whole file`: todos los archivos se analizaron como PE y se extrajo su sección de código.
- No debe aparecer `warning: ... exceed the ... window`. Con zstd la ventana es de 128 MiB.

Resultado esperado (85 de entrenamiento, 40 de prueba): GBDT sobre EMBER 0.956 de macro-F1, NCD entre 0.74 y 0.86 según el códec, k-NN euclídeo sobre EMBER 0.615. La tabla completa está en el README.

## 2. Malware de MalwareBazaar

### Seguridad

- Trabaja en una VM sin carpetas compartidas ni portapapeles compartido, y quita la red después de descargar.
- `fetch_bazaar.py` guarda cada muestra sin extensión, de solo lectura y con su SHA-256 como nombre. El proyecto nunca ejecuta ni mapea las muestras, pero un doble clic sí podría.
- El antivirus del host borra o pone en cuarentena las muestras. Hazlo dentro de la VM o excluye solo esa carpeta.
- Nunca dentro del repositorio.

### Descarga

La API exige una Auth-Key gratuita de https://auth.abuse.ch/. Elige familias con muchas muestras PE; verifica el nombre exacto de la firma en bazaar.abuse.ch (búsqueda `signature:`).

```bash
export ABUSECH_AUTH_KEY=...
python python/fetch_bazaar.py --out data/bazaar --per-family 40 \
    --families AgentTesla Formbook RemcosRAT AsyncRAT SnakeKeylogger RedLineStealer LummaStealer QuasarRAT
```

Por familia toma `--per-family` muestras repartidas a lo largo de su historial (`first_seen`), para que la partición temporal tenga sentido. Si se corta, vuelve a ejecutarlo: no repite lo ya descargado. La API tiene un límite diario de descargas.

### Evaluación

```bash
python python/make_split.py --meta data/bazaar/metadata.csv --id-col sha256 --label-col family \
    --time-col first_seen --protocol temporal --min-per-family 20 \
    --samples-dir data/bazaar/samples --link-to data/bazaar/corpus --out data/bazaar/splits.csv
ncd eval data/bazaar/corpus --split-file data/bazaar/splits.csv -k 1 -c zstd:9 -m conditional \
    --json results/bazaar/ncd_zstd9_cond_code.json
ncd eval data/bazaar/corpus --split-file data/bazaar/splits.csv -k 1 -c zstd:9 -m conditional -v whole \
    --json results/bazaar/ncd_zstd9_cond_whole.json
.venv-ember/bin/python python/ember_features.py data/bazaar/samples --out data/bazaar/ember
python python/baseline_malware.py --features data/bazaar/ember.npy --index data/bazaar/ember.index.csv \
    --split data/bazaar/splits.csv --k 1 --out-dir results/bazaar
python python/compare.py results/bazaar --out results/bazaar/summary.md
```

Repite también con `--protocol random` en `make_split.py`. La diferencia entre la partición aleatoria y la temporal mide cuánto se degrada cada modelo con muestras más nuevas que las de entrenamiento.

Puntos a vigilar:

- Muchas de estas familias son .NET: la sección `.text` contiene IL y metadatos, no código nativo. Compara `-v code` con `-v whole`.
- Un `fell back` alto indica archivos que no son PE válidos o están dañados.
- Si varias familias comparten empaquetador (crypter), la NCD agrupa por empaquetador y no por familia. La matriz de confusión lo delata: errores concentrados entre pares de familias concretos.

### BODMAS

BODMAS trae metadatos con familia y fecha, y características EMBER (`bodmas.npz`) para las mismas muestras. Los binarios se piden a los autores; consulta los términos en la página del conjunto. Comprueba los nombres de columna con `head -1 bodmas_metadata.csv` y pásalos a `make_split.py`. La línea base usa `--npz bodmas.npz --index bodmas_metadata.csv` en lugar de `ember_features.py`.

## 3. Tráfico: CIC-IDS2017

### Datos

Descarga desde http://cicresearch.ca/CICDataset/CIC-IDS-2017/ el archivo **GeneratedLabelledFlows.zip** (carpeta `TrafficLabelling/`). `MachineLearningCSV.zip` no sirve: no incluye `Source IP` ni `Timestamp`, y sin ellos no se pueden formar ventanas por host.

| día | contenido |
|---|---|
| lunes | solo tráfico benigno |
| martes | FTP-Patator, SSH-Patator |
| miércoles | DoS (slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed |
| jueves | ataques web (fuerza bruta, XSS, inyección SQL), infiltración |
| viernes | botnet Ares, escaneo de puertos, DDoS |

Los CSV tienen espacios al inicio de los nombres de columna, columnas duplicadas, valores `Infinity` y, en el archivo de ataques web del jueves, miles de filas vacías al final. `prepare_flows.py` maneja todo eso. En varios archivos `Timestamp` tiene resolución de minutos y no ordena los flujos dentro de un mismo minuto; usa `--order rows` (orden del CSV).

### Protocolo A: anomalías con el lunes como referencia

El lunes es todo benigno: sus ventanas son las referencias y el resto de la semana es la prueba. Ningún ataque se usa para ajustar el detector. La métrica es ROC AUC.

```bash
python python/prepare_flows.py data/cic/TrafficLabelling/*.csv --out data/cic/flows64 \
    --window 64 --order rows --split file --train-files Monday --max-per-label 3000
python python/baseline_flows.py data/cic/flows64 --out-dir results/cic
python python/eval_flows_ncd.py data/cic/flows64 --codec zstd:3 --max-refs 1000 --out-dir results/cic
python python/compare.py results/cic --out results/cic/summary.md     # tabla + roc.png
```

Con una sola clase en entrenamiento, los modelos supervisados se omiten y solo se reportan Isolation Forest, distancia k-NN euclídea y NCD.

### Protocolo B: clasificación de ataques

```bash
python python/prepare_flows.py data/cic/TrafficLabelling/*.csv --out data/cic/flows64r \
    --window 64 --order rows --split random --max-per-label 1000
python python/baseline_flows.py data/cic/flows64r --out-dir results/cic_cls
python python/eval_flows_ncd.py data/cic/flows64r --codec zstd:3 --max-refs 1000 --out-dir results/cic_cls
```

Es optimista: ventanas contiguas de una misma sesión de ataque caen a ambos lados. Repórtalo junto al protocolo A, no en su lugar.

### Variantes que conviene medir

- `--drop-port`: el puerto de destino identifica varios ataques por sí solo. Si el resultado cae mucho sin él, el modelo usaba ese atajo.
- `--window 32` y `--window 128`: el tamaño de ventana es el parámetro con más efecto en la NCD (ver `python/ablation_window.py`).
- LYCOS-IDS2017 o CSE-CIC-IDS2018: mismas órdenes. Si las columnas de IP de origen o de tiempo tienen otro nombre, añádelo a `SRC_COLS` o `TS_COLS` en `prepare_flows.py`.

`prepare_flows.py` carga todos los CSV en memoria: la semana completa necesita del orden de 8 GB de RAM. Con menos, procesa menos días.

## 4. Coste

La NCD calcula una compresión por par. Pares = consultas × referencias (`ncd eval` con partición: prueba × entrenamiento).

| objeto | códec y método | ms por par (2 núcleos) |
|---|---|---:|
| ventana de 64 flujos (~5 KB) | zstd:3 concat | 0.03 |
| | zstd:9 conditional | 0.13 |
| | zstd:9 concat | 0.19 |
| | zstd:19 concat | 0.90 |
| sección de código PE (~300 KB) | zstd:3 concat | 1.6 |
| | zstd:9 concat o conditional | 6.5 |
| | zstd:19 conditional | 70 |

Ejemplos:

- 8 familias × 40 muestras, 30 % de prueba: 96 × 224 ≈ 21 500 pares. zstd:3 tarda unos 35 s; zstd:9, 2.5 min; zstd:19, 25 min.
- CIC-IDS2017, protocolo A con 15 000 ventanas de prueba y 1000 referencias: 15 M de pares. zstd:3 tarda unos 8 min; zstd:9, 50 min.

Empieza con zstd:3 y sube de nivel solo si la diferencia lo justifica. En el ensayo con PyPI, zstd:19 no superó a zstd:3.

## 5. Qué reportar

Por cada conjunto, una tabla de `compare.py` con:

- NCD (códec, método, vista), GBDT y k-NN euclídeo sobre las mismas muestras y la misma partición.
- Macro-F1 para familias o ataques y ROC AUC para anomalías, con el tamaño de prueba (con 40 muestras, una sola es 2.5 puntos).
- Partición temporal o por archivo, y la aleatoria solo como referencia.
- Tiempo de cómputo, y las advertencias de ventana o de `fell back` si aparecieron.

## 6. Errores comunes

| mensaje | causa y solución |
|---|---|
| `LibraryNotFoundError: Error detecting the version of libcrypto` | oscrypto de PyPI con OpenSSL ≥ 3.0.10; instala el commit indicado en la sección 0 |
| `cannot import name 'SignedPEFile'` | `signify>=0.8`; instala `signify<0.8` |
| `401` o `missing Auth-Key` en `fetch_bazaar.py` | falta la Auth-Key de abuse.ch |
| `NotImplementedError: compression type 99` | zip AES sin pyzipper; `pip install pyzipper` |
| `no samples found under ...` | ruta del corpus equivocada; el corpus de `make_split.py --link-to` contiene enlaces simbólicos |
| `split file: 0 train / 0 test` | los nombres de archivo no coinciden con la columna `id` o `sha256` del CSV de partición |
| `--train-files must match some files but not all` | el patrón de `--train-files` no aparece en ningún nombre de archivo, o aparece en todos |
| `warning: ... exceed the ... window` | códec con ventana pequeña (DEFLATE, bzip2); usa zstd o xz |
