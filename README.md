# ncd-threat

**Compression-based threat classification in Rust.** Normalized Compression
Distance (NCD) — a computable upper-bound approximation of the normalized
information distance from algorithmic information theory — applied to
malware-family classification and network-anomaly detection, benchmarked
sample-for-sample against classical machine-learning baselines.

```
            ┌──────────── ncd-core (Rust) ─────────────┐
 binaries ─▶│ extract: PE/ELF code sections, defensive │
 flows ────▶│ codec:   zstd · xz · bzip2 · deflate      │──▶ distance matrices (.npy)
            │ ncd:     cached, parallel (rayon),        │──▶ k-NN / anomaly scores
            │          concat | conditional (zstd dict) │──▶ metrics (acc, F1, ROC AUC)
            └────────┬──────────────────────┬──────────┘
                ncd (CLI)          ncdthreat (PyO3, zero-copy, GIL released)
                                            │
                    python/: splits · EMBER/flow baselines · flow windowing · plots
```

## Why compression?

The Kolmogorov complexity `K(x)` is the length of the shortest program that
outputs `x`. The **normalized information distance**

```
NID(x, y) = max{K(x|y), K(y|x)} / max{K(x), K(y)}
```

minorizes every admissible normalized distance: whatever feature two objects
share, NID "sees" it (Li, Chen, Li, Ma & Vitányi, 2004). `K` is uncomputable,
but any real compressor gives an upper bound `C(x) ≥ K(x)` (up to a constant),
and `C(xy) − C(x)` approximates `K(y|x)`. This gives the **NCD** (Cilibrasi &
Vitányi, 2005):

```
NCD(x, y) = (C(xy) − min{C(x), C(y)}) / max{C(x), C(y)}
```

For threat detection this is attractive: no hand-engineered features, no
parser per file format, and similarity comes from shared *structure* —
reused code, a repeated attack loop — rather than from statistics someone
decided to extract. It is also fragile in ways worth measuring, which is what this project does:
real compressors violate the metric axioms, have finite windows, and pay a
fixed per-object overhead.

## What is implemented

| Component | Notes |
|---|---|
| **4 compressors** (`codec.rs`) | zstd 1–22 (128 MiB window), xz/LZMA2 0–9, bzip2 1–9, raw DEFLATE 0–9. Lengths counted through a null sink — nothing is allocated for the output. Frame checksums/headers disabled where possible (constant noise for NCD). |
| **Two estimators** (`ncd.rs`) | `concat`: classic `C(xy)` over the upper triangle (+ `concat-sym`, averaging `C(xy)` and `C(yx)`). `conditional`: the simpler object of each pair becomes a **prepared zstd raw-content dictionary** and the more complex one is compressed against it, estimating `max{K(x\|y), K(y\|x)}` — the NID numerator — directly. Each dictionary is built once and reused across a whole row. |
| **Window diagnostics** | Every run counts the pairs whose `\|x\|+\|y\|` exceeds the codec window (NCD silently saturates to ≈1 there) and warns. |
| **Defensive extraction** (`extract.rs`) | `code` view = executable sections (PE `CNT_CODE`/`MEM_EXECUTE`, ELF `SHF_EXECINSTR`, else `PF_X` segments); `no-overlay`; `whole`. Bounds-checked slicing, parser under `catch_unwind`, 256 MiB size cap, fallback to raw bytes. Tested against truncated, mutated and hostile-offset PEs. Samples are only ever read as bytes. |
| **Classification** (`knn.rs`, `metrics.rs`) | k-NN (leave-one-out or hold-out), anomaly score = mean NCD to the k nearest benign references, accuracy / macro-F1 / confusion matrix / ROC AUC (Mann–Whitney, tie-aware). |
| **CLI** `ncd` | `dist`, `matrix`, `eval`, `score`, `bench`, `synth`. |
| **Python bindings** `ncdthreat` | PyO3 + maturin, abi3 wheel. Bytes borrowed zero-copy (`PyBackedBytes`), GIL released during compression, NumPy `float32` matrices out. |
| **Evaluation pipeline** (`python/`) | Shared train/test split files (random or **temporal**), EMBER-feature baselines (GBDT, Euclidean k-NN), CICFlowMeter → flow-window objects, flow baselines (GBDT, k-NN, Isolation Forest), result aggregation and plots. |
| **Reproducibility** | Nix flake (package, checks, dev shell), GitHub Actions (fmt, clippy `-D warnings`, tests, end-to-end smoke test, `nix flake check`), deterministic synthetic corpus. |

## Quick start

```bash
nix develop                     # or: rustup + python3 with python/requirements.txt
cargo build --release
cargo test --release            # 25 unit/integration tests

# Python bindings
maturin develop --release -m crates/ncd-py/Cargo.toml

# Synthetic demo: benchmark, leave-one-out k-NN, figures in docs/img
./scripts/demo.sh
```

```bash
ncd dist a.exe b.exe -c zstd:19 -m conditional -v code
ncd eval corpus/ -k 1 --split-file splits.csv --json results/malware_ncd.json
ncd score --refs benign_train/ --queries test/ -k 5 -o scores.csv
ncd bench corpus/ --codecs zstd:3,zstd:19,xz:9,bzip2:9,deflate:9
```

```python
import ncdthreat as nt
D = nt.pairwise(objs, codec="zstd:19", method="conditional")   # (n, n) float32
s = nt.anomaly_scores(queries, benign_refs, k=5)                # (m,)  float32
```

A corpus is a directory `ROOT/<label>/<sample>`; the file stem (usually the
SHA-256) is the sample id.

## Engineering findings so far

These come from the synthetic corpus and fabricated traffic (both
reproducible from this repository). They are properties of the *method and
the implementation*, not claims about real-world detection rates.

**1. The conditional estimator is ≈3.4× faster at zstd:19, with the same accuracy.**
Building one dictionary per object and reusing it across the row avoids
recompressing `x` inside every `C(xy)`:

Synthetic corpus, 150 samples (6 families × 20 + 30 benign, 22 KB average),
11,175 pairs, 2 CPU cores (`./scripts/demo.sh`):

| codec | estimator | C(x) MiB/s | mean C(x)/\|x\| | mean NCD(x,x) | pairs over window | matrix time | 1-NN LOO acc |
|---|---|---:|---:|---:|---:|---:|---:|
| zstd:3 | concat | 253.0 | 0.532 | 0.0008 | 0 | 1.3 s | 1.000 |
| zstd:3 | conditional | 239.7 | 0.532 | 0.0015 | 0 | 1.1 s | 1.000 |
| zstd:19 | concat | 7.6 | 0.447 | 0.0009 | 0 | 58.8 s | 1.000 |
| **zstd:19** | **conditional** | 7.6 | 0.447 | 0.0019 | 0 | **17.2 s** | 1.000 |
| xz:9 | concat | 8.0 | 0.444 | 0.0063 | 0 | 57.1 s | 1.000 |
| bzip2:9 | concat | 11.2 | 0.434 | **0.2697** | 0 | 32.9 s | 1.000 |
| deflate:9 | concat | 41.4 | 0.478 | 0.0456 | **11,175 / 11,175** | 11.5 s | 1.000 |

<p align="center">
  <img src="docs/img/synthetic_heatmap.png" width="48%" alt="NCD matrix grouped by label: dark diagonal blocks per family">
  <img src="docs/img/synthetic_mds.png" width="48%" alt="2-D MDS embedding: one tight cluster per family">
</p>

On the synthetic corpus every compressor separates the families perfectly —
it validates the pipeline, nothing more. The informative columns are cost
and the diagnostics.

**2. DEFLATE's 32 KiB window silently breaks NCD.** Above it `C(xy) ≈ C(x) + C(y)`
no matter how similar the objects are: for 64 KiB of random bytes,
`NCD_deflate(x, x) > 0.9` while `NCD_zstd(x, x) < 0.05` (unit test
`deflate_window_pitfall`). The CLI counts and reports these pairs.

**3. bzip2 violates the identity axiom.** Its self-distance `NCD(x, x)` on
incompressible data is ≈0.22, two orders of magnitude above the LZ codecs:
the Burrows–Wheeler transform does not encode a verbatim repeat as a cheap
back-reference.

**4. On traffic, object size decides — and the two estimators fail
differently.** On fabricated CICFlowMeter traffic (`python/ablation_window.py`),
16-flow windows (≈180 bytes) make `concat` k-NN collapse below chance, while
the dictionary estimator still reaches 0.71 macro-F1. From 32 flows up the
picture flips: `concat` reaches perfect k-NN separation and its anomaly AUC
climbs with window size, whereas `conditional` k-NN *degrades* as windows grow
even though its anomaly AUC keeps up. Defaults follow the data: 64-flow
windows and `concat` for traffic, `conditional` for binaries (where it matched
`concat` accuracy at a third of the cost). Why the dictionary estimator loses
neighbour ranking on mid-sized, highly repetitive objects is an open question
worth a closer look (entropy-coder statistics are not transferred by a raw
dictionary, only LZ matches).

| flows/window | bytes/object | k-NN macro-F1 concat | k-NN macro-F1 conditional | anomaly AUC concat | anomaly AUC conditional |
|---:|---:|---:|---:|---:|---:|
| 16 | 176 | 0.106 | **0.705** | 0.764 | 0.776 |
| 32 | 348 | **0.921** | 0.856 | 0.839 | 0.729 |
| 64 | 699 | **1.000** | 0.797 | 0.934 | 0.925 |
| 128 | 1,399 | **1.000** | 0.732 | 0.990 | 1.000 |

(zstd:19, k = 5, 8 fabricated captures, 4 classes; the Euclidean k-NN and
Isolation-Forest baselines on aggregate flow statistics score ≈1.0 on this
easy fabricated traffic — the real comparison needs real traffic.)

**5. Two independent code paths agree.** The anomaly AUC from the Rust CLI
(`ncd score`) and from the Python bindings + scikit-learn matches to four
decimals in the smoke test, and `ncd::pairwise` is cross-checked entry-by-entry
against single-pair calls and against `ncd::cross` in the test-suite.

**6. Memory blow-up in xz, fixed.** xz presets 7–9 allocate a 16–64 MiB
dictionary per call; compressing thousands of small objects spent 11 minutes
in the kernel zeroing memory. Clamping the dictionary to the input size gives
byte-identical output and a **32× speed-up** (366 s → 11 s on an 80-sample
matrix).

## Running on real data

### Malware families

You need **raw binaries with family labels** — NCD cannot run on feature
vectors.

* **BODMAS** (Yang et al., DLS 2021): 57,293 malware + 77,142 benign PE files
  (2019–2020), 581 families, timestamps, EMBER-format feature vectors —
  ideal because the NCD and the EMBER baseline can share the exact same
  samples. Check the dataset page for binary-access terms.
* **EMBER2024** (Joyce et al., 2025): ~3.2 M files, 6,787 families, but the
  original files are *not* redistributed (hashes + features only); samples
  must be fetched separately (VirusTotal).
* **MalwareBazaar**: per-sample downloads with family signatures.

```bash
python python/make_split.py --meta bodmas_metadata.csv --id-col sha --label-col family \
    --time-col timestamp --protocol temporal --min-per-family 20 --max-per-family 60 \
    --samples-dir /data/bodmas/raw --link-to corpus/ --out splits.csv
ncd eval corpus/ --split-file splits.csv -k 1 -m conditional --json results/malware_ncd.json
python python/baseline_malware.py --npz bodmas.npz --index bodmas_metadata.csv \
    --split splits.csv --out-dir results/
python python/compare.py results/
```

The temporal protocol trains on older samples and tests on newer ones
(closed-world: unseen test families are dropped), following TESSERACT
(Pendlebury et al., 2019). Random splits are reported too, but they leak
campaign-level similarity and are optimistic.

**Handling live malware.** Work in an isolated VM without shared folders;
keep samples in password-protected archives at rest and outside this
repository (`.gitignore` refuses binaries); this project never executes or
maps samples, but antivirus software on the host will react to them.

### Network anomalies

```bash
python python/prepare_flows.py MachineLearningCSV/*.csv --out flows/ --window 64
python python/baseline_flows.py flows/ --out-dir results/
python python/eval_flows_ncd.py flows/ --codec zstd:19 --max-refs 2000 --out-dir results/
python python/compare.py results/      # table + ROC curves
```

The anomaly detector is fitted on **benign training windows only**.
CIC-IDS2017 has documented problems — duplicated and miscomputed features,
TCP flows split incorrectly, doubtful labels (Engelen et al., 2021; Rosay et
al., 2022). The pipeline drops the duplicated columns, but a corrected
release such as **LYCOS-IDS2017** is preferable when results are meant to be
reported. `--drop-port` removes the destination port, a known shortcut
feature.

### Results on real data

*To be filled after running the commands above.* Results tables and ROC
curves are generated by `python/compare.py` into `results/summary.md` and
`results/roc.png`.

## Repository layout

```
crates/ncd-core/   library: codecs, NCD matrices, extraction, k-NN, metrics, synth
crates/ncd-cli/    `ncd` binary
crates/ncd-py/     `ncdthreat` Python module (PyO3 + maturin)
python/            splits, baselines, flow windowing, NCD evaluation, plots, smoke test
scripts/demo.sh    synthetic end-to-end demo
flake.nix          package, checks and dev shell
```

## Limitations and next steps

* Pairwise cost is quadratic: ~11 k pairs of 22 KB objects take ~1 min at
  zstd:19 on two cores (17 s with `conditional`). Next: prototype
  selection / medoids per family, and pruning candidates with a cheap
  LSH (ssdeep/TLSH-style) pre-filter before exact NCD.
* Packed samples compress poorly by construction; the `code` view helps only
  when the unpacking stub is not the whole code section.
* NCD is not adversarially robust: appending benign code moves a sample
  toward benign neighbours. Measuring that attack (and padding-based evasion
  in general) is a natural follow-up.
* Fuzzing `extract` with `cargo-fuzz` (the current tests are deterministic
  mutation sweeps).

## References

* M. Li, X. Chen, X. Li, B. Ma, P. Vitányi. *The similarity metric.* IEEE Trans. Inf. Theory 50(12), 2004.
* R. Cilibrasi, P. Vitányi. *Clustering by compression.* IEEE Trans. Inf. Theory 51(4), 2005.
* M. Cebrián, M. Alfonseca, A. Ortega. *Common pitfalls using the normalized compression distance.* Commun. Inf. Syst. 5(4), 2005.
* H. Anderson, P. Roth. *EMBER: An open dataset for training static PE malware machine learning models.* arXiv:1804.04637, 2018.
* L. Yang et al. *BODMAS: An open dataset for learning based temporal analysis of PE malware.* DLS 2021.
* R. J. Joyce et al. *EMBER2024 — A benchmark dataset for holistic evaluation of malware classifiers.* arXiv:2506.05074, 2025.
* F. Pendlebury et al. *TESSERACT: Eliminating experimental bias in malware classification across space and time.* USENIX Security 2019.
* I. Sharafaldin, A. H. Lashkari, A. Ghorbani. *Toward generating a new intrusion detection dataset and intrusion traffic characterization.* ICISSP 2018.
* G. Engelen, V. Rimmer, W. Joosen. *Troubleshooting an intrusion detection dataset: the CICIDS2017 case study.* IEEE SPW 2021.
* A. Rosay, E. Cheval, F. Carlier, P. Leroux. *Network intrusion detection: a comprehensive analysis of CIC-IDS2017.* ICISSP 2022.
