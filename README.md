# NCD

Normalized Compression Distance for malware family classification and network anomaly detection,
with a Rust core and classical baselines evaluated on the same samples and splits.

```
NCD(x, y) = (C(xy) - min{C(x), C(y)}) / max{C(x), C(y)}
```

`C` is a real compressor standing in for Kolmogorov complexity (Cilibrasi and Vitányi, 2005).
Theory and design: [docs/FUNDAMENTOS.md](docs/FUNDAMENTOS.md). Runbook: [docs/EXPERIMENTOS.md](docs/EXPERIMENTOS.md).

## Layout

```
crates/ncd-core     codecs (zstd, xz, bzip2, DEFLATE), NCD matrices, PE/ELF views, k-NN, metrics
crates/ncd-cli      `ncd`: dist, matrix, eval, score, bench, synth
crates/ncd-py       `ncdthreat`: Python bindings, zero-copy, GIL released
python/ncdeval      data fetchers, splits, flow windows, baselines, reports
python/ncdeval/lm   language models with arithmetic coding as NCD compressors
python/tests        end-to-end smoke test, LM codec test
scripts             demo, malware families, CIC-IDS2017
```

## Build

```bash
cargo build --release && cargo test --release
python -m venv .venv && . .venv/bin/activate
pip install -e 'python[lm,fetch]' maturin
maturin develop --release -m crates/ncd-py/Cargo.toml
python python/tests/smoke.py && python python/tests/test_lm.py
```

## Usage

A corpus is `ROOT/<label>/<sample>`; the file stem is the sample id.

```bash
ncd dist a.exe b.exe -c zstd:19 -m conditional -v code
ncd eval corpus/ -k 1 --split-file splits.csv --json ncd.json
ncd score --refs benign/ --queries test/ -k 5 -o scores.csv
ncd bench corpus/ --codecs zstd:3,zstd:19,xz:9,bzip2:9,deflate:9
```

```python
import ncdthreat as nt
D = nt.pairwise(objs, codec="zstd:19", method="conditional")
s = nt.anomaly_scores(queries, benign_refs, k=5)
```

Python modules accept `--codec lm:model.pt` or `lm:hf:ORG/MODEL` wherever they take a codec.

## Results

**Synthetic** (`scripts/demo.sh`, 150 samples, 2 cores). All codecs reach 1.0 leave-one-out
accuracy; the corpus checks the pipeline and exposes compressor defects.

| codec | estimator | NCD(x,x) | pairs over window | time |
|---|---|---:|---:|---:|
| zstd:3 | concat | 0.0008 | 0 | 1.3 s |
| zstd:19 | concat | 0.0009 | 0 | 58.8 s |
| zstd:19 | conditional | 0.0019 | 0 | 17.2 s |
| xz:9 | concat | 0.0063 | 0 | 57.1 s |
| bzip2:9 | concat | 0.2697 | 0 | 32.9 s |
| deflate:9 | concat | 0.0456 | 11,175 | 11.5 s |

<p align="center"><img src="docs/img/synthetic_heatmap.png" width="55%" alt="NCD matrix of the synthetic corpus"></p>

**MalwareBazaar**, 8 families × 40 PE samples, 96 test samples (±10 points at 95%).

| model | input | random | temporal per family |
|---|---|---:|---:|
| GBDT | EMBER v3 | **0.635** | 0.500 |
| 1-NN NCD, zstd:9 conditional | whole file | 0.552 | **0.510** |
| 1-NN NCD, zstd:9 conditional | code sections | 0.469 | **0.510** |
| 1-NN Euclidean | EMBER v3 | 0.521 | 0.479 |

GBDT wins on the random split (+0.167 over NCD on code, p = 0.002). On newer samples both are
level (+0.010, 95% CI −0.07 to +0.09): GBDT drops 13.5 points, NCD 4. Both send AgentTesla
to Formbook. Appending a foreign family's code at the sample's size lowers NCD accuracy from
0.46 to 0.33; benign padding has no effect.

**CIC-IDS2017**, anomaly detection with benign Monday as reference, 64-flow windows.

| model | ROC AUC |
|---|---:|
| k-NN distance, flow statistics | 0.960 |
| Isolation Forest, flow statistics | 0.957 |
| 5-NN NCD, zstd:3 | 0.686 |
| 5-NN NCD, zstd:3, no destination port | 0.664 |

On quantized flow windows NCD stays well below baselines on aggregate statistics.

**Language models as compressors** (tag `text-experiment`). Few-shot AG News, 1-NN on NCD,
chance 0.25. GPT-2 reaches 0.588 against 0.447 for zstd:19. Single-document bits per character
do not predict accuracy (ρ = −0.02); the class separation d′ of conditional lengths does
(ρ = 0.89). Byte LMs without a match model stay at chance because C(xy) ≈ C(x) + C(y). Pythia
70M to 1B compress better as they grow but classify worse (0.657 → 0.602).

<p align="center"><img src="docs/img/text_acc_vs_bpc.png" width="90%" alt="Accuracy against bits per character and against class separation"></p>

## Limitations

- Pairwise cost is quadratic in the number of samples.
- Packers and shared crypters dominate the distance between binaries.
- Appended code from another family moves a sample toward it.
- `extract` is mutation-tested, not fuzzed.

## References

- M. Li, X. Chen, X. Li, B. Ma, P. Vitányi. The similarity metric. IEEE Trans. Inf. Theory 50(12), 2004.
- R. Cilibrasi, P. Vitányi. Clustering by compression. IEEE Trans. Inf. Theory 51(4), 2005.
- M. Cebrián, M. Alfonseca, A. Ortega. Common pitfalls using the normalized compression distance. Commun. Inf. Syst. 5(4), 2005.
- G. Delétang et al. Language modeling is compression. ICLR 2024.
- H. Anderson, P. Roth. EMBER. arXiv:1804.04637, 2018.
- R. J. Joyce et al. EMBER2024. arXiv:2506.05074, 2025.
- L. Yang et al. BODMAS. DLS 2021.
- F. Pendlebury et al. TESSERACT. USENIX Security 2019.
- G. Engelen, V. Rimmer, W. Joosen. Troubleshooting an intrusion detection dataset: the CICIDS2017 case study. IEEE SPW 2021.

## License

MIT
