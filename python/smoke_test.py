#!/usr/bin/env python3
"""End-to-end smoke test of the whole pipeline on fabricated data.

Real malware and the real CIC-IDS CSVs are not redistributable, so this
script fabricates stand-ins with the same *shape* (file layout, column
names, CICFlowMeter quirks such as leading spaces, duplicated columns and a
Latin-1 label) and runs every script on them. It checks that the plumbing
works — the numbers it prints say nothing about real-world performance.

Requires the `ncd` binary (cargo build --release) and the `ncdthreat`
bindings (maturin develop --release -m crates/ncd-py/Cargo.toml).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(*cmd: str | Path) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def py(script: str, *args) -> None:
    run(sys.executable, HERE / script, *args)


def fake_flows(path: Path, rng: np.random.Generator) -> None:
    """CICFlowMeter-like CSV: 3 benign hosts + DoS, port scan, web brute force."""
    rows = []
    t0 = pd.Timestamp("2017-07-05 09:00")

    def flows(src, label, n, dur, pkts, port_fn, iat):
        t = t0 + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        for _ in range(n):
            t += pd.Timedelta(seconds=float(rng.exponential(iat)))
            fwd = max(1, int(rng.poisson(pkts)))
            rows.append({
                " Source IP": src, " Timestamp": t.strftime("%d/%m/%Y %H:%M:%S"),
                " Destination Port": port_fn(), " Flow Duration": float(rng.lognormal(dur, 1.0)),
                " Total Fwd Packets": fwd, " Total Backward Packets": max(0, fwd - int(rng.integers(0, 3))),
                "Total Length of Fwd Packets": fwd * float(rng.lognormal(5, 0.6)),
                " Fwd Header Length": fwd * 20, " Fwd Header Length.1": fwd * 20,
                " Flow IAT Mean": float(rng.lognormal(dur - 2, 1.0)),
                " SYN Flag Count": int(rng.random() < 0.5), " Init_Win_bytes_forward": int(rng.choice([8192, 29200, 65535])),
                "Flow Bytes/s": np.inf if rng.random() < 0.01 else float(rng.lognormal(8, 2)),
                " Label": label,
            })

    for h in range(3):
        flows(f"192.168.10.{h + 5}", "BENIGN", 900, 11, 12, lambda: int(rng.choice([80, 443, 53, 22, 8080])), 4)
    flows("172.16.0.1", "DoS Hulk", 500, 9, 6, lambda: 80, 0.05)
    flows("172.16.0.2", "PortScan", 500, 3, 1, lambda: int(rng.integers(1, 1024)), 0.01)
    flows("172.16.0.3", "Web Attack \x96 Brute Force", 400, 10, 9, lambda: 80, 0.5)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="latin-1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncd", type=Path, default=ROOT / "target" / "release" / "ncd")
    ap.add_argument("--keep", type=Path, default=None, help="keep outputs in this dir")
    args = ap.parse_args()
    if not args.ncd.exists():
        raise SystemExit(f"{args.ncd} not found — run `cargo build --release` first")
    work = args.keep or Path(tempfile.mkdtemp(prefix="ncd-smoke-"))
    if work.exists() and args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    res = work / "results"
    rng = np.random.default_rng(7)

    # ---- malware-family track -----------------------------------------------
    run(args.ncd, "synth", work / "synth", "--families", "4", "--per-family", "12", "--benign", "0",
        "--template-len", "8000")
    flat = work / "flat"
    flat.mkdir()
    meta = []
    for f in sorted((work / "synth").glob("*/*.bin")):
        shutil.copy(f, flat / f.stem)
        meta.append({"sha": f.stem, "family": f.parent.name.upper(),
                     "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))})
    pd.DataFrame(meta).to_csv(work / "meta.csv", index=False)
    py("make_split.py", "--meta", work / "meta.csv", "--time-col", "timestamp", "--protocol", "temporal",
       "--min-per-family", "5", "--samples-dir", flat, "--link-to", work / "corpus", "--out", work / "splits.csv")
    run(args.ncd, "eval", work / "corpus", "--split-file", work / "splits.csv", "-k", "1", "-m", "conditional",
        "--json", res / "malware_ncd.json")

    # byte-histogram + size features stand in for EMBER vectors
    ids = sorted(p.name for p in flat.iterdir())
    X = np.stack([np.r_[np.bincount(np.frombuffer((flat / i).read_bytes(), np.uint8), minlength=256) / (flat / i).stat().st_size,
                        np.log((flat / i).stat().st_size)] for i in ids]).astype(np.float32)
    np.save(work / "X.npy", X)
    pd.DataFrame({"sha": ids}).to_csv(work / "X_index.csv", index=False)
    py("baseline_malware.py", "--features", work / "X.npy", "--index", work / "X_index.csv",
       "--split", work / "splits.csv", "--out-dir", res)

    run(args.ncd, "matrix", work / "corpus", "-o", work / "malware.npy", "-c", "zstd:9")
    py("plot_matrix.py", work / "malware.npy", "--title", "Synthetic families")

    # ---- network track ---------------------------------------------------------
    # several "captures" so 64-flow windows still yield enough samples
    frames = []
    for rep in range(4):
        fake_flows(work / f"flows{rep}.csv", rng)
        df = pd.read_csv(work / f"flows{rep}.csv", encoding="latin-1")
        df[" Source IP"] = df[" Source IP"] + f"_{rep}"
        frames.append(df)
    pd.concat(frames).to_csv(work / "flows.csv", index=False, encoding="latin-1")
    win = work / "windows"
    py("prepare_flows.py", work / "flows.csv", "--out", win, "--window", "64")
    py("baseline_flows.py", win, "--out-dir", res)
    py("eval_flows_ncd.py", win, "--codec", "zstd:9", "--out-dir", res)

    # CLI anomaly scoring: references = benign *training* windows only,
    # queries = test windows (symlinked layouts, no copies, no leakage).
    meta_w = pd.read_csv(win / "windows.csv")
    for row in meta_w.itertuples():
        if row.split == "train" and row.label != "benign":
            continue
        d = work / ("refs" if row.split == "train" else "queries") / row.label
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{row.id}.bin").symlink_to((win / row.label / f"{row.id}.bin").resolve())
    run(args.ncd, "score", "--refs", work / "refs", "--queries", work / "queries",
        "-o", work / "scores.csv", "-c", "zstd:9", "-v", "whole")

    py("compare.py", res, "--out", res / "summary.md")
    print(f"\nsmoke test OK — outputs in {work}")


if __name__ == "__main__":
    main()
