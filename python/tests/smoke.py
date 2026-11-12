"""End-to-end run on generated data. Needs the `ncd` binary and `ncdthreat`."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ncdeval.synth import fake_captures

ROOT = Path(__file__).resolve().parents[2]


def run(*cmd: str | Path) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def py(module: str, *args) -> None:
    run(sys.executable, "-m", f"ncdeval.{module}", *args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncd", type=Path, default=ROOT / "target" / "release" / "ncd")
    ap.add_argument("--keep", type=Path, default=None, help="keep outputs in this dir")
    args = ap.parse_args()
    if not args.ncd.exists():
        raise SystemExit(f"{args.ncd} not found, run `cargo build --release` first")
    work = args.keep or Path(tempfile.mkdtemp(prefix="ncd-smoke-"))
    if work.exists() and args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    res = work / "results"
    rng = np.random.default_rng(7)

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
    py("split", "--meta", work / "meta.csv", "--time-col", "timestamp", "--protocol", "temporal",
       "--min-per-family", "5", "--samples-dir", flat, "--link-to", work / "corpus", "--out", work / "splits.csv")
    run(args.ncd, "eval", work / "corpus", "--split-file", work / "splits.csv", "-k", "1", "-m", "conditional",
        "--json", res / "malware_ncd.json")

    # byte histogram stands in for EMBER features
    ids = sorted(p.name for p in flat.iterdir())
    X = np.stack([np.r_[np.bincount(np.frombuffer((flat / i).read_bytes(), np.uint8), minlength=256) / (flat / i).stat().st_size,
                        np.log((flat / i).stat().st_size)] for i in ids]).astype(np.float32)
    np.save(work / "X.npy", X)
    pd.DataFrame({"sha": ids}).to_csv(work / "X_index.csv", index=False)
    py("baseline_malware", "--features", work / "X.npy", "--index", work / "X_index.csv",
       "--split", work / "splits.csv", "--out-dir", res)

    run(args.ncd, "matrix", work / "corpus", "-o", work / "malware.npy", "-c", "zstd:9")
    py("plot", work / "malware.npy", "--title", "Synthetic families")

    fake_captures(work / "flows.csv", 4, rng)
    win = work / "windows"
    py("flows", work / "flows.csv", "--out", win, "--window", "64")
    py("baseline_flows", win, "--out-dir", res)
    py("eval_flows", win, "--codec", "zstd:9", "--out-dir", res)

    # refs: benign train windows; queries: test windows
    meta_w = pd.read_csv(win / "windows.csv")
    for row in meta_w.itertuples():
        if row.split == "train" and row.label != "benign":
            continue
        d = work / ("refs" if row.split == "train" else "queries") / row.label
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{row.id}.bin").symlink_to((win / row.label / f"{row.id}.bin").resolve())
    run(args.ncd, "score", "--refs", work / "refs", "--queries", work / "queries",
        "-o", work / "scores.csv", "-c", "zstd:9", "-v", "whole")

    py("compare", res, "--out", res / "summary.md")
    print(f"\nsmoke test OK, outputs in {work}")


if __name__ == "__main__":
    main()
