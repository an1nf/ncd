#!/usr/bin/env python3
"""EMBER v3 feature vectors (thrember) for a flat sample store, for baseline_malware.py.
Writes <out>.npy (float32, n x 2568) and <out>.index.csv (column `sha`).
Install: pip install "EMBER2024 @ git+https://github.com/FutureComputing4AI/EMBER2024.git" "signify<0.8"
"""

from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_extractor = None


def _init() -> None:
    global _extractor
    import thrember

    _extractor = thrember.PEFeatureExtractor()


def _vector(path: Path) -> tuple[str, np.ndarray | None, str]:
    try:
        return path.name.lower(), _extractor.feature_vector(path.read_bytes()), ""
    except Exception as e:  # noqa: BLE001
        return path.name.lower(), None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", type=Path, help="directory of samples named by sha256")
    ap.add_argument("--out", type=Path, required=True, help="output prefix, e.g. data/ember")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(p for p in args.samples.iterdir() if p.is_file() and not p.name.startswith("."))
    ids, rows, failed = [], [], 0
    with Pool(args.jobs, initializer=_init) as pool:
        for i, (sha, vec, err) in enumerate(pool.imap(_vector, paths, chunksize=4), 1):
            if vec is None:
                failed += 1
                print(f"{sha[:16]}: {err}", file=sys.stderr)
            else:
                ids.append(sha)
                rows.append(vec)
            if i % 200 == 0:
                print(f"{i}/{len(paths)}", file=sys.stderr)
    if not rows:
        raise SystemExit("no feature vectors extracted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix(".npy"), np.stack(rows).astype(np.float32))
    with open(args.out.with_suffix(".index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha"])
        w.writerows([s] for s in ids)
    print(f"{len(ids)} vectors ({failed} failed) -> {args.out.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
