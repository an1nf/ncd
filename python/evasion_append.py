#!/usr/bin/env python3
"""Evasion by appending benign code: does 1-NN NCD still find a test sample's family?

Each test sample (code view) gets benign bytes appended, a fraction f of its own size taken
from files under --benign (also code view), and is classified against the unchanged training
samples. f = 0 is the clean baseline. The padding of each sample comes from a seeded random
benign file and offset, the same for every codec, so the curves are comparable.

    python python/evasion_append.py ~/ncd-data/bazaar/corpus_temporal \\
        --split ~/ncd-data/bazaar/splits_temporal.csv --benign ~/ncd-data/pypi/samples \\
        --fractions 0 0.1 0.25 0.5 1 2 --codec zstd:9 --method conditional --out results/bazaar/evasion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ncd_backend as nt
from common import knn_predict, report_dict


def load_corpus(root: Path, view: str) -> tuple[list[str], list[str], list[bytes]]:
    ids, labels, objs = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            ids.append(f.name)
            labels.append(d.name)
            objs.append(nt_load(f, view))
    return ids, labels, objs


def nt_load(path: Path, view: str) -> bytes:
    import ncdthreat

    return ncdthreat.load(str(path), view)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, help="ROOT/<label>/<sample>, e.g. make_split.py --link-to output")
    ap.add_argument("--split", type=Path, required=True, help="CSV with id,split")
    ap.add_argument("--benign", type=Path, required=True, help="directory of benign files to take padding from")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2])
    ap.add_argument("--codec", default="zstd:9")
    ap.add_argument("--method", default="conditional")
    ap.add_argument("--view", default="code")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids, labels, objs = load_corpus(args.corpus, args.view)
    split = pd.read_csv(args.split)
    id_col = "id" if "id" in split.columns else split.columns[0]
    where = dict(zip(split[id_col].astype(str), split["split"]))
    tr = [i for i, s in enumerate(ids) if where.get(s) == "train"]
    te = [i for i, s in enumerate(ids) if where.get(s) == "test"]
    benign_files = sorted(p for p in args.benign.rglob("*") if p.is_file())
    if not benign_files:
        raise SystemExit(f"no files under {args.benign}")
    pool = [nt_load(p, args.view) for p in benign_files]
    pool = [b for b in pool if len(b) > 0]
    rng = np.random.default_rng(args.seed)
    picks = [(int(rng.integers(len(pool))), float(rng.random())) for _ in te]

    def padding(k: int, n: int) -> bytes:
        src, off = pool[picks[k][0]], picks[k][1]
        if n <= 0:
            return b""
        reps = -(-n // len(src)) + 1
        long = src * reps
        start = int(off * len(src))
        return long[start:start + n]

    ytr = [labels[i] for i in tr]
    yte = [labels[i] for i in te]
    rows = []
    for f in args.fractions:
        queries = [objs[i] + padding(k, int(f * len(objs[i]))) for k, i in enumerate(te)]
        d = nt.cross(queries, [objs[i] for i in tr], codec=args.codec, method=args.method)
        pred = knn_predict(np.asarray(d), ytr, 1)
        rep = report_dict(yte, pred)
        rows.append({"fraction": f, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]})
        print(f"f={f:<5} accuracy {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"codec": args.codec, "method": args.method, "view": args.view,
                                    "n_train": len(tr), "n_test": len(te), "benign_files": len(pool),
                                    "rows": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
