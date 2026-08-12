#!/usr/bin/env python3
"""Train/test split shared by `ncd eval` and the baselines.
random: stratified by family. temporal: split by first-seen time, unseen test families dropped.
--link-to builds CORPUS/<family>/<id> symlinks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import sanitize_label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--id-col", default="sha")
    ap.add_argument("--label-col", default="family")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--protocol", choices=["random", "temporal"], default="random")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--min-per-family", type=int, default=10, help="drop rarer families")
    ap.add_argument("--max-per-family", type=int, default=None, help="subsample (NCD cost is quadratic)")
    ap.add_argument("--samples-dir", type=Path, default=None, help="flat dir of samples named by id; keep only ids present")
    ap.add_argument("--link-to", type=Path, default=None, help="create CORPUS/<label>/<id> symlinks")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.meta)
    for col in (args.id_col, args.label_col, args.time_col):
        if col and col not in df.columns:
            raise SystemExit(f"column `{col}` not in {list(df.columns)}")
    df = df.dropna(subset=[args.label_col])
    df = df[df[args.label_col].astype(str).str.strip() != ""]
    df = pd.DataFrame({
        "id": df[args.id_col].astype(str).str.strip().str.lower(),
        "label": df[args.label_col].map(sanitize_label),
        **({"time": pd.to_datetime(df[args.time_col], errors="coerce")} if args.time_col else {}),
    }).drop_duplicates("id")

    if args.samples_dir:
        present = {p.name.lower().split(".")[0] for p in args.samples_dir.iterdir() if p.is_file()}
        before = len(df)
        df = df[df["id"].isin(present)]
        print(f"{len(df)}/{before} metadata rows have a sample in {args.samples_dir}")

    counts = df["label"].value_counts()
    df = df[df["label"].isin(counts[counts >= args.min_per_family].index)]
    if args.max_per_family:
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.max_per_family), random_state=args.seed)))

    if args.protocol == "random":
        df["split"] = "train"
        for _, g in df.groupby("label"):
            n_test = int(round(len(g) * args.test_frac))
            n_test = min(max(n_test, 1), len(g) - 1)
            df.loc[rng.permutation(g.index)[:n_test], "split"] = "test"
    else:
        if "time" not in df:
            raise SystemExit("--protocol temporal needs --time-col")
        df = df.dropna(subset=["time"])
        cut = df["time"].quantile(1 - args.test_frac)
        df["split"] = np.where(df["time"] < cut, "train", "test")
        seen = set(df.loc[df["split"] == "train", "label"])
        unseen = (df["split"] == "test") & ~df["label"].isin(seen)
        print(f"temporal cut at {cut}; dropping {unseen.sum()} test samples from families unseen in training")
        df = df[~unseen]

    df = df.sort_values(["split", "label", "id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    s = df.groupby("split")["label"].agg(["count", "nunique"])
    print(f"wrote {args.out}\n{s.to_string()}")

    if args.link_to:
        if not args.samples_dir:
            raise SystemExit("--link-to needs --samples-dir")
        files = {p.name.lower().split(".")[0]: p.resolve() for p in args.samples_dir.iterdir() if p.is_file()}
        for row in df.itertuples():
            d = args.link_to / row.label
            d.mkdir(parents=True, exist_ok=True)
            link = d / row.id
            if not link.exists():
                os.symlink(files[row.id], link)
        print(f"linked {len(df)} samples under {args.link_to}")


if __name__ == "__main__":
    main()
