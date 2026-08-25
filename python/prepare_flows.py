#!/usr/bin/env python3
"""CICFlowMeter CSVs to byte windows for NCD and aggregate features for baselines.
One byte per feature (quantile bin fitted on train flows), 0xFF between flows,
--window consecutive flows per source host.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import BENIGN, sanitize_label

ID_COLS = {"flow id", "source ip", "src ip", "source port", "src port", "destination ip", "dst ip", "timestamp", "label"}


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        df["__file"] = p.name
        df["__row"] = np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # duplicated columns, e.g. "Fwd Header Length.1"
    df = df.loc[:, [c for c in df.columns if not c.endswith(".1")]]
    return df


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=64,
                    help="flows per window; below ~64 objects get too small for NCD (see README)")
    ap.add_argument("--bins", type=int, default=16, help="quantile bins per feature (≤255)")
    ap.add_argument("--min-attack-frac", type=float, default=0.5)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--max-per-label", type=int, default=None, help="subsample windows per label")
    ap.add_argument("--drop-port", action="store_true", help="exclude Destination Port (a known shortcut feature)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not 2 <= args.bins <= 255:
        raise SystemExit("--bins must be in 2..255 (0xFF is the record separator)")
    rng = np.random.default_rng(args.seed)

    df = read_csvs(args.csv)
    label_col = find_col(df, "label")
    if label_col is None:
        raise SystemExit("no Label column")
    src_col = find_col(df, "source ip", "src ip")
    ts_col = find_col(df, "timestamp")
    df["__label"] = df[label_col].map(sanitize_label)
    print(f"{len(df):,} flows from {len(args.csv)} file(s); labels: {df['__label'].value_counts().to_dict()}")

    feat_cols = [c for c in df.columns
                 if c.lower() not in ID_COLS and not c.startswith("__")
                 and pd.api.types.is_numeric_dtype(df[c])]
    if args.drop_port:
        feat_cols = [c for c in feat_cols if c.lower() not in {"destination port", "dst port"}]
    feats = df[feat_cols].to_numpy(dtype=np.float64)
    feats[~np.isfinite(feats)] = 0.0
    print(f"{len(feat_cols)} numeric features")

    df["__src"] = df[src_col].astype(str) if src_col else "all"
    if ts_col:
        df["__t"] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
        order = df.sort_values(["__src", "__t", "__file", "__row"]).index.to_numpy()
    else:
        order = df.sort_values(["__file", "__src", "__row"]).index.to_numpy()
    srcs = df["__src"].to_numpy()[order]
    labels = df["__label"].to_numpy()[order]
    files = df["__file"].to_numpy()[order]

    windows: list[np.ndarray] = []
    meta = []
    dropped = 0
    start = 0
    n = len(order)
    while start < n:
        # run of one source (and one file without timestamps)
        end = start
        while end < n and srcs[end] == srcs[start] and (ts_col or files[end] == files[start]):
            end += 1
        for w0 in range(start, end, args.window):
            w1 = min(w0 + args.window, end)
            if w1 - w0 < max(2, args.window // 2):
                continue
            wl = labels[w0:w1]
            attack = wl[wl != BENIGN]
            frac = len(attack) / len(wl)
            if frac == 0:
                lab = BENIGN
            elif frac >= args.min_attack_frac:
                vals, cnt = np.unique(attack, return_counts=True)
                lab = vals[cnt.argmax()]
            else:
                dropped += 1
                continue
            windows.append(order[w0:w1])
            first_t = df["__t"].iloc[order[w0]] if ts_col else pd.NaT
            meta.append({"label": lab, "source": srcs[w0], "n_flows": w1 - w0,
                         "attack_frac": round(frac, 4), "t0": first_t})
        start = end
    meta = pd.DataFrame(meta)
    print(f"{len(meta):,} windows ({dropped} ambiguous dropped)")

    if args.max_per_label:
        keep = (meta.groupby("label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), args.max_per_label), random_state=args.seed)).index)
        keep = np.sort(keep.to_numpy())
        meta = meta.loc[keep].reset_index(drop=True)
        windows = [windows[i] for i in keep]

    if args.split == "temporal":
        if not ts_col or meta["t0"].isna().all():
            raise SystemExit("--split temporal needs timestamps")
        cut = meta["t0"].quantile(1 - args.test_frac)
        meta["split"] = np.where(meta["t0"] < cut, "train", "test")
    else:
        meta["split"] = "train"
        for _, g in meta.groupby("label"):
            k = min(max(int(round(len(g) * args.test_frac)), 1), max(len(g) - 1, 0))
            meta.loc[rng.permutation(g.index)[:k], "split"] = "test"

    # bins fitted on training flows only
    train_rows = np.concatenate([w for w, s in zip(windows, meta["split"]) if s == "train"])
    qs = np.linspace(0, 1, args.bins + 1)[1:-1]
    edges = [np.unique(np.quantile(feats[train_rows, j], qs)) for j in range(feats.shape[1])]
    binned = np.empty(feats.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        binned[:, j] = np.searchsorted(e, feats[:, j], side="right")

    args.out.mkdir(parents=True, exist_ok=True)
    agg = np.empty((len(windows), 4 * feats.shape[1]), dtype=np.float32)
    sep = np.full((1,), 0xFF, dtype=np.uint8)
    ids = []
    for i, (rows, lab) in enumerate(zip(windows, meta["label"])):
        wid = f"w{i:07d}"
        ids.append(wid)
        blob = b"".join(np.concatenate([binned[r], sep]).tobytes() for r in rows)
        d = args.out / lab
        d.mkdir(exist_ok=True)
        (d / f"{wid}.bin").write_bytes(blob)
        f = feats[rows]
        agg[i] = np.concatenate([f.mean(0), f.std(0), f.min(0), f.max(0)])
    meta.insert(0, "id", ids)
    meta.drop(columns="t0").to_csv(args.out / "windows.csv", index=False)
    np.save(args.out / "features.npy", agg)
    (args.out / "feature_names.txt").write_text("\n".join(feat_cols) + "\n")
    print(meta.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(f"wrote {len(meta):,} windows to {args.out}")


if __name__ == "__main__":
    main()
