#!/usr/bin/env python3
"""NCD k-NN and anomaly scores (benign references only) on flow windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ncdthreat as nt
from common import BENIGN, auc_vs_benign, knn_predict, print_summary, report_dict, save_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="output dir of prepare_flows.py")
    ap.add_argument("--codec", default="zstd:19")
    # concat beats conditional from 32 flows/window up (ablation_window.py)
    ap.add_argument("--method", default="concat", choices=["concat", "concat-sym", "conditional"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=None, help="subsample benign references (cost ∝ refs)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    meta = pd.read_csv(args.data / "windows.csv")
    objs = [(args.data / lab / f"{wid}.bin").read_bytes() for wid, lab in zip(meta["id"], meta["label"])]
    tr = np.flatnonzero(meta["split"] == "train")
    te = np.flatnonzero(meta["split"] == "test")
    labels = meta["label"].to_numpy()
    tag = f"ncd-{args.codec.replace(':', '')}-{args.method}"
    print(f"{len(tr)} train / {len(te)} test windows; codec {args.codec}, method {args.method}")

    t = time.perf_counter()
    D = nt.cross([objs[i] for i in te], [objs[i] for i in tr], codec=args.codec, method=args.method)
    pred = knn_predict(D, labels[tr], args.k)
    secs = time.perf_counter() - t
    metrics = report_dict(labels[te], pred, sorted(set(labels)))
    print_summary(f"{tag} {args.k}-NN", metrics)
    save_result(args.out_dir / f"flows_{tag}-knn{args.k}.json", model=f"{tag}-knn{args.k}",
                task="flow-classification", metrics=metrics, seconds=secs,
                predictions=[{"id": meta["id"][i], "true": labels[i], "pred": p} for i, p in zip(te, pred)])

    refs = tr[labels[tr] == BENIGN]
    if args.max_refs and len(refs) > args.max_refs:
        refs = np.random.default_rng(args.seed).choice(refs, args.max_refs, replace=False)
    t = time.perf_counter()
    scores = nt.anomaly_scores([objs[i] for i in te], [objs[i] for i in refs], k=args.k,
                               codec=args.codec, method=args.method)
    secs = time.perf_counter() - t
    auc = auc_vs_benign(labels[te], scores)
    print_summary(f"{tag} anomaly (k={args.k}, {len(refs)} benign refs)", None, auc)
    save_result(args.out_dir / f"flows_anomaly_{tag}.json", model=f"{tag}-knn{args.k}-dist",
                task="flow-anomaly", auc=auc, seconds=secs, n_refs=len(refs),
                labels=labels[te].tolist(), scores=scores.tolist())


if __name__ == "__main__":
    main()
