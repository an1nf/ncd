#!/usr/bin/env python3
"""Baselines on per-window flow statistics: GBDT, k-NN, Isolation Forest, k-NN distance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BENIGN, auc_vs_benign, print_summary, report_dict, save_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="output dir of prepare_flows.py")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    meta = pd.read_csv(args.data / "windows.csv")
    X = np.load(args.data / "features.npy")
    X = np.sign(X) * np.log1p(np.abs(X))  # signed log
    tr = (meta["split"] == "train").to_numpy()
    te = ~tr
    ytr, yte = meta.loc[tr, "label"].to_numpy(), meta.loc[te, "label"].to_numpy()
    labels = sorted(meta["label"].unique())

    sup = {
        "gbdt-flowstats": HistGradientBoostingClassifier(max_iter=300, random_state=args.seed),
        f"knn{args.k}-flowstats-euclidean": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=args.k)),
    }
    for name, model in sup.items():
        t = time.perf_counter()
        pred = model.fit(X[tr], ytr).predict(X[te])
        metrics = report_dict(yte, pred, labels)
        print_summary(name, metrics)
        save_result(args.out_dir / f"flows_{name}.json", model=name, task="flow-classification",
                    metrics=metrics, seconds=time.perf_counter() - t,
                    predictions=[{"id": i, "true": a, "pred": b} for i, a, b in zip(meta.loc[te, "id"], yte, pred)])

    # anomaly models see benign training windows only
    ben = tr & (meta["label"] == BENIGN).to_numpy()
    scaler = StandardScaler().fit(X[ben])
    Xb, Xt = scaler.transform(X[ben]), scaler.transform(X[te])
    anom = {}
    t = time.perf_counter()
    iso = IsolationForest(n_estimators=300, random_state=args.seed).fit(Xb)
    anom["iforest-flowstats"] = (-iso.score_samples(Xt), time.perf_counter() - t)
    t = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=min(args.k, len(Xb))).fit(Xb)
    anom[f"knn{args.k}-dist-flowstats-euclidean"] = (nn.kneighbors(Xt)[0].mean(1), time.perf_counter() - t)
    for name, (scores, secs) in anom.items():
        auc = auc_vs_benign(yte, scores)
        print_summary(name, None, auc)
        save_result(args.out_dir / f"flows_anomaly_{name}.json", model=name, task="flow-anomaly",
                    auc=auc, seconds=secs, labels=yte.tolist(), scores=scores.tolist())


if __name__ == "__main__":
    main()
