#!/usr/bin/env python3
"""Ablation: how many flows per window does NCD need?

Fabricates CICFlowMeter-like traffic (the generator from smoke_test.py,
fixed seed), builds windows of several sizes and reports NCD k-NN accuracy
and anomaly AUC for the `concat` and `conditional` estimators, next to the
Euclidean k-NN baseline on aggregate features. Fabricated data: the point is
the *trend* with object size, not the absolute numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import ncdthreat as nt
from common import BENIGN, auc_vs_benign, knn_predict, report_dict
from smoke_test import fake_flows

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--codec", default="zstd:19")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--captures", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="ncd-ablation-"))
    rng = np.random.default_rng(args.seed)

    frames = []
    for rep in range(args.captures):
        fake_flows(work / "tmp.csv", rng)
        df = pd.read_csv(work / "tmp.csv", encoding="latin-1")
        df[" Source IP"] = df[" Source IP"] + f"_{rep}"
        frames.append(df)
    pd.concat(frames).to_csv(work / "flows.csv", index=False, encoding="latin-1")

    rows = []
    for w in args.windows:
        out = work / f"w{w}"
        subprocess.run([sys.executable, HERE / "prepare_flows.py", work / "flows.csv", "--out", out,
                        "--window", str(w)], check=True, stdout=subprocess.DEVNULL)
        meta = pd.read_csv(out / "windows.csv")
        objs = [(out / l / f"{i}.bin").read_bytes() for i, l in zip(meta["id"], meta["label"])]
        lab = meta["label"].to_numpy()
        tr, te = np.flatnonzero(meta["split"] == "train"), np.flatnonzero(meta["split"] == "test")
        refs = tr[lab[tr] == BENIGN]
        Q, T, R = [objs[i] for i in te], [objs[i] for i in tr], [objs[i] for i in refs]
        for method in ("concat", "conditional"):
            D = nt.cross(Q, T, codec=args.codec, method=method)
            acc = report_dict(lab[te], knn_predict(D, lab[tr], args.k))["macro_f1"]
            auc = auc_vs_benign(lab[te], nt.anomaly_scores(Q, R, k=args.k, codec=args.codec, method=method))
            rows.append({"flows/window": w, "bytes/window": int(np.mean([len(o) for o in objs])),
                         "method": f"NCD {method}", "k-NN macro-F1": acc, "anomaly AUC": auc,
                         "windows": len(meta)})
        print(f"window {w}: done", file=sys.stderr)

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["flows/window", "bytes/window"], columns="method",
                          values=["k-NN macro-F1", "anomaly AUC"])
    print("| flows/window | bytes/object | k-NN macro-F1 concat | k-NN macro-F1 conditional "
          "| anomaly AUC concat | anomaly AUC conditional |")
    print("|---:|---:|---:|---:|---:|---:|")
    for (w, b), r in wide.iterrows():
        print(f"| {w} | {b} | {r[('k-NN macro-F1', 'NCD concat')]:.3f} | {r[('k-NN macro-F1', 'NCD conditional')]:.3f} "
              f"| {r[('anomaly AUC', 'NCD concat')]:.3f} | {r[('anomaly AUC', 'NCD conditional')]:.3f} |")


if __name__ == "__main__":
    main()
