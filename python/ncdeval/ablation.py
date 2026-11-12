"""NCD k-NN macro-F1 and anomaly AUC against flow window size, on fabricated traffic."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import ncdthreat as nt
import numpy as np
import pandas as pd

from .common import BENIGN, auc_vs_benign, knn_predict, report_dict
from .synth import fake_captures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--codec", default="zstd:19")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--captures", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="ncd-ablation-"))
    fake_captures(work / "flows.csv", args.captures, np.random.default_rng(args.seed))

    rows = []
    for w in args.windows:
        out = work / f"w{w}"
        subprocess.run([sys.executable, "-m", "ncdeval.flows", work / "flows.csv", "--out", out,
                        "--window", str(w)], check=True, stdout=subprocess.DEVNULL)
        meta = pd.read_csv(out / "windows.csv")
        objs = [(out / l / f"{i}.bin").read_bytes() for i, l in zip(meta["id"], meta["label"])]
        lab = meta["label"].to_numpy()
        tr, te = np.flatnonzero(meta["split"] == "train"), np.flatnonzero(meta["split"] == "test")
        refs = tr[lab[tr] == BENIGN]
        Q, T, R = [objs[i] for i in te], [objs[i] for i in tr], [objs[i] for i in refs]
        for method in ("concat", "conditional"):
            D = nt.cross(Q, T, codec=args.codec, method=method)
            f1 = report_dict(lab[te], knn_predict(D, lab[tr], args.k))["macro_f1"]
            auc = auc_vs_benign(lab[te], nt.anomaly_scores(Q, R, k=args.k, codec=args.codec, method=method))
            rows.append({"w": w, "bytes": int(np.mean([len(o) for o in objs])), "method": method,
                         "f1": f1, "auc": auc})
        print(f"window {w}: done", file=sys.stderr)

    t = pd.DataFrame(rows).pivot_table(index=["w", "bytes"], columns="method", values=["f1", "auc"])
    print("| flows/window | bytes/window | macro-F1 concat | macro-F1 conditional | AUC concat | AUC conditional |")
    print("|---:|---:|---:|---:|---:|---:|")
    for (w, b), r in t.iterrows():
        print(f"| {w} | {b} | {r[('f1', 'concat')]:.3f} | {r[('f1', 'conditional')]:.3f} "
              f"| {r[('auc', 'concat')]:.3f} | {r[('auc', 'conditional')]:.3f} |")


if __name__ == "__main__":
    main()
