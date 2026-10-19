#!/usr/bin/env python3
"""Collect result JSONs (Rust ``ncd eval --json`` and the Python scripts)
into one Markdown table, plus ROC curves for the anomaly-detection results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

import style
from style import plt


def describe(doc: dict) -> tuple[str, str]:
    """(task, model) for both JSON flavours."""
    if "codec" in doc:  # Rust CLI
        return "malware-family", f"ncd-{doc['codec'].replace(':', '')}-{doc['method']}-knn{doc['k']}"
    return doc.get("task", "?"), doc.get("model", "?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, nargs="+", help="JSON files or directories")
    ap.add_argument("--out", type=Path, default=Path("results/summary.md"))
    args = ap.parse_args()
    style.apply()

    files = []
    for p in args.results:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    rows, rocs = [], []
    for f in files:
        doc = json.loads(f.read_text())
        task, model = describe(doc)
        m = doc.get("metrics") or {}
        rows.append((task, model, m.get("accuracy"), m.get("macro_f1"), doc.get("auc"), doc.get("seconds")))
        if doc.get("scores") is not None and doc.get("auc") is not None:
            rocs.append((model, doc["labels"], np.asarray(doc["scores"]), doc["auc"]))

    fmt = lambda v, p=4: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
    lines = ["| task | model | accuracy | macro-F1 | ROC AUC | seconds |", "|---|---|---:|---:|---:|---:|"]
    for task, model, acc, f1, auc, secs in sorted(rows, key=lambda r: (r[0], -(r[3] or r[4] or 0))):
        lines.append(f"| {task} | {model} | {fmt(acc)} | {fmt(f1)} | {fmt(auc)} | {fmt(secs, 1)} |")
    table = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    print(table)

    if rocs:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot([0, 1], [0, 1], color=style.AXIS, lw=1, ls="--", label="chance")
        for color, (model, labels, scores, auc) in zip(style.CATEGORICAL, sorted(rocs, key=lambda r: -r[3])):
            fpr, tpr, _ = roc_curve([l != "benign" for l in labels], scores)
            ax.plot(fpr, tpr, color=color, label=f"{model}  (AUC {auc:.3f})")
        ax.set_xlabel("false-positive rate (benign windows flagged)")
        ax.set_ylabel("true-positive rate (attacks caught)")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
        ax.legend(loc="lower right")
        ax.set_title("Anomaly detection trained on benign traffic only", loc="left")
        fig.tight_layout()
        out = args.out.with_name("roc.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
