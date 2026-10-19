"""Shared helpers: metrics in the same JSON schema the Rust CLI emits."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

BENIGN = "benign"


def sanitize_label(label: str) -> str:
    """'Web Attack \x96 Brute Force' -> 'web_attack_brute_force'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(label).strip().lower()).strip("_")
    return s or "unknown"


def report_dict(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> dict:
    """Accuracy, macro/weighted F1, per-class metrics and confusion matrix."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f * s).sum() / max(s.sum(), 1)),
        "per_class": [
            {"label": l, "precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
            for l, pi, ri, fi, si in zip(labels, p, r, f, s)
        ],
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def auc_vs_benign(labels: Iterable[str], scores: np.ndarray) -> float | None:
    positive = np.asarray([l != BENIGN for l in labels])
    if positive.all() or not positive.any():
        return None
    return float(roc_auc_score(positive, scores))


def knn_predict(dist: np.ndarray, train_labels: Sequence[str], k: int) -> list[str]:
    """Majority vote of the k nearest columns; ties go to the nearest class."""
    train_labels = np.asarray(train_labels, dtype=object)
    k = min(k, dist.shape[1])
    nn = np.argpartition(dist, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(dist, nn, axis=1).argsort(axis=1, kind="stable")
    nn = np.take_along_axis(nn, order, axis=1)
    preds = []
    for row in nn:
        votes: dict[str, list[int]] = {}
        for rank, j in enumerate(row):
            votes.setdefault(train_labels[j], []).append(rank)
        preds.append(max(votes.items(), key=lambda kv: (len(kv[1]), -kv[1][0]))[0])
    return preds


def save_result(path: Path, *, model: str, task: str, metrics: dict | None = None, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"model": model, "task": task, "metrics": metrics, **extra}
    path.write_text(json.dumps(doc, indent=2, default=_jsonable))
    print(f"wrote {path}")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def print_summary(name: str, metrics: dict | None, auc: float | None = None) -> None:
    parts = [name]
    if metrics:
        parts.append(f"acc={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")
    if auc is not None:
        parts.append(f"AUC={auc:.4f}")
    print("  ".join(parts))
