"""Bag-of-words baselines on the same few-shot AG News split (fit on the references only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

from common import knn_predict, report_dict
from run_text_experiment import load_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    task = load_task(args.task_dir / "task.jsonl")
    rx = [r["text"] for r in task["ref"]]
    ry = [r["label"] for r in task["ref"]]
    qx = [r["text"] for r in task["query"]]
    qy = [r["label"] for r in task["query"]]

    rows = []
    for name, vec in (("tfidf word 1-2gram", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                      ("tfidf char 2-5gram", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))):
        r = vec.fit_transform(rx)
        q = vec.transform(qx)
        pred = knn_predict(cosine_distances(q, r), ry, 1)
        rows.append({"name": f"1-NN cosine, {name}", **{k: report_dict(qy, pred)[k] for k in ("accuracy", "macro_f1")}})
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(r, ry)
        pred = list(lr.predict(q))
        rows.append({"name": f"logistic regression, {name}", **{k: report_dict(qy, pred)[k] for k in ("accuracy", "macro_f1")}})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "text_baselines.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['name']}: acc {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
