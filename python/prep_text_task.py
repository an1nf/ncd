"""Build the text task: few-shot AG News classification with NCD k-NN.

Writes `task.jsonl` (references from the AG train split, queries and a held-out set from the
test split, balanced by class), `heldout_enwik8.bin` (the first bytes of the enwik8 test
split) and, for news-domain language models, `lm_train.txt` / `lm_valid.txt`: AG train
documents in random order, without the references and without any document identical to
an evaluation document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from text_data import ENWIK8_TEST, balanced_sample, dedup, enwik8, load_ag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-dir", type=Path, required=True, help="directory with train.csv and test.csv")
    ap.add_argument("--enwik8", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs-per-class", type=int, default=16)
    ap.add_argument("--queries-per-class", type=int, default=75)
    ap.add_argument("--heldout-per-class", type=int, default=250)
    ap.add_argument("--enwik8-bytes", type=int, default=100_000)
    ap.add_argument("--cap", type=int, default=255)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tr_docs, tr_lab = load_ag(args.ag_dir / "train.csv", args.cap)
    te_docs, te_lab = load_ag(args.ag_dir / "test.csv", args.cap)
    tr_keep, te_keep = dedup(tr_docs), dedup(te_docs)

    refs = balanced_sample(tr_lab, args.refs_per_class, rng, tr_keep)
    queries = balanced_sample(te_lab, args.queries_per_class, rng, te_keep)
    rest = sorted(set(te_keep) - set(queries))
    heldout = balanced_sample(te_lab, args.heldout_per_class, rng, rest)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "task.jsonl", "w", encoding="utf-8") as f:
        for role, docs, labs, idx in (("ref", tr_docs, tr_lab, refs), ("query", te_docs, te_lab, queries),
                                      ("heldout", te_docs, te_lab, heldout)):
            for i in idx:
                f.write(json.dumps({"role": role, "label": labs[i], "src_index": int(i),
                                    "text": docs[i].decode("utf-8")}) + "\n")

    banned = {tr_docs[i] for i in refs} | {te_docs[i] for i in queries + heldout}
    pool = [tr_docs[i] for i in tr_keep if tr_docs[i] not in banned]
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]
    (args.out / "lm_valid.txt").write_bytes(b"".join(pool[:2000]))
    (args.out / "lm_train.txt").write_bytes(b"".join(pool[2000:]))

    raw = enwik8(args.enwik8)
    (args.out / "heldout_enwik8.bin").write_bytes(raw[ENWIK8_TEST:ENWIK8_TEST + args.enwik8_bytes])
    print(f"refs {len(refs)}, queries {len(queries)}, heldout {len(heldout)}, "
          f"lm_train {len(pool) - 2000} docs ({sum(map(len, pool[2000:])) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
