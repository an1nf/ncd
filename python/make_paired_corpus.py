"""Training text where each document is followed by its most similar document (TF-IDF cosine).

Plain news text gives a language model no reason to use the previous document: consecutive
documents are unrelated. In this corpus the second document of each pair is lexically related
to the first, so a model can lower its loss by looking back, which is what NCD needs from
`C(y | x)`. No labels are used. Exact and near-exact duplicates (cosine > 0.9) are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lm_train", type=Path, help="one document per line (prep_text_task.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=30_000)
    ap.add_argument("--max-sim", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = [d + b"\n" for d in args.lm_train.read_bytes().split(b"\n") if d][: args.docs]
    text = [d.decode("utf-8", errors="ignore") for d in docs]
    x = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english").fit_transform(text)
    nn = np.zeros(len(docs), dtype=np.int64)
    sims = np.zeros(len(docs))
    for i in range(0, len(docs), 2000):
        s = (x[i:i + 2000] @ x.T).toarray()
        s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])] = -1
        s[s > args.max_sim] = -1
        nn[i:i + 2000] = s.argmax(1)
        sims[i:i + 2000] = s.max(1)
    order = np.random.default_rng(args.seed).permutation(len(docs))
    args.out.write_bytes(b"".join(docs[i] + docs[nn[i]] for i in order))
    print(f"{len(docs)} pairs, median neighbour cosine {np.median(sims):.2f}, "
          f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
