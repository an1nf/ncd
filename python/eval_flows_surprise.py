#!/usr/bin/env python3
"""Anomaly score without references: bits per byte of each test window under a language
model trained only on benign traffic (make_lm_corpus.py + train_byte_lm.py).

Cost is linear in the number of test windows, against test x references for NCD k-NN.
Repetitive attacks (floods, scans) can be *more* predictable than normal traffic, so a second,
two-sided score measures the distance from the benign median in units of its MAD, both taken
from benign training windows. Writes the eval_flows_ncd.py result schema, so compare.py lists both.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import BENIGN, auc_vs_benign, print_summary, save_result
from lmcodec import LMCodec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="output dir of prepare_flows.py")
    ap.add_argument("--model", type=Path, required=True, help="lmcodec checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--max-benign", type=int, default=2000, help="benign training windows for the two-sided score")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    meta = pd.read_csv(args.data / "windows.csv")
    labels_all = meta["label"].to_numpy()
    te = np.flatnonzero(meta["split"] == "test")
    tr = np.flatnonzero((meta["split"] == "train") & (labels_all == BENIGN))
    tr = np.random.default_rng(args.seed).permutation(tr)[: args.max_benign]
    load = lambda i: (args.data / labels_all[i] / f"{meta['id'][i]}.bin").read_bytes()  # noqa: E731
    codec = LMCodec.load(args.model, device=args.device)

    t = time.perf_counter()
    scores = bits_per_byte(codec, [load(i) for i in te])
    secs = time.perf_counter() - t
    labels = labels_all[te]
    tag = f"surprise-{args.model.stem}"
    auc = auc_vs_benign(labels, scores)
    print_summary(f"{tag} ({len(te)} test windows)", None, auc)
    save_result(args.out_dir / f"flows_anomaly_{tag}.json", model=tag, task="flow-anomaly", auc=auc,
                seconds=secs, labels=labels.tolist(), scores=scores.tolist())

    ref = bits_per_byte(codec, [load(i) for i in tr])
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) or 1e-9
    two = np.abs(scores - med) / mad
    auc2 = auc_vs_benign(labels, two)
    print_summary(f"{tag}-twosided (benign median {med:.3f} bits/byte from {len(tr)} windows)", None, auc2)
    save_result(args.out_dir / f"flows_anomaly_{tag}-twosided.json", model=f"{tag}-twosided", task="flow-anomaly",
                auc=auc2, seconds=secs, labels=labels.tolist(), scores=two.tolist(), benign_median=med, benign_mad=mad)


def bits_per_byte(codec: LMCodec, objs: list[bytes]) -> np.ndarray:
    fits = [len(codec.bos) + len(o) <= codec.context for o in objs]
    bits = np.zeros(len(objs))
    short = [i for i, f in enumerate(fits) if f]
    if short:
        bits[short] = codec.bits([objs[i] for i in short])
    for i, f in enumerate(fits):
        if not f:
            bits[i] = codec.stream_bits(objs[i])[0]
    return bits / np.array([max(1, len(o)) for o in objs])


if __name__ == "__main__":
    main()
