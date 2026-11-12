#!/usr/bin/env python3
"""Byte corpora for training a language-model compressor on the project's own data.

--flows DIR   benign training windows from prepare_flows.py, in file order (no test windows,
              no attacks), so the model learns only what normal traffic looks like.
--files DIR   every file under DIR, concatenated (e.g. benign code sections or documents).

Writes OUT/train.bin and OUT/valid.bin (the last --valid-frac of the objects).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BENIGN = "benign"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--flows", type=Path)
    src.add_argument("--files", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--max-bytes", type=float, default=200e6)
    args = ap.parse_args()

    if args.flows:
        meta = pd.read_csv(args.flows / "windows.csv")
        meta = meta[(meta["split"] == "train") & (meta["label"] == BENIGN)]
        paths = [args.flows / lab / f"{wid}.bin" for wid, lab in zip(meta["id"], meta["label"])]
    else:
        paths = sorted(p for p in args.files.rglob("*") if p.is_file())
    if len(paths) < 2:
        raise SystemExit("need at least 2 objects")
    objs, total = [], 0
    for p in paths:
        b = p.read_bytes()
        objs.append(b)
        total += len(b)
        if total >= args.max_bytes:
            break
    k = max(1, int(len(objs) * args.valid_frac))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.bin").write_bytes(b"".join(objs[:-k]))
    (args.out / "valid.bin").write_bytes(b"".join(objs[-k:]))
    lens = sorted(len(o) for o in objs)
    print(f"{len(objs) - k} train / {k} valid objects, {total / 1e6:.1f} MB; median object {lens[len(lens) // 2]} bytes, "
          f"max {lens[-1]}. For NCD the model context (--block) must exceed 2 x object size + 1.")


if __name__ == "__main__":
    main()
