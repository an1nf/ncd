"""Drop-in for the `ncdthreat` functions used by the scripts, plus language-model codecs.

`codec` is either an ncd-core codec (`zstd:19`, `xz:9`, ...), handled by the Rust bindings,
or `lm:PATH` for a lmcodec checkpoint (byte-level models, or GPT-2 with `encoder.json` and
`vocab.bpe` next to the checkpoint or in `$LMCODEC_TOKENIZER_DIR`), or `lm:hf:ORG/MODEL`.
`$LMCODEC_DEVICE` picks cpu or cuda (default: cuda if available).

    import ncd_backend as nt
    D = nt.cross(queries, refs, codec="lm:models/agnews/b6.5m.pt")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


def is_lm(codec: str) -> bool:
    return codec.startswith("lm:")


@lru_cache(maxsize=4)
def _lm(codec: str):
    from lmcodec import LMCodec

    device = os.environ.get("LMCODEC_DEVICE")
    if codec[3:].startswith("hf:"):
        return LMCodec.load(codec[3:], device=device)
    path = Path(codec[3:])
    tok = path.parent if (path.parent / "encoder.json").exists() else os.environ.get("LMCODEC_TOKENIZER_DIR")
    return LMCodec.load(path, Path(tok) if tok else None, device=device)


def compressed_len(data: bytes, codec: str = "zstd:19") -> int:
    if is_lm(codec):
        return int(_lm(codec).lengths([data])[0])
    import ncdthreat

    return ncdthreat.compressed_len(data, codec)


def cross(queries: Sequence[bytes], refs: Sequence[bytes], codec: str = "zstd:19", method: str = "concat") -> np.ndarray:
    if is_lm(codec):
        from lmcodec import ncd

        return ncd.cross(_lm(codec), queries, refs, method)
    import ncdthreat

    return ncdthreat.cross(queries, refs, codec=codec, method=method)


def pairwise(objs: Sequence[bytes], codec: str = "zstd:19", method: str = "concat") -> np.ndarray:
    if is_lm(codec):
        from lmcodec import ncd

        return ncd.pairwise(_lm(codec), objs, method)
    import ncdthreat

    return ncdthreat.pairwise(objs, codec=codec, method=method)


def anomaly_scores(queries: Sequence[bytes], refs: Sequence[bytes], k: int = 5, codec: str = "zstd:19",
                   method: str = "concat") -> np.ndarray:
    """Mean NCD to the k nearest references, as in ncd-core."""
    if not is_lm(codec):
        import ncdthreat

        return ncdthreat.anomaly_scores(queries, refs, k=k, codec=codec, method=method)
    d = cross(queries, refs, codec, method)
    k = min(k, d.shape[1])
    return np.sort(d, axis=1)[:, :k].mean(axis=1)
