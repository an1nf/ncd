"""`ncdthreat` interface with language-model codecs.

`codec` is an ncd-core spec (`zstd:19`, `xz:9`, ...) or `lm:PATH` for a byte-level checkpoint
(ncdeval.lm.train) or `lm:hf:ORG/MODEL` for a Hugging Face causal LM.
`$LMCODEC_DEVICE` selects cpu or cuda.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import ncdthreat
import numpy as np


def is_lm(codec: str) -> bool:
    return codec.startswith("lm:")


@lru_cache(maxsize=4)
def _lm(codec: str):
    from .lm import LMCodec

    return LMCodec.load(codec[3:], device=os.environ.get("LMCODEC_DEVICE"))


def compressed_len(data: bytes, codec: str = "zstd:19") -> int:
    if is_lm(codec):
        return int(_lm(codec).lengths([data])[0])
    return ncdthreat.compressed_len(data, codec)


def cross(queries: Sequence[bytes], refs: Sequence[bytes], codec: str = "zstd:19",
          method: str = "concat") -> np.ndarray:
    if is_lm(codec):
        from .lm import ncd

        return ncd.cross(_lm(codec), queries, refs, method)
    return ncdthreat.cross(queries, refs, codec=codec, method=method)


def pairwise(objs: Sequence[bytes], codec: str = "zstd:19", method: str = "concat") -> np.ndarray:
    if is_lm(codec):
        from .lm import ncd

        return ncd.pairwise(_lm(codec), objs, method)
    return ncdthreat.pairwise(objs, codec=codec, method=method)


def anomaly_scores(queries: Sequence[bytes], refs: Sequence[bytes], k: int = 5, codec: str = "zstd:19",
                   method: str = "concat") -> np.ndarray:
    """Mean NCD to the k nearest references."""
    if not is_lm(codec):
        return ncdthreat.anomaly_scores(queries, refs, k=k, codec=codec, method=method)
    d = cross(queries, refs, codec, method)
    return np.sort(d, axis=1)[:, :min(k, d.shape[1])].mean(axis=1)
