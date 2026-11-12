"""NCD estimators over an LMCodec.

concat: C(rq), reference first so its key/value cache serves every query (ncd-core puts the
query first). concat-sym: mean of both orders. conditional: C(y|x)/C(y) with the simpler
object as x. Prefixes that do not fit the context are cut from the left and counted.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from .arith import coded_bytes
from .codec import LMCodec


class _Lens:
    def __init__(self, codec: LMCodec, objs: Sequence[bytes]) -> None:
        self.toks = [codec.tokens(o) for o in objs]
        too_long = [len(t) for t in self.toks if len(codec.bos) + len(t) > codec.context]
        if too_long:
            raise ValueError(f"{len(too_long)} objects exceed the {codec.context}-token context of {codec.name} "
                             f"(longest {max(too_long)})")
        self.bits = codec._target_bits(codec.bos, self.toks)
        self.c = np.array([coded_bytes(b, len(t)) for b, t in zip(self.bits, self.toks)], float)


def _cond(codec: LMCodec, prefixes: list[list[int]], targets: list[list[int]], stats: dict) -> np.ndarray:
    """bits(target | BOS + prefix) for all pairs; prefixes are cut from the left to fit."""
    out = np.zeros((len(prefixes), len(targets)))
    room = codec.context - len(codec.bos)
    longest = max(len(t) for t in targets)
    for i, p in enumerate(prefixes):
        keep = max(0, min(len(p), room - longest))
        if keep < len(p):
            stats["overflow"] += sum(len(p) + len(t) > room for t in targets)
        out[i] = codec._target_bits(codec.bos + p[len(p) - keep:], targets)
    return out


def cross(codec: LMCodec, queries: Sequence[bytes], refs: Sequence[bytes], method: str = "concat") -> np.ndarray:
    """queries x refs NCD matrix (float32), not clamped."""
    q, r = _Lens(codec, queries), _Lens(codec, refs)
    stats = {"overflow": 0}
    lo = np.minimum(q.c[:, None], r.c[None, :])
    hi = np.maximum(q.c[:, None], r.c[None, :])
    nq = np.array([len(t) for t in q.toks])[:, None]
    nr = np.array([len(t) for t in r.toks])[None, :]

    def joint(first_bits, cond_bits, n):
        return np.vectorize(coded_bytes)(first_bits + cond_bits, n).astype(float)

    if method in ("concat", "concat-sym", "conditional"):
        c_rq = joint(r.bits[None, :], _cond(codec, r.toks, q.toks, stats).T, nq + nr)
    if method in ("concat-sym", "conditional"):
        c_qr = joint(q.bits[:, None], _cond(codec, q.toks, r.toks, stats), nq + nr)
    if method == "concat":
        d = (c_rq - lo) / hi
    elif method == "concat-sym":
        d = ((c_rq + c_qr) / 2 - lo) / hi
    elif method == "conditional":
        # Prefix = simpler object; then (C(xy) - C(x)) / C(y) = C(y|x) / C(y).
        use_r = r.c[None, :] <= q.c[:, None]
        d = np.where(use_r, c_rq - r.c[None, :], c_qr - q.c[:, None]) / hi
    else:
        raise ValueError(f"unknown method {method!r}")
    if stats["overflow"]:
        print(f"warning: {stats['overflow']} pairs exceed the {codec.context}-token context of {codec.name}; "
              "their prefix was cut", file=sys.stderr)
    return d.astype(np.float32)


def pairwise(codec: LMCodec, objs: Sequence[bytes], method: str = "concat") -> np.ndarray:
    """Symmetric matrix: the mean of both orders for `concat`, zero diagonal."""
    m = "concat-sym" if method == "concat" else method
    d = cross(codec, objs, objs, m)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d
