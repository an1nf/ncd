"""One interface over the Rust codecs (via `ncdthreat`) and language-model codecs.

A spec is `zstd:19`, `xz:9`, `bzip2:9`, `deflate:9`, `lm:NAME=PATH` for a lmcodec
checkpoint or `lm:NAME=hf:ORG/MODEL` for a Hugging Face model, or `lm+match:NAME=PATH` for the same model mixed with a match model (match.py),
whose weights are fit on `calibration` pairs. `ref_concat_lengths(refs, queries)` returns `C(r q)` with the reference first, so
a language model codes each reference once and reuses its key/value cache for all queries.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from lmcodec.arith import coded_bytes


class Compressor:
    name: str
    family: str
    params: int | None = None

    def lengths(self, docs: Sequence[bytes]) -> np.ndarray: ...
    def stream_bytes(self, data: bytes) -> float: ...
    def ref_concat_lengths(self, refs: Sequence[bytes], queries: Sequence[bytes]) -> np.ndarray: ...


class RustCodec(Compressor):
    def __init__(self, spec: str) -> None:
        import ncdthreat as nt

        self.nt = nt
        self.name = spec
        self.family = spec.split(":")[0]

    def lengths(self, docs):
        return np.array([self.nt.compressed_len(d, self.name) for d in docs])

    def stream_bytes(self, data):
        return float(self.nt.compressed_len(data, self.name))

    def ref_concat_lengths(self, refs, queries):
        out = np.zeros((len(refs), len(queries)))
        for i, r in enumerate(refs):
            for j, q in enumerate(queries):
                out[i, j] = self.nt.compressed_len(r + q, self.name)
        return out


class LMCompressor(Compressor):
    def __init__(self, name: str, path: str, tokenizer_dir: Path | None, family: str,
                 calibration: Sequence[tuple[bytes, bytes]] | None = None, device: str | None = None) -> None:
        from lmcodec import LMCodec

        self.codec = LMCodec.load(path, tokenizer_dir, name=name, device=device)
        self.name = name
        self.family = family
        self.params = self.codec.model.n_params()
        self.match_weights = None
        if calibration is not None:
            self.match_weights = self.codec.calibrate_match(calibration).tolist()

    def _ntok(self, docs):
        return [len(self.codec.tokens(d)) for d in docs]

    def lengths(self, docs):
        return self.codec.lengths(docs)

    def stream_bytes(self, data):
        bits, n = self.codec.stream_bits(data)
        return float(coded_bytes(bits, n))

    def ref_concat_lengths(self, refs, queries):
        br = self.codec.bits(refs)
        cb = self.codec.cond_bits(refs, queries)
        nr, nq = self._ntok(refs), self._ntok(queries)
        out = np.zeros_like(cb)
        for i in range(len(refs)):
            for j in range(len(queries)):
                out[i, j] = coded_bytes(br[i] + cb[i, j], nr[i] + nq[j])
        return out


def make(spec: str, tokenizer_dir: Path | None = None,
         calibration: Sequence[tuple[bytes, bytes]] | None = None, device: str | None = None) -> Compressor:
    """`lm:NAME=PATH`, `lm:NAME=hf:ORG/MODEL` or the same with `lm+match:`; else a Rust codec.

    The family (for grouping in analyze_text.py) is the checkpoint's directory name, `gpt2` for
    the converted GPT-2, or the model name up to its first dash for Hugging Face models."""
    if spec.startswith(("lm:", "lm+match:")):
        kind, rest = spec.split(":", 1)
        name, path = rest.split("=", 1)
        if path.startswith("hf:"):
            family = path[3:].split("/")[-1].split("-")[0].lower()
        else:
            family = "gpt2" if "gpt2" in name else Path(path).parent.name
        if kind == "lm+match":
            if calibration is None:
                raise ValueError("lm+match needs calibration pairs")
            return LMCompressor(name, path, tokenizer_dir, family + "+match", calibration, device)
        return LMCompressor(name, path, tokenizer_dir, family, device=device)
    return RustCodec(spec)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t
