"""A language model plus arithmetic coding as an NCD compressor.

`C(x)` is the length in bytes of the arithmetic code of `x`: the symbol count as a varint,
then the coder output. With ideal code lengths `C(xy) = C(x) + C(y | x)`, so a model that
uses its context makes `C(xy)` small when `x` and `y` share information.

Lengths for NCD come from one batched forward pass per prefix (`coded_bytes` of the
information content). `encode` and `decode` produce real codes with incremental decoding,
which is what a decoder can reproduce; `verify` checks that both agree.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import match
from .arith import FREQ_BITS, coded_bytes, decode_symbols, encode_symbols
from .bpe import EOT, GPT2Tokenizer
from .model import bf16_ok, load_checkpoint

LN2 = float(np.log(2.0))
NEWLINE = 10


@dataclass
class Stats:
    tokens: int = 0
    seconds: float = 0.0


class LMCodec:
    def __init__(self, model, name: str, tokenizer=None, bos: Sequence[int] = (NEWLINE,),
                 batch_tokens: int = 8192, bf16: bool = True) -> None:
        """`model(idx, past) -> (logits, present)` (model.py, or hf.HFAdapter); `tokenizer` has
        `encode(bytes) -> ids` and `decode(ids) -> bytes`, or is None for bytes."""
        self.model = model.eval()
        self.device = next(model.parameters()).device
        # bf16 for the transformer body (AMX on recent Xeons), float32 for the output layer.
        self.bf16 = bf16
        # Match-model weights per match-length bucket (match.py); None = language model only.
        self.match_w: np.ndarray | None = None
        self.name = name
        self.tok = tokenizer
        self.bos = list(bos)
        # Tokens per forward pass; raise it on a large GPU with LMCODEC_BATCH_TOKENS.
        self.batch_tokens = int(os.environ.get("LMCODEC_BATCH_TOKENS", batch_tokens))
        self.stats = Stats()

    def _amp(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16 and bf16_ok(self.device))

    def _t(self, x) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.long, device=self.device)

    @classmethod
    def load(cls, path, tokenizer_dir: Path | None = None, name: str | None = None,
             bf16: bool | None = None, device: str | None = None) -> "LMCodec":
        """A lmcodec checkpoint, or `hf:ORG/MODEL` for a Hugging Face causal LM (hf.py).

        `device` defaults to CUDA when available. bf16 defaults to on for lmcodec checkpoints
        (their output layer stays in float32) and off for Hugging Face models."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(path).startswith("hf:"):
            from .hf import load_hf

            model, tok, bos = load_hf(str(path)[3:], device)
            return cls(model, name or str(path)[3:].split("/")[-1], tok, bos=bos, batch_tokens=4096,
                       bf16=bool(bf16))
        bf16 = True if bf16 is None else bf16
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = load_checkpoint(path).to(device)
        name = name or Path(path).stem
        if ck.get("tokenizer") == "gpt2":
            if tokenizer_dir is None:
                raise ValueError("GPT-2 checkpoints need --tokenizer-dir")
            tok = GPT2Tokenizer(tokenizer_dir / "encoder.json", tokenizer_dir / "vocab.bpe")
            return cls(model, name, tok, bos=(EOT,), batch_tokens=2048, bf16=bf16)
        return cls(model, name, bf16=bf16)

    @property
    def context(self) -> int:
        return self.model.cfg.block_size

    def tokens(self, data: bytes) -> list[int]:
        return self.tok.encode(data) if self.tok else list(data)

    def detok(self, ids: Sequence[int]) -> bytes:
        return self.tok.decode(list(ids)) if self.tok else bytes(ids)

    # ---- information content -------------------------------------------------

    @torch.no_grad()
    def _target_lp(self, prefix: list[int], targets: list[list[int]]) -> list[np.ndarray]:
        """Natural-log probability of each token of each target, given `prefix` (with BOS)."""
        out: list[np.ndarray] = [np.zeros(0)] * len(targets)
        with self._amp():
            logits_p, past = self.model(self._t([prefix]))
        first = F.log_softmax(logits_p[0, -1].float(), -1)
        order = np.argsort([len(t) for t in targets], kind="stable")
        bsz = max(1, self.batch_tokens // max(1, max((len(t) for t in targets), default=1)))
        for i in range(0, len(order), bsz):
            idx = order[i:i + bsz]
            lens = [len(targets[j]) for j in idx]
            lmax = max(lens)
            if lmax == 0:
                continue
            if len(prefix) + lmax > self.context:
                raise ValueError(f"{self.name}: {len(prefix) + lmax} tokens exceed context {self.context}")
            tgt = torch.zeros(len(idx), lmax, dtype=torch.long)
            for r, j in enumerate(idx):
                tgt[r, :lens[r]] = torch.tensor(targets[j])
            tgt = tgt.to(self.device)
            lp0 = first[tgt[:, 0]]
            if lmax > 1:
                with self._amp():
                    logits, _ = self.model(tgt[:, :-1], past)
                lp = F.log_softmax(logits.float(), -1).gather(-1, tgt[:, 1:, None])[..., 0]
                lp = torch.cat([lp0[:, None], lp], 1)
            else:
                lp = lp0[:, None]
            lp = lp.double().cpu().numpy()
            for r, j in enumerate(idx):
                out[j] = lp[r, :lens[r]]
            self.stats.tokens += sum(lens) + len(prefix)
        return out

    def _bits(self, lp: np.ndarray, seq: list[int], start: int) -> float:
        """Bits for tokens seq[start:] with natural-log LM probabilities `lp`, mixing in the
        match model if enabled. The quantized coder never spends more than FREQ_BITS per symbol."""
        if self.match_w is not None:
            pred, mlen = match.predictions(seq, start)
            w = match.weights_for(mlen, self.match_w)
            hit = pred == np.asarray(seq[start:])
            p = (1 - w) * np.exp(lp) + w * hit
            b = -np.log2(np.maximum(p, 1e-300))
        else:
            b = -lp / LN2
        return float(np.minimum(b, FREQ_BITS).sum())

    def _target_bits(self, prefix: list[int], targets: list[list[int]]) -> np.ndarray:
        """Bits of each target sequence given `prefix` (which includes BOS)."""
        lps = self._target_lp(prefix, targets)
        return np.array([self._bits(lp, prefix + t, len(prefix)) for lp, t in zip(lps, targets)])

    def bits(self, docs: Sequence[bytes]) -> np.ndarray:
        t0 = time.perf_counter()
        b = self._target_bits(self.bos, [self.tokens(d) for d in docs])
        self.stats.seconds += time.perf_counter() - t0
        return b

    def lengths(self, docs: Sequence[bytes]) -> np.ndarray:
        """`C(x)` in bytes for each doc."""
        b = self.bits(docs)
        return np.array([coded_bytes(bi, len(self.tokens(d))) for bi, d in zip(b, docs)])

    def cond_bits(self, prefixes: Sequence[bytes], targets: Sequence[bytes]) -> np.ndarray:
        """(P, T) matrix of bits of each target coded right after BOS + prefix."""
        t0 = time.perf_counter()
        tt = [self.tokens(t) for t in targets]
        m = np.stack([self._target_bits(self.bos + self.tokens(p), tt) for p in prefixes])
        self.stats.seconds += time.perf_counter() - t0
        return m

    @torch.no_grad()
    def stream_bits(self, data: bytes, stride: int | None = None) -> tuple[float, int]:
        """Bits for a long text coded left to right with a sliding window.

        The first window codes all its tokens; later windows code their last `stride` tokens
        with `context - stride` tokens of history. A decoder can follow the same schedule.
        """
        ids = self.bos + self.tokens(data)
        w = self.context
        stride = stride or w // 2
        starts = [0] + list(range(stride, max(1, len(ids) - w + stride), stride))
        jobs = []
        for s in starts:
            seg = ids[s:s + w]
            first_coded = 1 if s == 0 else w - stride
            if len(seg) <= first_coded and s > 0:
                break
            jobs.append((s, seg, first_coded))
        lp_all = np.zeros(len(ids))  # lp_all[t] = log p(ids[t] | history)
        bsz = max(1, self.batch_tokens // w)
        for i in range(0, len(jobs), bsz):
            chunk = jobs[i:i + bsz]
            lmax = max(len(seg) for _, seg, _ in chunk)
            x = torch.zeros(len(chunk), lmax, dtype=torch.long)
            for r, (_, seg, _) in enumerate(chunk):
                x[r, :len(seg)] = torch.tensor(seg)
            x = x.to(self.device)
            with self._amp():
                logits, _ = self.model(x[:, :-1])
            lp = F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None])[..., 0].double().cpu().numpy()
            for r, (s0, seg, fc) in enumerate(chunk):
                lp_all[s0 + fc:s0 + len(seg)] = lp[r, fc - 1:len(seg) - 1]
            self.stats.tokens += int(x.numel())
        n0 = len(self.bos)
        # The match model sees the whole history, like an LZ window, not just the model context.
        return self._bits(lp_all[n0:], ids, n0), len(ids) - n0

    # ---- real codes ----------------------------------------------------------

    def _stepper(self):
        state = {"past": None, "match": match.MatchState()}
        bos = self.bos

        @torch.no_grad()
        def next_probs(prev):
            inp = self._t([bos]) if prev is None else self._t([[prev]])
            for t in (bos if prev is None else [prev]):
                state["match"].update(int(t))
            with self._amp():
                logits, state["past"] = self.model(inp, state["past"])
            p = torch.softmax(logits[0, -1].double(), -1).cpu().numpy()
            if self.match_w is not None:
                pred, length = state["match"].predict()
                if pred >= 0:
                    wt = self.match_w[match.bucket(length)]
                    p = (1 - wt) * p
                    p[pred] += wt
            return p

        return next_probs

    def calibrate_match(self, pairs: Sequence[tuple[bytes, bytes]]) -> np.ndarray:
        """Fit and enable match-model weights on (context, text) pairs, e.g. related documents."""
        p_all, hit_all, len_all = [], [], []
        for a, b in pairs:
            pre = self.bos + self.tokens(a)
            tb = self.tokens(b)
            lp = self._target_lp(pre, [tb])[0]
            pred, mlen = match.predictions(pre + tb, len(pre))
            p_all.append(np.exp(lp))
            hit_all.append(pred == np.asarray(tb))
            len_all.append(mlen)
        self.match_w = match.fit_weights(np.concatenate(p_all), np.concatenate(hit_all), np.concatenate(len_all))
        return self.match_w

    def encode(self, data: bytes) -> bytes:
        ids = self.tokens(data)
        if len(self.bos) + len(ids) > self.context:
            raise ValueError("input longer than the model context")
        return encode_symbols(ids, self._stepper())

    def decode(self, code: bytes) -> bytes:
        return self.detok(decode_symbols(code, self._stepper()))

    def verify(self, docs: Sequence[bytes]) -> list[dict]:
        """Real round trip; compares real code length with the batched estimate."""
        est = self.lengths(docs)
        rows = []
        for d, e in zip(docs, est):
            code = self.encode(d)
            back = self.decode(code)
            rows.append({"bytes": len(d), "code": len(code), "estimate": int(e), "roundtrip": back == d})
        return rows
