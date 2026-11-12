"""Language model + arithmetic coding as an NCD compressor.

C(x) is the byte length of the arithmetic code of x: a varint symbol count followed by the
coder output. With ideal code lengths C(xy) = C(x) + C(y|x).

NCD uses lengths computed from batched log-probabilities. `encode`/`decode` produce real codes
with incremental inference; `verify` checks that both agree.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import match
from .arith import FREQ_BITS, coded_bytes, decode_symbols, encode_symbols
from .model import bf16_ok, load_checkpoint

LN2 = float(np.log(2.0))
NEWLINE = 10


class LMCodec:
    def __init__(self, model, name: str, tokenizer=None, bos: Sequence[int] = (NEWLINE,),
                 batch_tokens: int = 8192, bf16: bool = True) -> None:
        """model(idx, past) -> (logits, present). tokenizer: encode(bytes)/decode(ids), None for bytes."""
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.bf16 = bf16
        self.match_w: np.ndarray | None = None  # match-model weight per length bucket
        self.name = name
        self.tok = tokenizer
        self.bos = list(bos)
        self.batch_tokens = int(os.environ.get("LMCODEC_BATCH_TOKENS", batch_tokens))

    @classmethod
    def load(cls, path, name: str | None = None, bf16: bool | None = None,
             device: str | None = None) -> LMCodec:
        """Byte-level checkpoint, or `hf:ORG/MODEL`. bf16 defaults on for checkpoints, off for HF."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(path).startswith("hf:"):
            from .hf import load_hf

            model, tok, bos = load_hf(str(path)[3:], device)
            return cls(model, name or str(path).split("/")[-1], tok, bos=bos, batch_tokens=4096, bf16=bool(bf16))
        model = load_checkpoint(path).to(device)
        return cls(model, name or Path(path).stem, bf16=True if bf16 is None else bf16)

    @property
    def context(self) -> int:
        return self.model.cfg.block_size

    def tokens(self, data: bytes) -> list[int]:
        return self.tok.encode(data) if self.tok else list(data)

    def detok(self, ids: Sequence[int]) -> bytes:
        return self.tok.decode(list(ids)) if self.tok else bytes(ids)

    def _amp(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16 and bf16_ok(self.device))

    def _t(self, x) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.long, device=self.device)

    # Information content

    @torch.no_grad()
    def _target_lp(self, prefix: list[int], targets: list[list[int]]) -> list[np.ndarray]:
        """ln p of each target token given prefix (with BOS); the prefix is encoded once."""
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
            lp = first[tgt[:, 0]][:, None]
            if lmax > 1:
                with self._amp():
                    logits, _ = self.model(tgt[:, :-1], past)
                lp = torch.cat([lp, F.log_softmax(logits.float(), -1).gather(-1, tgt[:, 1:, None])[..., 0]], 1)
            lp = lp.double().cpu().numpy()
            for r, j in enumerate(idx):
                out[j] = lp[r, :lens[r]]
        return out

    def _bits(self, lp: np.ndarray, seq: list[int], start: int) -> float:
        """Bits of seq[start:], mixed with the match model if enabled; capped at FREQ_BITS per symbol."""
        if self.match_w is None:
            b = -lp / LN2
        else:
            pred, mlen = match.predictions(seq, start)
            w = match.weights_for(mlen, self.match_w)
            b = -np.log2(np.maximum((1 - w) * np.exp(lp) + w * (pred == np.asarray(seq[start:])), 1e-300))
        return float(np.minimum(b, FREQ_BITS).sum())

    def _target_bits(self, prefix: list[int], targets: list[list[int]]) -> np.ndarray:
        lps = self._target_lp(prefix, targets)
        return np.array([self._bits(lp, prefix + t, len(prefix)) for lp, t in zip(lps, targets)])

    def bits(self, docs: Sequence[bytes]) -> np.ndarray:
        return self._target_bits(self.bos, [self.tokens(d) for d in docs])

    def cond_bits(self, prefixes: Sequence[bytes], targets: Sequence[bytes]) -> np.ndarray:
        """(P, T) bits of each target coded after BOS + prefix."""
        tt = [self.tokens(t) for t in targets]
        return np.stack([self._target_bits(self.bos + self.tokens(p), tt) for p in prefixes])

    def lengths(self, docs: Sequence[bytes]) -> np.ndarray:
        """C(x) in bytes. Inputs longer than the context are coded with a sliding window."""
        toks = [self.tokens(d) for d in docs]
        fits = [len(self.bos) + len(t) <= self.context for t in toks]
        bits = np.zeros(len(docs))
        short = [i for i, f in enumerate(fits) if f]
        if short:
            bits[short] = self._target_bits(self.bos, [toks[i] for i in short])
        for i, f in enumerate(fits):
            if not f:
                bits[i] = self.stream_bits(docs[i])[0]
        return np.array([coded_bytes(b, len(t)) for b, t in zip(bits, toks)])

    @torch.no_grad()
    def stream_bits(self, data: bytes, stride: int | None = None) -> tuple[float, int]:
        """(bits, tokens) left to right; each window codes its last `stride` tokens."""
        ids = self.bos + self.tokens(data)
        w = self.context
        stride = stride or w // 2
        jobs = []
        for s in [0] + list(range(stride, max(1, len(ids) - w + stride), stride)):
            seg = ids[s:s + w]
            first = 1 if s == 0 else w - stride
            if len(seg) <= first and s > 0:
                break
            jobs.append((s, seg, first))
        lp_all = np.zeros(len(ids))  # lp_all[t] = ln p(ids[t] | history)
        bsz = max(1, self.batch_tokens // w)
        for i in range(0, len(jobs), bsz):
            chunk = jobs[i:i + bsz]
            x = torch.zeros(len(chunk), max(len(seg) for _, seg, _ in chunk), dtype=torch.long)
            for r, (_, seg, _) in enumerate(chunk):
                x[r, :len(seg)] = torch.tensor(seg)
            x = x.to(self.device)
            with self._amp():
                logits, _ = self.model(x[:, :-1])
            lp = F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None])[..., 0].double().cpu().numpy()
            for r, (s0, seg, first) in enumerate(chunk):
                lp_all[s0 + first:s0 + len(seg)] = lp[r, first - 1:len(seg) - 1]
        n0 = len(self.bos)
        return self._bits(lp_all[n0:], ids, n0), len(ids) - n0  # match model sees the full history

    def calibrate_match(self, pairs: Sequence[tuple[bytes, bytes]]) -> np.ndarray:
        """Fit and enable match-model weights on (context, text) pairs."""
        p_all, hit_all, len_all = [], [], []
        for a, b in pairs:
            pre, tb = self.bos + self.tokens(a), self.tokens(b)
            lp = self._target_lp(pre, [tb])[0]
            pred, mlen = match.predictions(pre + tb, len(pre))
            p_all.append(np.exp(lp))
            hit_all.append(pred == np.asarray(tb))
            len_all.append(mlen)
        self.match_w = match.fit_weights(np.concatenate(p_all), np.concatenate(hit_all), np.concatenate(len_all))
        return self.match_w

    # Real codes

    def _stepper(self):
        state = {"past": None, "match": match.MatchState()}

        @torch.no_grad()
        def next_probs(prev):
            new = self.bos if prev is None else [prev]
            for t in new:
                state["match"].update(int(t))
            with self._amp():
                logits, state["past"] = self.model(self._t([new]), state["past"])
            p = torch.softmax(logits[0, -1].double(), -1).cpu().numpy()
            if self.match_w is not None:
                pred, length = state["match"].predict()
                if pred >= 0:
                    wt = self.match_w[match.bucket(length)]
                    p = (1 - wt) * p
                    p[pred] += wt
            return p

        return next_probs

    def encode(self, data: bytes) -> bytes:
        ids = self.tokens(data)
        if len(self.bos) + len(ids) > self.context:
            raise ValueError("input longer than the model context")
        return encode_symbols(ids, self._stepper())

    def decode(self, code: bytes) -> bytes:
        return self.detok(decode_symbols(code, self._stepper()))

    def verify(self, docs: Sequence[bytes]) -> list[dict]:
        """Real round trip and real vs. estimated code length."""
        rows = []
        for d, e in zip(docs, self.lengths(docs)):
            code = self.encode(d)
            rows.append({"bytes": len(d), "code": len(code), "estimate": int(e), "roundtrip": self.decode(code) == d})
        return rows
