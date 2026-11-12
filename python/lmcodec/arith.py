"""Integer arithmetic coder (Witten, Neal and Cleary, 1987) driven by model probabilities.

A model gives a float distribution over the alphabet at each step. `quantize` turns it into
integer frequencies with every symbol at least 1, so encoder and decoder agree exactly as long
as they see the same floats. The coder state has 48 bits and totals stay below 2^34, so the
loss from quantization is far below 0.01 bits per symbol even with a 50,257 token vocabulary.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

PREC = 48
FULL = (1 << PREC) - 1
HALF = 1 << (PREC - 1)
QUARTER = 1 << (PREC - 2)
FREQ_BITS = 32
FREQ_TOTAL = 1 << FREQ_BITS


def quantize(probs: np.ndarray) -> np.ndarray:
    """Cumulative integer frequencies, shape (V + 1,), cum[0] = 0."""
    p = np.asarray(probs, dtype=np.float64)
    v = p.shape[-1]
    f = np.floor(p * (FREQ_TOTAL - v)).astype(np.int64) + 1
    cum = np.zeros(v + 1, dtype=np.int64)
    np.cumsum(f, out=cum[1:])
    return cum


def quantized_bits(probs: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """-log2 of the quantized probability of each symbol. probs: (T, V), symbols: (T,)."""
    p = np.asarray(probs, dtype=np.float64)
    v = p.shape[-1]
    f = np.floor(p * (FREQ_TOTAL - v)) + 1
    tot = f.sum(axis=-1)
    fs = np.take_along_axis(f, symbols[:, None], axis=-1)[:, 0]
    return np.log2(tot) - np.log2(fs)


def coded_bytes(bits: float, n_symbols: int) -> int:
    """Length of `Encoder` output for a message with this information content.

    The varint header holds the symbol count; termination adds 2 bits; the tail is padded to
    a byte. Underflow handling can add at most one more byte, which the round-trip check bounds.
    """
    return varint_len(n_symbols) + math.ceil((bits + 2) / 8)


def varint_len(n: int) -> int:
    k = 1
    while n >= 0x80:
        n >>= 7
        k += 1
    return k


def write_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def read_varint(data: bytes) -> tuple[int, int]:
    n = shift = 0
    for i, b in enumerate(data):
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, i + 1
    raise ValueError("truncated varint")


class Encoder:
    def __init__(self) -> None:
        self.low, self.high, self.pending = 0, FULL, 0
        self.bits: list[int] = []

    def _emit(self, b: int) -> None:
        self.bits.append(b)
        if self.pending:
            self.bits.extend([1 - b] * self.pending)
            self.pending = 0

    def encode(self, cum: np.ndarray, sym: int) -> None:
        total = int(cum[-1])
        lo, hi = int(cum[sym]), int(cum[sym + 1])
        assert hi > lo, "zero-frequency symbol"
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


class Decoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.low, self.high = 0, FULL
        self.value = 0
        for _ in range(PREC):
            self.value = (self.value << 1) | self._bit()

    def _bit(self) -> int:
        i = self.pos
        self.pos += 1
        if i >= 8 * len(self.data):
            return 0
        return (self.data[i >> 3] >> (7 - (i & 7))) & 1

    def decode(self, cum: np.ndarray) -> int:
        total = int(cum[-1])
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // rng
        sym = int(np.searchsorted(cum, target, side="right")) - 1
        lo, hi = int(cum[sym]), int(cum[sym + 1])
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._bit()
        return sym


def encode_symbols(symbols: Sequence[int], next_probs: Callable[[int | None], np.ndarray]) -> bytes:
    """`next_probs(prev)` returns the distribution of the next symbol after feeding `prev`
    (None at the start). The output starts with the symbol count as a varint."""
    enc = Encoder()
    prev = None
    for s in symbols:
        enc.encode(quantize(next_probs(prev)), int(s))
        prev = int(s)
    return write_varint(len(symbols)) + enc.finish()


def decode_symbols(code: bytes, next_probs: Callable[[int | None], np.ndarray]) -> list[int]:
    n, k = read_varint(code)
    dec = Decoder(code[k:])
    out: list[int] = []
    prev = None
    for _ in range(n):
        s = dec.decode(quantize(next_probs(prev)))
        out.append(s)
        prev = s
    return out


def self_test(seed: int = 0) -> None:
    """Round trip with a random, skewed, context-free model over a large alphabet."""
    rng = np.random.default_rng(seed)
    for v, n in [(2, 500), (256, 400), (50257, 60)]:
        logits = rng.normal(size=(n, v)) * 4
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        syms = np.array([rng.choice(v, p=probs[t]) for t in range(n)])
        it: Iterable[np.ndarray] = iter(probs)
        code = encode_symbols(syms, lambda _p, it=it: next(it))
        it2 = iter(probs)
        back = decode_symbols(code, lambda _p, it2=it2: next(it2))
        assert list(syms) == back, f"round trip failed for V={v}"
        bits = quantized_bits(probs, syms).sum()
        pred = coded_bytes(bits, n)
        assert abs(len(code) - pred) <= 1, (len(code), pred)


if __name__ == "__main__":
    self_test()
    print("arith ok")
