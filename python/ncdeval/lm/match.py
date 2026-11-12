"""PAQ/lpaq match model: predict that the most recent repeat of the last MINLEN symbols continues.

Mixture with the network: p(a) = (1 - w_L) p_LM(a) + w_L [a = predicted], w_L fitted per
match-length bucket. Gives the model explicit access to its context, i.e. to C(y|x).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MINLEN = 4
BUCKETS = (0, 4, 6, 8, 12, 16, 24, 32, 1 << 30)  # bucket i covers [BUCKETS[i], BUCKETS[i+1])


def bucket(length: int) -> int:
    for i in range(len(BUCKETS) - 1):
        if length < BUCKETS[i + 1]:
            return i
    return len(BUCKETS) - 2


class MatchState:
    def __init__(self) -> None:
        self.seq: list[int] = []
        self.table: dict[tuple, int] = {}
        self.ptr = 0
        self.length = 0

    def predict(self) -> tuple[int, int]:
        """(predicted symbol or -1, match length) for the next position."""
        if self.length > 0:
            return self.seq[self.ptr], self.length
        return -1, 0

    def update(self, sym: int) -> None:
        s = self.seq
        if self.length > 0 and s[self.ptr] == sym:
            self.length += 1
            self.ptr += 1
        else:
            self.length = 0
        s.append(sym)
        t = len(s) - 1
        if t + 1 >= MINLEN:
            key = tuple(s[t + 1 - MINLEN:])
            if self.length == 0:
                cand = self.table.get(key)
                if cand is not None:
                    self.ptr, self.length = cand + 1, MINLEN
            self.table[key] = t


def predictions(seq: Sequence[int], start: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted symbol (-1 if none) and match length for each position t >= start."""
    st = MatchState()
    n = len(seq)
    pred = np.full(n - start, -1, dtype=np.int64)
    mlen = np.zeros(n - start, dtype=np.int64)
    for t, s in enumerate(seq):
        if t >= start:
            pred[t - start], mlen[t - start] = st.predict()
        st.update(int(s))
    return pred, mlen


def fit_weights(p_actual: np.ndarray, hit: np.ndarray, mlen: np.ndarray) -> np.ndarray:
    """Per-bucket w maximizing sum log((1 - w) p + w [hit]) on calibration positions."""
    grid = np.concatenate([[0.0], np.linspace(0.01, 0.99, 99)])
    b = np.array([bucket(int(m)) for m in mlen])
    w = np.zeros(len(BUCKETS) - 1)
    for i in range(1, len(w)):
        sel = b == i
        if sel.sum() < 20:
            w[i] = w[i - 1]
            continue
        ll = [np.log((1 - g) * p_actual[sel] + g * hit[sel]).sum() for g in grid]
        w[i] = grid[int(np.argmax(ll))]
    return w


def weights_for(mlen: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.array([w[bucket(int(m))] for m in mlen]) if len(mlen) else np.zeros(0)
