"""GPT-2 byte-level BPE (Radford et al., 2019). Lossless on arbitrary bytes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex

PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
EOT = 50256


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class GPT2Tokenizer:
    def __init__(self, encoder_json: Path, vocab_bpe: Path) -> None:
        self.encoder: dict[str, int] = json.loads(Path(encoder_json).read_text(encoding="utf-8"))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = Path(vocab_bpe).read_text(encoding="utf-8").split("\n")[1:-1]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[str]:
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            a, b = parts[i], parts[i + 1]
            out, j = [], 0
            while j < len(parts):
                if j < len(parts) - 1 and parts[j] == a and parts[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(parts[j])
                    j += 1
            parts = out
        return parts

    def encode(self, data: bytes) -> list[int]:
        # Byte-level: decode with surrogateescape so invalid UTF-8 still round-trips.
        text = data.decode("utf-8", errors="surrogateescape")
        ids: list[int] = []
        for piece in PAT.findall(text):
            if piece in self.cache:
                ids.extend(self.cache[piece])
                continue
            raw = piece.encode("utf-8", errors="surrogateescape")
            word = "".join(self.b2u[b] for b in raw)
            toks = [self.encoder[t] for t in self._bpe(word)]
            if len(self.cache) < 200_000:
                self.cache[piece] = toks
            ids.extend(toks)
        return ids

    def decode(self, ids: list[int]) -> bytes:
        return bytes(self.u2b[c] for i in ids for c in self.decoder[i])
