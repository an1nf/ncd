"""Text corpora for the compression experiments: AG News documents and enwik8 slices.

AG News (Zhang et al., 2015) has 4 classes. A document is `title. description` with the
corpus' escape artefacts removed, cut to `cap` bytes including a final newline, so that two
documents plus a start symbol fit in a 512 byte context. Every compressor sees the same bytes.
"""

from __future__ import annotations

import csv
import gzip
import html
import re
from pathlib import Path

import numpy as np

AG_CLASSES = ["world", "sports", "business", "scitech"]
ENWIK8_TRAIN = 90_000_000
ENWIK8_TEST = 95_000_000


def clean(s: str) -> str:
    s = s.replace("\\", " ").replace(" #36;", " $").replace("#36;", "$").replace("#39;", "'")
    s = s.replace("quot;", '"').replace("&amp;", "&")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def ag_doc(title: str, desc: str, cap: int) -> bytes:
    t, d = clean(title), clean(desc)
    text = f"{t}. {d}" if t and not t.endswith((".", "?", "!")) else f"{t} {d}"
    raw = text.strip().encode("utf-8")[: cap - 1]
    raw = raw.decode("utf-8", errors="ignore").rstrip().encode("utf-8")
    return raw + b"\n"


def load_ag(path: Path, cap: int = 255) -> tuple[list[bytes], list[str]]:
    docs, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            docs.append(ag_doc(row[1], row[2], cap))
            labels.append(AG_CLASSES[int(row[0]) - 1])
    return docs, labels


def dedup(docs: list[bytes]) -> list[int]:
    """Indices of first occurrences; AG News has repeated stories."""
    seen, keep = set(), []
    for i, d in enumerate(docs):
        if d not in seen:
            seen.add(d)
            keep.append(i)
    return keep


def balanced_sample(labels: list[str], per_class: int, rng: np.random.Generator,
                    allowed: list[int] | None = None) -> list[int]:
    pool = np.array(allowed if allowed is not None else range(len(labels)))
    lab = np.array(labels, dtype=object)[pool]
    out = []
    for c in AG_CLASSES:
        idx = pool[lab == c]
        out.extend(rng.choice(idx, size=per_class, replace=False).tolist())
    return out


def enwik8(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()
