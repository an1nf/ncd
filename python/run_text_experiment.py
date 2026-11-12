"""Bits per character and NCD matrices for a list of compressors on the text task.

For each compressor it measures:

- bits per character on held-out AG News documents compressed one at a time (`bpc_doc`,
  what NCD sees as C(x)), on the same documents as one stream (`bpc_stream`), and on
  100 KB of the enwik8 test split (`bpc_enwik8`);
- the query x reference NCD matrix with the reference first, `C(r q)`;
- for language models, a real arithmetic-coding round trip on a few documents.

Results go to OUT/<name>.json, OUT/<name>.ncd.npy and the raw lengths in OUT/<name>.lens.npz; `analyze_text.py` reads them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compressors import LMCompressor, Timer, make


def load_task(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.setdefault(r["role"], []).append(r)
    return rows


def calibration_pairs(task_dir: Path, n_near: int = 200, n_random: int = 100, seed: int = 0):
    """Related and unrelated document pairs from lm_valid.txt (no task documents, no labels)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [d + b"\n" for d in (task_dir / "lm_valid.txt").read_bytes().split(b"\n") if d]
    x = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform([d.decode("utf-8", "ignore") for d in docs])
    s = (x @ x.T).toarray()
    np.fill_diagonal(s, -1)
    s[s > 0.9] = -1
    rng = np.random.default_rng(seed)
    a = rng.choice(len(docs), size=n_near + n_random, replace=False)
    b = np.concatenate([s[a[:n_near]].argmax(1), rng.choice(len(docs), size=n_random)])
    return [(docs[i], docs[j]) for i, j in zip(a, b)]


def ncd_matrix(c_r: np.ndarray, c_q: np.ndarray, c_rq: np.ndarray) -> np.ndarray:
    """queries x refs."""
    lo = np.minimum(c_r[:, None], c_q[None, :])
    hi = np.maximum(c_r[:, None], c_q[None, :])
    return ((c_rq - lo) / hi).T.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path)
    ap.add_argument("--verify-docs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("compressors", nargs="+",
                    help="zstd:19, xz:9, ..., lm:NAME=checkpoint.pt, lm:NAME=hf:ORG/MODEL, lm+match:NAME=...")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    task = load_task(args.task_dir / "task.jsonl")
    enc = lambda rows: [r["text"].encode("utf-8") for r in rows]  # noqa: E731
    refs, queries, held = enc(task["ref"]), enc(task["query"]), enc(task["heldout"])
    stream = b"".join(held)
    wiki = (args.task_dir / "heldout_enwik8.bin").read_bytes()
    args.out.mkdir(parents=True, exist_ok=True)

    calib = None
    if any(c.startswith("lm+match:") for c in args.compressors):
        calib = calibration_pairs(args.task_dir)
    for spec in args.compressors:
        name = spec.split(":", 1)[1].split("=", 1)[0] if spec.startswith(("lm:", "lm+match:")) else spec
        out_json = args.out / f"{name}.json"
        print(f"== {name}", flush=True)
        if args.skip_existing and out_json.exists():
            print(f"{name}: exists, skipped")
            continue
        comp = make(spec, args.tokenizer_dir, calib, args.device)
        res: dict = {"name": comp.name, "spec": spec, "family": comp.family, "params": comp.params,
                     "match_weights": getattr(comp, "match_weights", None)}
        with Timer() as t:
            held_len = comp.lengths(held)
        res["bpc_doc"] = float(8 * held_len.sum() / len(stream))
        res["mean_doc_bytes"] = float(np.mean([len(d) for d in held]))
        res["mean_doc_code_bytes"] = float(held_len.mean())
        res["t_doc"] = t.seconds
        with Timer() as t:
            res["bpc_stream"] = 8 * comp.stream_bytes(stream) / len(stream)
            res["bpc_enwik8"] = 8 * comp.stream_bytes(wiki) / len(wiki)
        res["t_stream"] = t.seconds
        print(f"{comp.name}: bpc doc {res['bpc_doc']:.3f}, stream {res['bpc_stream']:.3f}, "
              f"enwik8 {res['bpc_enwik8']:.3f} ({res['t_doc'] + res['t_stream']:.0f}s)", flush=True)

        with Timer() as t:
            c_r, c_q = comp.lengths(refs), comp.lengths(queries)
            c_rq = comp.ref_concat_lengths(refs, queries)
        res["t_ncd"] = t.seconds
        d = ncd_matrix(c_r.astype(float), c_q.astype(float), c_rq)
        np.save(args.out / f"{comp.name}.ncd.npy", d)
        np.savez(args.out / f"{comp.name}.lens.npz", c_r=c_r, c_q=c_q, c_rq=c_rq)
        # How much of a query the reference explains, on average: 1 - C(q|r)/C(q).
        res["context_gain"] = float(np.mean(1 - (c_rq - c_r[:, None]) / c_q[None, :]))
        res["ncd_self"] = float(np.mean([(comp.ref_concat_lengths([q], [q])[0, 0] - cq) / cq
                                         for q, cq in zip(queries[:20], c_q[:20])]))
        print(f"  NCD {d.shape[0]}x{d.shape[1]} in {t.seconds:.0f}s, context gain {res['context_gain']:.3f}, "
              f"NCD(x,x) {res['ncd_self']:.3f}", flush=True)

        if isinstance(comp, LMCompressor) and args.verify_docs:
            rows = comp.codec.verify(held[: args.verify_docs])
            res["verify"] = rows
            ok = all(r["roundtrip"] for r in rows)
            gap = max(abs(r["code"] - r["estimate"]) for r in rows)
            print(f"  arithmetic coding round trip {'ok' if ok else 'FAILED'}, max |real - estimate| = {gap} bytes")
        out_json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
