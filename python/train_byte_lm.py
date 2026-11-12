"""Train a byte-level transformer language model on a byte stream (CPU or GPU).

The corpus is enwik8 (first 90 MB for training, the standard split) or any file of raw bytes
(`--corpus file`): text with one document per line, or binary data such as the benign flow
windows written by make_lm_corpus.py. Windows of `block` bytes are drawn at random. Every model
of a family sees the same number of bytes in the same order, so size is the only difference.
The context (`--block`) must hold two objects plus one start byte to be used for NCD.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lmcodec.model import GPT, GPTConfig, bf16_ok, save_checkpoint
from text_data import ENWIK8_TRAIN, enwik8


def batches(data: np.ndarray, block: int, batch: int, seed: int, device):
    rng = np.random.default_rng(seed)
    hi = len(data) - block - 1
    if hi <= 0:
        raise SystemExit(f"corpus of {len(data)} bytes is shorter than one block of {block}")
    while True:
        starts = rng.integers(0, hi, size=batch)
        x = torch.from_numpy(np.stack([data[s:s + block + 1] for s in starts]).astype(np.int64)).to(device)
        yield x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(model: GPT, data: np.ndarray, block: int, device) -> float:
    """Bits per byte on non-overlapping windows (a pessimistic, fast estimate)."""
    model.eval()
    if len(data) < 2:
        raise SystemExit("validation data is empty")
    block = min(block, len(data) - 1)  # a validation set shorter than one block is one window
    n = (len(data) - 1) // block
    x = torch.from_numpy(data[: n * block + 1].astype(np.int64)).to(device)
    tot = 0.0
    for i in range(0, n, 16):
        idx = torch.arange(i, min(n, i + 16))
        inp = torch.stack([x[j * block:(j + 1) * block] for j in idx])
        tgt = torch.stack([x[j * block + 1:(j + 1) * block + 1] for j in idx])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16_ok(device)):
            logits, _ = model(inp)
        tot += F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1), reduction="sum").item()
    model.train()
    return tot / (n * block) / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["enwik8", "file", "lines"], default="enwik8",
                    help="enwik8, or file (raw bytes; 'lines' is an alias)")
    ap.add_argument("--data", type=Path, required=True, help="enwik8.gz, or a raw byte file for --corpus file")
    ap.add_argument("--valid", type=Path, help="validation bytes for --corpus file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train-bytes", type=float, default=4e6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch's choice)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.corpus == "enwik8":
        raw = enwik8(args.data)
        train = np.frombuffer(raw[:ENWIK8_TRAIN], dtype=np.uint8)
        valid = np.frombuffer(raw[ENWIK8_TRAIN:ENWIK8_TRAIN + 131_072 + 1], dtype=np.uint8)
    else:
        train = np.frombuffer(args.data.read_bytes(), dtype=np.uint8)
        if args.valid is None:
            raise SystemExit("--corpus file needs --valid")
        valid = np.frombuffer(args.valid.read_bytes()[:131_073], dtype=np.uint8)

    cfg = GPTConfig(vocab=256, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    block_size=args.block)
    model = GPT(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    steps = int(args.train_bytes // (args.batch * args.block))
    print(f"{args.out.stem}: {model.n_params() / 1e6:.3f}M params "
          f"({model.n_params(non_embedding=True) / 1e6:.3f}M non-embedding), {steps} steps", flush=True)

    log = []
    t0 = time.time()
    it = batches(train, args.block, args.batch, args.seed, device)
    for step in range(1, steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(it)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16_ok(device)):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            v = evaluate(model, valid, args.block, device)
            el = time.time() - t0
            log.append({"step": step, "bytes": step * args.batch * args.block, "train_bpb": loss.item() / math.log(2),
                        "valid_bpb": v, "seconds": el})
            print(f"  step {step}/{steps} train {loss.item() / math.log(2):.3f} valid {v:.3f} bpb, {el:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.cpu(), args.out, tokenizer="bytes", corpus=args.corpus, train_bytes=steps * args.batch * args.block,
                    seconds=time.time() - t0, log=log, n_params=model.n_params())
    args.out.with_suffix(".json").write_text(json.dumps({"config": cfg.to_dict(), "n_params": model.n_params(),
                                                         "corpus": args.corpus, "log": log}, indent=1))


if __name__ == "__main__":
    main()
