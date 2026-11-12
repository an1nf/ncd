"""Decoder-only transformer with a key/value cache, shared by GPT-2 and the byte-level models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    block_size: int = 512
    gelu_tanh: bool = False  # GPT-2 uses the tanh approximation

    def to_dict(self) -> dict:
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, past=None):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        hd = c // self.n_head
        q, k, v = (z.view(b, t, self.n_head, hd).transpose(1, 2) for z in (q, k, v))
        if past is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pk, pv = past
            if pk.shape[0] != b:
                pk, pv = pk.expand(b, -1, -1, -1), pv.expand(b, -1, -1, -1)
            k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
            tp = pk.shape[2]
            # Query i sits at absolute position tp + i and sees keys 0..tp + i.
            mask = torch.ones(t, tp + t, dtype=torch.bool, device=x.device).tril(diagonal=tp)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.approx = "tanh" if cfg.gelu_tanh else "none"

    def forward(self, x, past=None):
        a, present = self.attn(self.ln_1(x), past)
        x = x + a
        x = x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate=self.approx))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel() + self.wte.weight.numel()
        return n

    def forward(self, idx, past=None):
        """idx: (B, T). past: list of per-layer (k, v) with batch 1 or B. Returns logits, present."""
        t0 = 0 if past is None else past[0][0].shape[2]
        t = idx.shape[1]
        if t0 + t > self.cfg.block_size:
            raise ValueError(f"sequence of {t0 + t} exceeds context {self.cfg.block_size}")
        pos = torch.arange(t0, t0 + t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        present = []
        for i, blk in enumerate(self.h):
            x, kv = blk(x, None if past is None else past[i])
            present.append(kv)
        # The output layer stays in float32 under bf16 autocast: GPT-2 logits reach |100|,
        # where bf16 resolution (0.5) would distort the probabilities the coder uses.
        with torch.autocast(device_type=idx.device.type, enabled=False):
            logits = F.linear(self.ln_f(x.float()), self.wte.weight.float())
        return logits, present


def bf16_ok(device) -> bool:
    """bf16 autocast only where it is native: CPUs, and NVIDIA GPUs from Ampere (sm_80) on.
    Older GPUs such as Pascal run float32 instead."""
    device = torch.device(device)
    if device.type != "cuda":
        return True
    return torch.cuda.get_device_capability(device)[0] >= 8


def load_checkpoint(path) -> GPT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def save_checkpoint(model: GPT, path, **extra) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), **extra}, path)
