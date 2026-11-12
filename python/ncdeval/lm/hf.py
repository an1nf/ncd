"""Hugging Face causal LMs as codec models (`hf:EleutherAI/pythia-160m`).

`past` is the token prefix: each call recomputes prefix + new tokens, which works for any
architecture at about twice the cost of a key/value cache. float32 by default
(`LMCODEC_HF_DTYPE=bfloat16` on Ampere+). Documents are tokenized separately and concatenated
as ids, a decodable code for the concatenation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class _Cfg:
    block_size: int


class HFAdapter(nn.Module):
    def __init__(self, model, block_size: int) -> None:
        super().__init__()
        self.hf = model
        self.cfg = _Cfg(block_size)

    def n_params(self, non_embedding: bool = False) -> int:
        return sum(p.numel() for p in self.hf.parameters())

    def forward(self, idx: torch.Tensor, past: torch.Tensor | None = None):
        dev = next(self.hf.parameters()).device
        idx = idx.to(dev)
        if past is not None:
            past = past.to(dev)
            full = torch.cat([past.expand(idx.shape[0], -1), idx], dim=1)
        else:
            full = idx
        if full.shape[1] > self.cfg.block_size:
            raise ValueError(f"sequence of {full.shape[1]} exceeds context {self.cfg.block_size}")
        logits = self.hf(input_ids=full, use_cache=False).logits.float()
        new = logits[:, full.shape[1] - idx.shape[1]:]
        return new, full[:1]


class HFTokenizer:
    """Lossless bytes <-> ids for UTF-8 text."""

    def __init__(self, tok) -> None:
        self.tok = tok

    def encode(self, data: bytes) -> list[int]:
        return self.tok.encode(data.decode("utf-8"), add_special_tokens=False)

    def decode(self, ids) -> bytes:
        return self.tok.decode(list(ids), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False).encode("utf-8")


def load_hf(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, os.environ.get("LMCODEC_HF_DTYPE", "float32"))
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    cfg = model.config
    block = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or 2048
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return HFAdapter(model, int(block)), HFTokenizer(tok), (int(bos),)
