#!/usr/bin/env python3
"""Checks for lmcodec with a small random byte model; no downloads, a few seconds on CPU.

- the arithmetic coder round-trips random distributions over 2, 256 and 50,257 symbols;
- encode/decode round-trips with the model and the real code length matches the estimate;
- batched C(y | x) equals C(xy) - C(x) from a single forward pass;
- the NCD estimators agree where the algebra says they must: `conditional` equals `concat`
  when the reference is the simpler object, `concat-sym` is the mean of both orders, and
  `pairwise` is symmetric with a zero diagonal;
- with the match model mixed in, real codes still round-trip, and a document after itself
  costs a small fraction of coding it alone, i.e. NCD(x, x) is near 0 even for this model;
- if transformers is installed, the Hugging Face adapter (hf.py) gives the same conditional
  bits as a single forward pass and round-trips, with a random tiny GPT-NeoX built locally.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lmcodec import LMCodec
from lmcodec import arith, ncd
from lmcodec.model import GPT, GPTConfig


def tiny_codec(seed: int = 0) -> LMCodec:
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab=256, n_layer=2, n_head=2, n_embd=32, block_size=256))
    return LMCodec(model, "tiny", bf16=False)


def main() -> None:
    arith.self_test()
    c = tiny_codec()
    docs = [b"the quick brown fox\n", b"jumps over the lazy dog\n", bytes(range(200, 256)) + b"\n"]

    for row in c.verify(docs):
        assert row["roundtrip"], row
        assert abs(row["code"] - row["estimate"]) <= 1, row

    x, y = docs[0], docs[1]
    cond = c.cond_bits([x], [y])[0, 0]
    ids = c.bos + c.tokens(x + y)
    with torch.no_grad():
        lp = F.log_softmax(c.model(torch.tensor([ids]))[0][0, :-1], -1)
    bits = -lp.gather(-1, torch.tensor(ids[1:])[:, None])[:, 0].double() / np.log(2)
    direct = float(bits[len(c.tokens(x)):].sum())
    assert abs(cond - direct) < 1e-3, (cond, direct)

    rng = np.random.default_rng(2)
    objs = [bytes(rng.integers(97, 123, size=n).tolist()) + b"\n" for n in (30, 45, 60, 75)]
    d = ncd.cross(c, objs[:2], objs[2:], "concat")
    dc = ncd.cross(c, objs[:2], objs[2:], "conditional")
    lens = c.lengths(objs)
    ref_simpler = lens[2:][None, :] <= lens[:2][:, None]
    assert np.allclose(d[ref_simpler], dc[ref_simpler]), (d, dc)
    ds = ncd.cross(c, objs[:2], objs[2:], "concat-sym")
    dq = ncd.cross(c, objs[2:], objs[:2], "concat").T  # query-first order
    assert np.allclose(ds, (d + dq) / 2, atol=1e-6), (ds, d, dq)
    p = ncd.pairwise(c, objs)
    assert np.allclose(p, p.T) and np.all(np.diag(p) == 0)
    w = c.calibrate_match([(o, o) for o in objs] + [(objs[0], objs[1])])
    assert w[-1] > 0.5, w
    for row in c.verify(docs + objs[:1]):
        assert row["roundtrip"], row
        assert abs(row["code"] - row["estimate"]) <= 1, row
    dm = ncd.cross(c, objs, objs, "concat")
    assert np.all(np.diag(dm) < 0.5) and np.all(dm[~np.eye(4, dtype=bool)] > 0.8), dm
    check_hf()
    print("lmcodec ok")


def check_hf() -> None:
    try:
        import transformers
    except ImportError:
        print("transformers not installed: skipping the Hugging Face adapter check")
        return
    import tempfile
    from pathlib import Path

    from lmcodec.hf import load_hf

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Byte-level BPE with no merges: every byte is a token, no download needed.
        from lmcodec.bpe import bytes_to_unicode

        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        vocab = {ch: i for i, ch in enumerate(bytes_to_unicode().values())}
        vocab["<|endoftext|>"] = len(vocab)
        t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        transformers.PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<|endoftext|>",
                                             eos_token="<|endoftext|>").save_pretrained(d)
        cfg = transformers.GPTNeoXConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, intermediate_size=64, max_position_embeddings=256,
                                         bos_token_id=len(vocab) - 1, eos_token_id=len(vocab) - 1)
        torch.manual_seed(0)
        transformers.GPTNeoXForCausalLM(cfg).save_pretrained(d)
        model, htok, bos = load_hf(str(d), "cpu")
        c = LMCodec(model, "hf-tiny", htok, bos=bos, bf16=False)
        x, y = b"the quick brown fox\n", b"jumps over the lazy dog\n"
        cond = c.cond_bits([x], [y])[0, 0]
        ids = list(bos) + c.tokens(x) + c.tokens(y)
        with torch.no_grad():
            lp = F.log_softmax(model.hf(input_ids=torch.tensor([ids])).logits[0, :-1].float(), -1)
        bits = -lp.gather(-1, torch.tensor(ids[1:])[:, None])[:, 0].double() / np.log(2)
        direct = float(bits[len(c.tokens(x)):].sum())
        assert abs(cond - direct) < 1e-2, (cond, direct)
        for row in c.verify([x, y]):
            assert row["roundtrip"] and abs(row["code"] - row["estimate"]) <= 1, row


if __name__ == "__main__":
    main()
