"""Turn the ONNX Model Zoo GPT-2 (gpt2-lm-head-10.onnx, 124M) into a lmcodec checkpoint.

The zoo file is the only GPT-2 copy reachable from package registries and GitHub. Its
initializers keep the Hugging Face names, with Conv1D weights stored as (in, out).
The converted model is checked against onnxruntime on a sample before saving.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from lmcodec.bpe import EOT, GPT2Tokenizer
from lmcodec.model import GPT, GPTConfig, save_checkpoint

SAMPLE = b"Wall St. Bears Claw Back Into the Black. Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    g = onnx.load(str(args.onnx)).graph
    w = {t.name: numpy_helper.to_array(t) for t in g.initializer if t.name.startswith("transformer.")}
    n_layer = 1 + max(int(m.group(1)) for k in w if (m := re.match(r"transformer\.h\.(\d+)\.", k)))
    cfg = GPTConfig(vocab=50257, n_layer=n_layer, n_head=12, n_embd=768, block_size=1024, gelu_tanh=True)
    model = GPT(cfg)

    sd = {"wte.weight": w["transformer.wte.weight"], "wpe.weight": w["transformer.wpe.weight"],
          "ln_f.weight": w["transformer.ln_f.weight"], "ln_f.bias": w["transformer.ln_f.bias"]}
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        for ln in ("ln_1", "ln_2"):
            sd[f"h.{i}.{ln}.weight"] = w[p + ln + ".weight"]
            sd[f"h.{i}.{ln}.bias"] = w[p + ln + ".bias"]
        for src, dst in (("attn.c_attn", "attn.c_attn"), ("attn.c_proj", "attn.c_proj"),
                         ("mlp.c_fc", "c_fc"), ("mlp.c_proj", "c_proj")):
            sd[f"h.{i}.{dst}.weight"] = w[p + src + ".weight"].T  # Conv1D (in, out) -> Linear (out, in)
            sd[f"h.{i}.{dst}.bias"] = w[p + src + ".bias"]
    model.load_state_dict({k: torch.from_numpy(np.array(v, copy=True)) for k, v in sd.items()})
    model.eval()

    tok = GPT2Tokenizer(args.tokenizer_dir / "encoder.json", args.tokenizer_dir / "vocab.bpe")
    ids = [EOT] + tok.encode(SAMPLE)
    assert tok.decode(ids[1:]) == SAMPLE
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.onnx))
    ref = sess.run(None, {sess.get_inputs()[0].name: np.array([[ids]], dtype=np.int64)})[0][0, 0]
    with torch.no_grad():
        out = model(torch.tensor([ids]))[0][0].numpy()
    err = float(np.abs(out - ref).max())
    lp = torch.log_softmax(torch.from_numpy(out), -1)
    nll = -lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].sum().item() / np.log(2)
    print(f"max |logit - onnxruntime| = {err:.2e}; sample: {8 * len(SAMPLE)} raw bits -> {nll:.0f} bits "
          f"({nll / len(SAMPLE):.3f} bits/char)")
    assert err < 1e-2, "converted weights disagree with onnxruntime"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.out, tokenizer="gpt2", source=args.onnx.name)
    print(f"saved {args.out} ({model.n_params() / 1e6:.1f}M parameters)")


if __name__ == "__main__":
    main()
