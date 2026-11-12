"""Does compression rate predict NCD k-NN accuracy? Reads the output of run_text_experiment.py.

Lengths are whole bytes and documents are short, so several references often tie at the
smallest distance (up to 79% of queries for the byte models). Taking the first one would favour
whichever class comes first; accuracy is instead the expected value under random tie-breaking
(a tie between k references counts 1/k per correct one), and macro-F1 uses a seeded random pick.

Per compressor: 1-NN accuracy and macro-F1 with a bootstrap interval over queries, accuracy
with 1 to 16 references per class (averaged over random subsets), the same-class AUC of the
distance and the class signal, i.e. how many bits per character a reference of the right
class saves over one of a wrong class when coding the query after it.

Across compressors: Spearman and Pearson correlation of accuracy with each compression
measure, overall and inside groups, with bootstrap intervals that resample queries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import report_dict
from run_text_experiment import load_task

# Paired comparisons on the same queries.
PAIRS = [("gpt2-124m", "zstd:19"), ("gpt2-124m", "xz:6"), ("agpair-b149k+match", "zstd:19"),
         ("agnews-b149k+match", "zstd:19"), ("enwik8-b2.8m+match", "enwik8-b2.8m"),
         ("agnews-b149k+match", "agnews-b2.8m+match"), ("agpair-b149k+match", "agpair-b2.8m+match"),
         ("enwik8-b2.8m+match", "enwik8-b037k+match"), ("zstd:19", "zstd:1")]
MEASURES = ["bpc_doc", "bpc_stream", "bpc_enwik8", "class_signal", "class_dprime", "same_class_auc"]


def tied_min(dist: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    return np.abs(dist - dist.min(axis=1, keepdims=True)) <= atol


def nn1_expected_hit(dist, ref_lab, q_lab) -> np.ndarray:
    """Per query: probability that 1-NN with random tie-breaking is right."""
    tied = tied_min(dist)
    return (tied & (ref_lab[None, :] == q_lab[:, None])).sum(1) / tied.sum(1)


def nn1_random(dist, ref_lab, rng) -> np.ndarray:
    tied = tied_min(dist)
    return np.array([ref_lab[rng.choice(np.flatnonzero(t))] for t in tied], dtype=object)


def shot_curve(dist, ref_lab, q_lab, shots, reps, rng):
    classes = sorted(set(ref_lab))
    by_c = {c: np.flatnonzero(ref_lab == c) for c in classes}
    out = {}
    for s in shots:
        accs = []
        for _ in range(reps if s < len(by_c[classes[0]]) else 1):
            cols = np.concatenate([rng.choice(by_c[c], size=s, replace=False) for c in classes])
            accs.append(nn1_expected_hit(dist[:, cols], ref_lab[cols], q_lab).mean())
        out[s] = float(np.mean(accs))
    return out


def class_signal(lens: dict, ref_lab, q_lab, q_bytes) -> float:
    """Mean over queries of [C(q|r) wrong class - C(q|r) right class] in bits per character."""
    cond = 8 * (lens["c_rq"] - lens["c_r"][:, None])  # refs x queries, bits
    same = ref_lab[:, None] == q_lab[None, :]
    diff = (np.where(~same, cond, 0).sum(0) / (~same).sum(0)) - (np.where(same, cond, 0).sum(0) / same.sum(0))
    return float(np.mean(diff / q_bytes))


def class_dprime(lens: dict, ref_lab, q_lab) -> float:
    """Same as class_signal, divided by the spread of C(q|r) over references (per query)."""
    cond = lens["c_rq"] - lens["c_r"][:, None]
    out = []
    for j in range(cond.shape[1]):
        same = ref_lab == q_lab[j]
        a, b = cond[same, j], cond[~same, j]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out.append((b.mean() - a.mean()) / sd if sd > 0 else 0.0)
    return float(np.mean(out))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    rho, p = stats.spearmanr(x[ok], y[ok])
    r = stats.pearsonr(x[ok], y[ok])[0]
    return float(rho), float(p), float(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    task = load_task(args.task_dir / "task.jsonl")
    ref_lab = np.array([r["label"] for r in task["ref"]], dtype=object)
    q_lab = np.array([r["label"] for r in task["query"]], dtype=object)
    q_bytes = np.array([len(r["text"].encode()) for r in task["query"]], float)
    nq = len(q_lab)
    boot_idx = rng.integers(0, nq, size=(args.boot, nq))

    rows, correct = [], {}
    for jf in sorted(args.results.glob("*.json")):
        res = json.loads(jf.read_text())
        name = res["name"]
        dist = np.load(args.results / f"{name}.ncd.npy")
        lens = dict(np.load(args.results / f"{name}.lens.npz"))
        pred = nn1_random(dist, ref_lab, np.random.default_rng(args.seed))
        rep = report_dict(list(q_lab), list(pred))
        hit = nn1_expected_hit(dist, ref_lab, q_lab)
        correct[name] = hit
        ba = hit[boot_idx].mean(1)
        same = (ref_lab[None, :] == q_lab[:, None]).ravel()
        row = {**{k: res.get(k) for k in ("name", "family", "params", "bpc_doc", "bpc_stream", "bpc_enwik8",
                                         "context_gain", "ncd_self", "t_ncd")},
               "accuracy": float(hit.mean()), "macro_f1": rep["macro_f1"],
               "ties": float((tied_min(dist).sum(1) > 1).mean()),
               "acc_lo": float(np.quantile(ba, 0.025)), "acc_hi": float(np.quantile(ba, 0.975)),
               "same_class_auc": float(roc_auc_score(same, -dist.ravel())),
               "class_signal": class_signal(lens, ref_lab, q_lab, q_bytes),
               "class_dprime": class_dprime(lens, ref_lab, q_lab),
               # Same reference subsets for every compressor, independent of which ones are present.
               "shots": shot_curve(dist, ref_lab, q_lab, [1, 2, 4, 8, 16], 200, np.random.default_rng(args.seed)),
               "confusion": rep["confusion"], "verify": res.get("verify")}
        rows.append(row)

    rows.sort(key=lambda r: (r["family"] != "gpt2", r["family"], r["params"] or 0, r["name"]))
    names = [r["name"] for r in rows]
    is_lm = np.array([r["params"] is not None for r in rows])
    acc = np.array([r["accuracy"] for r in rows])

    fam = [r["family"] for r in rows]
    has_match = np.array(["+match" in f for f in fam])
    groups = {"all": np.ones(len(rows), bool), "classical": ~is_lm,
              "LM alone": is_lm & ~has_match, "LM + match": has_match,
              "use context (classical, GPT-2, + match)": ~is_lm | has_match | np.array([f == "gpt2" for f in fam])}
    for f in sorted({f for f, l in zip(fam, is_lm) if l}):
        groups[f"family {f}"] = np.array([x == f for x in fam])
    hits = np.stack([correct[n] for n in names])  # compressors x queries
    boot_acc = np.stack([hits[:, b].mean(1) for b in boot_idx])  # boot x compressors

    corrs = []
    for g, m in groups.items():
        # Size only means something inside one family (same data, tokenizer and training).
        for meas in MEASURES + (["log_params"] if g.startswith("family") else []):
            if meas == "log_params":
                x = np.array([math.log10(r["params"]) for r, k in zip(rows, m) if k])
            else:
                x = np.array([r[meas] for r, k in zip(rows, m) if k], float)
            if m.sum() < 3:
                continue
            rho, p, r = corr(x, acc[m])
            bs = [stats.spearmanr(x, ba[m])[0] for ba in boot_acc[:500]]
            corrs.append({"group": g, "n": int(m.sum()), "measure": meas, "spearman": rho, "p": p, "pearson": r,
                          "spearman_lo": float(np.nanquantile(bs, 0.025)),
                          "spearman_hi": float(np.nanquantile(bs, 0.975))})

    paired = []
    for a, b in PAIRS:
        if a in correct and b in correct:
            d = correct[a] - correct[b]
            bd = d[boot_idx].mean(1)
            # Paired randomization test: flip the sign of each query's difference.
            flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
            p = float((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
            paired.append({"a": a, "b": b, "diff": float(d.mean()), "lo": float(np.quantile(bd, 0.025)),
                           "hi": float(np.quantile(bd, 0.975)), "p": max(p, 1 / 20000)})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "text_summary.json").write_text(json.dumps({"rows": rows, "correlations": corrs, "paired": paired}, indent=1))
    lines = ["| compressor | params | bpc doc | bpc stream | bpc enwik8 | NCD(x,x) | class signal | class d' | same-class AUC | ties | 1-NN acc [95% CI] | macro-F1 | 1-shot | 4-shot |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        p = "" if r["params"] is None else (f"{r['params'] / 1e6:.1f}M" if r["params"] >= 1e6 else f"{r['params'] / 1e3:.0f}K")
        lines.append(f"| {r['name']} | {p} | {r['bpc_doc']:.3f} | {r['bpc_stream']:.3f} | {r['bpc_enwik8']:.3f} | "
                     f"{r['ncd_self']:.3f} | {r['class_signal']:.3f} | {r['class_dprime']:.3f} | {r['same_class_auc']:.3f} | {r['ties']:.2f} | {r['accuracy']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}] | "
                     f"{r['macro_f1']:.3f} | {r['shots'][1]:.3f} | {r['shots'][4]:.3f} |")
    lines += ["", "| group | n | measure | Spearman ρ [95% CI] | p | Pearson r |", "|---|---:|---|---:|---:|---:|"]
    for c in corrs:
        lines.append(f"| {c['group']} | {c['n']} | {c['measure']} | {c['spearman']:+.2f} [{c['spearman_lo']:+.2f}, "
                     f"{c['spearman_hi']:+.2f}] | {c['p']:.3f} | {c['pearson']:+.2f} |")
    lines += ["", "| A | B | acc A - acc B [95% CI] | paired permutation p |", "|---|---|---:|---:|"]
    for c in paired:
        lines.append(f"| {c['a']} | {c['b']} | {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.2g} |")
    (args.out / "text_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
