"""Figures for the text experiment, from analyze_text.py's text_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = {"gpt2": "GPT-2 124M (BPE)", "enwik8": "bytes, trained on enwik8", "agnews": "bytes, trained on AG News",
        "agpair": "bytes, AG News related pairs"}
CLASSIC_LINES = {"zstd:19": ("-", 3), "xz:6": ("--", -10)}
LABELED = {"zstd:19", "xz:6", "bzip2:9", "gpt2-124m"}


def base(fam: str) -> str:
    return fam.replace("+match", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.summary.read_text())["rows"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lms = [r for r in rows if r["params"] is not None]
    classic = {r["name"]: r for r in rows if r["params"] is None}
    order = [f for f in BASE if any(base(r["family"]) == f for r in lms)]
    order += sorted({base(r["family"]) for r in lms} - set(order))
    color = {f: style.CATEGORICAL[i % len(style.CATEGORICAL)] for i, f in enumerate(order)}
    for f in order:
        BASE.setdefault(f, f)

    def col(r):
        return style.NEUTRAL if r["params"] is None else color[base(r["family"])]

    def marker(r):
        if r["params"] is None:
            return "s"
        if r["family"] == "gpt2":
            return "*"
        return "^" if "+match" in r["family"] else "o"

    # 1. Bits per character against model size.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, title in ((axes[0], "bpc_doc", "One AG News document at a time"),
                           (axes[1], "bpc_stream", "1000 AG News documents as one stream")):
        for fam in sorted({r["family"] for r in lms}):
            rs = sorted([r for r in lms if r["family"] == fam], key=lambda r: r["params"])
            ls = "--" if "+match" in fam else "-"
            if fam == "gpt2" and len(rs) == 1:
                ax.plot([rs[0]["params"]], [rs[0][key]], "*", ms=14, color=color["gpt2"])
            else:
                ax.plot([r["params"] for r in rs], [r[key] for r in rs], marker=marker(rs[0]), ls=ls,
                        ms=5, color=color[base(fam)])
        for name, (ls, dy) in CLASSIC_LINES.items():
            if name in classic:
                y = classic[name][key]
                ax.axhline(y, color=style.MUTED, lw=1.2, ls=ls)
                ax.annotate(name, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, dy),
                            textcoords="offset points", ha="right", fontsize=8, color=style.INK_2)
        ax.set_xscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel("bits per character")
        ax.set_title(title)
    handles = [Line2D([], [], color=color[f], marker="*" if f == "gpt2" else "o", ls="none" if f == "gpt2" else "-",
                      ms=9 if f == "gpt2" else 5, label=BASE[f]) for f in order]
    handles.append(Line2D([], [], color=style.INK_2, marker="^", ls="--", ms=5, label="+ match model"))
    axes[1].legend(handles=handles, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out_dir / "text_bpc_size.png", dpi=150)

    # 2. Accuracy against compression and against the class signal.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, key, xlabel, title in (
            (axes[0], "bpc_doc", "bits per character, one document at a time (C(x) in NCD)", "Single-document compression"),
            (axes[1], "class_dprime", "class signal d': same-class vs other-class C(q | r)", "Use of the reference")):
        for r in rows:
            ax.errorbar(r[key], r["accuracy"], yerr=[[r["accuracy"] - r["acc_lo"]], [r["acc_hi"] - r["accuracy"]]],
                        fmt=marker(r), ms=12 if r["family"] == "gpt2" else 6, color=col(r), ecolor=style.GRID,
                        elinewidth=2, capsize=0, zorder=3)
            if r["name"] in LABELED:
                left = r["name"] == "xz:6" and key == "bpc_doc"
                ax.annotate(r["name"].replace("-124m", ""), (r[key], r["accuracy"]), xytext=(-6 if left else 5, -3),
                            textcoords="offset points", fontsize=7, color=style.INK_2, ha="right" if left else "left",
                            zorder=5)
        ax.axhline(0.25, color=style.MUTED, lw=1, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("1-NN accuracy, 16 references per class")
    handles = [Line2D([], [], color=color[f], marker="*" if f == "gpt2" else "o", ls="none", ms=9 if f == "gpt2" else 6,
                      label=BASE[f]) for f in order]
    handles += [Line2D([], [], color=style.INK_2, marker="^", ls="none", ms=6, label="+ match model"),
                Line2D([], [], color=style.NEUTRAL, marker="s", ls="none", ms=6, label="zstd, xz, bzip2, deflate")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "text_acc_vs_bpc.png", dpi=150)
    # 3. Accuracy against model size.
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for fam in sorted({r["family"] for r in lms}):
        rs = sorted([r for r in lms if r["family"] == fam], key=lambda r: r["params"])
        if fam == "gpt2" and len(rs) == 1:
            ax.errorbar([rs[0]["params"]], [rs[0]["accuracy"]], yerr=[[rs[0]["accuracy"] - rs[0]["acc_lo"]],
                        [rs[0]["acc_hi"] - rs[0]["accuracy"]]], fmt="*", ms=14, color=color["gpt2"], ecolor=style.GRID)
            continue
        ax.plot([r["params"] for r in rs], [r["accuracy"] for r in rs], marker=marker(rs[0]),
                ls="--" if "+match" in fam else "-", ms=5, color=color[base(fam)])
    for name, (ls, dy) in CLASSIC_LINES.items():
        if name in classic:
            y = classic[name]["accuracy"]
            ax.axhline(y, color=style.MUTED, lw=1.2, ls=ls)
            ax.annotate(name, (0, y), xycoords=("axes fraction", "data"), xytext=(4, dy),
                        textcoords="offset points", fontsize=8, color=style.INK_2)
    ax.axhline(0.25, color=style.MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("1-NN accuracy, 16 references per class")
    ax.set_title("Bigger model, better compression, same or worse NCD")
    handles = [Line2D([], [], color=color[f], marker="*" if f == "gpt2" else "o", ls="none" if f == "gpt2" else "-",
                      ms=9 if f == "gpt2" else 5, label=BASE[f]) for f in order]
    handles.append(Line2D([], [], color=style.INK_2, marker="^", ls="--", ms=5, label="+ match model"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "text_acc_size.png", dpi=150)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
