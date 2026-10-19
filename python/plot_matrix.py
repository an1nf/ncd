#!/usr/bin/env python3
"""Visualize an NCD matrix written by ``ncd matrix``.

Produces ``<prefix>_heatmap.png`` (samples grouped by label; dark = similar)
and ``<prefix>_mds.png`` (2-D metric MDS embedding of the distances).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

import style
from style import plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix", type=Path, help=".npy from `ncd matrix`")
    ap.add_argument("--index", type=Path, default=None, help="defaults to <matrix>.index.csv")
    ap.add_argument("--out-prefix", type=Path, default=None)
    ap.add_argument("--title", default="NCD")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    index = args.index or args.matrix.with_suffix(".index.csv")
    prefix = args.out_prefix or args.matrix.with_suffix("")
    style.apply()

    D = np.load(args.matrix).astype(np.float64)
    idx = pd.read_csv(index)
    labels = idx["label"].astype(str).to_numpy()
    order = np.lexsort((np.arange(len(labels)), labels))
    D_sorted = D[np.ix_(order, order)]
    lab_sorted = labels[order]
    names = sorted(set(labels))
    colors = style.colors_for(names)

    # ---- heatmap ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(D_sorted, cmap=style.SIMILARITY_CMAP, vmin=0, vmax=max(1.0, np.percentile(D, 99)),
                   interpolation="nearest")
    ax.grid(False)
    bounds = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
    for b in bounds:
        ax.axhline(b - 0.5, color=style.SURFACE, lw=1.5)
        ax.axvline(b - 0.5, color=style.SURFACE, lw=1.5)
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(lab_sorted)]
    centers = (starts + ends - 1) / 2
    ax.set_yticks(centers, [lab_sorted[s] for s in starts])
    ax.set_xticks([])
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("NCD (0 = identical information)", color=style.INK_2)
    cb.ax.tick_params(colors=style.MUTED)
    ax.set_title(f"{args.title}: pairwise distances, grouped by label", loc="left")
    fig.tight_layout()
    fig.savefig(f"{prefix}_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- MDS ---------------------------------------------------------------
    Ds = (D + D.T) / 2
    np.fill_diagonal(Ds, 0)
    Ds = np.clip(Ds, 0, None)
    with warnings.catch_warnings():  # parameter renames across sklearn versions
        warnings.simplefilter("ignore", FutureWarning)
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed,
                 normalized_stress="auto", n_init=4).fit_transform(Ds)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name in names:
        m = labels == name
        ax.scatter(xy[m, 0], xy[m, 1], s=42, color=colors[name], edgecolor=style.SURFACE, linewidth=1.2,
                   label=f"{name} ({m.sum()})", zorder=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("MDS 1"), ax.set_ylabel("MDS 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"{args.title}: 2-D MDS embedding of the distance matrix", loc="left")
    fig.tight_layout()
    fig.savefig(f"{prefix}_mds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {prefix}_heatmap.png and {prefix}_mds.png")


if __name__ == "__main__":
    main()
