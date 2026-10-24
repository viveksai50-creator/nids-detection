"""Regenerate descriptive tables/figures from saved final JSONs; never train.

Usage: python scripts/render_benchmark_outputs.py
Single-seed results are observations, not significance or deployment claims.
The underlying aggregate excludes dummy fixtures and nested diagnostic folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from src import config as C
from src.aggregate import load_long, write_tables


def main():
    df = load_long()
    C.ensure_dirs()
    df.to_csv(C.TABLES_DIR / "all_runs_long.csv", index=False)
    write_tables(df, latex=True)
    primary = f"udr@{C.PRIMARY_FPR:g}"
    seeds = sorted(df.seed.unique())
    scope = f"Seed {seeds[0]} observations (n=1 per model/family)" if len(seeds) == 1 else f"Descriptive means over {len(seeds)} seeds"
    labels = {"ae": "Autoencoder", "vae": "VAE", "rf": "Random Forest",
              "xgb": "XGBoost", "iforest": "Isolation Forest"}

    mat = df.groupby(["family", "model"])[primary].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(mat.columns)), [labels.get(m, m) for m in mat.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            value = mat.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="black" if value > 0.65 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Unseen Detection Rate (recall on held-out family)")
    ax.set_title("UDR at the 1% benign-validation FPR target\n" + scope, fontsize=12)
    fig.savefig(C.FIGURES_DIR / "f1_udr_heatmap.png", dpi=200)
    plt.close(fig)

    g = df.groupby("model").agg(udr=(primary, "mean"), latency=("batch_us_per_row", "median"))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.scatter(g.latency, g.udr, s=45)
    for model, row in g.iterrows():
        ax.annotate(labels.get(model, model), (row.latency, row.udr),
                    xytext=(6, 7), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    lo, hi = g.latency.min() * .65, g.latency.max() * 1.8
    ticks = [t for t in ticks if lo <= t <= hi]
    if ticks:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.15)
    ax.set_xlabel("Median batch latency (microseconds per flow; log scale)")
    ax.set_ylabel("Mean UDR across seven held-out families")
    ax.set_title("Detection versus latency at the 1% validation FPR target\n" + scope, fontsize=11)
    fig.savefig(C.FIGURES_DIR / "f6_udr_vs_latency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for model, sub in df.groupby("model"):
        ax.plot(C.TARGET_FPRS, [sub[f"udr@{q}"].mean() for q in C.TARGET_FPRS],
                marker="o", label=labels.get(model, model))
    ax.set_xscale("log")
    ax.set_xticks(C.TARGET_FPRS, [f"{100*q:g}%" for q in C.TARGET_FPRS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target FPR on benign validation traffic (log scale)")
    ax.set_ylabel("Mean UDR across seven held-out families")
    ax.set_title("Threshold sensitivity — " + scope, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    fig.savefig(C.FIGURES_DIR / "f7_udr_vs_fpr.png", dpi=200)
    plt.close(fig)
    print(f"Rendered {len(df)} saved records; no fitting, tuning or statistical tests performed")


if __name__ == "__main__":
    main()
