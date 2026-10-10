"""Create reproducible EDA figures from the cleaned parquet dataset.

Run: ``python -m src.data.eda``.  Figures are intentionally data-root independent
and are saved under ``results/figures/eda`` in the repository.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data.split import META_COLUMNS, load_processed


EDA_DIR = C.FIGURES_DIR / "eda"


def _save(fig: plt.Figure, name: str) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(EDA_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _counts(df: pd.DataFrame) -> None:
    counts = df["family"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values, color="#2a6fbb")
    ax.set_xscale("log")
    ax.set_xlabel("Flows (log scale)")
    ax.set_title("Cleaned CIC-IDS2017 flows by family")
    _save(fig, "family_counts.png")

    labels = df["label_raw"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.32)))
    ax.barh(labels.index, labels.values, color="#4682b4")
    ax.set_xscale("log")
    ax.set_xlabel("Flows (log scale)")
    ax.set_title("Cleaned CIC-IDS2017 flows by raw label")
    _save(fig, "raw_label_counts.png")


def _feature_ranges(df: pd.DataFrame, features: list[str]) -> None:
    # Quantile spread makes columns with radically different scales comparable.
    q = df[features].quantile([0.01, 0.50, 0.99]).T
    q["spread"] = (q[0.99] - q[0.01]).abs()
    top = q.nlargest(20, "spread").sort_values("spread")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hlines(top.index, top[0.01], top[0.99], color="#4c78a8", linewidth=2)
    ax.plot(top[0.50], top.index, "o", color="#f58518", label="median")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Value (1st to 99th percentile; symlog scale)")
    ax.set_title("Twenty widest numeric feature ranges")
    ax.legend()
    _save(fig, "feature_ranges.png")


def _correlation(df: pd.DataFrame, features: list[str]) -> None:
    # Sampling bounds memory and keeps the same rows across reruns.
    sample = df[features].sample(n=min(50_000, len(df)), random_state=0)
    variances = sample.var().nlargest(30).index
    corr = sample[variances].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Correlation of 30 highest-variance features (50k-flow sample)")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    _save(fig, "correlation_heatmap.png")


def main() -> int:
    if not C.PROCESSED_FILE.exists():
        raise FileNotFoundError(f"{C.PROCESSED_FILE} missing; run python -m src.data.clean first")
    df = load_processed()
    features = [c for c in df.columns if c not in META_COLUMNS]
    _counts(df)
    _feature_ranges(df, features)
    _correlation(df, features)
    summary = {
        "rows": int(len(df)),
        "features": len(features),
        "families": df["family"].value_counts().to_dict(),
        "constant_columns": [c for c in features if df[c].nunique(dropna=False) <= 1],
        "near_constant_columns": [c for c in features if df[c].nunique(dropna=False) / len(df) < 0.001],
    }
    with open(EDA_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
