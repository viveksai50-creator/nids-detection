"""Statistical comparison (REVIEW-FEEDBACK.md §4.4): Friedman + Nemenyi, Wilcoxon + Holm.

Input: long DataFrame with columns family, model, seed, value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _pivot(df: pd.DataFrame, block: str, value: str = "value") -> pd.DataFrame:
    """rows = blocks (seed or family), cols = models, cells = value (mean over the other index)."""
    return df.pivot_table(index=block, columns="model", values=value, aggfunc="mean").dropna(axis=0, how="any")


def friedman(df: pd.DataFrame, block: str) -> dict:
    """Friedman test across models with 'block' ('seed' within a family, or 'family' overall)."""
    mat = _pivot(df, block)
    if mat.shape[1] < 3 or mat.shape[0] < 2:
        return {"n_blocks": int(mat.shape[0]), "n_models": int(mat.shape[1]), "statistic": float("nan"), "p_value": float("nan")}
    stat, p = stats.friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    ranks = mat.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    return {"n_blocks": int(mat.shape[0]), "n_models": int(mat.shape[1]), "statistic": float(stat),
            "p_value": float(p), "mean_rank": ranks.to_dict()}


def nemenyi(df: pd.DataFrame, block: str) -> pd.DataFrame | None:
    try:
        import scikit_posthocs as sp
    except ImportError:
        return None
    mat = _pivot(df, block)
    if mat.shape[1] < 3 or mat.shape[0] < 2:
        return None
    return sp.posthoc_nemenyi_friedman(mat.to_numpy()).set_axis(mat.columns, axis=0).set_axis(mat.columns, axis=1)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def paired_wilcoxon(df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Paired over (family, seed). Effect size: rank-biserial correlation."""
    a = df[df.model == model_a].set_index(["family", "seed"])["value"]
    b = df[df.model == model_b].set_index(["family", "seed"])["value"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    d = d[d != 0]
    if len(d) < 5:
        return {"a": model_a, "b": model_b, "n_pairs": int(len(d)), "p_value": float("nan"), "rank_biserial": float("nan")}
    stat, p = stats.wilcoxon(d, alternative="two-sided")
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    rb = (r_pos - r_neg) / (r_pos + r_neg)
    return {"a": model_a, "b": model_b, "n_pairs": int(len(d)), "mean_diff": float(np.mean(d)),
            "statistic": float(stat), "p_value": float(p), "rank_biserial": float(rb)}


def targeted_comparisons(df: pd.DataFrame, target: str, baselines: list[str]) -> list[dict]:
    res = [paired_wilcoxon(df, target, b) for b in baselines]
    adj = holm([r["p_value"] if not np.isnan(r["p_value"]) else 1.0 for r in res])
    for r, pa in zip(res, adj):
        r["p_holm"] = float(pa)
    return res