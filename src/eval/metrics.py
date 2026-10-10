"""All per-run metrics.

Positive class = attack. Scores: higher = more attack-like.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }


def thresholded_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (np.asarray(scores) >= threshold).astype(np.int8)
    c = confusion(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        **c,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def compute_run_metrics(
    scores_test: np.ndarray,
    y_test: np.ndarray,
    scores_unseen: np.ndarray,
    thresholds: dict[str, float],
    primary_fpr_key: str,
) -> dict:
    """Metrics on the final test set (test + unseen) plus unseen-only and known-only views."""
    scores_test = np.asarray(scores_test, dtype=np.float64)
    scores_unseen = np.asarray(scores_unseen, dtype=np.float64)
    y_test = np.asarray(y_test).astype(np.int8)

    s_final = np.concatenate([scores_test, scores_unseen])
    y_final = np.concatenate([y_test, np.ones(len(scores_unseen), dtype=np.int8)])
    benign_test = scores_test[y_test == 0]
    known_attack_test = scores_test[y_test == 1]

    out: dict = {
        "roc_auc": float(roc_auc_score(y_final, s_final)),
        "pr_auc": float(average_precision_score(y_final, s_final)),
        "unseen_roc_auc": float(
            roc_auc_score(
                np.concatenate([np.zeros(len(benign_test), dtype=np.int8), np.ones(len(scores_unseen), dtype=np.int8)]),
                np.concatenate([benign_test, scores_unseen]),
            )
        ) if len(scores_unseen) else float("nan"),
        "n_test": int(len(scores_test)),
        "n_unseen": int(len(scores_unseen)),
        "at_fpr": {},
    }

    for key, thr in thresholds.items():
        m = thresholded_metrics(y_final, s_final, thr)
        udr_k = int(np.sum(scores_unseen >= thr))
        lo, hi = wilson_interval(udr_k, len(scores_unseen))
        m.update({
            "threshold": float(thr),
            "udr": udr_k / len(scores_unseen) if len(scores_unseen) else float("nan"),
            "udr_ci95": [lo, hi],
            "known_attack_recall": float(np.mean(known_attack_test >= thr)) if len(known_attack_test) else float("nan"),
            "realised_fpr_benign_test": float(np.mean(benign_test >= thr)) if len(benign_test) else float("nan"),
        })
        out["at_fpr"][key] = m

    out["primary"] = out["at_fpr"][primary_fpr_key]
    return out


def val_pr_auc(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Tuning objective: PR-AUC on the validation split (benign + known attacks)."""
    return float(average_precision_score(np.asarray(y_val).astype(np.int8), np.asarray(scores_val, dtype=np.float64)))
