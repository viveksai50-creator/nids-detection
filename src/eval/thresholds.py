"""Select score thresholds to target a benign validation false-positive rate."""
from __future__ import annotations

import numpy as np


def threshold_at_fpr(benign_val_scores: np.ndarray, target_fpr: float) -> float:
    """Score threshold such that ~target_fpr of benign validation rows score >= threshold.

    Uses the (1 - q) quantile with the 'higher' interpolation, then advances one
    representable float.  The latter is essential for discrete scores (such as
    tree probabilities): with an inclusive ``score >= threshold`` decision rule,
    a large tie at the quantile would otherwise exceed the requested validation
    FPR.  Scores: higher = more attack-like.
    """
    s = np.asarray(benign_val_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no benign validation scores")
    boundary = float(np.quantile(s, 1.0 - target_fpr, method="higher"))
    return float(np.nextafter(boundary, np.inf))


def thresholds_for(benign_val_scores: np.ndarray, target_fprs: list[float]) -> dict[str, float]:
    return {f"{q:g}": threshold_at_fpr(benign_val_scores, q) for q in target_fprs}
