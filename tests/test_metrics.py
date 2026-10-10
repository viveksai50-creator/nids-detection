from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import compute_run_metrics, thresholded_metrics, val_pr_auc, wilson_interval
from src.eval.thresholds import threshold_at_fpr, thresholds_for


def test_threshold_meets_requested_validation_fpr() -> None:
    scores = np.linspace(0.0, 1.0, 1_000, endpoint=False)
    threshold = threshold_at_fpr(scores, 0.01)
    assert np.mean(scores >= threshold) <= 0.01


def test_perfect_scores_have_perfect_unseen_detection() -> None:
    out = compute_run_metrics(
        scores_test=np.array([0.01, 0.02, 0.03, 0.99]),
        y_test=np.array([0, 0, 0, 1]),
        scores_unseen=np.array([0.90, 0.95]),
        thresholds={"0.01": 0.5},
        primary_fpr_key="0.01",
    )
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["primary"]["udr"] == pytest.approx(1.0)
    assert out["primary"]["realised_fpr_benign_test"] == pytest.approx(0.0)


def test_metrics_helpers_handle_boundary_cases() -> None:
    assert thresholded_metrics(np.array([0, 1]), np.array([0.0, 1.0]), 0.5)["fpr"] == 0.0
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # NaN
    assert val_pr_auc(np.array([0.1, 0.9]), np.array([0, 1])) == pytest.approx(1.0)
    assert set(thresholds_for(np.array([0.1, 0.2]), [0.01])) == {"0.01"}
