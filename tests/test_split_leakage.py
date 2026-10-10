from __future__ import annotations

import numpy as np

from src.data.split import assert_no_leakage, build_split


def test_held_out_family_never_reaches_train_validation_or_test(synthetic_processed) -> None:
    split = build_split("DoS", seed=0, df=synthetic_processed, cache=False)
    assert_no_leakage(split)
    assert "DoS" not in split.fam_train
    assert "DoS" not in split.fam_val
    assert "DoS" not in split.fam_test
    assert np.all(split.fam_unseen == "DoS")


def test_scaler_is_fitted_on_training_rows_only(synthetic_processed) -> None:
    split = build_split("DDoS", seed=1, df=synthetic_processed, cache=False)
    reconstructed_train = split.X_train * split.scaler.scale_ + split.scaler.mean_
    assert np.allclose(reconstructed_train.mean(axis=0), split.scaler.mean_, atol=1e-3)
    assert len(split.X_unseen) == int((synthetic_processed.family == "DDoS").sum())
