"""Focused SCARF invariants on small synthetic data."""
from __future__ import annotations

import numpy as np
import torch

from src import config as C
from src import run_loaco as runner
from src.models import build_model
from src.models.ssl_scarf import nt_xent


def test_corruption_draws_each_feature_only_from_training_pool(monkeypatch):
    monkeypatch.setenv("NIDS_DEVICE", "cpu")
    model = build_model("scarf", 2, {"corruption_rate": 1.0})
    model.c = 1.0
    model.pool = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
    original = torch.full((8, 2), -999.0)
    corrupted = model._corrupt(original, torch.Generator().manual_seed(3))
    assert set(corrupted[:, 0].tolist()) <= {1.0, 2.0, 3.0}
    assert set(corrupted[:, 1].tolist()) <= {10.0, 20.0, 30.0}
    assert not torch.any(corrupted == -999.0)


def test_nt_xent_prefers_aligned_pairs():
    z = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    aligned = nt_xent(z, z.clone(), tau=0.2)
    misaligned = nt_xent(z, z.roll(1, dims=0), tau=0.2)
    assert torch.isfinite(aligned)
    assert aligned < misaligned


def test_scarf_embeddings_and_knn_scores_are_finite(monkeypatch):
    monkeypatch.setenv("NIDS_DEVICE", "cpu")
    rng = np.random.default_rng(9)
    benign = rng.normal(0, 0.4, size=(48, 5)).astype(np.float32)
    attacks = rng.normal(3, 0.4, size=(16, 5)).astype(np.float32)
    X_train = np.vstack([benign, attacks])
    y_train = np.r_[np.zeros(len(benign), dtype=np.int8), np.ones(len(attacks), dtype=np.int8)]
    X_val = np.vstack([rng.normal(0, 0.4, size=(16, 5)), rng.normal(3, 0.4, size=(8, 5))]).astype(np.float32)
    model = build_model("scarf", 0, {"hidden": 12, "embed_dim": 6, "proj_dim": 4,
                                      "epochs": 3, "k": 3})
    model.fit(X_train, y_train, X_val, np.r_[np.zeros(16), np.ones(8)])
    embeddings = model.embed(X_val)
    scores = model.score(X_val)
    assert embeddings.shape == (len(X_val), 6)
    assert scores.shape == (len(X_val),)
    assert np.isfinite(embeddings).all() and np.isfinite(scores).all()
    assert all(np.isfinite(row["train"]) and np.isfinite(row["val"]) for row in model.history)


def test_sanity_uses_no_test_or_held_out_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("NIDS_DEVICE", "cpu")
    monkeypatch.setattr(C, "CHECKPOINT_DIR", tmp_path)
    rng = np.random.default_rng(12)

    class TrainValOnlySplit:
        X_train = np.vstack([rng.normal(0, 0.3, (24, 4)), rng.normal(2, 0.3, (12, 4))]).astype(np.float32)
        y_train = np.r_[np.zeros(24, dtype=np.int8), np.ones(12, dtype=np.int8)]
        X_val = np.vstack([rng.normal(0, 0.3, (12, 4)), rng.normal(2, 0.3, (8, 4))]).astype(np.float32)
        y_val = np.r_[np.zeros(12, dtype=np.int8), np.ones(8, dtype=np.int8)]
        meta = {"n_train": 36, "n_val": 20}
        feature_names = ["a", "b", "c", "d"]

        @property
        def X_test(self):
            raise AssertionError("sanity must not access known test data")

        @property
        def X_unseen(self):
            raise AssertionError("sanity must not access held-out data")

    monkeypatch.setattr(runner, "build_split", lambda family, seed: TrainValOnlySplit())
    result = runner.run_scarf_sanity("DoS", 0, {"hidden": 8, "embed_dim": 4, "proj_dim": 3, "epochs": 2, "k": 3})
    assert result["kind"] == "diagnostic_sanity"
    assert np.isfinite(result["validation"]["linear_probe_roc_auc"])
    assert np.isfinite(result["validation"]["knn_roc_auc"])
