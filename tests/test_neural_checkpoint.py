"""Interrupted neural fits must resume their RNG and best-checkpoint state."""
import numpy as np
import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize("name", ["ae", "vae"])
def test_interrupted_fit_matches_uninterrupted_cpu(name, tmp_path, monkeypatch):
    monkeypatch.setenv("NIDS_DEVICE", "cpu")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(72, 6)).astype(np.float32)
    y = np.r_[np.zeros(60, dtype=np.int8), np.ones(12, dtype=np.int8)]
    hp = {"hidden": 8, "bottleneck": 2, "latent": 2, "dropout": 0.1, "epochs": 4}
    reference = build_model(name, 7, hp)
    reference.fit(X, y, X, y)
    path = tmp_path / "fit.pt"
    signature = {"model": name, "hp": hp, "seed": 7, "purpose": "test"}
    interrupted = build_model(name, 7, hp)
    interrupted.configure_checkpoint(path, signature)
    save = interrupted._save_checkpoint

    def interrupt_after_second_epoch(epoch, opt, stopper):
        save(epoch, opt, stopper)
        if epoch == 1:
            raise InterruptedError("simulated process interruption after atomic checkpoint")

    monkeypatch.setattr(interrupted, "_save_checkpoint", interrupt_after_second_epoch)
    with pytest.raises(InterruptedError):
        interrupted.fit(X, y, X, y)
    resumed = build_model(name, 7, hp)
    resumed.configure_checkpoint(path, signature)
    resumed.fit(X, y, X, y)
    np.testing.assert_array_equal(reference.score(X), resumed.score(X))
    for left, right in zip(reference.history, resumed.history):
        assert (left["train"], left["val"]) == (right["train"], right["val"])
    wrong = build_model(name, 7, hp)
    wrong.configure_checkpoint(path, {**signature, "seed": 8})
    with pytest.raises(ValueError, match="signature mismatch"):
        wrong.fit(X, y, X, y)


@pytest.mark.parametrize("name", ["ae", "vae"])
def test_neural_fit_ignores_attack_rows_in_train_and_early_stopping(name, monkeypatch):
    monkeypatch.setenv("NIDS_DEVICE", "cpu")
    rng = np.random.default_rng(8)
    X = rng.normal(size=(64, 6)).astype(np.float32)
    y = np.r_[np.zeros(48, dtype=np.int8), np.ones(16, dtype=np.int8)]
    changed = X.copy()
    changed[y == 1] = 1e6
    hp = {"hidden": 8, "bottleneck": 2, "latent": 2, "epochs": 2}
    first, second = build_model(name, 0, hp), build_model(name, 0, hp)
    first.fit(X, y, X, y)
    second.fit(changed, y, changed, y)
    np.testing.assert_array_equal(first.score(X), second.score(X))
