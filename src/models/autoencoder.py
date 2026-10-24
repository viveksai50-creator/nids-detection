"""Autoencoder anomaly detector: trained on benign train, early-stopped on benign val MSE.
Score = per-row reconstruction MSE. """
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import config as C
from src.models.base import BaseDetector
from src.models.torch_utils import (EarlyStopper, batched_apply, device,
                                    iterate_minibatches, mlp, set_seed,
                                    to_tensor)


class AE(nn.Module):
    def __init__(self, d: int, hidden: int, bottleneck: int, dropout: float):
        super().__init__()
        self.enc = mlp([d, hidden, bottleneck], dropout=dropout)
        self.dec = mlp([bottleneck, hidden, d], dropout=dropout)

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderDetector(BaseDetector):
    name = "ae"
    benign_only = True

    def configure_checkpoint(self, path: Path, signature: dict) -> None:
        """Configure an epoch-level, run-specific checkpoint.

        The runner supplies a signature tied to the held-out family, seed and
        hyperparameters, so a checkpoint can never be resumed for a different
        experiment.
        """
        self.checkpoint_path = Path(path)
        self.checkpoint_signature = signature

    def _save_checkpoint(self, epoch: int, opt, stopper: EarlyStopper) -> None:
        path = self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "signature": self.checkpoint_signature,
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "optimizer_state": opt.state_dict(),
            "history": self.history,
            "torch_rng": torch.get_rng_state(),
            "shuffle_rng": self.shuffle_generator.get_state(),
            "mps_rng": torch.mps.get_rng_state() if self.dev.type == "mps" else None,
            "stopper": {"best": stopper.best, "bad": stopper.bad, "best_state": stopper.best_state},
        }, tmp)
        os.replace(tmp, path)

    def _load_checkpoint(self, opt, stopper: EarlyStopper) -> int:
        path = getattr(self, "checkpoint_path", None)
        if path is None or not path.exists():
            return 0
        state = torch.load(path, map_location=self.dev, weights_only=False)
        if state.get("signature") != self.checkpoint_signature:
            raise ValueError(f"checkpoint signature mismatch: {path}")
        self.net.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        self.history = state["history"]
        saved_stopper = state["stopper"]
        stopper.best = saved_stopper["best"]
        stopper.bad = saved_stopper["bad"]
        stopper.best_state = saved_stopper["best_state"]
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
            self.shuffle_generator.set_state(state["shuffle_rng"].cpu())
            if self.dev.type == "mps" and state.get("mps_rng") is not None:
                torch.mps.set_rng_state(state["mps_rng"].cpu())
        return int(state["epoch"]) + 1

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        set_seed(self.seed)
        self.dev = device()
        hp = self.hp
        Xb = to_tensor(self.benign(X_train, y_train), self.dev)
        Xv = to_tensor(self.benign(X_val, y_val), self.dev) if X_val is not None else None
        d = Xb.shape[1]
        self.net = AE(d, int(hp.get("hidden", 128)), int(hp.get("bottleneck", 16)), float(hp.get("dropout", 0.0))).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(hp.get("lr", 1e-3)), weight_decay=1e-5)
        gen = torch.Generator().manual_seed(self.seed)
        self.shuffle_generator = gen
        stopper = EarlyStopper()
        self.history = []
        start_epoch = self._load_checkpoint(opt, stopper)
        if stopper.bad >= stopper.patience:
            start_epoch = int(hp.get("epochs", C.MAX_EPOCHS))
        for epoch in range(start_epoch, int(hp.get("epochs", C.MAX_EPOCHS))):
            epoch_start = time.perf_counter()
            self.net.train()
            total, count = 0.0, 0
            for xb in iterate_minibatches(Xb, C.TORCH_BATCH, shuffle=True, gen=gen):
                opt.zero_grad()
                loss = ((self.net(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(xb)
                count += len(xb)
            train_loss = total / max(1, count)
            val_loss = float(self._row_mse(Xv).mean()) if Xv is not None else train_loss
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise FloatingPointError("non-finite AE training or benign-validation loss")
            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                                 "seconds": time.perf_counter() - epoch_start})
            should_stop = stopper.step(val_loss, self.net)
            if hasattr(self, "checkpoint_path"):
                self._save_checkpoint(epoch, opt, stopper)
            if should_stop:
                break
        stopper.restore(self.net)
        self.net.eval()

    @torch.no_grad()
    def _row_mse(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        out = []
        for i in range(0, X.shape[0], 4096):
            xb = X[i : i + 4096]
            out.append(((self.net(xb) - xb) ** 2).mean(dim=1))
        return torch.cat(out)

    def score(self, X):
        return batched_apply(lambda xb: ((self.net(xb) - xb) ** 2).mean(dim=1), X, self.dev)
