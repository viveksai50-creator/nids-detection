"""SCARF self-supervised encoder + kNN anomaly head (REVIEW-FEEDBACK.md §2).

Pretraining: on ALL training rows (no labels). For each batch x:
    corrupted view x~ = x with a random 'corruption_rate' fraction of features replaced by
    values drawn from the empirical marginal (a random row of a fixed marginal pool).
    z = head(encoder(x)), z~ = head(encoder(x~)); NT-Xent loss with temperature.
Scoring: embedding e = encoder(x); score = mean distance to k nearest of REF (encoder
    embeddings of a seeded SCARF_REFERENCE_SIZE benign-train subsample). Higher = more anomalous.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import config as C
from src.models.base import BaseDetector
from src.models.torch_utils import (EarlyStopper, device, iterate_minibatches,
                                    mlp, set_seed, to_tensor)


class SCARFNet(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int, embedding_size: int, projection_size: int):
        super().__init__()
        self.encoder = mlp([feature_count, hidden_size, hidden_size, embedding_size], batchnorm=True)
        self.head = mlp([embedding_size, embedding_size, projection_size])

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n = z1.shape[0]
    sim = (z @ z.T) / tau
    eye = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SCARFDetector(BaseDetector):
    name = "scarf"

    def configure_checkpoint(self, path: Path, signature: dict) -> None:
        """Configure an atomic epoch checkpoint tied to one exact experiment."""
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
        saved = state["stopper"]
        stopper.best, stopper.bad, stopper.best_state = saved["best"], saved["bad"], saved["best_state"]
        torch.set_rng_state(state["torch_rng"].cpu())
        self.shuffle_generator.set_state(state["shuffle_rng"].cpu())
        if self.dev.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"].cpu())
        return int(state["epoch"]) + 1

    def _corrupt(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        batch_size, feature_count = x.shape
        marginal_pool = self.pool
        replacement_rows = torch.randint(
            0, marginal_pool.shape[0], (batch_size, feature_count), generator=gen, device="cpu"
        ).to(x.device)
        feature_columns = torch.arange(feature_count, device=x.device).unsqueeze(0).expand(batch_size, feature_count)
        replacements = marginal_pool[replacement_rows, feature_columns]
        corruption_mask = (torch.rand(batch_size, feature_count, generator=gen, device="cpu") < self.c).to(x.device)
        return torch.where(corruption_mask, replacements, x)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        set_seed(self.seed)
        self.dev = device()
        hyperparameters = self.hp
        self.c = float(hyperparameters.get("corruption_rate", 0.6))
        self.tau = float(hyperparameters.get("temperature", 0.5))
        self.k = int(hyperparameters.get("k", C.SCARF_KNN_K))
        train_features = to_tensor(X_train, self.dev)
        numpy_rng = np.random.default_rng(self.seed)
        pool_indices = numpy_rng.choice(len(X_train), size=min(C.SCARF_MARGINAL_POOL, len(X_train)), replace=False)
        self.pool = train_features[torch.as_tensor(pool_indices, device=self.dev)]
        validation_features = to_tensor(X_val, self.dev) if X_val is not None else None
        feature_count = train_features.shape[1]
        self.net = SCARFNet(feature_count, int(hyperparameters.get("hidden", 256)), int(hyperparameters.get("embed_dim", 128)), int(hyperparameters.get("proj_dim", 64))).to(self.dev)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(hyperparameters.get("lr", 1e-3)), weight_decay=1e-5)
        shuffle_rng = torch.Generator().manual_seed(self.seed)
        self.shuffle_generator = shuffle_rng
        val_gen_seed = self.seed + 10_000
        stopper = EarlyStopper()
        self.history = []
        start_epoch = self._load_checkpoint(optimizer, stopper)
        if stopper.bad >= stopper.patience:
            start_epoch = int(hyperparameters.get("epochs", C.MAX_EPOCHS))
        for epoch in range(start_epoch, int(hyperparameters.get("epochs", C.MAX_EPOCHS))):
            epoch_start = time.perf_counter()
            self.net.train()
            total, count = 0.0, 0
            for feature_batch in iterate_minibatches(train_features, C.TORCH_BATCH, shuffle=True, gen=shuffle_rng):
                if feature_batch.shape[0] < 4:
                    continue
                optimizer.zero_grad()
                loss = nt_xent(self.net(feature_batch), self.net(self._corrupt(feature_batch, shuffle_rng)), self.tau)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(feature_batch)
                count += len(feature_batch)
            train_loss = total / max(1, count)
            val_loss = self._val_loss(validation_features, val_gen_seed) if validation_features is not None else train_loss
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise FloatingPointError("non-finite SCARF training or validation contrastive loss")
            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                                 "seconds": time.perf_counter() - epoch_start})
            should_stop = stopper.step(val_loss, self.net)
            if hasattr(self, "checkpoint_path"):
                self._save_checkpoint(epoch, optimizer, stopper)
            print(f"  SCARF epoch {epoch + 1}: train={train_loss:.6g} val={val_loss:.6g}", flush=True)
            if should_stop:
                break
        stopper.restore(self.net)
        self.net.eval()

        # Reference set: benign train embeddings.
        benign_train_features = self.benign(X_train, y_train)
        reference_indices = numpy_rng.choice(len(benign_train_features), size=min(C.SCARF_REFERENCE_SIZE, len(benign_train_features)), replace=False)
        self.ref = self._embed(to_tensor(benign_train_features[reference_indices], self.dev))
        if not torch.isfinite(self.ref).all():
            raise FloatingPointError("non-finite SCARF benign reference embeddings")
        self.ref_sq = (self.ref ** 2).sum(dim=1)

    @torch.no_grad()
    def _val_loss(self, Xv: torch.Tensor, seed: int) -> float:
        self.net.eval()
        gen = torch.Generator().manual_seed(seed)
        tot, n = 0.0, 0
        for xb in iterate_minibatches(Xv, C.TORCH_BATCH, shuffle=False):
            if xb.shape[0] < 4:
                continue
            tot += float(nt_xent(self.net(xb), self.net(self._corrupt(xb, gen)), self.tau)) * len(xb)
            n += len(xb)
        value = tot / max(1, n)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite SCARF validation contrastive loss")
        return value

    @torch.no_grad()
    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        out = [self.net.encoder(X[i : i + 4096]) for i in range(0, X.shape[0], 4096)]
        embeddings = torch.cat(out)
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("non-finite SCARF embeddings")
        return embeddings

    @torch.no_grad()
    def score(self, X):
        out = []
        for i in range(0, len(X), C.SCARF_SCORE_BATCH):
            e = self._embed(to_tensor(X[i : i + C.SCARF_SCORE_BATCH], self.dev))
            # squared euclidean via ||a||^2 + ||b||^2 - 2ab (avoids cdist issues on MPS)
            d2 = (e ** 2).sum(dim=1, keepdim=True) + self.ref_sq.unsqueeze(0) - 2.0 * (e @ self.ref.T)
            d2 = d2.clamp_min(0.0)
            knn = torch.topk(d2, k=self.k, dim=1, largest=False).values.sqrt().mean(dim=1)
            out.append(knn.float().cpu().numpy())
        scores = np.concatenate(out)
        if not np.isfinite(scores).all():
            raise FloatingPointError("non-finite SCARF kNN scores")
        return scores

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Frozen embeddings (for the optional linear-probe head)."""
        return self._embed(to_tensor(X, self.dev)).cpu().numpy()
