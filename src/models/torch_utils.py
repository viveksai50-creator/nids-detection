"""Shared torch helpers: seeding, device, batching, early stopping. """
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from src import config as C


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device() -> torch.device:
    return C.get_device()


def to_tensor(X: np.ndarray, dev: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)


def iterate_minibatches(X: torch.Tensor, batch: int, shuffle: bool, gen: torch.Generator | None = None):
    n = X.shape[0]
    idx = torch.randperm(n, generator=gen, device="cpu") if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        j = idx[i : i + batch].to(X.device)
        yield X[j]


@torch.no_grad()
def batched_apply(fn, X: np.ndarray, dev: torch.device, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = to_tensor(X[i : i + batch], dev)
        out.append(fn(xb).detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class EarlyStopper:
    def __init__(self, patience: int = C.EARLY_STOP_PATIENCE, min_delta: float = 1e-5):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, value: float, model: nn.Module) -> bool:
        """Returns True when training should stop. Keeps the best state dict."""
        if value < self.best - self.min_delta:
            self.best, self.bad = value, 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def mlp(sizes: list[int], dropout: float = 0.0, batchnorm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if batchnorm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)