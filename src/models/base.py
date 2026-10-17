"""Common detector interface.

Every model: fit(X_train, y_train, X_val, y_val) then score(X) -> np.ndarray where
higher = more attack-like. Models that train on benign only filter inside fit().
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    name: str = "base"
    needs_labels: bool = False
    benign_only: bool = False

    def __init__(self, seed: int, **hp):
        self.seed = int(seed)
        self.hp = dict(hp)

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def benign(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X[np.asarray(y) == 0]

    def describe(self) -> dict:
        return {"name": self.name, "seed": self.seed, "hp": self.hp}
