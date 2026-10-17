"""Isolation Forest on benign training rows. """
from __future__ import annotations

from sklearn.ensemble import IsolationForest

from src.models.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "iforest"
    benign_only = True

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        hp = self.hp
        self.model = IsolationForest(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_samples=hp.get("max_samples", 256),
            max_features=float(hp.get("max_features", 1.0)),
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(self.benign(X_train, y_train))

    def score(self, X):
        # score_samples: higher = more normal. Negate so higher = more anomalous.
        return -self.model.score_samples(X)