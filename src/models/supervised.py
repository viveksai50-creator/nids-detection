"""Supervised binary baselines: Random Forest, XGBoost. """
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseDetector


class RandomForestDetector(BaseDetector):
    name = "rf"
    needs_labels = True

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        hp = self.hp
        self.model = RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=hp.get("max_depth", 20),
            max_features=hp.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(X_train, y_train)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBoostDetector(BaseDetector):
    name = "xgb"
    needs_labels = True

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from xgboost import XGBClassifier

        hp = self.hp
        y = np.asarray(y_train)
        n_pos = max(1, int(y.sum()))
        spw = float((len(y) - n_pos) / n_pos)
        self.model = XGBClassifier(
            n_estimators=int(hp.get("n_estimators", 400)),
            max_depth=int(hp.get("max_depth", 6)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            subsample=float(hp.get("subsample", 1.0)),
            scale_pos_weight=spw,
            tree_method="hist",
            # XGBoost's OpenMP runtime segfaulted on this macOS/ARM setup when
            # the grid moved from a multi-threaded Random Forest run to XGBoost.
            # One native worker is slower but stable and keeps the experiment
            # checkpointable; resource time is recorded in every result JSON.
            n_jobs=1,
            random_state=self.seed,
            eval_metric="aucpr",
        )
        self.model.fit(X_train, y)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]
