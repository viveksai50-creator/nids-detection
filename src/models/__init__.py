"""Model registry. """
from __future__ import annotations

from src import config as C
from src.models.base import BaseDetector


def build_model(name: str, seed: int, hp: dict | None = None) -> BaseDetector:
    merged = {**C.DEFAULT_HP.get(name, {}), **(hp or {})}
    if name == "rf":
        from src.models.supervised import RandomForestDetector as M
    elif name == "xgb":
        from src.models.supervised import XGBoostDetector as M
    elif name == "iforest":
        from src.models.isolation_forest import IsolationForestDetector as M
    elif name == "ocsvm":
        from src.models.ocsvm import OCSVMDetector as M
    elif name == "ae":
        from src.models.autoencoder import AutoencoderDetector as M
    elif name == "vae":
        from src.models.vae import VAEDetector as M
    elif name == "scarf":
        from src.models.ssl_scarf import SCARFDetector as M
    else:
        raise KeyError(f"unknown model {name}; choose from {C.MODELS}")
    return M(seed=seed, **merged)