"""Leave-one-family-out split builder with leakage assertions and on-disk cache.

For (family F, seed s):
    unseen U       = all rows with family == F
    rest           = everything else
    train/val/test = 60/20/20 of rest, stratified on is_attack, random_state = s
    scaler         = StandardScaler fitted on train ONLY, applied to val/test/U
Cache: $NIDS_DATA_ROOT/splits/{F}__seed{s}/{arrays.npz, scaler.joblib, meta.json}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config as C

META_COLUMNS = ["label_raw", "family", "is_attack"]


@dataclass
class Split:
    family: str
    seed: int
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    fam_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    fam_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    fam_test: np.ndarray
    X_unseen: np.ndarray
    fam_unseen: np.ndarray
    scaler: StandardScaler
    meta: dict = field(default_factory=dict)

    @property
    def y_unseen(self) -> np.ndarray:
        return np.ones(len(self.X_unseen), dtype=np.int8)


def load_processed() -> pd.DataFrame:
    if not C.PROCESSED_FILE.exists():
        raise FileNotFoundError(f"{C.PROCESSED_FILE} missing; run src.data.clean")
    return pd.read_parquet(C.PROCESSED_FILE)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def _cache_dir(family: str, seed: int) -> Path:
    return C.SPLITS_DIR / f"{family}__seed{seed}"


def assert_no_leakage(split: Split) -> None:
    F = split.family
    assert not np.any(split.fam_train == F), "held-out family present in TRAIN"
    assert not np.any(split.fam_val == F), "held-out family present in VAL"
    assert not np.any(split.fam_test == F), "held-out family present in TEST"
    assert np.all(split.fam_unseen == F), "UNSEEN contains rows of other families"
    n = len(split.X_train) + len(split.X_val) + len(split.X_test) + len(split.X_unseen)
    assert n == split.meta["n_total"], "row count mismatch after split"
    assert split.y_train.min() == 0 and split.y_train.max() == 1, "train must contain benign and known attacks"
    assert (split.y_val == 0).any(), "val must contain benign rows (threshold selection)"
    # Scaler must be fitted on train only: its mean must equal train mean (train is stored scaled,
    # so check the unscaled reconstruction instead).
    recon_mean = (split.X_train * split.scaler.scale_ + split.scaler.mean_).mean(axis=0)
    assert np.allclose(recon_mean, split.scaler.mean_, atol=1e-3), "scaler was not fitted on train"


def build_split(family: str, seed: int, df: pd.DataFrame | None = None, cache: bool = True) -> Split:
    if family not in C.FAMILIES:
        raise ValueError(f"{family} not in FAMILIES {C.FAMILIES}")
    cdir = _cache_dir(family, seed)
    if cache and (cdir / "arrays.npz").exists():
        return load_split(family, seed)

    df = load_processed() if df is None else df
    cols = feature_columns(df)
    fam = df["family"].to_numpy().astype(str)
    is_unseen = fam == family
    unseen = df.loc[is_unseen]
    rest = df.loc[~is_unseen]
    y_rest = rest["is_attack"].to_numpy().astype(np.int8)

    idx = np.arange(len(rest))
    tr_idx, tmp_idx = train_test_split(idx, train_size=C.SPLIT_FRACTIONS["train"], stratify=y_rest, random_state=seed)
    val_share = C.SPLIT_FRACTIONS["val"] / (C.SPLIT_FRACTIONS["val"] + C.SPLIT_FRACTIONS["test"])
    va_idx, te_idx = train_test_split(tmp_idx, train_size=val_share, stratify=y_rest[tmp_idx], random_state=seed)

    X_rest = rest[cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(X_rest[tr_idx])

    def scaled(a: np.ndarray) -> np.ndarray:
        return scaler.transform(a).astype(np.float32)

    fam_rest = rest["family"].to_numpy().astype(str)
    split = Split(
        family=family, seed=seed, feature_names=cols,
        X_train=scaled(X_rest[tr_idx]), y_train=y_rest[tr_idx], fam_train=fam_rest[tr_idx],
        X_val=scaled(X_rest[va_idx]), y_val=y_rest[va_idx], fam_val=fam_rest[va_idx],
        X_test=scaled(X_rest[te_idx]), y_test=y_rest[te_idx], fam_test=fam_rest[te_idx],
        X_unseen=scaled(unseen[cols].to_numpy(dtype=np.float32)), fam_unseen=unseen["family"].to_numpy().astype(str),
        scaler=scaler,
        meta={
            "n_total": int(len(df)), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)), "n_unseen": int(len(unseen)),
            "train_attack_share": float(y_rest[tr_idx].mean()),
        },
    )
    assert_no_leakage(split)
    if cache:
        save_split(split)
    return split


def save_split(split: Split) -> None:
    cdir = _cache_dir(split.family, split.seed)
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cdir / "arrays.npz",
        X_train=split.X_train, y_train=split.y_train, fam_train=split.fam_train,
        X_val=split.X_val, y_val=split.y_val, fam_val=split.fam_val,
        X_test=split.X_test, y_test=split.y_test, fam_test=split.fam_test,
        X_unseen=split.X_unseen, fam_unseen=split.fam_unseen,
    )
    joblib.dump(split.scaler, cdir / "scaler.joblib")
    with open(cdir / "meta.json", "w") as f:
        json.dump({"family": split.family, "seed": split.seed, "feature_names": split.feature_names, **split.meta}, f, indent=2)


def load_split(family: str, seed: int) -> Split:
    cdir = _cache_dir(family, seed)
    a = np.load(cdir / "arrays.npz", allow_pickle=False)
    with open(cdir / "meta.json") as f:
        meta = json.load(f)
    split = Split(
        family=family, seed=seed, feature_names=meta.pop("feature_names"),
        X_train=a["X_train"], y_train=a["y_train"], fam_train=a["fam_train"].astype(str),
        X_val=a["X_val"], y_val=a["y_val"], fam_val=a["fam_val"].astype(str),
        X_test=a["X_test"], y_test=a["y_test"], fam_test=a["fam_test"].astype(str),
        X_unseen=a["X_unseen"], fam_unseen=a["fam_unseen"].astype(str),
        scaler=joblib.load(cdir / "scaler.joblib"),
        meta={k: v for k, v in meta.items() if k not in ("family", "seed")},
    )
    assert_no_leakage(split)
    return split