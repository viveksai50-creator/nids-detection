"""Central configuration for the unknown-attack NIDS benchmark.

Every path, seed, family mapping and search space lives here so that a run is
fully described by (family, model, seed) + this file + the git commit hash.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("NIDS_DATA_ROOT", Path.home() / "Research" / "nids-data"))
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TUNING_DIR = RESULTS_DIR / "tuning"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
STATS_DIR = RESULTS_DIR / "stats"

PROCESSED_FILE = PROCESSED_DIR / "cicids2017_improved_clean.parquet"
PROCESSED_SUMMARY = PROCESSED_DIR / "cleaning_summary.json"

DATASET_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
DATASET_ZIP = RAW_DIR / "CICIDS2017_improved.zip"

# ----------------------------------------------------------------------------- protocol
SEEDS = [0, 1, 2, 3, 4]
TIER_A_SEEDS = [0, 1, 2]
TUNING_SEED = 0
TUNING_TRIALS = 10

SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
TARGET_FPRS = [0.001, 0.01, 0.05]
PRIMARY_FPR = 0.01

# PortScan stays separate (7,810 retained
# flows); small Botnet, WebAttack, and Infiltration folds require wide-CI reporting.
FAMILIES = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet", "Infiltration"]

# Ambiguous <attack> - Attempted rows are excluded.
ATTEMPTED_POLICY = "drop"
# Heartbleed is excluded because it is too small to evaluate alone.
HEARTBLEED_POLICY = "drop"

# Columns that identify hosts/time and must never be features (matched case-insensitively
# after stripping whitespace).
NON_FEATURE_COLUMNS = {
    "id", "flow id", "source ip", "src ip", "source port", "src port",
    "dst ip", "destination ip", "dst port", "destination port",
    "timestamp", "attempted category", "label",
}

LABEL_COLUMN_CANDIDATES = {"label"}

# ----------------------------------------------------------------------------- models
MODELS = ["rf", "xgb", "iforest", "ocsvm", "ae", "vae", "scarf"]
TIER_A_MODELS = ["rf", "xgb", "iforest", "ae", "scarf"]
TIER_B_MODELS = MODELS

OCSVM_BENIGN_SUBSAMPLE = 50_000
SCARF_REFERENCE_SIZE = 100_000
SCARF_MARGINAL_POOL = 200_000  # rows used to sample corrupted feature values
SCARF_KNN_K = 5
SCARF_SCORE_BATCH = 512       # lower to 256/128 on MPS OOM
TORCH_BATCH = 1024
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# Random-search spaces. Values are lists; a configuration picks one choice per key.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "rf": {"n_estimators": [100, 200], "max_depth": [10, 20, None], "max_features": ["sqrt", 0.3]},
    "xgb": {"n_estimators": [200, 400], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
    "iforest": {"n_estimators": [100, 200], "max_samples": [256, 1024], "max_features": [0.5, 1.0]},
    "ocsvm": {"nu": [0.01, 0.05], "gamma": ["scale", 0.01, 0.1]},
    "ae": {"bottleneck": [8, 16], "hidden": [64, 128], "dropout": [0.0, 0.1], "lr": [1e-3, 3e-4]},
    "vae": {"latent": [8, 16], "hidden": [64, 128], "beta": [0.5, 1.0], "lr": [1e-3, 1e-3]},
    "scarf": {"corruption_rate": [0.3, 0.6], "temperature": [0.1, 0.5, 1.0]},
}

# Defaults used when tuning is skipped (--no-tune) or a key is absent from a config.
DEFAULT_HP: dict[str, dict] = {
    "rf": {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt"},
    "xgb": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 1.0},
    "iforest": {"n_estimators": 200, "max_samples": 256, "max_features": 1.0},
    "ocsvm": {"nu": 0.01, "gamma": "scale"},
    "ae": {"bottleneck": 16, "hidden": 128, "dropout": 0.0, "lr": 1e-3},
    "vae": {"latent": 16, "hidden": 128, "beta": 1.0, "lr": 1e-3},
    "scarf": {"corruption_rate": 0.6, "temperature": 0.5, "hidden": 256, "embed_dim": 128, "proj_dim": 64, "lr": 1e-3, "k": SCARF_KNN_K},
}

# ----------------------------------------------------------------------------- label mapping
def strip_attempted(label: str) -> str:
    s = str(label)
    for token in (" - Attempted", " Attempted", "- Attempted", "Attempted"):
        s = s.replace(token, "")
    return s.strip(" -")


def label_to_family(raw_label: str) -> str:
    """Map a raw dataset label to a family name.

    Returns one of FAMILIES, or the sentinels "BENIGN", "ATTEMPTED", "HEARTBLEED",
    "UNMAPPED". Order of checks matters: the improved dataset files PortScan under
    Infiltration ("Infiltration - PortScan"), so Infiltration is tested before PortScan
    only when PortScan is not a separate family.
    """
    s = str(raw_label).strip().lower()
    if s == "benign":
        return "BENIGN"
    if "attempted" in s:
        return "ATTEMPTED"
    if "heartbleed" in s:
        return "HEARTBLEED"

    # PortScan: standalone label (original set) or "Infiltration - PortScan" (improved set).
    if ("portscan" in s or "port scan" in s) and "PortScan" in FAMILIES:
        return "PortScan"
    if "infiltration" in s:
        return "Infiltration"
    if "web attack" in s or "sql injection" in s or "xss" in s:
        return "WebAttack"
    if "ddos" in s:
        return "DDoS"
    if s.startswith("dos") or " dos " in f" {s} " or "slowloris" in s or "slowhttptest" in s or "hulk" in s or "goldeneye" in s:
        return "DoS"
    if "patator" in s or "brute force" in s or "bruteforce" in s:
        return "BruteForce"
    if "bot" in s:
        return "Botnet"
    return "UNMAPPED"

# ----------------------------------------------------------------------------- device
def get_device():
    """Return the torch device to use. NIDS_DEVICE=cpu forces CPU."""
    import torch  # imported lazily so non-torch code paths do not need torch

    forced = os.environ.get("NIDS_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, RAW_RESULTS_DIR, TUNING_DIR, TABLES_DIR, FIGURES_DIR, STATS_DIR):
        d.mkdir(parents=True, exist_ok=True)
