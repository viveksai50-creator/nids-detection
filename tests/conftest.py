from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_processed() -> pd.DataFrame:
    """Small, separable processed dataset suitable for leakage tests."""
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    counts = {"BENIGN": 4_000, "DoS": 700, "DDoS": 700, "PortScan": 700}
    for family, n in counts.items():
        attack = family != "BENIGN"
        shift = 3.0 if attack else 0.0
        for i in range(n):
            rows.append({
                "feature_a": rng.normal(shift, 1.0),
                "feature_b": rng.normal(shift, 1.0),
                "label_raw": family,
                "family": family,
                "is_attack": int(attack),
            })
    return pd.DataFrame(rows)
