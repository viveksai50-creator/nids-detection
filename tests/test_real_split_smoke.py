from __future__ import annotations

import pytest

from src import config as C
from src.data.split import assert_no_leakage, build_split


@pytest.mark.skipif(not C.PROCESSED_FILE.exists(), reason="cleaned dataset is not available")
def test_real_dos_seed0_split_has_no_held_out_leakage() -> None:
    split = build_split("DoS", seed=0, cache=True)

    assert_no_leakage(split)
    assert split.meta["n_unseen"] > 0
    assert split.meta["n_train"] > split.meta["n_val"] > 0
    assert split.meta["n_train"] > split.meta["n_test"] > 0
