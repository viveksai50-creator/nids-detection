"""Validate completed SCARF seed-0 records without fitting or selecting models."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.run_loaco import hp_id, sample_configs, write_json_atomically  # noqa: E402


def bounded(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def main() -> None:
    records = []
    for family in C.FAMILIES:
        result_path = C.RAW_RESULTS_DIR / f"{family}_scarf_seed0.json"
        tuning_path = C.TUNING_DIR / f"{family}_scarf.json"
        result = json.loads(result_path.read_text())
        tuning = json.loads(tuning_path.read_text())
        planned = sample_configs(
            C.SEARCH_SPACES["scarf"], C.tuning_trials_for("scarf"),
            zlib.crc32(f"{family}_scarf".encode()) & 0xFFFF,
        )
        trials = tuning["trials"]
        assert tuning.get("complete") and len(trials) == len(planned) == 6
        assert {hp_id(x["hp"]) for x in trials} == {hp_id(hp) for hp in planned}
        assert all(x.get("status") == "completed" and bounded(x.get("val_pr_auc")) for x in trials)
        best = max(trials, key=lambda x: x["val_pr_auc"])
        assert result["hp"] == tuning["best_hp"] == best["hp"]
        assert (result["family"], result["model"], result["seed"]) == (family, "scarf", 0)
        assert set(result["thresholds"]) == {"0.001", "0.01", "0.05"}
        assert all(math.isfinite(x) for x in result["thresholds"].values())
        assert all(bounded(result["validation"][key]) for key in ("roc_auc", "pr_auc"))
        assert all(bounded(result["metrics"][key]) for key in ("roc_auc", "pr_auc", "unseen_roc_auc"))
        assert result["metrics"]["primary"] == result["metrics"]["at_fpr"]["0.01"]
        assert 1 <= len(result["history"]) <= C.MAX_EPOCHS
        assert all(math.isfinite(row[key]) for row in result["history"] for key in ("train", "val", "seconds"))
        assert result["train_seconds"] >= sum(row["seconds"] for row in result["history"])
        for point in result["metrics"]["at_fpr"].values():
            assert all(bounded(point[key]) for key in ("udr", "realised_fpr_benign_test", "known_attack_recall"))
        records.append({
            "family": family,
            "result": str(result_path.relative_to(ROOT)),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "tuning": str(tuning_path.relative_to(ROOT)),
            "tuning_sha256": hashlib.sha256(tuning_path.read_bytes()).hexdigest(),
            "selected_hp": result["hp"],
            "udr_at_1pct": result["metrics"]["primary"]["udr"],
        })
        print(f"validated {family}/scarf: six unique trials and one frozen final record")
    write_json_atomically(C.RESULTS_DIR / "diagnostics" / "scarf_completion.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "completed_valid", "single_seed_only": True,
        "final_results": len(records), "tuning_trials": len(records) * 6,
        "records": records,
    })
    print("PASS: 7 SCARF finals and 42 distinct tuning trials")


if __name__ == "__main__":
    main()
