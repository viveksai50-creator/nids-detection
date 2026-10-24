"""Checkpoint-aware classical seed-0 grid supervisor.

Usage: python scripts/supervise_classical_grid.py --once

Run as a persistent process while a classical grid is incomplete.  It writes a
heartbeat/status JSON every poll, restarts the checkpointable runner after a
15-minute stale heartbeat, and exits cleanly once all expected records pass
schema/value validation.  It never deletes or reruns an existing result.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

STATUS = ROOT / "logs" / "classical_supervisor_status.json"
MODELS = ("rf", "xgb", "iforest")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_trials(model: str) -> int:
    return min(C.TUNING_TRIALS, math.prod(len(v) for v in C.SEARCH_SPACES[model].values()))


def validate() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    final = []
    expected = {(f, m) for f in C.FAMILIES for m in MODELS}
    for family, model in expected:
        rp = C.RAW_RESULTS_DIR / f"{family}_{model}_seed0.json"
        tp = C.TUNING_DIR / f"{family}_{model}.json"
        if not rp.exists():
            errors.append(f"missing final result: {rp.name}")
        else:
            try:
                r = json.loads(rp.read_text())
                primary = r["metrics"]["primary"]
                for key in ("udr", "realised_fpr_benign_test"):
                    value = primary[key]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        errors.append(f"{rp.name}: invalid {key}")
                # Test FPR is not expected to be exactly 1%, but a near-one
                # value is a threshold-tie failure, not normal split drift.
                if primary["realised_fpr_benign_test"] > 0.05:
                    errors.append(f"{rp.name}: pathological benign-test FPR")
                if r["metrics"]["n_unseen"] <= 0:
                    errors.append(f"{rp.name}: empty unseen set")
                final.append(rp.name)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{rp.name}: malformed ({exc})")
        if not tp.exists():
            errors.append(f"missing tuning record: {tp.name}")
        else:
            try:
                t = json.loads(tp.read_text())
                trials = t["trials"]
                if len(trials) != expected_trials(model):
                    errors.append(f"{tp.name}: expected {expected_trials(model)} trials, found {len(trials)}")
                if not trials or t["best_hp"] not in [x["hp"] for x in trials]:
                    errors.append(f"{tp.name}: invalid best_hp")
                if any(not isinstance(x["val_pr_auc"], (int, float)) or not 0 <= x["val_pr_auc"] <= 1 for x in trials):
                    errors.append(f"{tp.name}: invalid validation objective")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{tp.name}: malformed ({exc})")
    return final, errors


def save(state: str, final: list[str], errors: list[str], pid: int | None = None) -> None:
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps({
        "checked_at_utc": now(), "state": state, "watcher_pid": pid,
        "results": len(final), "expected_results": len(C.FAMILIES) * len(MODELS),
        "tuning_records": len(list(C.TUNING_DIR.glob("*.json"))),
        "expected_tuning_records": len(C.FAMILIES) * len(MODELS),
        "stall_limit_seconds": 900, "validation_errors": errors,
    }, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    command = [sys.executable, "-m", "src.run_loaco", "--families", *C.FAMILIES,
               "--models", *MODELS, "--seeds", "0"]
    child: subprocess.Popen[str] | None = None
    last_progress: tuple[int, int] | None = None
    last_change = time.monotonic()
    while True:
        final, errors = validate()
        complete = len(final) == len(C.FAMILIES) * len(MODELS) and not errors
        if complete:
            save("completed_valid", final, errors, child.pid if child else None)
            return 0
        current = (len(final), len(list(C.TUNING_DIR.glob("*.json"))))
        if current != last_progress:
            last_progress, last_change = current, time.monotonic()
        stalled = time.monotonic() - last_change >= 900
        if child is None or child.poll() is not None or stalled:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            child = subprocess.Popen(command, cwd=ROOT, stdout=(ROOT / "logs" / "classical_supervised_grid.log").open("a"), stderr=subprocess.STDOUT, text=True)
            last_change = time.monotonic()
        save("running", final, errors, child.pid)
        if a.once:
            return 0
        time.sleep(max(5, a.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
