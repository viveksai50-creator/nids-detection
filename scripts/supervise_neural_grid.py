"""Checkpoint-aware supervisor for an AE or VAE seed-0 grid (POSIX).

Usage: python scripts/supervise_neural_grid.py --model ae

It never deletes a result, tuning record, or epoch checkpoint.  Completed
records are validated before they count toward completion; an interrupted
runner is restarted from its own atomic checkpoints after 15 minutes without
saved progress.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
import time
import psutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

STATUS = ROOT / "logs" / "neural_ae_supervisor_status.json"
MODEL = "ae"


class AdoptedRunner:
    """Monitor the existing runner after its previous supervisor exited."""
    def __init__(self, process):
        self.process, self.pid = process, process.pid

    def poll(self):
        try:
            return None if self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE else 0
        except psutil.NoSuchProcess:
            return 0

    def terminate(self):
        self.process.terminate()

    def wait(self, timeout=30):
        return self.process.wait(timeout=timeout)


def adopt_runner():
    try:
        pid = json.loads(STATUS.read_text()).get("runner_pid")
        if not pid:
            return None
        process = psutil.Process(pid)
        command = process.cmdline()
        if "src.run_loaco" in command and "--models" in command and command[command.index("--models") + 1] == MODEL:
            if Path(process.cwd()).resolve() == ROOT.resolve():
                return AdoptedRunner(process)
    except (OSError, ValueError, KeyError, psutil.Error):
        pass
    return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate() -> tuple[list[str], list[str], int]:
    final, errors, trial_count = [], [], 0
    for family in C.FAMILIES:
        rp = C.RAW_RESULTS_DIR / f"{family}_{MODEL}_seed0.json"
        tp = C.TUNING_DIR / f"{family}_{MODEL}.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text())
                p = r["metrics"]["primary"]
                if r["family"] != family or r["model"] != MODEL or r["seed"] != 0:
                    errors.append(f"{rp.name}: wrong provenance")
                for key in ("udr", "realised_fpr_benign_test"):
                    if not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or not 0 <= p[key] <= 1:
                        errors.append(f"{rp.name}: invalid {key}")
                if p["realised_fpr_benign_test"] > 0.05 or r["metrics"]["n_unseen"] <= 0:
                    errors.append(f"{rp.name}: invalid final evaluation")
                final.append(rp.name)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{rp.name}: malformed ({exc})")
        if tp.exists():
            try:
                t = json.loads(tp.read_text())
                trials = t["trials"]
                trial_count += len(trials)
                if len(trials) > C.TUNING_TRIALS:
                    errors.append(f"{tp.name}: too many trials")
                if t.get("complete") and (len(trials) != C.TUNING_TRIALS or t.get("best_hp") not in [x["hp"] for x in trials]):
                    errors.append(f"{tp.name}: invalid completed tuning record")
                if any((not isinstance(x.get("val_pr_auc"), (int, float)) or not 0 <= x["val_pr_auc"] <= 1)
                       and not (x.get("status") == "failed_numerical" and x.get("error")) for x in trials):
                    errors.append(f"{tp.name}: invalid validation objective")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{tp.name}: malformed ({exc})")
    return final, errors, trial_count


def save(state: str, final: list[str], errors: list[str], trials: int, child: subprocess.Popen | None) -> None:
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps({
        "checked_at_utc": now(), "state": state,
        "runner_pid": child.pid if child and child.poll() is None else None,
        "results": len(final), "expected_results": len(C.FAMILIES),
        "completed_tuning_trials": trials, "expected_tuning_trials": len(C.FAMILIES) * C.TUNING_TRIALS,
        "stall_limit_seconds": 900, "validation_errors": errors,
    }, indent=2) + "\n")


def main() -> int:
    global MODEL, STATUS
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ae", "vae"], default="ae")
    p.add_argument("--poll-seconds", type=int, default=60)
    a = p.parse_args()
    MODEL = a.model
    STATUS = ROOT / "logs" / f"neural_{MODEL}_supervisor_status.json"
    command = [sys.executable, "-m", "src.run_loaco", "--families", *C.FAMILIES, "--models", MODEL, "--seeds", "0"]
    lock = (ROOT / "logs" / f"neural_{MODEL}_supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    child = adopt_runner()
    last_progress, last_change = None, time.monotonic()
    while True:
        final, errors, trials = validate()
        if len(final) == len(C.FAMILIES) and trials == len(C.FAMILIES) * C.TUNING_TRIALS and not errors:
            save("completed_valid", final, errors, trials, child)
            return 0
        checkpoints = list(C.CHECKPOINT_DIR.glob(f"*_{MODEL}_*.pt"))
        progress = (len(final), trials, max((p.stat().st_mtime_ns for p in checkpoints), default=0))
        if progress != last_progress:
            last_progress, last_change = progress, time.monotonic()
        if child is None or child.poll() is not None or time.monotonic() - last_change >= 900:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            log = (ROOT / "logs" / f"neural_{MODEL}_grid.log").open("a")
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            last_change = time.monotonic()
        save("running", final, errors, trials, child)
        time.sleep(max(5, a.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
