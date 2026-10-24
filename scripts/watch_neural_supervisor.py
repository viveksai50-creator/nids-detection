"""Outer POSIX watchdog: keep a neural supervisor alive until valid completion.

Usage: python scripts/watch_neural_supervisor.py --model ae
The matching supervisor reuses completed records and resumes incomplete fits.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "logs" / "neural_ae_watchdog_status.json"
MODEL = "ae"


def save(state: str, supervisor: subprocess.Popen | None, restarts: int) -> None:
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "state": state,
        "supervisor_pid": supervisor.pid if supervisor and supervisor.poll() is None else None,
        "supervisor_restarts": restarts,
    }, indent=2) + "\n")


def completed() -> bool:
    try:
        return json.loads((ROOT / "logs" / f"neural_{MODEL}_supervisor_status.json").read_text()).get("state") == "completed_valid"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    global MODEL, STATUS
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ae", "vae"], default="ae")
    p.add_argument("--poll-seconds", type=int, default=60)
    a = p.parse_args()
    MODEL = a.model
    STATUS = ROOT / "logs" / f"neural_{MODEL}_watchdog_status.json"
    lock = (ROOT / "logs" / f"neural_{MODEL}_watchdog.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    supervisor, restarts = None, 0
    while not completed():
        if supervisor is None or supervisor.poll() is not None:
            log = (ROOT / "logs" / f"neural_{MODEL}_supervisor.log").open("a")
            supervisor = subprocess.Popen([sys.executable, "scripts/supervise_neural_grid.py", "--model", MODEL], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            restarts += 1
        save("running", supervisor, restarts)
        time.sleep(max(5, a.poll_seconds))
    save("completed_valid", supervisor, restarts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
