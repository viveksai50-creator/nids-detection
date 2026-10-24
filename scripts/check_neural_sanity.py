"""Validation-only neural sanity checks and matched benign-train device timing.

Usage: python scripts/check_neural_sanity.py --sanity vae
       python scripts/check_neural_sanity.py --timing

No test or held-out-family array is scored by this script. Diagnostic JSONs
are stored outside the canonical raw-results directory.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data.split import build_split
from src.eval.metrics import val_pr_auc
from src.models import build_model
from src.run_loaco import write_json_atomically


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", choices=["ae", "vae"])
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    split = build_split("DoS", 0)
    out = C.RESULTS_DIR / "diagnostics"
    if args.sanity:
        name = args.sanity
        path = out / f"DoS_{name}_validation_sanity_seed0.json"
        if path.exists():
            print(f"preserve {path.name}", flush=True)
        else:
            model = build_model(name, 0)
            checkpoint = C.CHECKPOINT_DIR / f"sanity_DoS_{name}_seed0.pt"
            model.configure_checkpoint(checkpoint, {
                "purpose": "validation_sanity", "family": "DoS", "model": name,
                "seed": 0, "hp": model.hp, "n_train": len(split.X_train),
                "n_val": len(split.X_val), "n_features": len(split.feature_names),
            })
            model.fit(split.X_train, split.y_train, split.X_val, split.y_val)
            scores = model.score(split.X_val)
            assert np.isfinite(scores).all()
            history = model.history
            result = {
                "purpose": "validation_only_sanity", "family": "DoS", "model": name,
                "seed": 0, "hp": model.hp, "device": str(model.dev),
                "validation_roc_auc": float(roc_auc_score(split.y_val, scores)),
                "validation_pr_auc": val_pr_auc(scores, split.y_val),
                "history": history, "epochs": len(history),
                "loss_decreased": history[-1]["train"] < history[0]["train"],
                "early_stopped": len(history) < C.MAX_EPOCHS,
                "finite_losses": all(math.isfinite(x[k]) for x in history for k in ("train", "val")),
            }
            result["passes_sanity"] = result["loss_decreased"] and result["finite_losses"] and result["validation_roc_auc"] > 0.7
            write_json_atomically(path, result)
            checkpoint.unlink(missing_ok=True)
            print({k: v for k, v in result.items() if k != "history"}, flush=True)
    if args.timing:
        path = out / "ae_cpu_mps_epoch_timing_seed0.json"
        if path.exists():
            print(f"preserve {path.name}", flush=True)
            return
        rows = []
        for dev in ("cpu", "mps"):
            if dev == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable for matching device comparison")
            os.environ["NIDS_DEVICE"] = dev
            model = build_model("ae", 0, {"epochs": 3})
            model.fit(split.X_train, split.y_train, split.X_val, split.y_val)
            seconds = [x["seconds"] for x in model.history]
            rows.append({"device": dev, "epoch_seconds": seconds,
                         "median_seconds_after_first_epoch": statistics.median(seconds[1:])})
            print(rows[-1], flush=True)
        write_json_atomically(path, {
            "purpose": "device_timing_only", "family": "DoS", "seed": 0,
            "model": "ae", "hp": model.hp, "batch_size": C.TORCH_BATCH,
            "n_benign_train": int((split.y_train == 0).sum()),
            "n_benign_val": int((split.y_val == 0).sum()), "n_features": len(split.feature_names),
            "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(), "machine": platform.machine(), "timings": rows,
            "measurement": "full benign-train epoch plus benign-validation loss; first epoch excluded from median; no checkpoint I/O; three epochs per device; sequential devices",
        })


if __name__ == "__main__":
    main()
