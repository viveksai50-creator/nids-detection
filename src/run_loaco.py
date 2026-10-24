"""Leave-one-family-out runner with train-only fitting and validation-only selection.

Usage:
    python -m src.run_loaco --dry-run
    python -m src.run_loaco --tier A --seeds 0 1 2
    python -m src.run_loaco --families DoS --models rf xgb --seeds 0 --no-tune
    python -m src.run_loaco --families DoS --models ae --seeds 0 --no-tune --output-dir results/raw/sanity

One JSON per (family, model, seed) in results/raw/. Existing JSONs are skipped, so the
runner can be restarted at any time. Tuning (per family, model, seed=TUNING_SEED,
objective = validation PR-AUC) is cached in results/tuning/.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data.split import Split, build_split
from src.eval.metrics import compute_run_metrics, val_pr_auc
from src.eval.resources import (PeakMemorySampler, Timer,
                                measure_batch_latency,
                                measure_single_flow_latency,
                                torch_peak_memory_mb)
from src.eval.thresholds import thresholds_for
from src.models import build_model


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sample_configs(space: dict[str, list], n: int, seed: int) -> list[dict]:
    grid = [dict(zip(space.keys(), vals)) for vals in itertools.product(*space.values())]
    rng = random.Random(seed)
    rng.shuffle(grid)
    return grid[:n]


def write_json_atomically(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(value, fh, indent=2, default=str)
    os.replace(tmp, path)


def hp_id(hp: dict) -> str:
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:12]


def tune(family: str, model_name: str) -> dict:
    path = C.TUNING_DIR / f"{family}_{model_name}.json"
    if path.exists():
        with open(path) as f:
            cached = json.load(f)
        if cached.get("complete", "best_hp" in cached):
            return cached["best_hp"]
    split = build_split(family, C.TUNING_SEED)
    configs = sample_configs(C.SEARCH_SPACES[model_name], C.TUNING_TRIALS, seed=zlib.crc32(f"{family}_{model_name}".encode()) & 0xFFFF)
    trials = cached.get("trials", []) if path.exists() else []
    completed = {hp_id(t["hp"]) for t in trials}
    for hp in configs:
        if hp_id(hp) in completed:
            continue
        t0 = time.perf_counter()
        m = build_model(model_name, C.TUNING_SEED, hp)
        checkpoint_path = None
        if model_name in ("ae", "vae"):
            checkpoint_path = C.CHECKPOINT_DIR / f"tuning_{family}_{model_name}_{hp_id(hp)}.pt"
            m.configure_checkpoint(checkpoint_path, {
                "family": family, "model": model_name, "seed": C.TUNING_SEED, "hp": hp,
                "purpose": "tuning", "n_train": split.meta["n_train"],
                "n_val": split.meta["n_val"], "n_features": len(split.feature_names),
            })
        try:
            m.fit(split.X_train, split.y_train, split.X_val, split.y_val)
            scores = m.score(split.X_val)
            if not np.isfinite(scores).all():
                raise FloatingPointError("non-finite validation scores")
            obj = val_pr_auc(scores, split.y_val)
            trial = {"hp": hp, "val_pr_auc": obj, "status": "completed"}
        except FloatingPointError as exc:
            trial = {"hp": hp, "val_pr_auc": None, "status": "failed_numerical",
                     "error": str(exc)}
        trial.update({"seconds": time.perf_counter() - t0, "history": getattr(m, "history", None)})
        trials.append(trial)
        write_json_atomically(path, {
            "family": family, "model": model_name, "objective": "val_pr_auc",
            "trials": trials, "complete": False,
        })
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
        print(f"  tune {family}/{model_name} {hp} -> {trial['status']} val PR-AUC {trial['val_pr_auc']}", flush=True)
    valid_trials = [t for t in trials if isinstance(t.get("val_pr_auc"), (int, float)) and np.isfinite(t["val_pr_auc"])]
    if not valid_trials:
        raise RuntimeError(f"no finite validation trial for {family}/{model_name}; inspect saved failure records")
    best = max(valid_trials, key=lambda t: t["val_pr_auc"])
    write_json_atomically(path, {
        "family": family, "model": model_name, "objective": "val_pr_auc",
        "trials": trials, "best_hp": best["hp"], "complete": True,
    })
    return best["hp"]


def run_one(family: str, model_name: str, seed: int, hp: dict, measure_latency: bool = True) -> dict:
    split: Split = build_split(family, seed)
    model = build_model(model_name, seed, hp)
    checkpoint_path = None
    if model_name in ("ae", "vae"):
        checkpoint_path = C.CHECKPOINT_DIR / f"{family}_{model_name}_seed{seed}.pt"
        model.configure_checkpoint(checkpoint_path, {
            "family": family, "model": model_name, "seed": seed, "hp": hp,
            "n_train": split.meta["n_train"], "n_val": split.meta["n_val"],
            "n_features": len(split.feature_names),
        })
    with PeakMemorySampler() as mem, Timer() as t_fit:
        model.fit(split.X_train, split.y_train, split.X_val, split.y_val)
    scores_val = model.score(split.X_val)
    benign_val_scores = scores_val[split.y_val == 0]
    thresholds = thresholds_for(benign_val_scores, C.TARGET_FPRS)
    with Timer() as t_score:
        scores_test = model.score(split.X_test)
        scores_unseen = model.score(split.X_unseen)
    metrics = compute_run_metrics(scores_test, split.y_test, scores_unseen, thresholds, f"{C.PRIMARY_FPR:g}")
    latency = {}
    if measure_latency:
        X_lat = np.concatenate([split.X_test, split.X_unseen])
        latency.update(measure_batch_latency(model.score, X_lat))
        latency.update(measure_single_flow_latency(model.score, X_lat, seed=seed))
    dev = getattr(model, "dev", None)
    result = {
        "family": family, "model": model_name, "seed": seed, "hp": hp,
        "device": str(dev) if dev is not None else "cpu",
        "git": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "split": split.meta, "thresholds": thresholds,
        "validation": {
            "roc_auc": float(roc_auc_score(split.y_val, scores_val)),
            "pr_auc": val_pr_auc(scores_val, split.y_val),
        },
        "train_seconds": t_fit.seconds, "score_seconds_test_plus_unseen": t_score.seconds,
        "peak_rss_mb": mem.peak_rss_mb, "torch_peak_mb": torch_peak_memory_mb(),
        "latency": latency, "metrics": metrics,
        "history": getattr(model, "history", None),
    }
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(checkpoint_path)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="*", default=C.FAMILIES)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--tier", choices=["A", "B"], default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--output-dir", type=Path, default=C.RAW_RESULTS_DIR,
                   help="directory for result JSONs (use a non-root subdirectory for diagnostic runs)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    models = a.models or (C.TIER_A_MODELS if a.tier == "A" else C.TIER_B_MODELS if a.tier == "B" else C.TIER_A_MODELS)
    seeds = a.seeds if a.seeds is not None else (C.TIER_A_SEEDS if a.tier == "A" else C.SEEDS)
    C.ensure_dirs()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plan = [(f, m, s) for f in a.families for m in models for s in seeds]
    print(f"plan: {len(plan)} runs = {len(a.families)} families x {len(models)} models x {len(seeds)} seeds; tune={'no' if a.no_tune else 'yes'}")
    if a.dry_run:
        for f, m, s in plan:
            print(f"  {f:14s} {m:8s} seed={s}")
        return 0

    for f, m, s in plan:
        out = a.output_dir / f"{f}_{m}_seed{s}.json"
        if out.exists():
            print(f"skip {out.name}")
            continue
        hp = C.DEFAULT_HP[m] if a.no_tune else tune(f, m)
        print(f"run {f}/{m}/seed{s} hp={hp}")
        res = run_one(f, m, s, hp, measure_latency=not a.no_latency)
        write_json_atomically(out, res)
        checkpoint_path = res.get("checkpoint_path")
        if checkpoint_path:
            Path(checkpoint_path).unlink(missing_ok=True)
        pm = res["metrics"]["primary"]
        print(f"  UDR@{C.PRIMARY_FPR:g}={pm['udr']:.3f} unseenAUC={res['metrics']['unseen_roc_auc']:.3f} PR-AUC={res['metrics']['pr_auc']:.3f} FPR={pm['realised_fpr_benign_test']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
