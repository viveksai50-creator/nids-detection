"""Validate frozen AE/VAE seed-0 grids; never train or choose settings.

Usage: python scripts/validate_neural_results.py
Writes a local diagnostic manifest under results/diagnostics/.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.run_loaco import hp_id, sample_configs, write_json_atomically


def bounded(value):
    return isinstance(value, (int,float)) and math.isfinite(value) and 0 <= value <= 1


def main():
    final = []
    total_trials, failed_trials = 0, []
    for family in C.FAMILIES:
        for model in ("ae","vae"):
            rp=C.RAW_RESULTS_DIR/f"{family}_{model}_seed0.json"
            tp=C.TUNING_DIR/f"{family}_{model}.json"
            r,t=json.loads(rp.read_text()),json.loads(tp.read_text())
            assert t["complete"] and len(t["trials"])==C.TUNING_TRIALS
            planned=sample_configs(C.SEARCH_SPACES[model], C.TUNING_TRIALS,
                                   zlib.crc32(f"{family}_{model}".encode()) & 0xFFFF)
            assert {hp_id(x["hp"]) for x in t["trials"]}=={hp_id(hp) for hp in planned}
            assert len({hp_id(x["hp"]) for x in t["trials"]})==C.TUNING_TRIALS
            valid=[]
            for i,trial in enumerate(t["trials"]):
                total_trials+=1
                if trial.get("status")=="failed_numerical":
                    assert trial.get("val_pr_auc") is None and trial.get("error")
                    failed_trials.append({"family":family,"model":model,"trial":i,"error":trial["error"]})
                else:
                    assert bounded(trial["val_pr_auc"])
                    valid.append(trial)
            assert valid
            assert r["hp"]==t["best_hp"]==max(valid,key=lambda x:x["val_pr_auc"])["hp"]
            assert (r["family"],r["model"],r["seed"])==(family,model,0)
            assert r["device"]=="mps"
            assert r["metrics"]["n_unseen"]==r["split"]["n_unseen"]>0
            assert r["metrics"]["n_test"]==r["split"]["n_test"]>0
            assert sum(r["split"][f"n_{p}"] for p in ("train","val","test","unseen"))==r["split"]["n_total"]
            assert r["metrics"]["primary"]==r["metrics"]["at_fpr"]["0.01"]
            assert set(r["thresholds"])=={"0.001","0.01","0.05"}
            assert all(math.isfinite(v) for v in r["thresholds"].values())
            assert all(bounded(r["validation"][k]) for k in ("roc_auc","pr_auc"))
            assert all(bounded(r["metrics"][k]) for k in ("roc_auc","pr_auc","unseen_roc_auc"))
            assert 1<=len(r["history"])<=C.MAX_EPOCHS
            assert all(math.isfinite(x[k]) for x in r["history"] for k in ("train","val"))
            for point in r["metrics"]["at_fpr"].values():
                assert all(bounded(point[k]) for k in ("udr","realised_fpr_benign_test","known_attack_recall"))
            assert r["metrics"]["primary"]["realised_fpr_benign_test"]<0.05
            assert r["train_seconds"]>0 and r["peak_rss_mb"]>0
            assert r["latency"]["batch_rows"]==r["metrics"]["n_test"]+r["metrics"]["n_unseen"]
            final.append({"family":family,"model":model,"seed":0,
                          "result":str(rp.relative_to(ROOT)),"result_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),
                          "tuning":str(tp.relative_to(ROOT)),"tuning_sha256":hashlib.sha256(tp.read_bytes()).hexdigest(),
                          "udr":r["metrics"]["primary"]["udr"],
                          "benign_test_fpr":r["metrics"]["primary"]["realised_fpr_benign_test"]})
            print(f"validated {family}/{model}: ten configurations, selected HP, final metrics and resource fields",flush=True)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/"src").rglob("*.py"))}
    chip=platform.processor() or platform.machine()
    if platform.system()=="Darwin":
        chip=subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],capture_output=True,text=True,check=True).stdout.strip()
    manifest={"checked_at_utc":datetime.now(timezone.utc).isoformat(),"state":"completed_valid",
              "neural_final_results":len(final),"tuning_trials":total_trials,"failed_numerical_trials":failed_trials,
              "single_seed_only":True,"cpu_identifier":chip,"platform":platform.platform(),
              "records":final,"current_source_sha256":sources,
              "source_scope_note":"Source hashes are the current reproducibility manifest; older completed AE records predate the later checkpoint/RNG instrumentation. Existing result JSONs were preserved."}
    write_json_atomically(C.RESULTS_DIR/"diagnostics/neural_completion.json",manifest)
    print(f"PASS: {len(final)} neural finals, {total_trials} trials, {len(failed_trials)} explicitly recorded numerical failures")


if __name__=="__main__":
    main()
