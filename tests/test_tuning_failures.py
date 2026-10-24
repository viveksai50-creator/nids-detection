"""Numerical failures are auditable trials and cannot win validation selection."""
import json
from types import SimpleNamespace

import numpy as np

from src import config as C
from src import run_loaco as runner


def test_failed_trial_is_saved_and_best_hp_uses_validation_only(tmp_path, monkeypatch):
    X=np.zeros((4,2),dtype=np.float32)
    y=np.array([0,0,1,1],dtype=np.int8)
    # No test or unseen attributes are supplied: accessing either would fail.
    split=SimpleNamespace(X_train=X,y_train=y,X_val=X,y_val=y)
    calls=[]

    class Detector:
        def __init__(self,hp): self.hp=hp
        def fit(self,*args):
            assert args==(split.X_train,split.y_train,split.X_val,split.y_val)
            calls.append(self.hp)
            if self.hp['kind']=='bad': raise FloatingPointError('synthetic non-finite loss')
        def score(self,values):
            assert values is split.X_val
            return np.array([0.1,0.2,0.8,0.9])

    monkeypatch.setattr(C,'TUNING_DIR',tmp_path)
    monkeypatch.setattr(C,'SEARCH_SPACES',{'fixture':{'kind':['bad','good']}})
    monkeypatch.setattr(C,'TUNING_TRIALS',2)
    monkeypatch.setattr(runner,'build_split',lambda family,seed:split)
    monkeypatch.setattr(runner,'build_model',lambda name,seed,hp:Detector(hp))
    assert runner.tune('DoS','fixture')=={'kind':'good'}
    saved=json.loads((tmp_path/'DoS_fixture.json').read_text())
    assert saved['complete'] and len(saved['trials'])==2
    failed=next(x for x in saved['trials'] if x['hp']['kind']=='bad')
    assert failed['status']=='failed_numerical' and failed['val_pr_auc'] is None
    assert failed['error']=='synthetic non-finite loss'
    assert runner.tune('DoS','fixture')=={'kind':'good'}
    assert len(calls)==2  # A completed search is reused without refitting.
