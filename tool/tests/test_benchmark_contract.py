"""The measured campaign must never credit a candidate finished after a budget."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def runner():
    path=Path(__file__).resolve().parents[2]/'benchmarks/round4/frozen_runner.py'
    spec=importlib.util.spec_from_file_location('triangle_frozen_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_completed_after_deadline_is_not_credited(monkeypatch):
    mod=runner();clock=[0.]
    monkeypatch.setattr(mod.time,'perf_counter',lambda:clock[0])
    rec=mod.Recorder(np.array([0.]),(10,60))
    clock[0]=1.;rec.offer(np.array([.2]),'early')
    # Start scoring just before 10 s, finish after it.
    readings=iter([9.99,10.01])
    monkeypatch.setattr(mod.time,'perf_counter',lambda:next(readings))
    rec.offer(np.array([.1]),'late-for-first-budget')
    assert rec.snapshots[10]['payload']=='early'
    assert rec.best['payload']=='late-for-first-budget'
    monkeypatch.setattr(mod.time,'perf_counter',lambda:60.01)
    with pytest.raises(mod.Deadline):
        rec.offer(np.array([0.]),'late-for-both')
    assert rec.snapshots[60]['payload']=='late-for-first-budget'
    assert rec.snapshots[10]['candidate_seconds'] <= 10
    assert rec.snapshots[60]['candidate_seconds'] <= 60


def test_campaign_seed_domains_do_not_overlap():
    run_seeds=[];restart_seeds=[];heldout_seeds=[]
    for target in range(3):
        for method in range(3):
            for rep in range(10):
                seed=int(np.random.SeedSequence([20261011,target,method,rep]).generate_state(1)[0])
                run_seeds.append(seed)
                for restart in range(100):
                    restart_seeds.append(int(np.random.SeedSequence([seed,101,restart]).generate_state(1)[0]))
                for budget in [10,60]:
                    heldout_seeds.append(int(np.random.SeedSequence([seed,203,budget]).generate_state(1,dtype=np.uint64)[0]))
    assert len(set(run_seeds))==90
    assert len(set(restart_seeds))==9000
    assert len(set(heldout_seeds))==180
