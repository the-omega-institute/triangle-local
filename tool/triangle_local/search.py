"""Seeded finite-table search with perturbation and convex polishing."""
from time import perf_counter

import numpy as np

from .model import Model


def perturb(model, strength, rng):
    weights=[];responses=[]
    for w in model.weights:
        weights.append((1-strength)*w+strength*rng.dirichlet(np.ones(len(w))))
    for r in model.responses:
        random=rng.dirichlet(np.ones(r.shape[-1]),size=r.shape[:2])
        responses.append((1-strength)*r+strength*random)
    return Model(tuple(weights),tuple(responses))


def hybrid_fit(target, sizes, *, starts=4, steps=2500, seed=0, initial=None,
               hops=2, polish_steps=120, inner_steps=200, tolerance=1e-20,
               optimize_weights=True):
    from .optimize import fit, FitResult
    if hops<0 or polish_steps<0:
        raise ValueError('Nonnegative hops and polish_steps required')
    started=perf_counter();rng=np.random.default_rng(seed)
    best=initial
    best_value=float('inf') if best is None else float(np.sum((best.distribution()-target)**2))
    runs=[];seen=set()
    for hop in range(hops+1):
        if best_value<=tolerance:break
        warm=initial if hop==0 else perturb(best,[.03,.12,.3][(hop-1)%3],rng)
        result=fit(target,sizes,method='lbfgs',starts=starts if hop==0 else 1,
                   steps=steps,seed=seed+hop,initial=warm,tolerance=tolerance)
        if polish_steps:
            polished=fit(target,sizes,method='convex',starts=1,steps=polish_steps,
                         initial=result.model,inner_steps=inner_steps,tolerance=tolerance,
                         optimize_weights=optimize_weights)
            if polished.objective<result.objective:
                model,value=polished.model,polished.objective
            else:model,value=result.model,result.objective
        else:model,value=result.model,result.objective
        fingerprint=np.round(model.distribution(),7).tobytes()
        duplicate=fingerprint in seen
        seen.add(fingerprint)
        if value<best_value:best,best_value=model,value
        runs.append({'hop':hop,'objective':value,'duplicate_distribution':duplicate,
                     'joint':result.summary(),
                     'polish':polished.summary() if polish_steps else None})
    return FitResult(best,'l2',best_value,perf_counter()-started,runs)
