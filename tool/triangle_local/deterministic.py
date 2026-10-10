"""Discrete response coordinate descent with independent simplex source updates."""
import numpy as np

from .convex import simplex_least_squares
from .model import Model


def deterministic_descent(model, target, steps=100, optimize_weights=True, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    current = model.rounded()
    trace = [float(np.sum((current.distribution()-target)**2))]
    for sweep in range(steps):
        before = trace[-1]
        for party in rng.permutation(3):
            _, matrices = current.linear_maps()
            design = matrices[party]
            tables = [r.copy() for r in current.responses]
            rows = tables[party].reshape(-1, current.outputs[party])
            residual = (current.distribution()-target).ravel()
            for row in rng.permutation(len(rows)):
                old = int(rows[row].argmax())
                columns = design[:, row*rows.shape[1]:(row+1)*rows.shape[1]]
                changes = columns-columns[:,old,None]
                costs = 2*residual@changes + np.sum(changes*changes,axis=0)
                choice = int(costs.argmin())
                if costs[choice] < -1e-18:
                    residual += changes[:,choice]
                    rows[row] = np.eye(rows.shape[1])[choice]
            current = Model(current.weights, tuple(tables))
        if optimize_weights:
            for source in range(3):
                dw, _ = current.linear_maps()
                weights = list(current.weights)
                weights[source] = simplex_least_squares(dw[source],target.ravel(),weights[source],300)
                proposed = Model(tuple(weights),current.responses)
                if np.linalg.norm(proposed.distribution()-target) <= np.linalg.norm(current.distribution()-target):
                    current = proposed
        value = float(np.sum((current.distribution()-target)**2))
        trace.append(value)
        if before-value < 1e-17:
            break
    return current, sweep+1, value, trace
