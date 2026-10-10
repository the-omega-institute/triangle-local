"""Alternating convex simplex subproblems for untied finite models.

Each response or independent source enters linearly when the other five factors
are fixed. L2 uses projected accelerated gradient; L1 uses a sparse LP. Block
solves are numerical and may hit their cap; accepted updates cannot increase
the evaluated objective. No global convexity or optimality is asserted.
"""
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, kron, vstack

from .model import Model


def project_simplex(x):
    """Euclidean projection onto a simplex, independently along the last axis."""
    u = np.sort(x, axis=-1)[..., ::-1]
    css = np.cumsum(u, axis=-1)-1
    indices = np.arange(1, x.shape[-1]+1)
    valid = u-css/indices > 0
    rho = valid.sum(axis=-1, keepdims=True)-1
    theta = np.take_along_axis(css, rho, axis=-1)/(rho+1)
    return np.maximum(x-theta, 0)


def simplex_least_squares(design, target, initial, steps=300):
    shape = initial.shape
    x = initial.copy()
    y = x.copy()
    momentum = 1.
    lipschitz = float(np.linalg.eigvalsh(design @ design.T)[-1])
    if lipschitz < 1e-30:
        return x
    previous = float(np.sum((design @ x.ravel()-target)**2))
    for _ in range(steps):
        gradient = (design.T @ (design @ y.ravel()-target)).reshape(shape)
        candidate = project_simplex(y-gradient/lipschitz)
        value = float(np.sum((design @ candidate.ravel()-target)**2))
        if value > previous + 1e-20:
            y = x.copy()
            momentum = 1.
            gradient = (design.T @ (design @ y.ravel()-target)).reshape(shape)
            candidate = project_simplex(y-gradient/lipschitz)
            value = float(np.sum((design @ candidate.ravel()-target)**2))
        if np.max(np.abs(candidate-x)) < 1e-10:
            x = candidate
            break
        new_momentum = (1+np.sqrt(1+4*momentum**2))/2
        y = candidate + (momentum-1)/new_momentum*(candidate-x)
        x, momentum, previous = candidate, new_momentum, value
    return x


def simplex_l1(design, target, initial):
    rows = initial.size//initial.shape[-1]
    simplex = kron(eye(rows), np.ones((1, initial.shape[-1])), format="csr")
    d = csr_matrix(design)
    identity = eye(len(target), format="csr")
    # |D x-t| <= slack, with exact simplex equalities in the numerical LP.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
        result = linprog(np.r_[np.zeros(initial.size), np.ones(len(target))],
                         A_ub=vstack([hstack([d,-identity]), hstack([-d,-identity])]),
                         b_ub=np.r_[target,-target],
                         A_eq=hstack([simplex,csr_matrix((rows,len(target)))]),
                         b_eq=np.ones(rows), bounds=(0,None), method="highs",
                         options={"threads":2})
    if not result.success:
        return initial
    return project_simplex(result.x[:initial.size].reshape(initial.shape))


def alternating_convex(model, target, steps, tolerance, loss="l2", inner_steps=300,
                       optimize_weights=True):
    from .optimize import loss_gradient
    current = model
    trace = [loss_gradient(current.distribution(),target,loss)[0]]
    for sweep in range(steps):
        before = trace[-1]
        current_value = before
        # Rebuild the linear map after each accepted mutation.
        blocks = list(range(3,6)) + (list(range(3)) if optimize_weights else [])
        for block in blocks:
            dw, dr = current.linear_maps()
            matrices = [*dw,*dr]
            arrays = [*current.weights,*current.responses]
            if loss == "l2":
                proposed = simplex_least_squares(matrices[block],target.ravel(),arrays[block],inner_steps)
            else:
                proposed = simplex_l1(matrices[block],target.ravel(),arrays[block])
            arrays[block] = proposed
            candidate = Model(tuple(arrays[:3]),tuple(arrays[3:]))
            value = loss_gradient(candidate.distribution(),target,loss)[0]
            if value <= current_value:
                current = candidate
                current_value = value
        value = loss_gradient(current.distribution(),target,loss)[0]
        trace.append(value)
        if value <= tolerance or before-value < 1e-17:
            break
    return current, sweep+1, value, trace
