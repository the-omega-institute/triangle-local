"""Full finite contractions; multistart optimization is an inner approximation."""
from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np
from scipy.optimize import minimize, least_squares

from .model import Model, Parameterization
from .convex import alternating_convex
from .deterministic import deterministic_descent


def loss_gradient(p, target, loss="l2"):
    if loss == "l2":
        d = p - target
        return float(np.sum(d * d)), 2 * d
    if loss == "l1":
        return float(np.abs(p - target).sum()), np.sign(p - target)
    if loss == "kl":
        # Smooth, strictly positive floor; derivative matches this objective.
        eps = 1e-15
        mask = target > 0
        value = np.sum(target[mask] * np.log(target[mask] / (p[mask] + eps)))
        return float(value), -target / (p + eps)
    raise ValueError("Loss must be l1, l2 (squared Euclidean), or kl (target || model)")


@dataclass
class FitResult:
    model: Model
    loss: str
    objective: float
    elapsed_seconds: float
    runs: list

    def summary(self):
        return {k: v for k, v in asdict(self).items() if k != "model"}


def _em(model, target, steps, tolerance):
    """Exact latent EM: local response and independent source M-steps.

    Posterior expected counts are p(parameter) * d/dparameter sum P_t log P_m.
    Zero target entries contribute no counts. Initialization has full support.
    """
    current = model
    trace = []
    best_value = float("inf")
    for _ in range(steps):
        p = current.distribution()
        value, _ = loss_gradient(p, target, "kl")
        trace.append(value)
        if value <= tolerance:
            break
        dw, dr = current.derivatives(target / (p + 1e-15))
        ws = [w * d for w, d in zip(current.weights, dw)]
        rs = [r * d for r, d in zip(current.responses, dr)]
        ws = [w / w.sum() for w in ws]
        rs = [np.divide(r, r.sum(-1, keepdims=True), out=old.copy(),
                        where=r.sum(-1, keepdims=True) > 0)
              for r, old in zip(rs, current.responses)]
        candidate = Model(tuple(ws), tuple(rs))
        new_value, _ = loss_gradient(candidate.distribution(), target, "kl")
        if new_value > value + 1e-11:
            break
        current = candidate
        best_value = new_value
    value, _ = loss_gradient(current.distribution(), target, "kl")
    return current, len(trace), value, trace


def fit(target, sizes=(4, 4, 4), *, symmetry="none", method="lbfgs", loss="l2",
        starts=3, steps=500, seed=0, learning_rate=0.08, tolerance=1e-12,
        initial=None, inner_steps=300, optimize_weights=True, hops=2, polish_steps=120):
    target = np.asarray(target, dtype=float)
    if (target.ndim != 3 or not np.all(np.isfinite(target)) or np.any(target < 0)
            or not np.isclose(target.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Target must be a normalized nonnegative 3D tensor")
    if starts < 1 or steps < 1 or tolerance < 0 or learning_rate <= 0:
        raise ValueError("Positive starts, steps, learning rate and nonnegative tolerance required")
    if method not in ("adam", "lbfgs", "alternating", "em", "least-squares", "convex", "deterministic", "hybrid", "neural"):
        raise ValueError("Unknown optimization method")
    if method == "em" and (loss != "kl" or symmetry != "none"):
        raise ValueError("EM requires --loss kl and --symmetry none; use alternating for tied symmetry")
    if method == "least-squares" and loss != "l2":
        raise ValueError("Least-squares requires the l2 loss")
    if method == "convex" and (symmetry != "none" or loss not in ("l1", "l2")):
        raise ValueError("Convex block optimization requires untied parameters and l1 or l2")
    if inner_steps < 1:
        raise ValueError("Positive inner_steps required")
    if method == "deterministic" and (symmetry != "none" or loss != "l2"):
        raise ValueError("Deterministic descent requires untied parameters and l2")
    if method in ("hybrid", "neural") and (symmetry != "none" or loss != "l2"):
        raise ValueError("Hybrid and neural search require untied parameters and l2")
    loss_gradient(target, target, loss)  # Validate loss even on early convergence.
    parametrization = Parameterization(sizes, target.shape, symmetry)
    if initial is not None:
        parametrization.encode(initial)  # Validate before early returns or dispatch.
    if method == "hybrid":
        from .search import hybrid_fit
        recognized = None
        if initial is None:
            from .families import recognized_seed
            recognized,initial = recognized_seed(target,sizes)
        result = hybrid_fit(target,sizes,starts=starts,steps=steps,seed=seed,initial=initial,
                          hops=hops,polish_steps=polish_steps,inner_steps=inner_steps,
                          tolerance=tolerance,optimize_weights=optimize_weights)
        if recognized:
            result.runs.insert(0,{"recognized_control":recognized,"seed_only":True,
                                  "objective":float(np.sum((initial.distribution()-target)**2))})
        return result
    if method == "neural":
        if initial is not None:
            raise ValueError("Neural initialization uses network weights; table warm starts are unsupported")
        from .neural import neural_fit
        return neural_fit(target,sizes,starts=starts,steps=steps,seed=seed)
    rng = np.random.default_rng(seed)
    t0 = perf_counter()
    best, best_value = None, float("inf")
    runs = []
    for start in range(starts):
        x = parametrization.encode(initial) if start == 0 and initial is not None else rng.normal(
            0, 1.5, parametrization.dimension)
        evaluations = 0
        run_best, run_value = None, float("inf")

        def evaluate(z):
            nonlocal evaluations, run_best, run_value
            model = parametrization.decode(z)
            value, g = loss_gradient(model.distribution(), target, loss)
            evaluations += 1
            if value < run_value:
                run_value, run_best = value, model
            return value, parametrization.pullback(z, model, g)

        status = "iteration_limit"
        if method == "deterministic":
            starting = initial if start == 0 and initial is not None else parametrization.decode(x)
            run_best, nit, run_value, trace = deterministic_descent(
                starting, target, steps, optimize_weights, rng)
            evaluations = nit*6+1
            status = "coordinate_minimum_or_iteration_limit"
        elif method == "convex":
            starting = initial if start == 0 and initial is not None else parametrization.decode(x)
            run_best, nit, run_value, trace = alternating_convex(
                starting, target, steps, tolerance, loss, inner_steps, optimize_weights)
            evaluations = nit * (6 if optimize_weights else 3) + 1
            status = "tolerance" if run_value <= tolerance else "stalled_or_iteration_limit"
        elif method == "em":
            run_best, nit, run_value, trace = _em(parametrization.decode(x), target, steps, tolerance)
            evaluations = nit + 1
            status = "tolerance" if run_value <= tolerance else "iteration_limit"
        elif method == "least-squares":
            def residual(z):
                nonlocal evaluations, run_best, run_value
                model = parametrization.decode(z)
                d = (model.distribution() - target).ravel()
                value = float(d @ d)
                evaluations += 1
                if value < run_value:
                    run_best, run_value = model, value
                return d

            def jacobian(z):
                return parametrization.jacobian(z, parametrization.decode(z))

            result = least_squares(residual, x, jac=jacobian, tr_solver="lsmr",
                                   max_nfev=steps, ftol=1e-13, xtol=1e-13, gtol=1e-14)
            nit = int(result.nfev)
            status = str(result.message)
        elif method == "lbfgs":
            result = minimize(evaluate, x, jac=True, method="L-BFGS-B",
                              options={"maxiter": steps, "ftol": 1e-15,
                                       "gtol": 1e-10, "maxls": 30})
            nit = int(result.nit)
            status = str(result.message)
        elif method == "alternating":
            nit = 0
            # Each block is projected onto its simplex through softmax. Block
            # subproblems use the actual tied objective, including in S3 modes.
            for sweep in range(steps):
                previous, _ = evaluate(x)
                for lo, hi in zip(parametrization.cuts[:-1], parametrization.cuts[1:]):
                    base = x.copy()

                    def block(z):
                        full = base.copy()
                        full[lo:hi] = z
                        val, grad = evaluate(full)
                        return val, grad[lo:hi]

                    result = minimize(block, x[lo:hi], jac=True, method="L-BFGS-B",
                                      options={"maxiter": 12, "ftol": 1e-14, "gtol": 1e-10})
                    x[lo:hi] = result.x
                value, _ = evaluate(x)
                nit = sweep + 1
                if value <= tolerance or abs(previous - value) < 1e-15:
                    status = "tolerance" if value <= tolerance else "stalled"
                    break
        else:
            momentum = np.zeros_like(x)
            velocity = np.zeros_like(x)
            for nit in range(1, steps + 1):
                value, grad = evaluate(x)
                if value <= tolerance:
                    status = "tolerance"
                    break
                momentum = .9 * momentum + .1 * grad
                velocity = .999 * velocity + .001 * grad * grad
                x -= learning_rate * (momentum / (1 - .9 ** nit)) / (
                    np.sqrt(velocity / (1 - .999 ** nit)) + 1e-8)
            evaluate(x)  # Include the final updated iterate in candidate selection.
        if run_value < best_value:
            best_value, best = run_value, run_best
        runs.append({"start": start, "objective": run_value, "iterations": nit,
                     "evaluations": evaluations, "status": status})
    return FitResult(best, loss, best_value, perf_counter() - t0, runs)
