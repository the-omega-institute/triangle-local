"""Exact rational checking with integer contractions; does not trust search flags."""
from fractions import Fraction
from hashlib import sha256
from itertools import permutations, product
import json
from math import isfinite, lcm, prod

import numpy as np

from .model import Model, Parameterization
from .serialization import canonical_document

SCHEMA = "triangle-local-model/v1"


def fraction(x):
    # Explicitly ungroup rational strings for Python 3.10 compatibility.
    x = canonical_document(x)
    if isinstance(x, (float, np.floating)):
        if not isfinite(float(x)):
            raise ValueError("Nonfinite probability")
        return Fraction(str(float(x)))
    if isinstance(x, bool):
        raise ValueError("Boolean is not a probability")
    try:
        return Fraction(x)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid rational probability: {x!r}") from exc


def _simplex(row):
    vals = [fraction(x) for x in row]
    if not vals or any(x < 0 for x in vals) or sum(vals) != 1:
        raise ValueError("Every rational simplex must be nonnegative and sum exactly to one")
    return vals


def _rational_row(row, denominator):
    ps = np.asarray(row, dtype=float)
    candidates = [Fraction(float(x)).limit_denominator(denominator) for x in ps]
    if sum(candidates) == 1:
        return [str(x) for x in candidates]
    # Largest-remainder quantization gives exact sum one and denominator <= D.
    scaled = ps / ps.sum() * denominator
    counts = np.floor(scaled).astype(int)
    left = denominator - int(counts.sum())
    order = np.argsort(-(scaled - counts), kind="stable")
    counts[order[:left]] += 1
    return [str(Fraction(int(x), denominator)) for x in counts]


def rationalize(model, max_denominator=10000):
    if max_denominator < 1:
        raise ValueError("Denominator must be a positive integer")
    if model.symmetry in ("color-equivariant", "pair-equivariant"):
        from scipy.special import softmax
        space = Parameterization(model.sizes, model.outputs, model.symmetry)
        blocks = space.blocks(space.encode(model))
        rational_blocks = []
        for block in blocks:
            probs = softmax(block, axis=-1)
            rational_blocks.append(np.asarray([_rational_row(row, max_denominator)
                for row in probs.reshape(-1,probs.shape[-1])],object).reshape(probs.shape))
        # Quantize orbit masses, then split each mass uniformly within its
        # stabilizer orbit. This keeps symmetry exact at rational precision.
        flat = np.concatenate([p.ravel() for p in rational_blocks])
        weights_row = [str(fraction(p)/space.source_colors) for p in rational_blocks[0]]*space.source_colors
        table = np.full(space.maps.shape, "0", dtype=object)
        for x,y,z in np.ndindex(table.shape):
            index = space.maps[x,y,z]
            if index >= 0:
                multiplicity = int(round(1/space.coefficients[x,y,z]))
                table[x,y,z] = str(fraction(flat[index])/multiplicity)
        return {"schema":SCHEMA,"symmetry_parameterization":model.symmetry,
                "weights":{name:list(weights_row) for name in ("alpha","beta","gamma")},
                "responses":{name:table.tolist() for name in ("A","B","C")}}
    weights = {name: _rational_row(w, max_denominator) for name, w in
               zip(("alpha", "beta", "gamma"), model.weights)}
    responses = {name: [[_rational_row(row, max_denominator) for row in slab] for slab in r]
                 for name, r in zip(("A", "B", "C"), model.responses)}
    return {"schema": SCHEMA, "symmetry_parameterization": model.symmetry,
            "weights": weights, "responses": responses}


def apply_uniform_output_channel(model, ratio):
    """Apply the exact detector-noise map to every rational response row.

    For an output alphabet of size ``m``, ``K_ratio(x) = ratio*x +
    (1-ratio)/m``.  Source weights remain independent and unchanged.  This
    maps a local model Q_H to a local model Q_L. For the independent
    detector-noise target family, ||Q_L-P(v_L)||_2 <= ||Q_H-P(v_H)||_2
    when v_L = ratio*v_H. Exact target equality propagates only if the
    starting model equals its intended target exactly.
    """
    ratio = fraction(ratio)
    if not 0 <= ratio <= 1:
        raise ValueError("Noise-channel ratio must be in [0,1]")
    ws, rs = _parse(model)
    out = {"schema": SCHEMA,
           "symmetry_parameterization": model.get("symmetry_parameterization", "none"),
           "weights": {name: [str(x) for x in row] for name, row in
                       zip(("alpha", "beta", "gamma"), ws)},
           "responses": {}}
    for name, table in zip(("A", "B", "C"), rs):
        m = len(table[0][0])
        out["responses"][name] = [
            [[str(ratio * x + (1 - ratio) / m) for x in row]
             for row in slab]
            for slab in table]
    return out


def trim_zero_weight_symbols(model):
    """Remove zero-weight source symbols and return a compact rational model.

    Symbols are ordered by decreasing source weight, with their original index
    as a stable tie-breaker.  The three response tables are permuted along
    their incident source axes, so the resulting contraction is exactly the
    same distribution.  The returned metadata records the original indices.
    """
    ws, rs = _parse(model)
    active = [sorted((i for i, value in enumerate(w) if value),
                     key=lambda i: (-w[i], i)) for w in ws]
    if any(not indices for indices in active):
        raise ValueError("Cannot trim an empty source support")
    new = {"schema": SCHEMA,
           "symmetry_parameterization": model.get("symmetry_parameterization", "none"),
           "weights": {}, "responses": {},
           "trimmed_from_sizes": [len(w) for w in ws],
           "trimmed_permutations": active}
    for name, w, indices in zip(("alpha", "beta", "gamma"), ws, active):
        new["weights"][name] = [str(w[i]) for i in indices]
    # A[beta,gamma], B[gamma,alpha], C[alpha,beta].
    axes = {"A": (1, 2), "B": (2, 0), "C": (0, 1)}
    for name, table in zip(("A", "B", "C"), rs):
        left, right = axes[name]
        new["responses"][name] = [
            [[str(value) for value in table[i][j]]
             for j in active[right]]
            for i in active[left]
        ]
    return new


def _parse(model):
    if not isinstance(model,dict):
        raise ValueError("Rational model must be a JSON object")
    if model.get("schema") != SCHEMA:
        raise ValueError("Unsupported rational-model schema")
    try:
        if any(not isinstance(model["weights"][name], list) for name in ("alpha","beta","gamma")):
            raise ValueError("Rational source weights must be JSON arrays")
        ws = [_simplex(model["weights"][k]) for k in ("alpha", "beta", "gamma")]
        rs = []
        i, j, k = map(len, ws)
        for name, (x, y) in zip(("A", "B", "C"), ((j, k), (k, i), (i, j))):
            r = model["responses"][name]
            if (not isinstance(r,list) or len(r) != x or
                    any(not isinstance(s,list) or len(s) != y for s in r) or
                    any(not isinstance(row,list) for slab in r for row in slab)):
                raise ValueError("Rational response has incompatible hidden-variable axes")
            parsed = [[_simplex(row) for row in slab] for slab in r]
            m = len(parsed[0][0])
            if any(len(row) != m for slab in parsed for row in slab):
                raise ValueError("Ragged output alphabet")
            rs.append(parsed)
        return ws, rs
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed rational model") from exc


def numerical_model(document):
    ws, rs = _parse(document)
    return Model(tuple(np.asarray(w, float) for w in ws),
                 tuple(np.asarray(r, float) for r in rs),
                 document.get("symmetry_parameterization", "none"))


def canonical_hash(document):
    return sha256(json.dumps(canonical_document(document), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()).hexdigest()


def exact_distribution(model):
    ws, rs = _parse(model)
    shape = tuple(len(r[0][0]) for r in rs)
    # Lift each factor to a common integer denominator once. The entire finite
    # contraction then uses Python integers; reduce only its output entries.
    wd = [lcm(*(v.denominator for v in row)) for row in ws]
    rd = [lcm(*(v.denominator for slab in r for row in slab for v in row)) for r in rs]
    iw = [[v.numerator*(d//v.denominator) for v in row] for row,d in zip(ws,wd)]
    ir = [[[[v.numerator*(d//v.denominator) for v in row] for row in slab] for slab in r]
          for r,d in zip(rs,rd)]
    denominator = prod(wd)*prod(rd)
    counts = {abc: 0 for abc in product(*(range(n) for n in shape))}
    # The source-triple weights are PRODUCTS, never an arbitrary joint latent law.
    for i, j, k in product(*(range(len(w)) for w in ws)):
        weight = iw[0][i] * iw[1][j] * iw[2][k]
        if not weight:
            continue
        rows = (ir[0][j][k], ir[1][k][i], ir[2][i][j])
        supports = [[(a, x) for a, x in enumerate(row) if x] for row in rows]
        for (a, pa), (b, pb), (c, pc) in product(*supports):
            counts[a, b, c] += weight * pa * pb * pc
    p = {abc: Fraction(n,denominator) for abc,n in counts.items()}
    if sum(p.values()) != 1:
        raise ValueError("Exact model distribution did not normalize")
    return shape, p


def verify(model, target, *, tolerance="1/1000000"):
    """Return a reproducible certificate for the rational candidate versus target.

    target is an object array (or nested tensor) of exact rationals. Decimal input
    is interpreted literally; no normalization occurs here. All distances below
    are exact. A small distance certifies a nearby local model, not target locality.
    """
    shape, p = exact_distribution(model)
    t = np.asarray(target, dtype=object)
    if t.shape != shape:
        raise ValueError("Target and model output alphabets differ")
    vals = _simplex(t.ravel())
    eps = fraction(tolerance)
    if eps < 0:
        raise ValueError("Tolerance must be nonnegative")
    residuals = [p[abc] - v for abc, v in zip(p, vals)]
    l1 = sum(abs(x) for x in residuals)
    l2 = sum(x * x for x in residuals)
    linf = max(abs(x) for x in residuals)
    symmetric = (len(set(shape)) == 1 and all(
        p[abc] == p[tuple(abc[i] for i in perm)] for abc in p for perm in permutations(range(3))))
    target_doc = {"outputs": list(shape), "probabilities": [str(v) for v in vals]}
    return {"schema": "triangle-local-certificate/v1", "model_sha256": canonical_hash(model),
            "target_sha256": canonical_hash(target_doc), "outputs": list(shape),
            "sources_independent": True, "normalized_exactly": True,
            "model_S3_symmetric_exactly": symmetric,
            "exact_match": l1 == 0, "within_L1_tolerance": l1 <= eps,
            "L1_tolerance": str(eps), "L1": str(l1), "L2_squared": str(l2),
            "Linf": str(linf), "total_variation": str(l1 / 2),
            "status": "exact_match" if l1 == 0 else ("nearby_local_model" if l1 <= eps else "approximate_fit"),
            "model_probabilities": [str(x) for x in p.values()]}
