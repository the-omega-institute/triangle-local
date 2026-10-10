"""Axis convention: A[beta,gamma,a], B[gamma,alpha,b], C[alpha,beta,c]."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

import numpy as np
from scipy.special import softmax


def apply_uniform_output_channel(model, ratio):
    """Apply ``K_ratio = ratio I + (1-ratio)J/m`` to every response row.

    If ``model`` approximates an independently detector-noised target at
    visibility ``v_H``, the returned model approximates the same target at
    ``v_L = ratio*v_H``.  The source weights and their product law are kept
    unchanged.  The map is a stochastic contraction: on the uniform output
    direction its eigenvalue is one and on the orthogonal directions it is
    ``ratio``, so the three-party tensor channel cannot increase an L2 error.
    ``ratio`` may be a float or an exact rational; the numerical model API
    stores the resulting tables as floating-point arrays.
    """
    ratio = float(ratio)
    if not 0 <= ratio <= 1:
        raise ValueError("Noise-channel ratio must be in [0,1]")
    responses = []
    for table in model.responses:
        m = table.shape[-1]
        responses.append(ratio * table + (1 - ratio) / m)
    return Model(model.weights, tuple(responses), model.symmetry)


@lru_cache(maxsize=128)
def _path(expr, shapes):
    return np.einsum_path(expr, *(np.empty(s) for s in shapes), optimize="greedy")[0]


def _contract(expr, *args):
    return np.einsum(expr, *args, optimize=_path(expr, tuple(x.shape for x in args)))


@dataclass
class Model:
    weights: tuple
    responses: tuple
    symmetry: str = "none"

    def __post_init__(self):
        self.weights = tuple(np.asarray(x, dtype=float).copy() for x in self.weights)
        self.responses = tuple(np.asarray(x, dtype=float).copy() for x in self.responses)
        if len(self.weights) != 3 or len(self.responses) != 3:
            raise ValueError("Exactly three sources and responses are required")
        for w in self.weights:
            if w.ndim != 1 or not len(w):
                raise ValueError("Source alphabets must be nonempty vectors")
        i, j, k = (len(w) for w in self.weights)
        for r, shape in zip(self.responses, [(j, k), (k, i), (i, j)]):
            if r.ndim != 3 or r.shape[:2] != shape or r.shape[-1] < 1:
                raise ValueError("Response axes must be A[beta,gamma,a], B[gamma,alpha,b], C[alpha,beta,c]")
        for p in (*self.weights, *self.responses):
            if not np.all(np.isfinite(p)) or np.any(p < 0):
                raise ValueError("Probabilities must be finite and nonnegative")
            if not np.allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0):
                raise ValueError("Every source and conditional response must sum to one")

    @property
    def sizes(self):
        return tuple(len(w) for w in self.weights)

    @property
    def outputs(self):
        return tuple(r.shape[-1] for r in self.responses)

    def distribution(self):
        return _contract("i,j,k,jka,kib,ijc->abc", *self.weights, *self.responses)

    def derivatives(self, g):
        """Derivative of <g, P> with respect to unconstrained probabilities."""
        weights, responses = self.linear_maps()
        flat = np.asarray(g).ravel()
        dw = tuple(matrix.T @ flat for matrix in weights)
        dr = tuple((matrix.T @ flat).reshape(r.shape) for matrix,r in zip(responses,self.responses))
        return dw, dr

    def response_derivatives(self, g):
        _, responses = self.linear_maps(include_weights=False)
        return tuple((matrix.T @ np.asarray(g).ravel()).reshape(r.shape)
                     for matrix,r in zip(responses,self.responses))

    def linear_maps(self, include_weights=True):
        """Jacobians in probabilities; each individual factor enters linearly.

        Matrices have output triples as rows and the factor's flattened entries
        as columns. These also define convex least-squares block subproblems.
        """
        u, v, w = self.weights
        a, b, c = self.responses
        n = int(np.prod(self.outputs))
        weights = (
            _contract("j,k,jka,kib,ijc->abci", v, w, a, b, c).reshape(n, -1),
            _contract("i,k,jka,kib,ijc->abcj", u, w, a, b, c).reshape(n, -1),
            _contract("i,j,jka,kib,ijc->abck", u, v, a, b, c).reshape(n, -1),
        ) if include_weights else None
        ta = _contract("i,j,k,kib,ijc->jkbc", u, v, w, b, c)
        tb = _contract("i,j,k,jka,ijc->kiac", u, v, w, a, c)
        tc = _contract("i,j,k,jka,kib->ijab", u, v, w, a, b)
        responses = (
            np.einsum("jkbc,ad->abcjkd", ta, np.eye(self.outputs[0])).reshape(n, -1),
            np.einsum("kiac,bd->abckid", tb, np.eye(self.outputs[1])).reshape(n, -1),
            np.einsum("ijab,cd->abcijd", tc, np.eye(self.outputs[2])).reshape(n, -1),
        )
        return weights, responses

    def rounded(self):
        """Argmax rounding is a candidate only; it must be verified anew."""
        return Model(self.weights, tuple(np.eye(r.shape[-1])[r.argmax(axis=-1)]
                                        for r in self.responses), "none")


def equivariant_maps(m):
    """S_m-equivariant responses on ordered distinct pairs of outcome labels.

    With iid uniform sources and one shared response this gives joint outcome
    relabeling symmetry and cyclic party symmetry, hence full party S3 symmetry
    for the output triple.
    Entries map to logits within stabilizer-fixed output choices for each orbit.
    """
    if m < 3 or m > 5:
        raise ValueError("Outcome-equivariant mode supports 3 to 5 outcomes")
    symbols = list(permutations(range(m), 2))
    index = {x: i for i, x in enumerate(symbols)}
    todo = set(product(symbols, repeat=2))
    maps = np.full((len(symbols), len(symbols), m), -1, dtype=int)
    shapes = []
    offset = 0
    group = list(permutations(range(m)))
    act = lambda p, x: (p[x[0]], p[x[1]])
    while todo:
        x, y = min(todo)
        stabilizer = [p for p in group if act(p, x) == x and act(p, y) == y]
        fixed = [a for a in range(m) if all(p[a] == a for p in stabilizer)]
        if not fixed:
            raise ValueError("An orbit has no stabilizer-fixed output")
        for l, a in enumerate(fixed):
            for p in group:
                maps[index[act(p, x)], index[act(p, y)], p[a]] = offset + l
        todo -= {(act(p, x), act(p, y)) for p in group}
        shapes.append((len(fixed),))
        offset += len(fixed)
    return maps, shapes


class Parameterization:
    """Softmax blocks, with parameter sharing rather than output symmetrization."""
    def __init__(self, sizes=(4, 4, 4), outputs=(4, 4, 4), symmetry="none"):
        self.sizes, self.outputs, self.symmetry = tuple(sizes), tuple(outputs), symmetry
        if len(sizes) != 3 or len(outputs) != 3 or any(n < 1 for n in (*sizes, *outputs)):
            raise ValueError("Three positive source sizes and output counts are required")
        i, j, k = sizes
        a, b, c = outputs
        self.maps = None
        self.coefficients = None
        self.source_colors = None
        if symmetry == "none":
            self.shapes = [(i,), (j,), (k,), (j, k, a), (k, i, b), (i, j, c)]
        elif symmetry == "exchangeable":
            if len(set(sizes)) != 1 or len(set(outputs)) != 1:
                raise ValueError("Symmetry requires identical source sizes and output counts")
            self.shapes = [(i,), (i * (i + 1) // 2, a)]
            ids = np.empty((i, i), dtype=int)
            for n, (x, y) in enumerate(zip(*np.triu_indices(i))):
                ids[x, y] = ids[y, x] = n
            self.maps = ids
        elif symmetry == "outcome-equivariant":
            if len(set(outputs)) != 1 or tuple(sizes) != (a * (a - 1),) * 3:
                raise ValueError("Outcome-equivariant mode needs m(m-1) symbols per source")
            self.maps, self.shapes = equivariant_maps(a)
        elif symmetry == "color-equivariant":
            if len(set(outputs)) != 1 or len(set(sizes)) != 1 or a < 3 or i % a:
                raise ValueError("Color-equivariant mode needs equal sizes divisible by m >= 3 outputs")
            bins = i // a
            self.source_colors = a
            self.shapes = [(bins,), (bins, bins, 2), (bins, bins, 3)]
            self.maps = np.empty((i, i, a), dtype=int)
            self.coefficients = np.ones((i, i, a))
            for x, y, z in np.ndindex(self.maps.shape):
                color_x, bin_x = divmod(x, bins)
                color_y, bin_y = divmod(y, bins)
                if color_x == color_y:
                    choice = int(z != color_x)
                    self.maps[x, y, z] = bins + 2*(bin_x*bins+bin_y) + choice
                    self.coefficients[x, y, z] = 1 if choice == 0 else 1/(a-1)
                else:
                    choice = 0 if z == color_x else (1 if z == color_y else 2)
                    self.maps[x, y, z] = bins + 2*bins*bins + 3*(bin_x*bins+bin_y) + choice
                    self.coefficients[x, y, z] = 1 if choice < 2 else 1/(a-2)
        elif symmetry == "pair-equivariant":
            symbols = a*(a-1)
            if len(set(outputs)) != 1 or len(set(sizes)) != 1 or i % symbols:
                raise ValueError("Pair-equivariant sizes must be equal multiples of m(m-1)")
            old_maps, old_shapes = equivariant_maps(a)
            bins = i//symbols
            self.source_colors = symbols
            self.shapes = [(bins,)] + [(bins,bins,s[0]) for s in old_shapes]
            self.maps = np.full((i,i,a), -1, dtype=int)
            self.coefficients = np.ones((i,i,a))
            old_offset, offset = 0, bins
            for (choices,) in old_shapes:
                for x, y, z in np.ndindex(old_maps.shape):
                    choice = old_maps[x,y,z]-old_offset
                    if 0 <= choice < choices:
                        for bx, by in product(range(bins),repeat=2):
                            self.maps[x*bins+bx,y*bins+by,z] = offset+(bx*bins+by)*choices+choice
                old_offset += choices
                offset += bins*bins*choices
        else:
            raise ValueError("Unknown symmetry mode")
        self.lengths = [int(np.prod(s)) for s in self.shapes]
        self.cuts = np.cumsum([0] + self.lengths)
        self.dimension = int(self.cuts[-1])

    def blocks(self, x):
        if np.asarray(x).shape != (self.dimension,):
            raise ValueError("Wrong parameter vector length")
        return [np.asarray(x)[lo:hi].reshape(s) for lo, hi, s in
                zip(self.cuts[:-1], self.cuts[1:], self.shapes)]

    def decode(self, x):
        ps = [softmax(b, axis=-1) for b in self.blocks(x)]
        if self.symmetry == "none":
            return Model(tuple(ps[:3]), tuple(ps[3:]), self.symmetry)
        if self.symmetry == "exchangeable":
            r = ps[1][self.maps]
            return Model((ps[0],) * 3, (r,) * 3, self.symmetry)
        flat = np.concatenate([p.ravel() for p in ps])
        r = np.zeros(self.maps.shape)
        valid = self.maps >= 0
        r[valid] = flat[self.maps[valid]]
        if self.coefficients is not None:
            r *= self.coefficients
            w = np.tile(ps[0] / self.source_colors, self.source_colors)
        else:
            w = np.full(self.sizes[0], 1 / self.sizes[0])
        return Model((w,) * 3, (r,) * 3, self.symmetry)

    def encode(self, model):
        if model.sizes != self.sizes or model.outputs != self.outputs:
            raise ValueError("Warm-start model has incompatible alphabets")
        if self.symmetry == "none":
            ps = [*model.weights, *model.responses]
        elif self.symmetry == "exchangeable":
            if not all(np.allclose(model.weights[0], w) for w in model.weights):
                raise ValueError("Warm-start sources are not shared")
            r = model.responses[0]
            if not np.allclose(r, r.swapaxes(0, 1)) or not all(
                    np.allclose(r, q) for q in model.responses):
                raise ValueError("Warm-start responses are not exchangeable")
            ps = [model.weights[0], r[np.triu_indices(self.sizes[0])]]
        elif self.symmetry == "outcome-equivariant":
            # Encoding outside this fixed uniform-source family is rejected.
            if not all(np.allclose(w, 1 / len(w)) for w in model.weights):
                raise ValueError("Outcome-equivariant sources must be uniform")
            flat = np.zeros(self.dimension)
            r = model.responses[0]
            for n in range(self.dimension):
                flat[n] = r[self.maps == n].mean()
            ps = self.blocks(flat)
            candidate = self.decode(np.concatenate([np.log(np.maximum(p, 1e-14)).ravel() for p in ps]))
            if not all(np.allclose(p, q) for p, q in zip(candidate.responses, model.responses)):
                raise ValueError("Warm-start responses are not outcome-equivariant")
        else:
            m = self.source_colors
            bins = self.shapes[0][0]
            w = model.weights[0].reshape(m, bins)
            if not np.allclose(w, w[0]) or not all(np.allclose(model.weights[0], q) for q in model.weights):
                raise ValueError("Warm-start sources are not color-equivariant")
            flat = np.zeros(self.dimension)
            flat[:bins] = w.sum(axis=0)
            r = model.responses[0] / self.coefficients
            for n in range(bins, self.dimension):
                flat[n] = r[self.maps == n].mean()
            ps = self.blocks(flat)
            candidate = self.decode(np.concatenate([np.log(np.maximum(p, 1e-14)).ravel() for p in ps]))
            if not all(np.allclose(p, q) for p, q in zip(candidate.responses, model.responses)):
                raise ValueError("Warm-start responses are not color-equivariant")
        return np.concatenate([np.log(np.maximum(p, 1e-14)).ravel() for p in ps])

    def pullback(self, x, model, g):
        dw, dr = model.derivatives(g)
        if self.symmetry == "none":
            gs = [*dw, *dr]
            ps = [*model.weights, *model.responses]
        elif self.symmetry == "exchangeable":
            gr = np.zeros(self.shapes[1])
            np.add.at(gr, self.maps.ravel(), sum(dr).reshape(-1, self.outputs[0]))
            gs = [sum(dw), gr]
            ps = [model.weights[0], softmax(self.blocks(x)[1], axis=-1)]
        else:
            valid = self.maps >= 0
            flat = np.bincount(self.maps[valid], weights=sum(dr)[valid], minlength=self.dimension)
            if self.coefficients is not None:
                flat = np.bincount(self.maps[valid], weights=(sum(dr)*self.coefficients)[valid],
                                   minlength=self.dimension)
                bins = self.shapes[0][0]
                flat[:bins] = sum(dw).reshape(self.source_colors, bins).sum(axis=0) / self.source_colors
            gs = self.blocks(flat)
            ps = [softmax(b, axis=-1) for b in self.blocks(x)]
        return np.concatenate([(p * (d - (p * d).sum(axis=-1, keepdims=True))).ravel()
                               for p, d in zip(ps, gs)])

    def jacobian(self, x, model):
        """Full output Jacobian, including tied parameters and simplex maps."""
        dw, dr = model.linear_maps()
        n = int(np.prod(self.outputs))
        if self.symmetry == "none":
            matrices = [*dw, *dr]
        elif self.symmetry == "exchangeable":
            tied = np.zeros((n, *self.shapes[1]))
            for row in range(n):
                np.add.at(tied[row], self.maps.ravel(), sum(dr)[row].reshape(-1, self.outputs[0]))
            matrices = [sum(dw), tied.reshape(n, -1)]
        else:
            valid = self.maps >= 0
            total = sum(dr).reshape(n, *self.maps.shape)
            if self.coefficients is not None:
                total *= self.coefficients
            tied = np.asarray([np.bincount(self.maps[valid], weights=row[valid],
                                          minlength=self.dimension) for row in total])
            if self.coefficients is not None:
                bins = self.shapes[0][0]
                tied[:, :bins] = sum(dw).reshape(n, self.source_colors, bins).sum(axis=1) / self.source_colors
            matrices = [tied[:, lo:hi] for lo, hi in zip(self.cuts[:-1], self.cuts[1:])]
        result = []
        for matrix, block in zip(matrices, self.blocks(x)):
            p = softmax(block, axis=-1)
            d = matrix.reshape(n, *block.shape)
            result.append((p * (d - (d * p).sum(axis=-1, keepdims=True))).reshape(n, -1))
        return np.concatenate(result, axis=1)


def lift_model(model, sizes, *, jitter=0., seed=0):
    """Split each source's symbols independently, preserving its distribution.

    Local tables pull back along each edge map. There is no global mixture
    flag. Optional response jitter opens a basin for further optimization.
    """
    if len(sizes) != 3 or any(n < old for n, old in zip(sizes, model.sizes)):
        raise ValueError("Lifting requires three source sizes at least as large as the originals")
    if not 0 <= jitter <= 1:
        raise ValueError("Jitter must be in [0,1]")
    maps = [np.arange(n) % old for n, old in zip(sizes, model.sizes)]
    weights = [w[ids] / np.bincount(ids, minlength=len(w))[ids] for w, ids in zip(model.weights, maps)]
    edges = [(maps[1], maps[2]), (maps[2], maps[0]), (maps[0], maps[1])]
    rng = np.random.default_rng(seed)
    responses = []
    for r, (left, right) in zip(model.responses, edges):
        table = r[left[:, None], right[None, :]].copy()
        if jitter:
            random = rng.dirichlet(np.ones(r.shape[-1]), size=table.shape[:2])
            table = (1-jitter)*table + jitter*random
        responses.append(table)
    return Model(tuple(weights), tuple(responses))
