"""Published noise families and explicit local controls. See LITERATURE.md."""
from fractions import Fraction
from itertools import permutations, product
from math import isqrt, sqrt

import numpy as np

from .exact import fraction
from .model import Model


def _unit(x):
    x = fraction(x)
    if not 0 <= x <= 1:
        raise ValueError("Parameter must be in [0,1]")
    return x


def pattern_distribution(values, m=4):
    p = np.empty((m,) * 3, dtype=object)
    for abc in np.ndindex(p.shape):
        p[abc] = fraction(values[len(set(abc)) - 1])
    if sum(p.ravel()) != 1:
        raise ValueError("Pattern distribution is not normalized")
    return p


def note_target():
    return pattern_distribution([Fraction(123, 1728), Fraction(19, 1728), Fraction(23, 1728)])


def note_model():
    symbols = list(permutations(range(4), 2))
    r = np.zeros((12, 12, 4))
    for i, x in enumerate(symbols):
        for j, y in enumerate(symbols):
            a = x[1] if x[1] in y else x[0]
            r[i, j, a] = 1
    return Model((np.full(12, 1 / 12),) * 3, (r,) * 3, "outcome-equivariant")


def elegant(parameter="1", noise="source"):
    v = _unit(parameter)
    if noise == "source":
        nums = [4 + 9*v + 9*v*v + 3*v**3,
                4 + v - 3*v*v - v**3, 4 - 3*v + 3*v*v + v**3]
    elif noise == "detector":
        nums = [4 + 9*v*v + 12*v**3, 4 + v*v - 4*v**3, 4 - 3*v*v + 4*v**3]
    elif noise == "global":
        nums = [4 + 21*v, 4 - 3*v, 4 + v]
    else:
        raise ValueError("Noise must be source, detector, or global")
    return pattern_distribution([n / 256 for n in nums])


def fritz(correlation="1/2"):
    """P(2a+x,2b+y,2x+y)=[1-(-1)^(a+b+xy)t]/16.

    t is the CHSH correlator magnitude. The 2020 source visibility is v=sqrt(2)t.
    Rational t yields a rational target; t <= 1/2 admits the explicit model below.
    """
    t = _unit(correlation)
    p = np.full((4, 4, 4), Fraction(0), dtype=object)
    for a, b, x, y in product(range(2), repeat=4):
        p[2*a+x, 2*b+y, 2*x+y] = (1 - (-1)**(a+b+x*y)*t) / 16
    return p


def fritz_model(correlation="1/2"):
    t = _unit(correlation)
    if t > Fraction(1, 2):
        raise ValueError("The explicit Fritz construction applies only to t <= 1/2")
    strategies = [bits for bits in product(range(2), repeat=4)
                  if sum(-(-1)**(bits[x] + bits[2+y] + x*y)
                         for x, y in product(range(2), repeat=2)) == 2]
    assert len(strategies) == 8
    a = np.zeros((2, 8, 4)); b = np.zeros((8, 2, 4)); c = np.zeros((2, 2, 4))
    for x, n in product(range(2), range(8)):
        # Local randomization of Alice's result scales t without modifying settings.
        bit = strategies[n][x]
        a[x, n, 2*bit+x] = float(Fraction(1, 2) + t)
        a[x, n, 2*(1-bit)+x] = float(Fraction(1, 2) - t)
    for n, y in product(range(8), range(2)):
        b[n, y, 2*strategies[n][2+y]+y] = 1
    for y, x in product(range(2), repeat=2):
        c[y, x, 2*x+y] = 1
    return Model((np.full(2, .5), np.full(2, .5), np.full(8, .125)), (a, b, c))


def parity_model():
    r = np.eye(2)[np.bitwise_xor.outer(np.arange(2), np.arange(2))]
    return Model((np.full(2, .5),) * 3, (r,) * 3, "exchangeable")


def parity_target():
    p = np.empty((2,) * 3, dtype=object)
    for a, b, c in np.ndindex(p.shape):
        p[a, b, c] = Fraction(1, 4) if a ^ b ^ c == 0 else Fraction(0)
    return p


def _sqrt_rational(x):
    n, d = isqrt(x.numerator), isqrt(x.denominator)
    return Fraction(n, d) if n*n == x.numerator and d*d == x.denominator else None


def _quantize_tensor(p, denominator=10**12):
    """Explicit rational approximation when the quantum target is irrational."""
    from .exact import _rational_row
    return np.asarray([fraction(x) for x in _rational_row(np.asarray(p, float).ravel(), denominator)],
                      dtype=object).reshape(p.shape)


def renoup(u_squared="17/20"):
    """Renou's four-outcome ring family (2020 eigenstate ordering 01,10,chi0,chi1).

    Exact for rational u² with rational sqrt(u²(1-u²)); otherwise an explicitly
    quantized rational approximation with denominator 10^12.
    """
    t = _unit(u_squared)
    root = _sqrt_rational(t*(1-t))
    interference = 2*t*(1-t)*(root if root is not None else sqrt(float(t*(1-t))))
    p = np.full((4,)*3, Fraction(0), dtype=object)
    for a, b, c in product(range(4), repeat=3):
        if a >= 2 and b >= 2 and c >= 2:
            n = (a == 3) + (b == 3) + (c == 3)
            p[a, b, c] = (t**(3-n)*(1-t)**n + (1-t)**(3-n)*t**n
                          + (-1)**n*interference) / 8
    for abc in [(0,1,2), (1,0,3)]:
        for shift in range(3):
            p[tuple(abc[(j+shift)%3] for j in range(3))] = t / 8
    for abc in [(0,1,3), (1,0,2)]:
        for shift in range(3):
            p[tuple(abc[(j+shift)%3] for j in range(3))] = (1-t) / 8
    return p if root is not None else _quantize_tensor(p)


def detector_noise(p, parameter):
    v = _unit(parameter)
    p = np.asarray(p, dtype=object)
    channels = [np.asarray([[v*(a == x)+(1-v)/m for x in range(m)] for a in range(m)],
                           dtype=object) for m in p.shape]
    out = np.full(p.shape, Fraction(0), dtype=object)
    for abc in np.ndindex(p.shape):
        out[abc] = sum(p[xyz]*channels[0][abc[0],xyz[0]]*channels[1][abc[1],xyz[1]]*
                       channels[2][abc[2],xyz[2]] for xyz in np.ndindex(p.shape))
    return out


def renoup_sources(u_squared="17/20", visibility="1"):
    """6-qubit Born-rule contraction; finite decimal approximation, not exact quantum arithmetic."""
    t, v = float(_unit(u_squared)), float(_unit(visibility))
    if v == 1:
        return renoup(u_squared)
    if v == 0:
        return np.full((4,)*3, Fraction(1,64), dtype=object)
    u, s = sqrt(t), sqrt(1-t)
    vectors = np.asarray([[0,1,0,0], [0,0,1,0], [u,0,0,s], [s,0,0,-u]])
    effects = np.einsum("ax,ay->axy", vectors, vectors).reshape(4,2,2,2,2)
    phi = np.asarray([1,0,0,1]) / sqrt(2)
    rho = (v*np.outer(phi,phi)+(1-v)*np.eye(4)/4).reshape(2,2,2,2)
    p = np.einsum("axyXY,bzuZU,cvwVW,UVuv,WXwx,YZyz->abc",
                  effects, effects, effects, rho, rho, rho, optimize="greedy")
    if np.min(p) < -1e-12 or abs(p.sum()-1) > 1e-12:
        raise ValueError("Quantum distribution failed numerical sanity check")
    return _quantize_tensor(np.maximum(p,0))


def ring_endpoint_model():
    # u²=1: measurement labels 01->0,10->1,00->2,11->3.
    table = np.asarray([[2,0],[1,3]])
    r = np.eye(4)[table]
    return Model((np.full(2,.5),)*3, (r,)*3)


def ring_half_model():
    """Exact u²=1/2: independent token bits and independent edge parity bits."""
    symbols = list(product(range(2),repeat=2))
    r = np.zeros((4,4,4))
    for i,(x,s) in enumerate(symbols):
        for j,(y,t) in enumerate(symbols):
            a = (0 if x == 0 else 1) if x != y else 2+(s ^ t)
            r[i,j,a] = 1
    return Model((np.full(4,.25),)*3,(r,)*3)


FAMILIES = ("note", "elegant-source", "elegant-detector", "elegant-global",
            "fritz", "parity", "renou", "renou-source", "renou-detector")


def family(name, parameter="1", u_squared="17/20"):
    if name == "note": return note_target()
    if name == "parity": return parity_target()
    if name == "fritz": return fritz(parameter)
    if name.startswith("elegant-"): return elegant(parameter, name.split("-")[1])
    if name == "renou": return renoup(parameter)
    if name == "renou-source": return renoup_sources(u_squared, parameter)
    if name == "renou-detector": return detector_noise(renoup(u_squared), parameter)
    raise ValueError("Unknown family")


def recognized_seed(target, sizes):
    """Recognize checked controls before a hybrid search; certify them normally.

    Matching is numerical here. The rational verifier, not this recognizer,
    decides whether the returned candidate matches the actual stored target.
    """
    from .model import lift_model
    p=np.asarray(target,float)
    candidates=[]
    marginals=[p.sum(axis=axes) for axes in [(1,2),(0,2),(0,1)]]
    product_model=Model((np.ones(1),)*3,tuple(q.reshape(1,1,-1) for q in marginals))
    candidates.append(('product',product_model))
    if p.shape==(2,)*3:candidates.append(('parity',parity_model()))
    if p.shape==(4,)*3:
        candidates.extend([('note',note_model()),('ring-half',ring_half_model()),('ring-endpoint',ring_endpoint_model())])
        t=1-16*p[0,0,0]
        if 0<=t<=.5+1e-14:
            candidates.append(('fritz',fritz_model(str(min(t,.5)))))
    for name,model in candidates:
        if all(n>=old for n,old in zip(sizes,model.sizes)) and np.allclose(model.distribution(),p,atol=1e-14,rtol=0):
            return name,lift_model(model,sizes)
    return None,None
