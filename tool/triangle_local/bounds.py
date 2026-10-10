"""Validated target-approximation bounds for the published target families.

The certificates in :mod:`triangle_local.exact` are exact for the rational
target stored in a bundle.  This module evaluates the corresponding quantum
formula with ``mpmath.iv`` outward-rounded intervals and bounds the distance
to that stored target.  The interval calculation is deliberately separate
from the optimizer and from the integer certificate checker.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from math import isqrt

import mpmath as mp
import numpy as np

from .exact import fraction
from .families import detector_noise, elegant, renoup, renoup_sources

mp.iv.dps = 100


def _iv_fraction(value):
    value = fraction(value)
    return mp.iv.mpf(value.numerator) / mp.iv.mpf(value.denominator)


def _iv_zero(shape):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = mp.iv.mpf(0)
    return result


def _iv_renou(u_squared):
    """Renou's ideal distribution, evaluated without binary floats."""
    t = _iv_fraction(u_squared)
    one = mp.iv.mpf(1)
    root = mp.iv.sqrt(t * (one - t))
    interference = 2 * t * (one - t) * root
    p = _iv_zero((4, 4, 4))
    for a, b, c in product(range(4), repeat=3):
        if a >= 2 and b >= 2 and c >= 2:
            n = int(a == 3) + int(b == 3) + int(c == 3)
            p[a, b, c] = (t ** (3 - n) * (one - t) ** n
                           + (one - t) ** (3 - n) * t ** n
                           + (-1) ** n * interference) / 8
    for abc in ((0, 1, 2), (1, 0, 3)):
        for shift in range(3):
            p[tuple(abc[(j + shift) % 3] for j in range(3))] = t / 8
    for abc in ((0, 1, 3), (1, 0, 2)):
        for shift in range(3):
            p[tuple(abc[(j + shift) % 3] for j in range(3))] = (one - t) / 8
    return p


def _iv_detector(p, visibility):
    v = _iv_fraction(visibility)
    one = mp.iv.mpf(1)
    result = _iv_zero(p.shape)
    m = p.shape[0]
    for abc in np.ndindex(p.shape):
        total = mp.iv.mpf(0)
        for xyz in np.ndindex(p.shape):
            channel = [v * int(abc[d] == xyz[d]) + (one - v) / p.shape[d]
                       for d in range(3)]
            total += p[xyz] * channel[0] * channel[1] * channel[2]
        result[abc] = total
    return result


def _iv_source(u_squared, visibility):
    """Six-qubit Born contraction for independent source noise.

    The einsum is over interval objects.  ``mpmath.iv`` supplies outward
    rounding at every operation, including the square roots in the source
    states and in the three Werner density matrices.
    """
    v = _iv_fraction(visibility)
    one = mp.iv.mpf(1)
    z = lambda n: mp.iv.mpf(n)
    vectors = np.array([[z(0), z(1), z(0), z(0)],
                        [z(0), z(0), z(1), z(0)],
                        [mp.iv.sqrt(_iv_fraction(u_squared)), z(0), z(0),
                         mp.iv.sqrt(one - _iv_fraction(u_squared))],
                        [mp.iv.sqrt(one - _iv_fraction(u_squared)), z(0), z(0),
                         -mp.iv.sqrt(_iv_fraction(u_squared))]], dtype=object)
    effects = np.einsum("ax,ay->axy", vectors, vectors).reshape(4, 2, 2, 2, 2)
    phi = np.array([z(1), z(0), z(0), z(1)], dtype=object) / mp.iv.sqrt(z(2))
    rho2 = np.empty((4, 4), dtype=object)
    for i, j in product(range(4), repeat=2):
        rho2[i, j] = v * phi[i] * phi[j] + (one - v) * z(int(i == j)) / 4
    rho = rho2.reshape(2, 2, 2, 2)
    return np.einsum("axyXY,bzuZU,cvwVW,UVuv,WXwx,YZyz->abc",
                     effects, effects, effects, rho, rho, rho, optimize="greedy")


def quantum_interval(family, parameter, *, u_squared="17/20"):
    """Return an interval tensor for a target family.

    EJM is an exact rational polynomial.  Renou's ideal and detector families
    use the algebraic formula directly; source noise uses the six-qubit Born
    contraction.  Other families are intentionally rejected so callers cannot
    silently attach a bound to the wrong physical convention.
    """
    if family == "elegant-source":
        return np.asarray([_iv_fraction(x) for x in elegant(parameter, "source").ravel()],
                          dtype=object).reshape((4, 4, 4))
    if family == "elegant-detector":
        return np.asarray([_iv_fraction(x) for x in elegant(parameter, "detector").ravel()],
                          dtype=object).reshape((4, 4, 4))
    if family == "renou-u2":
        return _iv_renou(parameter)
    if family == "renou-detector":
        return _iv_detector(_iv_renou(u_squared), parameter)
    if family == "renou-source":
        return _iv_source(u_squared, parameter)
    raise ValueError(f"No validated quantum formula for {family!r}")


def _endpoint(interval, upper):
    """Read the exact binary endpoint, without a decimal string round trip."""
    return Fraction(*mp.libmp.to_rational(interval._mpi_[int(upper)]))


def _sqrt_fraction_upper(value, digits=120):
    """Ceiling of sqrt(value) on a decimal grid, using integer arithmetic."""
    value = Fraction(value)
    if value < 0:
        raise ValueError("A squared distance must be nonnegative")
    if not value:
        return Fraction(0)
    exponent = (len(str(value.numerator)) - len(str(value.denominator))) // 2
    quantum = Fraction(10) ** (exponent - digits)
    scaled = value / (quantum * quantum)
    root = isqrt(scaled.numerator // scaled.denominator)
    if root * root * scaled.denominator < scaled.numerator:
        root += 1
    return root * quantum


def _decimal_exact(value):
    # Callers supply a decimal-grid rational. Give Decimal enough precision to
    # express the numerator and the power-of-ten denominator without rounding.
    value = Fraction(value)
    with localcontext() as ctx:
        ctx.prec = len(str(abs(value.numerator))) + len(str(value.denominator)) + 5
        return Decimal(value.numerator) / Decimal(value.denominator)


def distance_bound(interval, stored):
    """Upper-bound ||P-P_stored||_2 from exact interval endpoints.

    Endpoint subtraction and the sum of squared coordinate bounds use Fraction.
    The final square root is rounded upward by an integer ceiling. No numerical
    point check or arbitrary safety epsilon is used as an enclosure.
    """
    stored = np.asarray(stored, dtype=object)
    interval = np.asarray(interval, dtype=object)
    if stored.shape != interval.shape:
        raise ValueError("Interval and stored target shapes must agree")
    total = Fraction(0)
    for value, exact in zip(interval.ravel(), stored.ravel()):
        point = fraction(exact)
        bound = max(abs(_endpoint(value, False) - point),
                    abs(_endpoint(value, True) - point))
        total += bound * bound
    return _decimal_exact(_sqrt_fraction_upper(total))


def upward_decimal(value, places=12):
    """Round a nonnegative rational/Decimal upward to significant digits."""
    value = Fraction(value)
    if value < 0 or places < 1:
        raise ValueError("Nonnegative bound and positive precision required")
    if not value:
        return "0"
    exponent = len(str(value.numerator)) - len(str(value.denominator))
    if value < Fraction(10) ** exponent:
        exponent -= 1
    quantum = Fraction(10) ** (exponent - places + 1)
    scaled = value / quantum
    ceiling = -(-scaled.numerator // scaled.denominator)
    return format(_decimal_exact(ceiling * quantum), ".15g")


def certificate_upper(l2_squared, delta_target):
    """An upward-rounded bound on sqrt(E) + delta, with exact rounding."""
    delta = Fraction(delta_target)
    if delta < 0:
        raise ValueError("Target error bound must be nonnegative")
    return upward_decimal(_sqrt_fraction_upper(fraction(l2_squared)) + delta,
                          places=12)
