"""Soundness checks for the endpoint-to-display part of certification."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

mp = pytest.importorskip('mpmath')
from triangle_local.bounds import (certificate_upper, distance_bound,
                                   _sqrt_fraction_upper, upward_decimal)


def test_square_root_upper_is_a_mathematical_bound_at_extreme_scales():
    for squared in [Fraction(0), Fraction(4,9), Fraction(2),
                    Fraction(1,10**400), Fraction(2,10**400), Fraction(10**400,3)]:
        upper = _sqrt_fraction_upper(squared)
        assert upper >= 0 and upper * upper >= squared
        if squared:
            assert (upper-Fraction(1,10**110)*upper)**2 < squared


def test_interval_distance_contains_every_corner():
    intervals = np.array([mp.iv.mpf(['0.1','0.2']), mp.iv.mpf(['-0.3','0.4'])],dtype=object)
    stored = np.array([Fraction(1,3),Fraction(1,7)],dtype=object)
    upper = Fraction(distance_bound(intervals,stored))
    for a in [Fraction(1,10),Fraction(1,5)]:
        for b in [Fraction(-3,10),Fraction(2,5)]:
            assert upper**2 >= (a-stored[0])**2+(b-stored[1])**2
    exact = np.array([mp.iv.mpf(0),mp.iv.mpf(1)],dtype=object)
    assert distance_bound(exact,np.array([0,1],dtype=object)) == 0


def test_certificate_display_rounds_up_even_next_to_decimal_boundary():
    for e, delta in [(Fraction(1,9),Decimal('1e-20')),
                     (Fraction(2),Decimal(0)),
                     (Fraction(1,10**300),Decimal('1e-160')),
                     (Fraction(0),Decimal('1e-200'))]:
        u = Fraction(certificate_upper(e,delta))
        assert u >= Fraction(delta)
        assert (u-Fraction(delta))**2 >= e
    assert certificate_upper(0,0) == '0'
    assert Fraction(upward_decimal(Fraction(1000000000001,10**12),12)) >= Fraction(1000000000001,10**12)


def test_mismatched_target_shape_rejected():
    with pytest.raises(ValueError,match='shapes'):
        distance_bound(np.array([mp.iv.mpf(1)],dtype=object),np.array([1,0]))
