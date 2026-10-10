from fractions import Fraction

import numpy as np

from triangle_local.exact import (apply_uniform_output_channel, exact_distribution,
                                  rationalize, trim_zero_weight_symbols, verify)
from triangle_local.families import detector_noise, renoup
from triangle_local.model import Model, apply_uniform_output_channel as apply_float_channel


def test_detector_channel_composes_and_preserves_locality():
    base = Model((np.full(2, .5),) * 3,
                 (np.eye(2)[np.bitwise_xor.outer(np.arange(2), np.arange(2))],) * 3)
    first = apply_float_channel(base, .8)
    second = apply_float_channel(first, .75)
    direct = apply_float_channel(base, .6)
    assert np.allclose(second.distribution(), direct.distribution(), atol=1e-14)


def test_exact_detector_channel_reproduces_target_channel():
    target = renoup("17/20")
    model = rationalize(Model((np.full(1, 1.),) * 3,
                              (np.full((1, 1, 4), .25),) * 3), max_denominator=100)
    transformed = apply_uniform_output_channel(model, "3/4")
    cert = verify(transformed, detector_noise(np.full((4, 4, 4), Fraction(1, 64), dtype=object), "3/4"))
    assert cert["exact_match"]
    shape, distribution = exact_distribution(transformed)
    assert shape == (4, 4, 4)
    assert sum(distribution.values()) == 1
    assert target.shape == (4, 4, 4)


def test_zero_weight_trim_preserves_certificate():
    import gzip
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = root / "benchmarks/models/117-renou-detector-93-100-hybrid-reverse.json.gz"
    with gzip.open(source, "rt") as stream:
        bundle = json.load(stream)
    compact = trim_zero_weight_symbols(bundle["model"])
    target = np.asarray(bundle["target"]["probabilities"], dtype=object).reshape(4, 4, 4)
    cert = verify(compact, target, tolerance=bundle["certificate"]["L1_tolerance"])
    assert [len(compact["weights"][name]) for name in ("alpha", "beta", "gamma")] == [8, 9, 2]
    assert Fraction(cert["L1"]) == Fraction(bundle["certificate"]["L1"].replace("_", ""))


def test_detector_transport_controls_approximate_error_without_claiming_equality():
    from triangle_local.families import elegant
    from triangle_local.model import Parameterization
    space = Parameterization((2,2,2),(4,4,4))
    model = space.decode(np.random.default_rng(984).normal(size=space.dimension))
    high = np.asarray(elegant("9/10", "detector"),float)
    low = np.asarray(elegant("3/5", "detector"),float)
    transported = apply_float_channel(model,2/3)
    high_error = np.linalg.norm(model.distribution()-high)
    low_error = np.linalg.norm(transported.distribution()-low)
    assert 0 < low_error <= high_error
