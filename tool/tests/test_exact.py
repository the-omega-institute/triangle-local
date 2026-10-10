from copy import deepcopy
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pytest

from triangle_local.cli import bundle, main
from triangle_local.exact import exact_distribution, numerical_model, rationalize, verify
from triangle_local.families import (note_model, note_target, parity_model, parity_target,
                                    fritz_model, fritz, ring_endpoint_model, ring_half_model, renoup)
from triangle_local.io import dump_json, load_distribution
from triangle_local.model import Model, Parameterization


@pytest.mark.parametrize("model,target", [(note_model(), note_target()),
                         (parity_model(), parity_target()), (fritz_model(), fritz()),
                         (ring_endpoint_model(), renoup("1")),
                         (ring_half_model(), renoup("1/2"))])
def test_explicit_models(model, target):
    cert = verify(rationalize(model), target)
    assert cert["exact_match"] and cert["L1"] == cert["L2_squared"] == "0"
    assert cert["normalized_exactly"] and cert["sources_independent"]


def test_note_counts_and_task_discrepancy():
    shape, p = exact_distribution(rationalize(note_model()))
    assert shape == (4,4,4)
    assert sum(p[k,k,k] for k in range(4)) == Fraction(41,144)
    assert sum(p[k,k,k] for k in range(4)) != Fraction(11,36)
    for abc, q in p.items():
        assert q == Fraction({1:123,2:19,3:23}[len(set(abc))],1728)
    assert verify(rationalize(note_model()), note_target())["model_S3_symmetric_exactly"]


@pytest.mark.parametrize("kind", ["negative", "unnormalized", "shape", "ragged", "schema", "zero-row"])
def test_reject_invalid_rational_model(kind):
    doc = rationalize(parity_model())
    if kind == "negative": doc["weights"]["alpha"] = ["-1", "2"]
    if kind == "unnormalized": doc["weights"]["beta"] = ["1/3", "1/3"]
    if kind == "shape": doc["responses"]["A"].pop()
    if kind == "ragged": doc["responses"]["A"][0][0] = ["1"]
    if kind == "schema": doc["schema"] = "unknown"
    if kind == "zero-row": doc["responses"]["B"][0][0] = ["0","0"]
    with pytest.raises(ValueError): verify(doc, parity_target())


def test_general_unequal_axes_against_direct_fraction_enumeration():
    rng = np.random.default_rng(123)
    space = Parameterization((2,3,4), (2,3,2))
    model = space.decode(rng.normal(size=space.dimension))
    doc = rationalize(model, 23)
    shape, p = exact_distribution(doc)
    w = [[Fraction(x) for x in doc["weights"][name]] for name in ("alpha","beta","gamma")]
    r = [np.asarray(doc["responses"][name],object) for name in ("A","B","C")]
    for abc in product(*(range(m) for m in shape)):
        expected = sum(w[0][i]*w[1][j]*w[2][k]*Fraction(r[0][j,k,abc[0]])*
                       Fraction(r[1][k,i,abc[1]])*Fraction(r[2][i,j,abc[2]])
                       for i,j,k in product(range(2),range(3),range(4)))
        assert p[abc] == expected
    np.testing.assert_allclose(numerical_model(doc).distribution(),
                               np.asarray(list(p.values()),float).reshape(shape), atol=1e-14)


def test_exact_distance_and_tolerance():
    model = Model((np.ones(1),)*3, (np.asarray([[[1.,0.]]]),)*3)
    target = np.full((2,)*3,Fraction(0),dtype=object)
    target[0,0,0], target[1,1,1] = Fraction(3,4), Fraction(1,4)
    cert = verify(rationalize(model), target, tolerance="1/2")
    assert cert["L1"] == "1/2" and cert["L2_squared"] == "1/8"
    assert cert["Linf"] == cert["total_variation"] == "1/4"
    assert cert["within_L1_tolerance"] and not cert["exact_match"]
    assert cert["status"] == "nearby_local_model"
    with pytest.raises(ValueError): verify(rationalize(model), target, tolerance="-1")


def test_certification_does_not_trust_symmetry_metadata():
    model = Model((np.ones(1),)*3, (np.asarray([[[1.,0.]]]),np.asarray([[[0.,1.]]]),
                                   np.asarray([[[1.,0.]]])), "exchangeable")
    doc = rationalize(model)
    target = np.full((2,)*3,Fraction(0),dtype=object);target[0,1,0]=Fraction(1)
    assert not verify(doc,target)["model_S3_symmetric_exactly"]


@pytest.mark.parametrize("document",[None,[],"bad",1])
def test_reject_nonobject_model(document):
    with pytest.raises(ValueError): verify(document,parity_target())


@pytest.mark.parametrize("mode,sizes", [("exchangeable",(3,3,3)),("outcome-equivariant",(12,12,12))])
def test_symmetry_survives_rationalization(mode,sizes):
    m = 4
    space = Parameterization(sizes,(m,)*3,mode)
    model = space.decode(np.random.default_rng(32).normal(size=space.dimension))
    doc = rationalize(model, 31)
    shape, p = exact_distribution(doc)
    target = np.asarray(list(p.values()),object).reshape(shape)
    assert verify(doc,target)["model_S3_symmetric_exactly"]


def test_json_csv_and_certificate_tamper(tmp_path):
    target = parity_target()
    path = tmp_path/"target.json"
    dump_json(path, {"outputs":[2,2,2],"probabilities":[str(x) for x in target.ravel()]})
    assert np.array_equal(load_distribution(path), target)
    csv = tmp_path/"target.csv"
    csv.write_text("a,b,c,p\n0,0,0,1/4\n0,1,1,1/4\n1,0,1,1/4\n1,1,0,1/4\n")
    assert np.array_equal(load_distribution(csv),target)
    result = tmp_path/"model.json"
    doc = bundle(parity_model(), target)
    dump_json(result,doc)
    assert main(["verify",str(result),"--input",str(path)]) == 0
    doc["certificate"]["L1"] = "1"
    dump_json(result,doc)
    with pytest.raises(SystemExit) as e: main(["verify",str(result)])
    assert e.value.code == 2
    csv.write_text("a,b,c,p\n0,0,0,1/2\n0,0,0,1/2\n")
    with pytest.raises(ValueError): load_distribution(csv)


def test_rational_input_not_silently_normalized(tmp_path):
    path=tmp_path/"p.json"
    path.write_text('{"outputs":[1,1,2],"probabilities":[0.333,0.666]}')
    with pytest.raises(ValueError): load_distribution(path)
    p=load_distribution(path,normalize=True)
    assert list(p.ravel())==[Fraction(1,3),Fraction(2,3)]


def test_solve_and_verify_explicit_normalization(tmp_path):
    path=tmp_path/"p.json"
    path.write_text('{"outputs":[1,1,2],"probabilities":[0.333,0.666]}')
    result=tmp_path/"result.json"
    assert main(["solve",str(path),"--normalize-input","--sizes","1","1","1",
                 "--starts","1","--steps","50","--out",str(result)])==0
    assert main(["verify",str(result),"--input",str(path)])==0
    doc=json.loads(result.read_text())
    assert doc["source_input"]["normalization_requested"]
    assert doc["certificate"]["exact_match"]
