from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from triangle_local.families import (detector_noise, elegant, fritz, fritz_model,
                                    parity_target, renoup, renoup_sources)
from triangle_local.model import Parameterization
from triangle_local.optimize import fit, loss_gradient


@pytest.mark.parametrize("mode,sizes,outputs", [("none",(2,3,2),(2,3,2)),
                       ("exchangeable",(2,2,2),(2,2,2)),
                       ("outcome-equivariant",(6,6,6),(3,3,3))])
@pytest.mark.parametrize("loss", ["l1","l2","kl"])
def test_analytic_gradient(mode,sizes,outputs,loss):
    rng=np.random.default_rng(49)
    space=Parameterization(sizes,outputs,mode)
    x=rng.normal(size=space.dimension)
    t=rng.uniform(size=outputs);t/=t.sum()
    def objective(z):return loss_gradient(space.decode(z).distribution(),t,loss)[0]
    model=space.decode(x)
    _,g=loss_gradient(model.distribution(),t,loss)
    actual=space.pullback(x,model,g)
    expected=approx_derivative(objective,x,method="3-point").ravel()
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-9)


def test_em_improves_kl():
    target=np.asarray(parity_target(),float)
    space=Parameterization((2,)*3,(2,)*3)
    initial=space.decode(np.random.default_rng(88).normal(size=space.dimension))
    before=loss_gradient(initial.distribution(),target,"kl")[0]
    result=fit(target,(2,)*3,method="em",loss="kl",starts=1,steps=80,initial=initial)
    assert result.objective < before*.1


@pytest.mark.parametrize("method",["adam","lbfgs","alternating"])
def test_recover_biased_product(method):
    probs=([.2,.8],[.3,.7],[.4,.6])
    target=np.einsum("a,b,c->abc",*probs)
    result=fit(target,(1,)*3,method=method,starts=1,steps=250,seed=2)
    assert np.linalg.norm(result.model.distribution()-target)<1e-5


def test_noise_models_and_fritz_convention():
    for v in ("0","1/2","4/5","1"):
        for noise in ("source","detector","global"):
            p=elegant(v,noise)
            assert sum(p.ravel())==1 and min(p.ravel())>=0
        assert np.array_equal(elegant(v,"detector"),detector_noise(elegant(),v))
    assert elegant("1")[0,0,0]==Fraction(25,256)
    for t in ("0","1/4","1/2"):
        np.testing.assert_allclose(fritz_model(t).distribution(),np.asarray(fritz(t),float),atol=1e-14)


def test_ring_quantum_and_endpoint_normalization():
    for t in ("1/2","4/5","17/20","1"):
        p=renoup(t)
        assert sum(p.ravel())==1 and min(p.ravel())>=0
        assert np.count_nonzero(p)<=20
        q=renoup_sources(t,"9/10")
        assert sum(q.ravel())==1 and min(q.ravel())>=0
    # A direct pure-state amplitude contraction independently checks ordering.
    u=np.sqrt(.8);s=np.sqrt(.2)
    basis=np.array([[0,1,0,0],[0,0,1,0],[u,0,0,s],[s,0,0,-u]]).reshape(4,2,2)
    amplitudes=np.einsum("ajk,bki,cij->abc",basis,basis,basis)/np.sqrt(8)
    np.testing.assert_allclose(amplitudes**2,np.asarray(renoup("4/5"),float),atol=1e-14)
