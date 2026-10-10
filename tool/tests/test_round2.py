import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from triangle_local.convex import project_simplex, simplex_least_squares, alternating_convex
from triangle_local.exact import rationalize, verify
from triangle_local.families import parity_target, ring_half_model, renoup
from triangle_local.model import Parameterization, lift_model
from triangle_local.optimize import fit


@pytest.mark.parametrize('mode,sizes,outputs', [
    ('none',(2,3,2),(2,3,2)), ('exchangeable',(3,)*3,(2,)*3),
    ('outcome-equivariant',(6,)*3,(3,)*3), ('pair-equivariant',(12,)*3,(3,)*3),
    ('color-equivariant',(6,)*3,(3,)*3)])
def test_full_output_jacobian(mode,sizes,outputs):
    space=Parameterization(sizes,outputs,mode)
    x=np.random.default_rng(17).normal(size=space.dimension)
    actual=space.jacobian(x,space.decode(x))
    expected=approx_derivative(lambda z:space.decode(z).distribution().ravel(),x)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-5)
    g=np.random.default_rng(18).normal(size=outputs)
    np.testing.assert_allclose(space.pullback(x,space.decode(x),g),actual.T@g.ravel(),atol=1e-13)


@pytest.mark.parametrize('mode,n',[('color-equivariant',8),('pair-equivariant',24)])
def test_new_symmetries_exact_after_rationalization(mode,n):
    space=Parameterization((n,)*3,(4,)*3,mode)
    model=space.decode(np.random.default_rng(19).normal(size=space.dimension))
    recovered=space.decode(space.encode(model))
    np.testing.assert_allclose(model.distribution(),recovered.distribution(),atol=1e-14)
    target=np.full((4,)*3,'1/64',object)
    assert verify(rationalize(model,31),target)['model_S3_symmetric_exactly']


def test_lifting_preserves_distribution_with_unequal_axes():
    space=Parameterization((2,3,2),(2,3,2))
    model=space.decode(np.random.default_rng(20).normal(size=space.dimension))
    lifted=lift_model(model,(5,7,4))
    np.testing.assert_allclose(lifted.distribution(),model.distribution(),atol=1e-14)
    with pytest.raises(ValueError):lift_model(model,(1,3,2))


def test_simplex_projection_kkt_and_degenerate_design():
    x=np.array([[3.,1.,-2.],[.2,.2,.2]])
    p=project_simplex(x)
    np.testing.assert_allclose(p,[[1,0,0],[1/3]*3])
    initial=np.array([.2,.8])
    np.testing.assert_array_equal(simplex_least_squares(np.zeros((2,2)),np.ones(2),initial),initial)
    d=np.eye(2)
    np.testing.assert_allclose(simplex_least_squares(d,np.array([.9,.1]),initial),[.9,.1])


@pytest.mark.parametrize('loss',['l1','l2'])
def test_convex_blocks_decrease_actual_loss(loss):
    space=Parameterization((2,3,2),(2,)*3)
    model=space.decode(np.random.default_rng(21).normal(size=space.dimension))
    target=np.asarray(parity_target(),float)
    result,_,_,trace=alternating_convex(model,target,8,1e-20,loss,80)
    assert np.all(np.diff(trace)<=1e-13)
    assert np.linalg.norm(result.distribution()-target)<np.linalg.norm(model.distribution()-target)


def test_deterministic_search_and_exact_endpoint():
    result=fit(np.asarray(renoup('1/2'),float),(4,)*3,method='deterministic',initial=ring_half_model(),
               starts=1,steps=10)
    assert result.objective==0
    assert verify(rationalize(result.model),renoup('1/2'))['exact_match']
    for table in result.model.responses:assert np.all((table==0)|(table==1))


def test_least_squares_recovers_small_product():
    target=np.einsum('a,b,c->abc',[.2,.8],[.3,.7],[.4,.6])
    result=fit(target,(1,)*3,method='least-squares',starts=1,steps=80,seed=4)
    assert np.sqrt(result.objective)<1e-8


def test_hybrid_preserves_exact_seed_and_checks_shapes():
    result=fit(np.asarray(renoup('1/2'),float),(4,)*3,method='hybrid',initial=ring_half_model(),starts=2,steps=30)
    assert result.objective==0
    with pytest.raises(ValueError):
        fit(np.asarray(renoup('1/2'),float),(5,)*3,method='hybrid',initial=ring_half_model())


def test_compressed_cli_certificate_and_sparse_unused_labels(tmp_path):
    from triangle_local.cli import bundle,main
    from triangle_local.io import dump_json
    from triangle_local.model import Model
    target=np.full((2,)*3,'0',object);target[0,0,0]='1'
    model=Model((np.ones(1),)*3,(np.array([[[1.,0.]]]),)*3)
    path=tmp_path/'model.json.gz';dump_json(path,bundle(model,target))
    csv=tmp_path/'target.csv';csv.write_text('a,b,c,p\n0,0,0,1\n')
    assert main(['verify',str(path),'--input',str(csv)])==0


def test_default_hybrid_recognizes_local_half_and_not_perturbed_target():
    from triangle_local.families import recognized_seed
    target=np.asarray(renoup('1/2'),float)
    result=fit(target,(12,)*3,method='hybrid',starts=1,steps=30)
    assert result.runs[0]['recognized_control']=='ring-half'
    assert verify(rationalize(result.model),renoup('1/2'))['exact_match']
    changed=target.copy();changed[2,2,2]-=.001;changed[2,2,3]+=.001
    assert recognized_seed(changed,(12,)*3)==(None,None)
