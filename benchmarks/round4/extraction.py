"""Reload a preselected native LHV-Net checkpoint, extract and certify a finite model.

Run after the comparison. This demonstration and its random controls are outside
all timed search scores. No selection based on held-out evaluation is performed.
"""
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from run_comparison import build_neural, digest, Recorder, Deadline
sys.path.insert(0,str(ROOT/'tool'))
import numpy as np
from triangle_local.bounds import certificate_upper, distance_bound, quantum_interval
from triangle_local.exact import rationalize, verify, numerical_model, fraction
from triangle_local.families import family
from triangle_local.io import distribution_document, dump_json
from triangle_local.model import Model, Parameterization
from triangle_local.optimize import fit


def timed_polish(initial,p,seconds):
    original=Model.distribution
    rec=Recorder(p,(seconds,))
    def tracked(model):
        rec.tick();q=original(model);rec.offer(q,model);return q
    Model.distribution=tracked
    try:
        current=initial
        current.distribution()
        while True:
            current=fit(p,(4,4,4),method='convex',starts=1,steps=120,initial=current,
                        inner_steps=120,tolerance=0,optimize_weights=True).model
    except Deadline:
        pass
    finally:
        Model.distribution=original
    return rec.snapshots[seconds]['payload'],rec.elapsed()


def certify(model,p,delta,name,denominator):
    t0=time.perf_counter();rat=rationalize(model,max_denominator=denominator)
    rtime=time.perf_counter()-t0
    t0=time.perf_counter();cert=verify(rat,p);ctime=time.perf_counter()-t0
    u=certificate_upper(cert['L2_squared'],delta)
    dest=HERE/'interoperability'/(name+'.json.gz')
    dump_json(dest,{'schema':'triangle-local-bundle/v1','target':distribution_document(p),'model':rat,
                   'certificate':cert,'bound':{'delta_target_upper':str(delta),'U':u}})
    return numerical_model(rat),{'U':u,'stored_target_residual':float(fraction(cert['L2_squared']))**.5,
            'rationalization_seconds':rtime,'verification_seconds':ctime,'bundle':dest.name,
            'bundle_bytes':dest.stat().st_size,'bundle_sha256':digest(dest),'denominator_cap':denominator}


def main():
    protocol=json.loads((HERE/'interoperability-protocol.json').read_text())
    stem='renou-detector-093__LHV-Net-2020__00'
    run=json.loads((HERE/'runs'/(stem+'.json')).read_text())
    cp=run['checkpoints']['60'];checkpoint=HERE/'runs'/(stem+'__60s.weights.h5')
    assert digest(checkpoint)==cp['checkpoint_sha256']
    builder=build_neural(run['seed']);model=builder();model.load_weights(checkpoint)
    p=family('renou-detector','93/100');pf=np.asarray(p,float)
    delta=distance_bound(quantum_interval('renou-detector','93/100'),p)
    start=time.perf_counter()
    points=np.array([.125,.375,.625,.875],dtype=np.float32)
    def response(a,b,c):
        x=np.array([[(a-.5)*np.sqrt(12),(b-.5)*np.sqrt(12),(c-.5)*np.sqrt(12)]],dtype=np.float32)
        return model(x,training=False).numpy()[0].astype(float)
    A=np.empty((4,4,4));B=np.empty((4,4,4));C=np.empty((4,4,4))
    for j,b in enumerate(points):
        for k,c in enumerate(points): A[j,k]=response(.5,b,c)[:4]
    for k,c in enumerate(points):
        for i,a in enumerate(points): B[k,i]=response(a,.5,c)[4:8]
    for i,a in enumerate(points):
        for j,b in enumerate(points): C[i,j]=response(a,b,.5)[8:]
    # Correct float32 softmax row-sum roundoff before rationalization.
    for table in [A,B,C]: table/=table.sum(axis=-1,keepdims=True)
    finite=Model((np.full(4,.25),)*3,(A,B,C))
    extraction_seconds=time.perf_counter()-start
    initial,before=certify(finite,p,delta,'extracted',10**6)
    polished,seconds=timed_polish(initial,pf,protocol['polish_seconds'])
    _,after=certify(polished,p,delta,'polished',10**9)
    after['polish_seconds']=seconds
    controls=[]
    space=Parameterization((4,4,4),(4,4,4))
    for rep in range(10):
        seed=protocol['random_seed_base']+rep
        initial=space.decode(np.random.default_rng(seed).normal(0,1.5,space.dimension))
        initial,control_before=certify(initial,p,delta,f'random-{rep:02d}-initial',10**6)
        polished,seconds=timed_polish(initial,pf,protocol['polish_seconds'])
        _,control_after=certify(polished,p,delta,f'random-{rep:02d}-polished',10**9)
        controls.append({'rep':rep,'seed':seed,'initial':control_before,'polished':control_after,'polish_seconds':seconds})
    receipt={'checkpoint':checkpoint.name,'checkpoint_sha256':digest(checkpoint),
        'source_revision':json.loads((HERE/'protocol.json').read_text())['revisions']['LHV-Net-2020'],
        'training_seconds':run['search_seconds'],'selected_checkpoint_seconds':cp['candidate_seconds'],
        'native_heldout':cp,'extraction_seconds':extraction_seconds,'support_points':points.tolist(),
        'cardinalities':[4,4,4],'extracted':before,'polished':after,'random_controls':controls,
        'interpretation':'The finite extraction is a new local model, not the unchanged continuous network. Random controls have equal polishing time; this one neural checkpoint is an interoperability demonstration, not evidence of typical initialization benefit.'}
    (HERE/'interoperability.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
