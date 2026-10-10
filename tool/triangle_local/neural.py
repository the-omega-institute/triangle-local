"""Small NumPy response networks on independent finite quadrature grids.

This is deterministic quadrature, not the Monte Carlo training implementation
of Krivachy et al. It needs no deep-learning runtime. The resulting ordinary
finite response tables use the same exact certificate path as every model.
"""
from time import perf_counter

import numpy as np
from scipy.special import softmax

from .model import Model


def neural_fit(target, sizes=(24,24,24), *, starts=3, steps=3000, seed=0,
               width=24, learning_rate=.02):
    from .optimize import FitResult
    rng=np.random.default_rng(seed)
    grid=[np.linspace(-1,1,n) for n in sizes]
    inputs=[]
    for left,right in [(grid[1],grid[2]),(grid[2],grid[0]),(grid[0],grid[1])]:
        x,y=np.meshgrid(left,right,indexing='ij')
        inputs.append(np.stack([x,y],axis=-1).reshape(-1,2))
    weights=tuple(np.full(n,1/n) for n in sizes)
    best,best_value=None,float('inf')
    runs=[];started=perf_counter()
    for start in range(starts):
        networks=[]
        for outputs in target.shape:
            networks.append([rng.normal(size=(2,width)),np.zeros(width),
                             rng.normal(0,np.sqrt(2/width),size=(width,width)),np.zeros(width),
                             rng.normal(0,np.sqrt(2/width),size=(width,outputs)),np.zeros(outputs)])
        momentum=[[np.zeros_like(p) for p in net] for net in networks]
        velocity=[[np.zeros_like(p) for p in net] for net in networks]
        run_value=float('inf');run_model=None
        for step in range(1,steps+1):
            activations=[];tables=[]
            for x,net,shape in zip(inputs,networks,[(sizes[1],sizes[2]),(sizes[2],sizes[0]),(sizes[0],sizes[1])]):
                w1,b1,w2,b2,w3,b3=net
                h1=np.maximum(x@w1+b1,0)
                h2=np.maximum(h1@w2+b2,0)
                r=softmax(h2@w3+b3,axis=-1)
                activations.append((x,h1,h2,r))
                tables.append(r.reshape(*shape,-1))
            model=Model(weights,tuple(tables))
            residual=model.distribution()-target
            value=float(np.sum(residual**2))
            if value<run_value:run_value,run_model=value,model
            if value<1e-20:break
            derivatives=model.response_derivatives(2*residual)
            for party,(net,acts,d) in enumerate(zip(networks,activations,derivatives)):
                x,h1,h2,r=acts
                d=d.reshape(r.shape)
                g3=r*(d-(r*d).sum(axis=-1,keepdims=True))
                g2=(g3@net[4].T)*(h2>0)
                g1=(g2@net[2].T)*(h1>0)
                grads=[x.T@g1,g1.sum(0),h1.T@g2,g2.sum(0),h2.T@g3,g3.sum(0)]
                for k,grad in enumerate(grads):
                    momentum[party][k]=.9*momentum[party][k]+.1*grad
                    velocity[party][k]=.999*velocity[party][k]+.001*grad**2
                    net[k]-=learning_rate*(momentum[party][k]/(1-.9**step))/(np.sqrt(velocity[party][k]/(1-.999**step))+1e-8)
        if run_value<best_value:best,best_value=run_model,run_value
        runs.append({'start':start,'objective':run_value,'iterations':step,'evaluations':step,
                     'status':'tolerance' if run_value<1e-20 else 'iteration_limit'})
    return FitResult(best,'l2',best_value,perf_counter()-started,runs)
