"""Reproducible sequential benchmark; one optimizer runs at a time."""
import argparse
import csv
from .io import load_csv, dump_csv
from fractions import Fraction
import platform
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import scipy

from .cli import bundle
from .exact import fraction
from .families import (elegant, fritz, fritz_model, note_model, note_target,
                       parity_model, parity_target, renoup, renoup_sources,
                       detector_noise, ring_endpoint_model)
from .io import dump_json
from .model import Model
from .optimize import fit


def run(out_dir="benchmarks", starts=4, steps=1000, seed=20261009):
    out = Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows, best, controls = [], [], []
    started = perf_counter()

    def record(name,param,config,model,target,search=None,published_reference=None):
        t0=perf_counter()
        document=bundle(model,target,max_denominator=10**8,tolerance="1/1000000",search=search)
        cert=document["certificate"]
        document["benchmark"]={"family":name,"parameter":str(param),"configuration":config}
        filename=f"{name}_{str(param).replace('/','-')}_{config}.json"
        dump_json(out/"models"/filename,document)
        p=model.distribution();t=np.asarray(target,float)
        row={"family":name,"parameter":str(param),"configuration":config,
             "sizes":"x".join(str(x) for x in model.sizes),"symmetry":model.symmetry,
             "L2_float":float(np.linalg.norm(p-t)),"L1_float":float(np.abs(p-t).sum()),
             "L1_rational":float(fraction(cert["L1"])),
             "L2_rational":float(np.sqrt(float(fraction(cert["L2_squared"])))),
             "exact_match":cert["exact_match"],"S3_exact":cert["model_S3_symmetric_exactly"],
             "search_seconds":0. if search is None else search["elapsed_seconds"],
             "certificate_seconds":perf_counter()-t0,
             "published_reference_L2":published_reference if published_reference is not None else "",
             "model_file":"models/"+filename}
        rows.append(row)
        print(f"{name:18s} {str(param):7s} {config:20s} L2={row['L2_float']:.6g} exact={cert['exact_match']} "
              f"time={row['search_seconds']:.2f}s",flush=True)
        return row

    def search(name,param,target,sizes=(8,8,8),symmetry="none",method="lbfgs",loss="l2",
               nstarts=None,nsteps=None,initial=None):
        config=f"{method}-{symmetry}-{sizes[0]}"
        taskseed=seed+len(rows)
        result=fit(np.asarray(target,float),sizes,symmetry=symmetry,method=method,loss=loss,
                   starts=nstarts or starts,steps=nsteps or steps,seed=taskseed,initial=initial)
        info={**result.summary(),"seed":taskseed,"method":method,"symmetry":symmetry,
              "sizes":list(sizes),"starts":nstarts or starts,"max_steps":nsteps or steps,
              "warm_start":initial is not None}
        return record(name,param,config,result.model,target,info),result.model

    for name,param,model,target in [
        ("note","41-144",note_model(),note_target()),
        ("parity","1",parity_model(),parity_target()),
        *[("fritz-local",t,fritz_model(t),fritz(t)) for t in ("0","1/4","1/2")],
        ("ring-endpoint","1",ring_endpoint_model(),renoup("1")),
    ]:
        controls.append(record(name,param,"explicit",model,target))

    # Blind numerical recovery is reported separately from replaying a witness.
    product_target=np.einsum("a,b,c->abc",np.array([Fraction(1,5),Fraction(4,5)],object),
                             np.array([Fraction(3,10),Fraction(7,10)],object),
                             np.array([Fraction(2,5),Fraction(3,5)],object))
    for method,loss in [("adam","l2"),("lbfgs","l2"),("alternating","l2"),("em","kl")]:
        row,_=search("biased-product","1",product_target,(1,1,1),method=method,loss=loss,
                     nstarts=2,nsteps=350)
        controls.append(row)
    for name,target,n,mode in [("parity-blind",parity_target(),2,"none"),
                               ("note-blind",note_target(),12,"outcome-equivariant")]:
        row,_=search(name,"1",target,(n,)*3,symmetry=mode,nstarts=max(6,starts))
        controls.append(row)

    # Source and detector noise are the two different families in 2020 Figure 4.
    grid=["0","1/2","7/10","3/4","4/5","21/25","43/50","9/10","1"]
    for noise,vstar,theta in [("source",.80,50),("detector",.86,60)]:
        warm={}
        for v in grid:
            name="elegant-"+noise;target=elegant(v,noise)
            candidates=[]
            for n,mode in [(8,"none"),(12,"outcome-equivariant")]:
                row,model=search(name,v,target,(n,)*3,symmetry=mode,initial=warm.get(mode),
                                 nstarts=starts if mode=="none" else max(starts,6))
                warm[mode]=model
                reference=(0. if float(fraction(v))<=vstar else
                           float(np.linalg.norm(np.asarray(target,float)-np.asarray(elegant(str(vstar),noise),float))
                                 *np.sin(np.deg2rad(theta))))
                row["published_reference_L2"]=reference
                candidates.append(row)
            best.append(min(candidates,key=lambda r:r["L2_float"]).copy())

    # The explicit CHSH construction also supplies a useful reference candidate.
    for visibility in [0.,.5,1/np.sqrt(2),.8,1.]:
        t=Fraction(str(visibility/np.sqrt(2)))
        target=fritz(t);model=fritz_model(min(t,Fraction(1,2)))
        reference=max(float(t)-.5,0)/4
        row=record("fritz-visibility",f"{visibility:.12g}","explicit-clipped",model,target,
                   published_reference=reference)
        fitted,_=search("fritz-visibility",f"{visibility:.12g}",target,(2,2,8),initial=model)
        fitted["published_reference_L2"]=reference
        best.append(min([row,fitted],key=lambda r:r["L2_float"]).copy())

    # Ring/Renou Figure 5: noiseless u² scan and the u²=.85 noise scans.
    warm=None
    for t in ["1/2","63/100","157/200","17/20","1"]:
        row,warm=search("renou-u2",t,renoup(t),(6,)*3,initial=warm)
        best.append(row.copy())
    for noise,grid in [("source",["0","4/5","89/100","19/20","1"]),
                       ("detector",["0","4/5","91/100","19/20","1"])]:
        warm=None
        for v in grid:
            target=(renoup_sources("17/20",v) if noise=="source" else detector_noise(renoup("17/20"),v))
            row,warm=search("renou-"+noise,v,target,(6,)*3,initial=warm)
            best.append(row.copy())

    def write_csv(filename,data):
        dump_csv(out/filename,data)
    write_csv("all-fits.csv",rows)
    write_csv("best-fits.csv",best)
    write_csv("controls.csv",controls)
    elapsed=perf_counter()-started
    dump_json(out/"environment.json",{"python":platform.python_version(),"numpy":np.__version__,
               "scipy":scipy.__version__,"platform":platform.platform(),"machine":platform.machine(),
               "seed":seed,"starts":starts,"steps":steps,"total_wall_seconds":elapsed,
               "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024),
               "processes":1,"max_source_size":12,"max_outputs":4,
               "rational_denominator_cap":10**8,
               "notes":"One optimizer at a time. Timings include full contractions; certificates separately timed."
               })
    print(f"Saved {len(rows)} candidates; total {elapsed:.2f}s",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",default="benchmarks")
    p.add_argument("--starts",type=int,default=4)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--seed",type=int,default=20261009)
    a=p.parse_args()
    run(a.out_dir,a.starts,a.steps,a.seed)


if __name__=="__main__":main()
