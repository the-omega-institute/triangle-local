"""Sequential, checkpointed round-2 benchmark and component comparisons.

No optimizer subprocesses or worker pools. Complete intermediate bundles are
regenerable in runs/; selected certified candidates are promoted to models/.
"""
import argparse
import csv
from triangle_local.io import load_csv, dump_csv
from fractions import Fraction
import os
from pathlib import Path
import platform
import resource
import shutil
from time import perf_counter

import numpy as np
import scipy

from triangle_local.cli import bundle
from triangle_local.exact import fraction, numerical_model
from triangle_local.families import (elegant, renoup, renoup_sources, detector_noise,
                                    note_model, note_target, ring_half_model, ring_endpoint_model,
                                    parity_model, parity_target, fritz, fritz_model)
from triangle_local.io import dump_json, load_json
from triangle_local.model import Model, lift_model
from triangle_local.optimize import fit


class Runner:
    def __init__(self,out,seed,starts,steps,polish):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.cache={};self.counter=0
        self.seed=seed;self.starts=starts;self.steps=steps;self.polish=polish
        self.started=perf_counter()

    def record(self,name,param,config,model,target,search=None,group='sweep',reference=''):
        started=perf_counter()
        doc=bundle(model,target,max_denominator=10**9,tolerance='1/1000000',search=search)
        doc['benchmark']={'family':name,'parameter':str(param),'configuration':config,'group':group}
        if name == 'fritz-visibility':
            doc['benchmark'].update(
                parameter_role='visibility display label rounded to eight significant digits',
                target_correlation=str(Fraction(1)-16*target[0,0,0]))
            if str(param) == '0.70710678':
                doc['benchmark']['visibility_expression'] = '1/sqrt(2)'
        filename=f'runs/{self.counter:03d}-{name}-{str(param).replace("/","-")}-{config}.json.gz'
        self.counter+=1;dump_json(self.out/filename,doc)
        cert=doc['certificate'];p=model.distribution();t=np.asarray(target,float)
        row={'family':name,'parameter':str(param),'configuration':config,'group':group,
             'sizes':'x'.join(map(str,model.sizes)),'symmetry':model.symmetry,
             'L2_float':float(np.linalg.norm(p-t)),'L1_float':float(np.abs(p-t).sum()),
             'L2_rational':float(np.sqrt(float(fraction(cert['L2_squared'])))),
             'L1_rational':float(fraction(cert['L1'])),'exact_match':cert['exact_match'],
             'S3_exact':cert['model_S3_symmetric_exactly'],
             'search_seconds':0 if search is None else search['elapsed_seconds'],
             'certificate_seconds':perf_counter()-started,'published_reference_L2':reference,
             'target_sha256':cert['target_sha256'],'model_file':filename}
        self.rows.append(row)
        h=cert['target_sha256']
        if h not in self.cache or row['L2_rational']<self.cache[h]['L2_rational']:
            self.cache[h]=row
        print(f'{name:18s} {str(param):8s} {config:24s} d={row["L2_float"]:.6g} '
              f'rational={row["L2_rational"]:.6g} t={row["search_seconds"]:.2f}s',flush=True)
        self.checkpoint()
        return row,model

    def search(self,name,param,target,*,initial=None,method='hybrid',sizes=(12,)*3,
               symmetry='none',starts=None,steps=None,polish=None,hops=0,group='sweep',config=None,
               optimize_weights=True,reference='',loss='l2'):
        taskseed=self.seed # paired component comparisons use the same RNG seed
        kwargs=dict(sizes=sizes,method=method,symmetry=symmetry,initial=initial,loss=loss,
                    starts=self.starts if starts is None else starts,steps=self.steps if steps is None else steps,
                    seed=taskseed,tolerance=1e-20,polish_steps=self.polish if polish is None else polish,
                    inner_steps=150,hops=hops,optimize_weights=optimize_weights)
        result=fit(np.asarray(target,float),**kwargs)
        info={**result.summary(),**{k:v for k,v in kwargs.items() if k!='initial'},
              'warm_start':initial is not None}
        return self.record(name,param,config or f'{method}-{symmetry}-{sizes[0]}',
                           result.model,target,info,group,reference)

    def checkpoint(self):
        self.csv('all-fits.csv',self.rows)
        dump_json(self.out/'progress.json',{'elapsed_seconds':perf_counter()-self.started,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
            'candidate_count':len(self.rows)})

    def csv(self,name,rows):
        if rows:
            dump_csv(self.out/name,rows)

    def nearest(self,target):
        # Reuse a single actual model, never average distributions.
        t=np.asarray(target,float)
        eligible=[r for r in self.cache.values() if r['sizes']=='12x12x12']
        if not eligible:return None
        candidates=[]
        for row in eligible:
            model=numerical_model(load_json(self.out/row['model_file'])['model'])
            candidates.append((np.linalg.norm(model.distribution()-t),model))
        return min(candidates,key=lambda pair:pair[0])[1]

    def finish(self,best,controls):
        promoted=set()
        for row in best+controls:
            selected=self.cache[row['target_sha256']] if row in best else row
            family,param,reference=row['family'],row['parameter'],row['published_reference_L2']
            row.update(selected);row.update(family=family,parameter=param,published_reference_L2=reference)
            src=self.out/row['model_file'];dest=self.out/'models'/src.name
            dest.parent.mkdir(exist_ok=True)
            if src.resolve()!=dest.resolve():shutil.copy2(src,dest)
            promoted.add(dest.name)
            row['model_file']='models/'+src.name
        for obsolete in (self.out/'models').glob('*.json.gz'):
            if obsolete.name not in promoted:obsolete.unlink()
        self.csv('best-fits.csv',best);self.csv('controls.csv',controls)
        self.csv('ablation.csv',[r for r in self.rows if r['group']=='ablation'])
        dump_json(self.out/'environment.json',{
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'platform':platform.platform(),'seed':self.seed,'starts':self.starts,'steps':self.steps,
            'polish_steps':self.polish,'total_wall_seconds':perf_counter()-self.started,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
            'optimizer_processes':1,'max_source_size':24,'rational_denominator_cap':10**9,
            'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
            'candidate_count':len(self.rows),'selected_target_count':len(best),
            'notes':'RSS is process high-water mark. Full contractions; no Monte Carlo. Neural comparator is not a rerun of 2020 code.'})


def run(out='benchmarks',seed=20261010,starts=2,steps=1800,polish=200):
    r=Runner(out,seed,starts,steps,polish);best=[];controls=[]
    uniform=Model((np.ones(1),)*3,(np.full((1,1,4),.25),)*3)
    for name,param,model,target in [
        ('note','41/144',note_model(),note_target()),('parity','1',parity_model(),parity_target()),
        ('ring-half','1/2',ring_half_model(),renoup('1/2')),
        ('ring-endpoint','1',ring_endpoint_model(),renoup('1')),
        ('uniform','0',uniform,elegant('0')),
        *[('fritz-local',t,fritz_model(t),fritz(t)) for t in ['0','1/4','1/2']]]:
        row,_=r.record(name,param,'explicit',model,target,group='control');controls.append(row.copy())

    # Controlled EJM component comparisons at the reported source transition.
    target=elegant('4/5');base=lift_model(note_model(),(12,)*3,jitter=.15,seed=seed)
    for method,n,mode,nstarts,nsteps in [
        ('lbfgs',12,'none',2,steps),('lbfgs',24,'none',2,steps),
        ('least-squares',12,'none',2,300),('deterministic',12,'none',2,80),
        ('deterministic',24,'none',2,80),('lbfgs',12,'outcome-equivariant',6,steps),
        ('lbfgs',24,'pair-equivariant',6,steps),('lbfgs',24,'color-equivariant',6,steps),
        ('neural',24,'none',2,3000)]:
        r.search('elegant-source','4/5',target,method=method,sizes=(n,)*3,symmetry=mode,
                 starts=nstarts,steps=nsteps,group='ablation',config=f'blind-{method}-{mode}-{n}')
    seeded,model=r.search('elegant-source','4/5',target,initial=base,method='lbfgs',starts=1,
                          group='ablation',config='seeded-lbfgs-12')
    for fixed in [True,False]:
        r.search('elegant-source','4/5',target,initial=model,method='convex',starts=1,
                 steps=200,group='ablation',config='polish-fixed-weights' if fixed else 'polish-free-weights',
                 optimize_weights=not fixed)
    r.search('elegant-source','4/5',target,initial=base,starts=1,hops=2,
             group='ablation',config='seeded-hybrid-hops-12')
    # Explicit seeds and blind recovery are reported separately.
    r.search('renou-u2','1/2',renoup('1/2'),method='lbfgs',starts=2,group='ablation',config='blind-lbfgs-12')
    for name,target in [('renou-source',renoup_sources('17/20','89/100')),
                        ('renou-detector',detector_noise(renoup('17/20'),'91/100')),
                        ('renou-u2',renoup('17/20'))]:
        param={'renou-source':'89/100','renou-detector':'91/100','renou-u2':'17/20'}[name]
        base=lift_model(ring_half_model(),(12,)*3,jitter=.03,seed=1)
        _,model=r.search(name,param,target,method='lbfgs',initial=base,starts=2,group='ablation',config='token-seeded-lbfgs')
        r.search(name,param,target,method='convex',initial=model,starts=1,steps=200,
                 group='ablation',config='token-seeded-polish')

    families=[
        ('elegant-source',['0','1/2','7/10','3/4','39/50','79/100','4/5','81/100','21/25','43/50','9/10','1'],lambda v:elegant(v,'source'),.8,50),
        ('elegant-detector',['0','1/2','7/10','4/5','21/25','17/20','43/50','87/100','9/10','1'],lambda v:elegant(v,'detector'),.86,60),
        ('renou-u2',['1/2','3/5','63/100','7/10','3/4','157/200','4/5','17/20','9/10','19/20','1'],renoup,None,None),
        ('renou-source',['0','4/5','87/100','89/100','9/10','19/20','1'],lambda v:renoup_sources('17/20',v),.89,6),
        ('renou-detector',['0','4/5','89/100','91/100','93/100','19/20','1'],lambda v:detector_noise(renoup('17/20'),v),.91,6)]
    requested=[]
    for name,grid,generator,vstar,theta in families:
        for direction,order in [('forward',grid),('reverse',list(reversed(grid)))]:
            for param in order:
                target=generator(param)
                reference='' if vstar is None else (0 if float(Fraction(param))<=vstar else
                    float(np.linalg.norm(np.asarray(target,float)-np.asarray(generator(str(vstar)),float))*np.sin(np.deg2rad(theta))))
                # Endpoints retain their actual exact witnesses.
                explicit=(uniform if param=='0' else ring_half_model() if name=='renou-u2' and param=='1/2'
                          else ring_endpoint_model() if name=='renou-u2' and param=='1' else None)
                if explicit is not None:
                    row,_=r.record(name,param,'explicit-'+direction,explicit,target,reference=reference)
                else:
                    warm=r.nearest(target)
                    if warm is None:
                        seedmodel=note_model() if name.startswith('elegant') else ring_half_model()
                        warm=lift_model(seedmodel,(12,)*3,jitter=.1,seed=seed)
                    row,_=r.search(name,param,target,initial=warm,config='hybrid-'+direction,reference=reference)
                if direction=='forward':requested.append(row.copy())
    # Longer boundary polishing, shared across the source/detector seed bank.
    for name,param,target in [('elegant-source','4/5',elegant('4/5')),
                              ('elegant-detector','43/50',elegant('43/50','detector')),
                              ('renou-source','89/100',renoup_sources('17/20','89/100')),
                              ('renou-detector','91/100',detector_noise(renoup('17/20'),'91/100'))]:
        r.search(name,param,target,initial=r.nearest(target),method='convex',starts=1,steps=1200,
                 config='boundary-polish',group='refinement')
    for visibility in [0.,.5,1/np.sqrt(2),.8,1.]:
        t=Fraction(str(visibility/np.sqrt(2)))
        reference=max(float(t)-.5,0)/4
        row,_=r.record('fritz-visibility',f'{visibility:.8g}','clipped-CHSH',fritz_model(min(t,Fraction(1,2))),
                       fritz(t),reference=reference)
        requested.append(row.copy())
    best=requested;r.finish(best,controls)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',default='benchmarks');p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--starts',type=int,default=2);p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--polish',type=int,default=200)
    a=p.parse_args();run(a.out_dir,a.seed,a.starts,a.steps,a.polish)
