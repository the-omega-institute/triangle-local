"""Final adaptive refinement and retention of the earlier ideal EJM candidate."""
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import csv

from round2 import Runner
from refine import read
from triangle_local.exact import numerical_model
from triangle_local.families import elegant,renoup_sources,renoup
from triangle_local.io import load_json,dump_json
from triangle_local.model import lift_model


def run(directory='benchmarks'):
    out=Path(directory);env=load_json(out/'environment.json');started=perf_counter()
    r=Runner(out,env['seed'],env['starts'],env['steps'],env['polish_steps'])
    r.rows=read(out/'all-fits.csv');r.counter=len(r.rows)
    for row in r.rows:
        h=row['target_sha256']
        if h not in r.cache or row['L2_rational']<r.cache[h]['L2_rational']:r.cache[h]=row
    best=read(out/'best-fits.csv');controls=read(out/'controls.csv')
    target=elegant('43/50','detector');model=r.nearest(target)
    row,model=r.search('elegant-detector','43/50',target,initial=model,method='convex',starts=1,
                       steps=2000,config='long-boundary-polish',group='refinement')
    if row['L2_rational']>1e-7:
        _,lp_model=r.search('elegant-detector','43/50',target,initial=model,method='convex',loss='l1',starts=1,
                            steps=80,config='boundary-linear-programs',group='refinement')
        row,model=r.search('elegant-detector','43/50',target,initial=lp_model,method='convex',starts=1,
                           steps=400,config='after-linear-programs',group='refinement')
    if row['L2_rational']>1e-7:
        seed=lift_model(model,(24,)*3,jitter=.01,seed=r.seed)
        r.search('elegant-detector','43/50',target,initial=seed,sizes=(24,)*3,starts=1,
                 steps=2400,polish=600,hops=2,config='lifted-boundary-24',group='refinement')
    # A stronger search retains the previously certified ideal candidate.
    seed=numerical_model(load_json(Path(__file__).resolve().parent.parent/'examples'/'ejm-ideal-seed.json.gz')['model'])
    target=elegant('1')
    r.record('elegant-source','1','retained-round1-seed',seed,target,group='refinement')
    r.search('elegant-source','1',target,initial=lift_model(seed,(12,)*3),starts=1,
             steps=3500,polish=400,hops=2,config='ideal-seeded-hybrid',group='refinement')
    critical=min([row for row in r.rows if row['family']=='renou-u2' and Fraction(row['parameter'])==Fraction('157/200')],
                 key=lambda row:row['L2_rational'])
    seed=numerical_model(load_json(out/critical['model_file'])['model'])
    r.search('renou-u2','17/20',renoup('17/20'),initial=seed,starts=1,steps=3500,polish=400,hops=2,
             config='critical-point-continuation',group='refinement')
    target=renoup_sources('17/20','87/100')
    r.search('renou-source','87/100',target,initial=r.nearest(target),method='convex',starts=1,
             steps=1000,config='low-noise-consistency-polish',group='refinement')
    r.finish(best,controls)
    new=load_json(out/'environment.json')
    new.update(total_wall_seconds=env['total_wall_seconds']+perf_counter()-started,
               peak_RSS_bytes=max(env['peak_RSS_bytes'],new['peak_RSS_bytes']),
               base_sweep_wall_seconds=env['base_sweep_wall_seconds'],
               refinement_wall_seconds=env.get('refinement_wall_seconds',0)+perf_counter()-started,
               refinement_recipe=env['refinement_recipe']+'; adaptive 24-symbol detector boundary; retained ideal seed; consistency polish')
    for key in ['memory_measurement_note','refinement_timing_source']:
        if key in env:new[key]=env[key]
    dump_json(out/'environment.json',new)


if __name__=='__main__':run()
