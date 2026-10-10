"""Targeted follow-up to the base sweep; one optimizer at a time.

Keeps all failures in all-fits.csv and promotes the best verified candidates.
It adds the symmetric fallback, extra boundary hops, and numerical Fritz checks.
"""
import csv
from triangle_local.io import load_csv, dump_csv
from fractions import Fraction
from pathlib import Path
import resource
import platform
from time import perf_counter

import numpy as np

from round2 import Runner
from triangle_local.exact import numerical_model
from triangle_local.families import elegant, renoup, renoup_sources, detector_noise, note_model
from triangle_local.io import load_json,dump_json,from_document
from triangle_local.model import lift_model


def read(path):
    rows=load_csv(path)
    for row in rows:
        for key in ['L2_float','L1_float','L2_rational','L1_rational','search_seconds','certificate_seconds']:
            row[key]=float(row[key])
        for key in ['exact_match','S3_exact']:row[key]=row[key]=='True'
        if row['published_reference_L2']:row['published_reference_L2']=float(row['published_reference_L2'])
    return rows


def run(directory='benchmarks'):
    out=Path(directory);env=load_json(out/'environment.json')
    started=perf_counter()
    r=Runner(out,env['seed'],env['starts'],env['steps'],env['polish_steps'])
    r.rows=read(out/'all-fits.csv');r.counter=len(r.rows)
    for row in r.rows:
        h=row['target_sha256']
        if h not in r.cache or row['L2_rational']<r.cache[h]['L2_rational']:r.cache[h]=row
    best=read(out/'best-fits.csv');controls=read(out/'controls.csv')
    for row in best:
        v=float(Fraction(row['parameter']))
        if row['family'].startswith('elegant') and v>0 and row['L2_rational']>1e-7:
            target=elegant(row['parameter'],row['family'].split('-')[1])
            r.search(row['family'],row['parameter'],target,method='lbfgs',sizes=(12,)*3,
                     symmetry='outcome-equivariant',starts=8,steps=1800,
                     config='symmetric-fallback',group='refinement')
    # Boundary continuation is pooled with perturbations to leave poor basins.
    for name,param,target in [('elegant-detector','43/50',elegant('43/50','detector')),
                              ('elegant-detector','17/20',elegant('17/20','detector')),
                              ('renou-u2','157/200',renoup('157/200')),
                              ('renou-u2','17/20',renoup('17/20'))]:
        r.search(name,param,target,initial=r.nearest(target),starts=2,steps=3500,polish=400,hops=4,
                 config='extra-hops',group='refinement')
    for row in best:
        if row['family']=='fritz-visibility':
            doc=load_json(out/row['model_file']);model=numerical_model(doc['model'])
            r.search(row['family'],row['parameter'],from_document(doc['target']),initial=model,
                     sizes=model.sizes,starts=2,steps=1200,polish=120,hops=0,
                     config='seeded-numerical-CHSH',group='refinement',reference=row['published_reference_L2'])
    r.finish(best,controls)
    new=load_json(out/'environment.json')
    new.update(total_wall_seconds=env['total_wall_seconds']+perf_counter()-started,
               peak_RSS_bytes=max(env['peak_RSS_bytes'],new['peak_RSS_bytes']),
               base_sweep_wall_seconds=env.get('base_sweep_wall_seconds',env['total_wall_seconds']),
               refinement_wall_seconds=perf_counter()-started,
               refinement_recipe='symmetric fallback; 4 extra hops (3500 joint, 400 convex sweeps); seeded numerical Fritz checks')
    dump_json(out/'environment.json',new)


if __name__=='__main__':run()
