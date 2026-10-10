"""Propagate the successful critical-region seed across the noiseless scan."""
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from round2 import Runner
from refine import read
from triangle_local.exact import numerical_model
from triangle_local.families import renoup
from triangle_local.io import load_json,dump_json


def run(directory='benchmarks'):
    out=Path(directory);env=load_json(out/'environment.json');started=perf_counter()
    r=Runner(out,env['seed'],env['starts'],env['steps'],env['polish_steps'])
    r.rows=read(out/'all-fits.csv');r.counter=len(r.rows)
    for row in r.rows:
        h=row['target_sha256']
        if h not in r.cache or row['L2_rational']<r.cache[h]['L2_rational']:r.cache[h]=row
    best=read(out/'best-fits.csv');controls=read(out/'controls.csv')
    critical=min([row for row in r.rows if row['family']=='renou-u2' and Fraction(row['parameter'])==Fraction('157/200')],
                 key=lambda row:row['L2_rational'])
    original=numerical_model(load_json(out/critical['model_file'])['model'])
    for direction,grid in [('up',['4/5','17/20','9/10','19/20']),
                           ('down',['3/4','7/10','63/100','3/5'])]:
        warm=original
        for param in grid:
            _,warm=r.search('renou-u2',param,renoup(param),initial=warm,starts=1,steps=2500,polish=350,hops=1,
                            config='critical-continuation-'+direction,group='refinement')
    r.finish(best,controls)
    new=load_json(out/'environment.json')
    new.update(total_wall_seconds=env['total_wall_seconds']+perf_counter()-started,
               peak_RSS_bytes=max(env['peak_RSS_bytes'],new['peak_RSS_bytes']),
               base_sweep_wall_seconds=env['base_sweep_wall_seconds'],
               refinement_wall_seconds=env.get('refinement_wall_seconds',0)+perf_counter()-started,
               refinement_recipe=env['refinement_recipe']+'; noiseless continuation from critical seed in both directions')
    for key in ['memory_measurement_note','refinement_timing_source']:
        if key in env:new[key]=env[key]
    dump_json(out/'environment.json',new)


if __name__=='__main__':run()
