"""Audit every saved outcome and build the repeated-comparison tables.

Run after the serial campaign; this script does not search or select new models.
"""
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tool'))
sys.path.insert(0,str(ROOT/'benchmarks'))
import numpy as np
from triangle_local.bounds import certificate_upper, distance_bound, quantum_interval
from triangle_local.exact import rationalize, verify, fraction
from triangle_local.families import family
from triangle_local.io import distribution_document, dump_json
from triangle_local.model import Model
from independent_check import audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    threshold=protocol['success_threshold_l2']
    targets={t['id']:t for t in protocol['targets']}
    assert sha(HERE/'frozen_runner.py')==protocol['runner_sha256']
    seeds=set()
    docs={}
    for ident,t in targets.items():
        assert sha(HERE/'targets'/(ident+'.npy'))==t['sha256']
        p=family(t['family'],t['visibility'])
        common=np.load(HERE/'targets'/(ident+'.npy')).reshape(4,4,4)
        assert np.array_equal(np.asarray(p,float),common)
        delta=0 if t['family'].startswith('elegant') else distance_bound(quantum_interval(t['family'],t['visibility']),p)
        docs[ident]=(p,delta,common)
        dump_json(HERE/'targets'/(ident+'-rational.json'),distribution_document(p))
    rows=[]; checks=[]
    certdir=HERE/'certificates';certdir.mkdir(exist_ok=True)
    for ident in targets:
        for method in protocol['methods']:
            for rep in range(protocol['repetitions']):
                stem=f'{ident}__{method}__{rep:02d}'
                runpath=HERE/'runs'/(stem+'.json')
                run=json.loads(runpath.read_text())
                assert (run['target'],run['method'],run['rep'])==(ident,method,rep)
                expected_seed=int(np.random.SeedSequence([20261011,list(targets).index(ident),protocol['methods'].index(method),rep]).generate_state(1)[0])
                assert run['seed']==expected_seed and expected_seed not in seeds
                seeds.add(expected_seed)
                assert run['target_sha256']==targets[ident]['sha256']
                assert run['protocol_sha256']==sha(HERE/'protocol.json')
                p,delta,common=docs[ident]
                for budget in protocol['budgets_seconds']:
                    cp=run['checkpoints'][str(budget)]
                    path=HERE/'runs'/(stem+f'__{budget}s.npz')
                    assert sha(path)==cp['output_sha256']
                    output=np.load(path,allow_pickle=False)
                    q=output['q'];l2=float(np.linalg.norm(q-common))
                    assert abs(l2-cp['l2']) < 1e-14
                    assert cp['candidate_seconds']<=budget
                    row={'target':ident,'method':method,'rep':rep,'seed':run['seed'],
                        'budget_seconds':budget,'l2':l2,'ci95_low':'','ci95_high':'',
                        'mc_rms_l2':'','candidate_seconds':cp['candidate_seconds'],
                        'search_seconds':run['search_seconds'],'evaluation_seconds':cp.get('evaluation_seconds',0),
                        'success':l2<=threshold,'uncertain_success':False,'stored_target_residual':'',
                        'delta_target_upper':'','U':'','rationalization_seconds':'','verification_seconds':'',
                        'target_sha256':run['target_sha256'],'output_sha256':cp['output_sha256']}
                    if method=='LHV-Net-2020':
                        assert cp['samples']==1048576
                        assert sha(HERE/'runs'/(stem+f'__{budget}s.weights.h5'))==cp['checkpoint_sha256']
                        heldout_seed=int(np.random.SeedSequence([run['seed'],203,budget]).generate_state(1,dtype=np.uint64)[0])
                        assert heldout_seed==cp['heldout_seed']
                        means=output['block_means'];assert means.shape==(32,64)
                        trace=float(means.var(axis=0,ddof=1).sum()/32)
                        assert abs(trace-cp['l2_squared_variance_correction'])<1e-20
                        bootstrap=np.random.default_rng(heldout_seed+1).integers(0,32,(2000,32))
                        norms=np.linalg.norm(means[bootstrap].mean(axis=1)-common.ravel(),axis=1)
                        lo,hi=np.quantile(norms,[.025,.975])
                        ci=[max(0.,float(2*l2-hi)),max(0.,float(2*l2-lo))]
                        assert np.allclose(ci,cp['ci95'],rtol=0,atol=1e-14)
                        assert np.allclose(q.ravel(),output['block_means'].mean(axis=0),rtol=0,atol=1e-15)
                        row.update(ci95_low=cp['ci95'][0],ci95_high=cp['ci95'][1],mc_rms_l2=cp['mc_rms_l2'],
                            success=cp['ci95'][1]<=threshold,
                            uncertain_success=cp['ci95'][0]<=threshold<cp['ci95'][1])
                    else:
                        factors=[output[f'factor_{i}'] for i in range(6)]
                        model=Model(tuple(factors[:3]),tuple(factors[3:]))
                        # Separate direct contraction, without the cached optimizer path.
                        recomputed=np.einsum('i,j,k,jka,kib,ijc->abc',*factors,optimize=False)
                        assert np.allclose(q,recomputed,rtol=0,atol=2e-14)
                        rmse=float(np.sqrt(np.mean((q-common)**2)))
                        assert abs(l2-8*rmse)<1e-15 and abs(rmse-cp['rmse'])<1e-15
                        t0=time.perf_counter();rat=rationalize(model,max_denominator=10**9)
                        rtime=time.perf_counter()-t0
                        t0=time.perf_counter();cert=verify(rat,p);vtime=time.perf_counter()-t0
                        u=certificate_upper(cert['L2_squared'],delta)
                        bundle={'schema':'triangle-local-bundle/v1','model':rat,'target':distribution_document(p),
                            'certificate':cert,'bound':{'delta_target_upper':str(delta),'U':u},
                            'search':{'method':method,'cold_start':True,'rep':rep,'budget_seconds':budget,
                                'common_target_sha256':run['target_sha256'],'native_output_sha256':cp['output_sha256'],
                                'rationalization_seconds':rtime,'verification_seconds':vtime}}
                        certpath=certdir/(stem+f'__{budget}s.json.gz');dump_json(certpath,bundle)
                        independent=audit(certpath)
                        assert independent['all_probabilities_and_distances_match']
                        row.update(stored_target_residual=float(fraction(cert['L2_squared']))**.5,
                            delta_target_upper=str(delta),U=u,rationalization_seconds=rtime,verification_seconds=vtime)
                        checks.append({'file':certpath.name,'sha256':sha(certpath),'independent_verified':True,
                                       'native_distribution_max_difference':float(np.max(abs(q-recomputed))),
                                       'l2_equals_8_rmse':True})
                    rows.append(row)
    assert len(rows)==180
    with (HERE/'outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for ident in targets:
        for method in protocol['methods']:
            for budget in protocol['budgets_seconds']:
                rs=[r for r in rows if (r['target'],r['method'],r['budget_seconds'])==(ident,method,budget)]
                values=[r['l2'] for r in rs]
                summaries.append({'target':ident,'method':method,'budget_seconds':budget,'n':len(rs),
                    'median_l2':float(np.median(values)),'q1_l2':float(np.quantile(values,.25)),
                    'q3_l2':float(np.quantile(values,.75)),'min_l2':min(values),'max_l2':max(values),
                    'successes':sum(r['success'] for r in rs),'uncertain':sum(r['uncertain_success'] for r in rs),
                    'success_threshold_l2':threshold})
    (HERE/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (HERE/'verification.json').write_text(json.dumps({'outcomes':180,'finite_certificates':len(checks),
        'checks':checks,'target_identity_verified':True,'checkpoint_deadlines_verified':True,'run_seeds_verified':len(seeds),'heldout_intervals_recomputed':60},indent=2)+'\n')
    print('Audited 180 outcomes, 120 exact finite certificates, normalization and shared-target hashes.')


if __name__=='__main__': main()
