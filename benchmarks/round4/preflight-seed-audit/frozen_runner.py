"""Frozen, single-worker cold-start comparison; no target-specific initialization.

Run `prepare`, then `run`. External repositories are read-only and pinned.
All methods load the same little-endian float64 64-entry target artifact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, str(ROOT / 'tool'))
TARGETS = [('ejm-source-080', 'elegant-source', '4/5'),
           ('ejm-detector-086', 'elegant-detector', '43/50'),
           ('renou-detector-093', 'renou-detector', '93/100')]
METHODS = ['triangle-local', 'LHV-Net-2020', 'localmodels']
REVISIONS = {'LHV-Net-2020': '9460fe5340016e9a2f954613a2c07e633449d455',
             'localmodels': '28fd13ec30761d81eaa434a8545e02db5fe51ed7'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare():
    import numpy as np
    from triangle_local.families import family
    if (HERE / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; do not overwrite an active experiment')
    targets = []
    for name, fam, par in TARGETS:
        p = np.asarray(family(fam, par), dtype='<f8').reshape(64)
        path = HERE / 'targets' / (name + '.npy')
        np.save(path, p, allow_pickle=False)
        doc = {'id': name, 'family': fam, 'visibility': par,
               'u_squared': '17/20' if fam.startswith('renou') else None,
               'ordering': 'index = 16*a + 4*b + c; a,b,c in {0,1,2,3}',
               'shape': [64], 'dtype': '<f8', 'sha256': digest(path),
               'array_bytes_sha256': hashlib.sha256(p.tobytes()).hexdigest(),
               'probabilities': p.tolist()}
        write(path.with_suffix('.json'), doc)
        targets.append(doc)
    protocol = {'schema': 'triangle-local-comparison/v1', 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'targets': targets, 'methods': METHODS, 'repetitions': 10, 'budgets_seconds': [10, 60],
        'success_threshold_l2': 1e-3, 'initialization': 'cold; no recognized, fitted, analytic or neighboring target seeds',
        'timing': 'wall clock from initialization after imports and target loading; include initialization, optimization and checkpoint selection; final held-out evaluation, serialization and certification charged separately',
        'finite_checkpoint_policy': 'best feasible output fully evaluated before each deadline',
        'neural_checkpoint_policy': 'lowest L2 on fixed 4096-sample validation set, checked every 50 steps; selection and copies inside budget',
        'neural_evaluation': {'fresh_samples_per_checkpoint': 1048576, 'blocks': 32, 'block_size': 32768,
            'uncertainty': 'approximate 95% basic block-bootstrap interval, 2000 resamples; trace variance correction and RMS Monte Carlo error also retained',
            'success': 'upper endpoint of held-out 95% interval <= threshold; unresolved intervals reported separately'},
        'triangle_local': {'cardinalities': [7,7,7], 'method': 'hybrid_fit directly (no recognizer)', 'joint_steps': 400, 'polish_steps': 120, 'inner_steps': 120, 'hops': 2, 'starts': 1, 'tolerance': 0, 'restart_after_completion': True},
        'localmodels': {'cardinalities': [7,7,7], 'optimizer': 'public trust-constr', 'trials_per_call': 1, 'tol': 0, 'restart_after_completion': True, 'feasibility': 'reject candidate if any source/response entry < 0 or row sum differs from 1 by > 1e-12'},
        'lhv_net': {'depth': 3, 'width': 16, 'batch_size': 256, 'optimizer': 'Adadelta', 'learning_rate': 2, 'objective': 'KL', 'dtype': 'float32 network, float64 contraction/scoring'},
        'threads': THREAD_ENV, 'parallel_workers': 1, 'revisions': REVISIONS,
        'hardware': {'machine': platform.machine(), 'platform': platform.platform(), 'cpu': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True).strip(), 'memory_bytes': int(subprocess.check_output(['sysctl','-n','hw.memsize']))},
        'runner_sha256': digest(__file__), 'base_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'note': 'Working tree contains round-3 changes; a content-hashed source snapshot accompanies results. Three targets preserve ten repetitions within compute cap; no 300-second arm.'}
    write(HERE / 'protocol.json', protocol)
    print('Frozen protocol:', digest(HERE / 'protocol.json'))


class Deadline(Exception):
    pass


class Recorder:
    def __init__(self, target, budgets=(10,60)):
        self.target = target
        self.budgets = budgets
        self.start = time.perf_counter()
        self.best = None
        self.value = float('inf')
        self.snapshots = {}
        self.evaluations = 0

    def elapsed(self):
        return time.perf_counter() - self.start

    def tick(self):
        now = self.elapsed()
        for budget in self.budgets:
            if now >= budget and budget not in self.snapshots:
                if self.best is None:
                    raise RuntimeError('No feasible incumbent by deadline')
                self.snapshots[budget] = self.best
        if now >= max(self.budgets):
            raise Deadline()
        return now

    def offer(self, q, payload):
        import numpy as np
        self.tick()
        value = float(np.linalg.norm(q - self.target))
        now = self.tick()  # Score must have completed before the deadline.
        self.evaluations += 1
        if value < self.value:
            self.value = value
            self.best = {'l2': value, 'candidate_seconds': now, 'q': q.copy(), 'payload': payload}


def finite_run(method, p, seed, budgets):
    import numpy as np
    from triangle_local.model import Model
    rec = Recorder(p, budgets)
    if method == 'triangle-local':
        from triangle_local.search import hybrid_fit
        original = Model.distribution
        def tracked(model):
            rec.tick()
            q = original(model)
            rec.offer(q, model)
            return q
        Model.distribution = tracked
        try:
            restart = 0
            while True:
                hybrid_fit(p, (7,7,7), starts=1, steps=400, seed=seed+restart,
                    hops=2, polish_steps=120, inner_steps=120, tolerance=0)
                restart += 1
        except Deadline:
            pass
        finally:
            Model.distribution = original
    else:
        sys.path.insert(0, os.environ['LOCALMODELS_SOURCE'])
        from triangle import TrilocalModel
        np.random.seed(seed)
        original = TrilocalModel.cost_for_optimizer
        rejected = 0
        def tracked(x, target, *sizes):
            nonlocal rejected
            rec.tick()
            model = TrilocalModel(*sizes, x)
            q = model.behavior()
            cost = float(np.sum((q-target)**2))
            arrays = (model.p_alpha, model.p_beta, model.p_gamma,
                      np.moveaxis(model.p_a,0,-1), np.moveaxis(model.p_b,0,-1), np.moveaxis(model.p_c,0,-1))
            feasible = all(np.all(a >= 0) and np.allclose(a.sum(axis=-1),1,atol=1e-12,rtol=0) for a in arrays)
            if feasible:
                rec.offer(q, arrays)
            else:
                rejected += 1
                rec.tick()
            return cost
        TrilocalModel.cost_for_optimizer = staticmethod(tracked)
        try:
            while True:
                TrilocalModel.uniform(7,7,7,4,4,4).optimize(p, number_of_trials=1, tol=0)
        except Deadline:
            pass
        finally:
            TrilocalModel.cost_for_optimizer = original
        rec.rejected_infeasible = rejected
    return rec


def build_neural(seed):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    sys.path.insert(0, str(Path(os.environ['LHV_NET_SOURCE']) / 'sample_code'))
    import config as cf
    class P:
        pass
    cf.pnn = P()
    for key, value in dict(inputsize=3,latin_depth=3,latin_width=16,batch_size=256,
        weight_init_scaling=2.,greek_depth=0,greek_width=1,activ='tanh',activ2='softmax',
        kernel_reg=None,a_outputsize=4,b_outputsize=4,c_outputsize=4).items():
        setattr(cf.pnn,key,value)
    from utils_nn import build_model
    return build_model


def evaluate_neural(model, p, seed, blocks=32, block_size=32768):
    import numpy as np
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(blocks):
        x = (rng.random((block_size,3),dtype=np.float32)-.5)*np.float32(np.sqrt(12))
        y = model(x,training=False).numpy().astype(np.float64)
        # Sum products in float64; public network weights/activations remain float32.
        q = np.einsum('na,nb,nc->abc',y[:,:4],y[:,4:8],y[:,8:])/block_size
        means.append(q.ravel())
    means = np.asarray(means)
    q = means.mean(axis=0)
    l2 = float(np.linalg.norm(q-p.ravel()))
    trace = float(means.var(axis=0,ddof=1).sum()/blocks)
    boot = np.random.default_rng(seed+1).integers(0,blocks,(2000,blocks))
    norms = np.linalg.norm(means[boot].mean(axis=1)-p.ravel(),axis=1)
    lo, hi = np.quantile(norms,[.025,.975])
    ci = [max(0.,float(2*l2-hi)),max(0.,float(2*l2-lo))]
    return {'q': q.reshape(4,4,4), 'block_means': means, 'l2': l2, 'ci95': ci,
            'mc_rms_l2': trace**.5, 'l2_squared_variance_correction': trace,
            'bias_corrected_l2': max(0,l2*l2-trace)**.5, 'samples': blocks*block_size,
            'evaluation_seconds': time.perf_counter()-started, 'heldout_seed': seed}


def neural_run(p, seed, budgets):
    import numpy as np
    import tensorflow as tf
    builder = build_neural(seed)
    started = time.perf_counter()
    model = builder()
    opt = tf.keras.optimizers.Adadelta(learning_rate=2.)
    target = tf.constant(p,dtype=tf.float32)
    rng = np.random.default_rng(seed)
    valid_rng = np.random.default_rng(seed+1000000)
    valid_x = (valid_rng.random((4096,3),dtype=np.float32)-.5)*np.float32(np.sqrt(12))
    best, best_score, best_at = None, float('inf'), None
    snapshots = {}
    steps = 0
    # The public architecture and KL training are unchanged; eager GradientTape
    # replaces removed Keras generator methods in the 2020 sample code.
    while time.perf_counter()-started < max(budgets):
        x = (rng.random((256,3),dtype=np.float32)-.5)*np.float32(np.sqrt(12))
        with tf.GradientTape() as tape:
            y = model(x,training=True)
            q = tf.einsum('na,nb,nc->abc',y[:,:4],y[:,4:8],y[:,8:])/256
            safe = tf.where(target>0,target,tf.ones_like(target))
            loss = tf.reduce_sum(tf.where(target>0,target*tf.math.log(safe/tf.maximum(q,1e-7)),tf.zeros_like(target)))
        grads = tape.gradient(loss,model.trainable_variables)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        steps += 1
        if steps == 1 or steps % 50 == 0:
            y = model(valid_x,training=False).numpy().astype(np.float64)
            q = np.einsum('na,nb,nc->abc',y[:,:4],y[:,4:8],y[:,8:])/4096
            score = float(np.linalg.norm(q-p))
            weights = model.get_weights() if score < best_score else None
            now = time.perf_counter()-started
            # Freeze old incumbent before considering anything completed too late.
            for budget in budgets:
                if now >= budget and budget not in snapshots:
                    if best is None:
                        raise RuntimeError('No neural checkpoint before deadline')
                    snapshots[budget] = (best, best_score, best_at)
            if score < best_score and now < max(budgets):
                best, best_score, best_at = weights, score, now
    for budget in budgets:
        snapshots.setdefault(budget,(best,best_score,best_at))
    training_seconds = time.perf_counter()-started
    return model, snapshots, training_seconds, steps


def worker(args):
    import numpy as np
    import resource
    protocol = json.loads((HERE/'protocol.json').read_text())
    target_meta = next(t for t in protocol['targets'] if t['id']==args.target)
    target_path = HERE/'targets'/(args.target+'.npy')
    assert digest(target_path)==target_meta['sha256']
    p = np.load(target_path,allow_pickle=False).reshape(4,4,4)
    seed = 410000 + TARGETS.index(next(t for t in TARGETS if t[0]==args.target))*10000 + METHODS.index(args.method)*1000 + args.rep*10
    stem = f'{args.target}__{args.method}__{args.rep:02d}'
    out = HERE/'runs'/stem
    budgets = (10,60) if not args.smoke else (1,2)
    result = {'target':args.target,'method':args.method,'rep':args.rep,'seed':seed,
        'target_sha256':digest(target_path),'protocol_sha256':digest(HERE/'protocol.json'),
        'checkpoints':{},'versions':{'python':platform.python_version(),'numpy':np.__version__}}
    if args.method == 'LHV-Net-2020':
        model, snapshots, duration, steps = neural_run(p,seed,budgets)
        result.update(search_seconds=duration,steps=steps)
        for budget,(weights,score,at) in snapshots.items():
            model.set_weights(weights)
            checkpoint = str(out)+f'__{budget}s.weights.h5'
            model.save_weights(checkpoint)
            ev = evaluate_neural(model,p,seed+2000000+budget,
                                  blocks=32 if not args.smoke else 4,
                                  block_size=32768 if not args.smoke else 1024)
            arrays = str(out)+f'__{budget}s.npz'
            np.savez_compressed(arrays,q=ev.pop('q'),block_means=ev.pop('block_means'))
            result['checkpoints'][str(budget)] = {**ev,'candidate_seconds':at,'selection_l2':score,
                'checkpoint_sha256':digest(checkpoint),'output_sha256':digest(arrays)}
        import tensorflow as tf
        result['versions']['tensorflow']=tf.__version__
    else:
        rec = finite_run(args.method,p,seed,budgets)
        result.update(search_seconds=rec.elapsed(),evaluations=rec.evaluations,
                      rejected_infeasible=getattr(rec,'rejected_infeasible',0))
        for budget,snapshot in rec.snapshots.items():
            payload = snapshot['payload']
            arrays = (*payload.weights,*payload.responses) if args.method=='triangle-local' else payload
            saved = str(out)+f'__{budget}s.npz'
            np.savez_compressed(saved,q=snapshot['q'],**{f'factor_{i}':a for i,a in enumerate(arrays)})
            # Scoring is recomputed from the actual saved distribution.
            q = np.load(saved)['q']
            l2 = float(np.linalg.norm(q-p))
            rmse = float(np.sqrt(np.mean((q-p)**2)))
            assert np.isclose(l2,8*rmse,rtol=1e-14,atol=1e-16)
            result['checkpoints'][str(budget)]={'l2':l2,'rmse':rmse,
                'l2_equals_8_rmse':True,'candidate_seconds':snapshot['candidate_seconds'],
                'output_sha256':digest(saved)}
    import scipy
    result['versions']['scipy']=scipy.__version__
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write(str(out)+('.smoke.json' if args.smoke else '.json'),result)
    print(json.dumps(result),flush=True)


def run(args):
    import random
    for method, key in [('LHV-Net-2020','LHV_NET_SOURCE'),('localmodels','LOCALMODELS_SOURCE')]:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ[key],text=True).strip()
        assert revision==REVISIONS[method], (method,revision)
    jobs = [(t[0],m,r) for t in TARGETS for m in METHODS for r in range(10)]
    random.Random(20261011).shuffle(jobs)
    started = time.monotonic()
    for index,(target,method,rep) in enumerate(jobs):
        stem = f'{target}__{method}__{rep:02d}'
        path = HERE/'runs'/(stem+'.json')
        if path.exists():
            continue
        command = [sys.executable,__file__,'worker','--target',target,'--method',method,'--rep',str(rep)]
        print(f'{index+1}/90 {stem}',flush=True)
        with (HERE/'runs'/(stem+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ,timeout=180)
        write(HERE/'progress.json',{'completed':index+1,'total':90,'wall_seconds':time.monotonic()-started,'last':stem})
    write(HERE/'completion.json',{'jobs':90,'wall_seconds':time.monotonic()-started,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','worker','run'])
    parser.add_argument('--target',choices=[t[0] for t in TARGETS])
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--rep',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='worker': worker(args)
    else: run(args)
