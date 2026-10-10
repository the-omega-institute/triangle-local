import json, os, signal, sys, time, resource
from pathlib import Path
sys.path.insert(0, os.environ.get('LOCALMODELS_SOURCE', '/tmp/localmodels')); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tool'))
import numpy as np
from triangle import TrilocalModel
from triangle_local.families import family
name,param=sys.argv[1],sys.argv[2]; outpath=Path(sys.argv[3]); limit=float(sys.argv[4]); run_seed=int(sys.argv[5]) if len(sys.argv)>5 else 20262019; rep=int(sys.argv[6]) if len(sys.argv)>6 else 2
np.random.seed(run_seed); p=np.asarray(family('renou' if name=='renou' else name,param),float)
state={'best':float('inf'),'checkpoints':{},'start':time.monotonic()}
orig=TrilocalModel.cost_for_optimizer
def tracked(x,*args):
 val=float(orig(x,*args)); state['best']=min(state['best'],val); elapsed=time.monotonic()-state['start']
 for budget in (10,60,300):
  if elapsed>=budget and str(budget) not in state['checkpoints']:
   l2=state['best']**.5; state['checkpoints'][str(budget)]={'elapsed_seconds':elapsed,'l2':l2,'rms':l2/8}
 return val
TrilocalModel.cost_for_optimizer=staticmethod(tracked)
def alarm(sig,frame): raise TimeoutError(f'{limit}s limit')
signal.signal(signal.SIGALRM,alarm); signal.alarm(int(limit))
status='completed'; error=None
try:
 m=TrilocalModel.uniform(7,7,7,4,4,4); model=m.optimize(p,number_of_trials=1,tol=0); error=float(model.cost(p))**.5
except TimeoutError as exc: status='timeout'; error=state['best']**.5 if state['best']<float('inf') else None
except Exception as exc: status='error:'+repr(exc)
finally: signal.alarm(0); TrilocalModel.cost_for_optimizer=orig
elapsed=time.monotonic()-state['start']
rec={'solver':'localmodels','revision':'28fd13ec30761d81eaa434a8545e02db5fe51ed7','family':name,'parameter':param,'u_squared':'17/20' if name.startswith('renou') else None,'rep':rep,'seed':run_seed,'cardinalities':[7,7,7],'outputs':[4,4,4],'status':status,'elapsed_seconds':elapsed,'best_l2':error,'best_rms':None if error is None else error/8,'checkpoints':state['checkpoints'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with outpath.open('a') as f: f.write(json.dumps(rec)+'\n')
print(json.dumps(rec),flush=True)
