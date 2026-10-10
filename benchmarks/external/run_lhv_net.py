import json, os, sys, time, resource
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2'); os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2'); os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
sys.path.insert(0,os.environ.get('LHV_NET_SOURCE', '/tmp/lhv-net/sample_code')); sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'tool'))
import numpy as np
import tensorflow as tf
import config as cf
class P: pass
cf.pnn=P(); pnn=cf.pnn
pnn.inputsize=3;pnn.latin_depth=3;pnn.latin_width=16;pnn.batch_size=256;pnn.weight_init_scaling=2.;pnn.greek_depth=0;pnn.greek_width=1;pnn.activ='tanh';pnn.activ2='softmax';pnn.kernel_reg=None;pnn.a_outputsize=pnn.b_outputsize=pnn.c_outputsize=4
from utils_nn import build_model
from triangle_local.families import family
name,param=sys.argv[1],sys.argv[2]; max_seconds=float(sys.argv[3]) if len(sys.argv)>3 else 70; run_seed=int(sys.argv[4]) if len(sys.argv)>4 else 20261010
p=np.asarray(family('renou' if name=='renou' else name,param),dtype=np.float32)
# tensorflow defaults are too aggressive on a shared 16GB host
try:
 tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError: pass
model=build_model(); opt=tf.keras.optimizers.Adadelta(learning_rate=2.0)
target=tf.constant(p,dtype=tf.float32)
rng=np.random.default_rng(run_seed)
state={'best':float('inf'),'checkpoints':{},'start':time.monotonic()}
def evaluate():
 x=(rng.random((4096,3),dtype=np.float32)-.5)/np.float32(1/np.sqrt(12))
 y=model(x,training=False); a=y[:,:4]; b=y[:,4:8]; c=y[:,8:]
 q=tf.einsum('na,nb,nc->abc',a,b,c)/tf.cast(tf.shape(y)[0],tf.float32)
 return float(tf.sqrt(tf.reduce_sum(tf.square(q-target))).numpy())
def record():
 e=time.monotonic()-state['start']; d=evaluate(); state['best']=min(state['best'],d)
 for bgt in (10,60,300):
  if e>=bgt and str(bgt) not in state['checkpoints']: state['checkpoints'][str(bgt)]={'elapsed_seconds':e,'l2':state['best']}
 return e,d
status='completed'; steps=0
while time.monotonic()-state['start'] < max_seconds:
 e=time.monotonic()-state['start']
 x=(rng.random((pnn.batch_size,3),dtype=np.float32)-.5)/np.float32(1/np.sqrt(12))
 with tf.GradientTape() as tape:
  y=model(x,training=True); a=y[:,:4]; b=y[:,4:8]; c=y[:,8:]
  q=tf.einsum('na,nb,nc->abc',a,b,c)/tf.cast(tf.shape(y)[0],tf.float32)
  # Original code optimizes KL; preserve that native objective.
  safe_target=tf.where(target>0,target,tf.ones_like(target)); loss=tf.reduce_sum(tf.where(target>0,target*tf.math.log(safe_target/tf.maximum(q,1e-7)),tf.zeros_like(target)))
 grads=tape.gradient(loss,model.trainable_variables); opt.apply_gradients(zip(grads,model.trainable_variables)); steps+=1
 if steps%100==0: record()
try: e,d=record()
except Exception: e=time.monotonic()-state['start'];d=float('nan')
rec={'solver':'LHV-Net-2020','family':name,'parameter':param,'rep':2,'seed':run_seed,'status':status,'elapsed_seconds':e,'steps':steps,'best_l2':state['best'],'final_l2':d,'checkpoints':state['checkpoints'],'batch_size':pnn.batch_size,'depth':pnn.latin_depth,'width':pnn.latin_width,'objective':'KL training, independently evaluated full Cartesian Monte Carlo L2','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
print(json.dumps(rec),flush=True)
