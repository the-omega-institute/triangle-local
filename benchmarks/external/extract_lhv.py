import gzip,json,os,sys,time,resource
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
sys.path.insert(0,os.environ.get('LHV_NET_SOURCE', '/tmp/lhv-net/sample_code')); sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'tool'))
import numpy as np, tensorflow as tf
import config as cf
class P: pass
cf.pnn=P(); pnn=cf.pnn
pnn.inputsize=3;pnn.latin_depth=3;pnn.latin_width=16;pnn.batch_size=256;pnn.weight_init_scaling=2.;pnn.greek_depth=0;pnn.greek_width=1;pnn.activ='tanh';pnn.activ2='softmax';pnn.kernel_reg=None;pnn.a_outputsize=pnn.b_outputsize=pnn.c_outputsize=4
from utils_nn import build_model
from triangle_local.families import family
from triangle_local.model import Model
from triangle_local.exact import rationalize,verify,numerical_model
from triangle_local.io import distribution_document,dump_json
from triangle_local.optimize import fit
name,param='renou-detector','93/100'; target=np.asarray(family(name,param),dtype=object); target_float=np.asarray(target,float)
try: tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError: pass
model=build_model(); opt=tf.keras.optimizers.Adadelta(learning_rate=2.0); target_tf=tf.constant(target_float,dtype=tf.float32); rng=np.random.default_rng(20261010)
start=time.monotonic(); limit=65; steps=0
while time.monotonic()-start<limit:
 x=(rng.random((pnn.batch_size,3),dtype=np.float32)-.5)/np.float32(1/np.sqrt(12))
 with tf.GradientTape() as tape:
  y=model(x,training=True);a=y[:,:4];b=y[:,4:8];c=y[:,8:];q=tf.einsum('na,nb,nc->abc',a,b,c)/tf.cast(tf.shape(y)[0],tf.float32)
  safe=tf.where(target_tf>0,target_tf,tf.ones_like(target_tf));loss=tf.reduce_sum(tf.where(target_tf>0,target_tf*tf.math.log(safe/tf.maximum(q,1e-7)),tf.zeros_like(target_tf)))
 grads=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(grads,model.trainable_variables));steps+=1
train_seconds=time.monotonic()-start
# Four independent support points per source; each response is evaluated on its full pair grid.
support=np.array([.125,.375,.625,.875],dtype=np.float32); scale=np.float32(1/np.sqrt(12)); feed=lambda a,b,c: np.array([[(a-.5)/scale,(b-.5)/scale,(c-.5)/scale]],dtype=np.float32)
def pred(x): return model(tf.constant(x),training=False).numpy()[0]
A=np.empty((4,4,4));B=np.empty((4,4,4));C=np.empty((4,4,4))
for j,beta in enumerate(support):
 for k,gamma in enumerate(support): A[j,k]=pred(feed(0.5,beta,gamma))[:4]
for k,gamma in enumerate(support):
 for i,alpha in enumerate(support): B[k,i]=pred(feed(alpha,0.5,gamma))[4:8]
for i,alpha in enumerate(support):
 for j,beta in enumerate(support): C[i,j]=pred(feed(alpha,beta,0.5))[8:]
A /= A.sum(axis=-1,keepdims=True); B /= B.sum(axis=-1,keepdims=True); C /= C.sum(axis=-1,keepdims=True); extracted=Model((np.full(4,.25),)*3,(A,B,C)); extract_seconds=time.monotonic()-start-train_seconds
rat=rationalize(extracted,max_denominator=1000000); cert_before=verify(rat,target,tolerance='1/1000000');before_seconds=time.monotonic()-start
before={'schema':'triangle-local-bundle/v1','target':distribution_document(target),'model':rat,'certificate':cert_before,'search':{'source':'LHV-Net-2020','supports':[str(x) for x in support],'extraction_seconds':extract_seconds,'train_seconds':train_seconds}}
dump_json('benchmarks/external/lhv-net-extracted-renou-detector-93.json.gz',before)
polish_start=time.monotonic(); result=fit(target_float,(4,4,4),method='convex',loss='l2',starts=1,steps=120,seed=20261010,initial=numerical_model(rat),inner_steps=120,optimize_weights=True); polish_seconds=time.monotonic()-polish_start
pol_rat=rationalize(result.model,max_denominator=1000000000);cert_after=verify(pol_rat,target,tolerance='1/1000000');after={'schema':'triangle-local-bundle/v1','target':distribution_document(target),'model':pol_rat,'certificate':cert_after,'search':{'source':'triangle-local convex polish','steps':120,'inner_steps':120,'polish_seconds':polish_seconds,'initial_bundle':'lhv-net-extracted-renou-detector-93.json.gz'}};dump_json('benchmarks/external/lhv-net-polished-renou-detector-93.json.gz',after)
receipt={'schema':'triangle-local-neural-extraction/v1','target':{'family':name,'parameter':param,'u_squared':'17/20'},'training':{'revision':'9460fe5340016e9a2f954613a2c07e633449d455','seconds':train_seconds,'steps':steps,'batch_size':256,'depth':3,'width':16,'objective':'KL'},'extraction':{'supports':[str(x) for x in support],'cardinalities':[4,4,4],'seconds':extract_seconds,'certified_L2':float(__import__('fractions').Fraction(cert_before['L2_squared']))**.5,'certified_L1':float(__import__('fractions').Fraction(cert_before['L1'])),'status':cert_before['status']},'polish':{'method':'triangle-local convex','seconds':polish_seconds,'certified_L2':float(__import__('fractions').Fraction(cert_after['L2_squared']))**.5,'certified_L1':float(__import__('fractions').Fraction(cert_after['L1'])),'status':cert_after['status']},'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path('benchmarks/external/lhv-extraction-polish.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
