"""Cross-check the actual shared arrays against pinned reference target code.

The recorded spelling patch is applied to a disposable downstream copy; neither
external checkout nor its publication contract is modified.
"""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
import numpy as np
HERE=Path(__file__).resolve().parent
source=Path(os.environ['LHV_NET_SOURCE'])
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
protocol=json.loads((HERE/'protocol.json').read_text())
assert revision==protocol['revisions']['LHV-Net-2020']
reference=source/'sample_code/targets.py'
with tempfile.TemporaryDirectory(prefix='triangle-reference-') as temp:
    root=Path(temp);(root/'sample_code').mkdir()
    copied=root/'sample_code/targets.py';copied.write_bytes(reference.read_bytes())
    subprocess.run(['patch','-p1','-i',str(HERE/'patches/lhv-net-target-spelling.patch')],cwd=root,check=True,capture_output=True)
    namespace={};exec(compile(copied.read_text(),str(copied),'exec'),namespace)
    generator=namespace['target_distribution_gen']
    names={'elegant-source':'elegant-visibility','elegant-detector':'elegant-localnoise','renou-detector':'Renou-localnoise'}
    checks=[]
    for t in protocol['targets']:
        p=np.load(HERE/'targets'/(t['id']+'.npy'))
        ref=generator(names[t['family']],np.sqrt(.85),float(Fraction(t['visibility'])))
        discrepancy=float(np.max(abs(p-ref)))
        assert discrepancy < 1e-12
        checks.append({'target':t['id'],'sha256':t['sha256'],'max_entry_discrepancy':discrepancy,
                       'l2_discrepancy':float(np.linalg.norm(p-ref))})
receipt={'reference_revision':revision,'original_source_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
         'patch_sha256':hashlib.sha256((HERE/'patches/lhv-net-target-spelling.patch').read_bytes()).hexdigest(),
         'checks':checks,'meaning':'Point cross-checks; not target interval enclosures.'}
(HERE/'reference-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
