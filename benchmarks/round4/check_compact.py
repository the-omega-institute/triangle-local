"""Measure the exact check with the compressed bundle already read into memory."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tool'))
from triangle_local.exact import verify
from triangle_local.io import from_document, load_json
p=ROOT/'benchmarks/models/renou-detector-93-100-trimmed-8-9-2.json.gz'
bundle=load_json(p)
target=from_document(bundle['target'])
times=[]
for _ in range(11):
    start=time.perf_counter()
    cert=verify(bundle['model'],target,tolerance=bundle['certificate']['L1_tolerance'])
    times.append(time.perf_counter()-start)
    assert cert==bundle['certificate']
receipt={'bundle_bytes':p.stat().st_size,'repetitions':11,
         'median_check_seconds':statistics.median(times),'check_seconds':times,
         'scope':'verify(): rational model parsing, simplex checks, integer contraction, errors, symmetry and hashes; excludes gzip/JSON IO and prior target parsing',
         'active_sizes':[len(bundle['model']['weights'][s]) for s in ['alpha','beta','gamma']]}
(ROOT/'benchmarks/round4/compact-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
