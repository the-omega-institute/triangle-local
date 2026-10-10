"""Recompute every selected certificate; no optimization or mutation of models."""
import argparse
from pathlib import Path
from triangle_local.exact import verify
from triangle_local.io import load_json, dump_json, from_document


def run(directory='benchmarks'):
    out=Path(directory);files=sorted((out/'models').glob('*.json*'))
    if not files:raise ValueError('No selected bundles found')
    exact=0
    for file in files:
        doc=load_json(file)
        saved=doc['certificate']
        actual=verify(doc['model'],from_document(doc['target']),tolerance=saved['L1_tolerance'])
        if actual!=saved:raise ValueError('Certificate mismatch: '+str(file))
        exact+=actual['exact_match']
    result={'independently_recomputed_certificates':len(files),'all_matched':True,'exact_matches':exact}
    dump_json(out/'verification.json',result)
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',default='benchmarks')
    run(p.parse_args().directory)
