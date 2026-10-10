"""Build a content-hashed local Round 4 reproducibility archive, without publishing."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def main():
    names=['README.md','RESULTS.md','BENCHMARK.md','LITERATURE.md','CHANGES.md','HEADLINE.md','LICENSE','.gitignore','note.tex','note.pdf']
    files=[ROOT/name for name in names]
    files += [ROOT/'tool'/name for name in ['pyproject.toml','README.md','LICENSE']]
    patterns=['tool/triangle_local/*.py','tool/tests/test*.py','benchmarks/*.py',
       'benchmarks/*.csv','benchmarks/models/*.json.gz','benchmarks/*verification*.json',
       'benchmarks/target-bounds.json','benchmarks/environment.json','benchmarks/validation.json',
       'benchmarks/external/*.py','benchmarks/external/*.json','benchmarks/external/*.json.gz',
       'benchmarks/round4/*.py','benchmarks/round4/*.sh','benchmarks/round4/*.json','benchmarks/round4/*.md',
       'benchmarks/round4/*.csv','benchmarks/round4/*.tex','benchmarks/round4/patches/*.patch',
       'benchmarks/round4/targets/*.json','benchmarks/round4/targets/*.npy',
       'benchmarks/round4/runs/*.json','benchmarks/round4/runs/*.npz','benchmarks/round4/runs/*.h5',
       'benchmarks/round4/certificates/*.json.gz','benchmarks/round4/interoperability/*.json.gz',
       'docs/*.py','docs/*.json','docs/figures/*.md']
    for pattern in patterns: files += list(ROOT.glob(pattern))
    for name in ['headline','baseline-time','local-model']:
        for ext in ['png','svg']: files.append(ROOT/'docs/figures'/(name+'.'+ext))
    for name in ['best-fits.csv','all-fits.csv','controls.csv','environment.json','verification.json','validation.json']:
        files.append(ROOT/'benchmarks/round1'/name)
    for name in ['parity.csv','elegant-source-v07.json','elegant-source-v08.json','ejm-ideal-seed.json.gz']:
        files.append(ROOT/'examples'/name)
    for name in ['family-crosschecks.json','round2-family-crosschecks.json','verified-metadata.json',
                 'verified-arxiv.xml','witness-note-round1.tex','independent-target-checks.json','trimmed-witness.json']:
        files.append(ROOT/'research'/name)
    files=sorted({p for p in files if '__99' not in p.name and p.name not in ['manifest.json','progress.json']})
    missing=[str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing: raise ValueError('Missing public files: '+str(missing))
    manifest={'artifact_version':'round4-2026-10-11','publication_status':'local, not pushed',
       'files':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}}
    mp=ROOT/'benchmarks/round4/manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');files.append(mp)
    dest=ROOT/'output/triangle-local-round4-2026-10-11.zip';dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:
            info=zipfile.ZipInfo('triangle-local/'+str(path.relative_to(ROOT)),date_time=(2026,10,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        for relative,expected in manifest['files'].items():
            assert sha256(archive.read('triangle-local/'+relative)).hexdigest()==expected
    (dest.with_suffix('.zip.sha256')).write_text(sha256(dest.read_bytes()).hexdigest()+'  '+dest.name+'\n')
    print(f'{len(files)} files; {dest.stat().st_size/1e6:.2f} MB; archive and all hashes verified: {dest}')


if __name__=='__main__': main()
