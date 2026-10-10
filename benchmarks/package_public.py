"""Build a small public source archive with an explicit file allowlist."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def main():
    files=[ROOT/name for name in ['README.md','RESULTS.md','LITERATURE.md','LICENSE','.gitignore','note.tex','note.pdf']]
    files += [ROOT/'tool'/name for name in ['pyproject.toml','README.md','LICENSE']]
    for pattern in ['tool/triangle_local/*.py','tool/tests/test*.py','benchmarks/*.py',
                    'benchmarks/*.csv','benchmarks/models/*.json.gz']:
        files += list(ROOT.glob(pattern))
    for name in ['environment.json','verification.json','validation.json','independent-verification.json','noise-curves.png','noise-curves.svg']:
        files.append(ROOT/'benchmarks'/name)
    for name in ['best-fits.csv','all-fits.csv','controls.csv','environment.json','verification.json','validation.json']:
        files.append(ROOT/'benchmarks'/'round1'/name)
    for name in ['parity.csv','elegant-source-v07.json','elegant-source-v08.json','ejm-ideal-seed.json.gz']:
        files.append(ROOT/'examples'/name)
    for name in ['family-crosschecks.json','round2-family-crosschecks.json','verified-metadata.json',
                 'verified-arxiv.xml','witness-note-round1.tex','independent-target-checks.json']:
        files.append(ROOT/'research'/name)
    files=sorted(set(files))
    missing=[str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:raise ValueError('Missing public files: '+str(missing))
    destination=ROOT/'output'/'triangle-local-public-source.zip';destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:
            info=zipfile.ZipInfo('triangle-local/'+str(path.relative_to(ROOT)),date_time=(2026,10,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,path.read_bytes())
    print(f'{len(files)} public files; {destination.stat().st_size/1e6:.2f} MB: {destination}')


if __name__=='__main__':main()
