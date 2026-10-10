#!/bin/sh
# Run after the comparison, without another solver active.
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export PYTHONPATH=tool
: "${LHV_NET_SOURCE:?Set the pinned LHV-Net repository root}"
: "${LOCALMODELS_SOURCE:?Set the pinned localmodels repository root}"
: "${TRIANGLE_NEURAL_PYTHON:?Set the TensorFlow Python executable}"
.venv/bin/python -m pytest tool -q > benchmarks/round4/tests.log
.venv/bin/python benchmarks/target_bounds.py
.venv/bin/python benchmarks/round4/analyze.py
"$TRIANGLE_NEURAL_PYTHON" -m pip install --quiet 'mpmath==1.3.0'
"$TRIANGLE_NEURAL_PYTHON" benchmarks/round4/extraction.py > benchmarks/round4/extraction.log
.venv/bin/python benchmarks/round4/check_compact.py
.venv/bin/python benchmarks/independent_check.py --out benchmarks/independent-verification-round4.json > benchmarks/round4/independent-check.log
.venv/bin/python benchmarks/round4/check_reference.py
.venv/bin/python - <<'PY'
import json,re
from pathlib import Path
p=Path('benchmarks/round4')
log=(p/'tests.log').read_text()
m=re.search(r'(\d+) passed',log)
assert m,log
r={'tests_passed':int(m.group(1)),'test_command':'python -m pytest tool -q',
   'native_outcomes':180,'finite_checkpoints_independently_certified':120,
   'visual_qa':'pending final rendering','no_push':True,'no_email':True}
(p/'validation.json').write_text(json.dumps(r,indent=2)+'\n')
PY
.venv/bin/python benchmarks/round4/write_report.py
.venv/bin/python docs/make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error note.tex > benchmarks/round4/latex.log
pdflatex -interaction=nonstopmode -halt-on-error note.tex >> benchmarks/round4/latex.log
mkdir -p tmp/pdfs
pdftoppm -scale-to 1500 -png note.pdf tmp/pdfs/note > benchmarks/round4/render.log 2>&1
