# triangle-local

Search finite independent-source triangle models, export rational candidates,
and recompute exact model-to-target distances with integer arithmetic.

Install a locally built wheel with
`pip install path/to/triangle_local-0.2.0-py3-none-any.whl`, or run
`pip install -e './tool[test]'` from the source repository. Use
`triangle-local --help` for the command line. NumPy and SciPy are the only
runtime dependencies. The repository README documents model axes, JSON/CSV
inputs, certificates, optimizer options and reproducible benchmarks.

An exact-match certificate establishes a local witness for the stored rational
target. A small residual establishes a nearby local distribution; optimizer
failure does not establish nonlocality. The verifier is tested software, not
a formally verified decision procedure. Licensed under MIT.
