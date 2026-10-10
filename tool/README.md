# triangle-local

A certification and interoperability layer for finite triangle-local models.
Export independent source laws and response tables as rational witnesses, then
recompute every probability, L1 error and squared L2 error exactly with integer
arithmetic. Search can produce candidates; the certificate checker does not
trust its result flags.

Install a locally built wheel with
`pip install path/to/triangle_local-0.2.0-py3-none-any.whl`, or run
`pip install -e './tool[test]'` from the source repository. Use
`triangle-local --help` for the command line. NumPy and SciPy are the base
runtime dependencies. The optional `bounds` extra adds mpmath for intended-target
interval enclosures. The repository README documents the axes, input formats,
certificate schema and reproducible experiments.

For a stored rational target, the checker computes E = ||P_stored−Q||₂²
exactly and establishes that Q is triangle-local. A validated enclosure
δ ≥ ||P−P_stored||₂ supplies an upward-rounded bound U ≥ sqrt(E)+δ to the
intended target. Exact stored-target equality proves intended-target locality
only when target identity is exact. Positive U establishes a nearby local
model; a positive residual or optimizer failure does not establish nonlocality
or optimality. The verifier is tested software, not a formally verified
implementation. Licensed under MIT.
