# triangle-local

**Find an explicit classical model for a three-source quantum network, and check
its distance to the target exactly.** Given a joint distribution `P(a,b,c)`,
triangle-local searches for finite hidden-variable models in which each party
sees only its two incident, independent sources. It exports the source weights,
stochastic response tables, and a rational certificate that independently
recomputes every output probability. The saved models fit all four noise
transitions estimated by Kriváchy et al. (2020) to within `3.27e-8`, and remain
particularly close to the Renou targets beyond those estimates. The package
uses NumPy and SciPy on a CPU.

**Authors:** Haobo Ma and Wenlin Zhang. **License:** [MIT](LICENSE).

![Certified local-model distances and 2020 estimates for Renou and EJM source and detector noise, with log-scale detail](docs/figures/headline.png)

The Renou detector model at `u² = 0.85`, efficiency `v = 0.93`, has certified
distance `8.207624541e-9`, while the 2020 Eq. (4) estimate is `0.001260030627`.
Solid markers show saved local models; dashed lines connect approximate
analytic estimates evaluated at the sampled parameters, **not digitized
neural-network data**. Dotted lines mark the 2020 estimated transitions.
The lower panels resolve residuals hidden on a linear scale.

## Results

Here `d = sqrt(sum((P_target - P_model)²))`, the Euclidean distance used in the
2020 paper's Eq. (3). Each candidate is certified local, so **its distance is an
upper bound on distance to the unrestricted local set**. The displayed decimals
have ten significant digits; the certificates retain exact rational L1 and
squared L2 distances.

| Target | Certified distance `d` | Comparison |
|---|---:|---|
| EJM source, `v = 0.80` (2020 estimated transition) | `5.773336608e-11` | Round 1: `0.00537512503` |
| EJM detector, `v = 0.86` (2020 estimated transition) | `1.509705617e-9` | Round 1: `0.01088298077` |
| Renou source, `v = 0.89` (2020 estimated transition) | `3.269903713e-8` | Round 1: `0.001073918666` |
| Renou detector, `v = 0.91` (2020 estimated transition) | `5.909466315e-9` | Round 1: `8.289418225e-4` |
| Renou source, `v = 0.90` | `8.809609828e-6` | 2020 Eq. (4): `4.739586006e-4` |
| Renou source, `v = 0.95` | `3.037946704e-5` | 2020 Eq. (4): `0.002952369414` |
| Renou detector, `v = 0.93` | `8.207624541e-9` | 2020 Eq. (4): `0.001260030627` |
| Renou detector, `v = 0.95` | `5.581997047e-8` | 2020 Eq. (4): `0.002568680931` |
| Noiseless Renou, `u² = 0.85` | `0.002240845347` | 2020 independent runs: about `0.0034` and `0.0055`; round 1: `0.006201116283` |
| Noiseless Renou, `u² = 1/2` | **`0` — exact witness** | Independent token/parity model; four symbols per source |

All noisy Renou rows fix `u² = 0.85`. The source sweep has no stored sample at
`v = 0.93`, and the detector sweep has none at `v = 0.90`. General Renou targets
are stored rational approximations to the quantum probabilities; certificates
refer to those stored targets. **Small positive residuals certify nearby local
distributions, and do not establish exact target locality or a new threshold.**
Beating an Eq. (4) estimate does not necessarily beat the authors' best saved
model. The round and component comparisons use different search settings and
are descriptive, rather than equal-budget comparisons.

The reference is Tamás Kriváchy et al., “A neural network oracle for quantum
nonlocality problems in networks,” *npj Quantum Information* **6**, 70 (2020),
[doi:10.1038/s41534-020-00305-x](https://doi.org/10.1038/s41534-020-00305-x).
See [RESULTS.md](RESULTS.md) for the complete sweeps and target-approximation
checks, [the technical note](note.pdf) for the mathematical constructions, and
[LITERATURE.md](LITERATURE.md) for conventions and reference checks. All
[52 target summaries](benchmarks/best-fits.csv) point to
[50 selected bundles](benchmarks/models/); independent recomputation checks
every selected certificate, including **8 exact matches**.

## Quick start

Run these five lines from the repository root with Python 3.10 or later. They
install the package, solve the Renou half endpoint, and verify its certificate
against the generated input:

```sh
python3 -m venv .venv && .venv/bin/python -m pip install -e './tool[test,plot]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m triangle_local generate renou --parameter 1/2 --out tmp/renou-half.json
.venv/bin/python -m triangle_local solve tmp/renou-half.json --sizes 4 4 4 --starts 1 --steps 20 --out tmp/renou-half-fit.json.gz
.venv/bin/python -m triangle_local verify tmp/renou-half-fit.json.gz --input tmp/renou-half.json
```

The verifier prints `"status": "exact_match"`, `"exact_match": true`,
`"L1": "0"`, and `"L2_squared": "0"`. This endpoint uses the default hybrid
method's known-control recognizer; it demonstrates an exact construction.
The [command-line example below](#command-line) performs a numerical EJM fit.
For an approximate fit, successful verification means the model and its
distance were recomputed correctly; inspect `exact_match` for target equality.

## Inspect a found model

![Independent source weights and the three parties' four response probabilities for a saved Renou detector model at efficiency 0.93](docs/figures/local-model.png)

This is the [saved Renou detector model at `v = 0.93`](benchmarks/models/117-renou-detector-93-100-hybrid-reverse.json.gz).
Each source has 12 stored symbols. The heatmaps show **all four response
probabilities**, retaining the original symbol order; grey cells have an
incident source symbol with zero weight. Stochastic responses are preserved.
The model's certified L1 distance is `3.848594064e-8`.

The [round and component figure](docs/figures/search-improvement.png) shows how
seeded search, convex polishing, and perturbations improve on the earlier
fits. [Figure captions and provenance](docs/figures/README.md) explain the
comparisons. Regenerate all three PNG/SVG pairs from committed data, without
running any optimization:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python docs/make_figures.py
```

Round 2 also provides deterministic coordinate descent, seed lifting,
bidirectional continuation, and a small NumPy response-network experiment.
The detailed interface, assumptions, and reproduction commands follow.

## Install and test

Use Python 3.10 or later from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e './tool[test,plot]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest tool -q
```

The `plot` extra is optional. Runtime dependencies are NumPy and SciPy; `test`
adds pytest and `plot` adds Matplotlib. `uv pip install --python
.venv/bin/python -e './tool[test,plot]'` is an alternative installer.
The software and original documentation are licensed under [MIT](LICENSE).
Third-party full texts and the authors' target-generator source retain their
own licenses and are excluded from the public archive. Bibliographic receipts
and source hashes identify the cited originals.

## Command line

The installed command is `triangle-local`; `python -m triangle_local` is
identical. For a published EJM source target at visibility 0.80:

```sh
.venv/bin/python -m triangle_local generate elegant-source \
  --parameter 4/5 --out examples/elegant-source-v08.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m triangle_local solve examples/elegant-source-v08.json \
  --sizes 12 12 12 --method hybrid --seed-model note --jitter 0.15 \
  --starts 1 --steps 1800 --hops 2 --polish-steps 200 --inner-steps 150 \
  --seed 20261010 --max-denominator 1000000000 \
  --out examples/elegant-fit.json.gz
.venv/bin/python -m triangle_local verify examples/elegant-fit.json.gz \
  --input examples/elegant-source-v08.json
```

JSON and gzip-compressed JSON bundles have the same schema. Verification checks
an approximate-fit certificate successfully when its model and distances are
correct; inspect `exact_match` to determine whether it certifies the target.

For a general distribution, pass its JSON or CSV file to `solve`, choose the
three source alphabet sizes, and use `--symmetry none`. The default `hybrid`
method combines joint L-BFGS-B, convex polishing, and two perturbation searches.
`--initial` accepts a saved bundle or model, automatically splitting symbols
independently when the requested alphabets are larger. `--seed-model` selects
`note`, `ring-half`, `ring-endpoint`, `parity`, or the CHSH-boundary `fritz`
construction. `--jitter` perturbs responses after lifting. The numerical hybrid
recognizer also checks known controls and product distributions before blind
optimization; recognition itself never determines the certificate status.

An exact Renou endpoint can be recovered with the default hybrid recognizer:

```sh
.venv/bin/python -m triangle_local generate renou --parameter 1/2 \
  --out examples/renou-half.json
.venv/bin/python -m triangle_local solve examples/renou-half.json \
  --sizes 12 12 12 --out examples/renou-half-fit.json
.venv/bin/python -m triangle_local verify examples/renou-half-fit.json \
  --input examples/renou-half.json
```

The old blind endpoint failure is retained in the component comparison. The
recognized or explicitly seeded result is labeled separately.

## Input format

JSON accepts a nested three-dimensional tensor or this row-major representation
with `c` varying fastest. Rational strings preserve the intended input exactly:

```json
{
  "outputs": [2, 2, 2],
  "probabilities": ["1/4", "0", "0", "1/4", "0", "1/4", "1/4", "0"]
}
```

JSON decimals mean their exact decimal values. Inputs must be nonnegative and
sum exactly to one. `--normalize-input` explicitly divides every entry by the
exact total; the bundle records this choice, and `verify --input` repeats it.

CSV uses the exact header `a,b,c,p` and zero-based integer labels. Zero cells may
be omitted. Duplicate or negative indices are rejected. Alphabet sizes are
inferred; use `--outputs A B C` on `solve` when unused trailing labels must be
preserved. External verification uses the certified output sizes, including
those unused labels. [examples/parity.csv](examples/parity.csv) is an example.

## Model and search

A finite model has three source vectors and row-stochastic response tables:

\[
A[j,k,a]=P_A(a\mid\beta=j,\gamma=k),\quad
B[k,i,b]=P_B(b\mid\gamma=k,\alpha=i),\quad
C[i,j,c]=P_C(c\mid\alpha=i,\beta=j).
\]

Its distribution is the complete finite contraction

\[
P_M(a,b,c)=\sum_{i,j,k}q_\alpha(i)q_\beta(j)q_\gamma(k)
 A[j,k,a]B[k,i,b]C[i,j,c].
\]

The source law is always the product of the three source vectors. Full
contractions remove Monte Carlo sampling error for the chosen finite model.
Source lifting divides each original symbol's weight among copies and pulls
back the adjacent response tables. It preserves the original distribution and
never adds a global mixture flag.

Available methods:

| Method | Use and scope |
|---|---|
| `hybrid` | Joint L-BFGS-B, convex polishing, then perturbed searches; best evaluated model retained. Requires untied parameters and `l2`. |
| `convex` | Alternates one response table or source vector while fixing the other factors. `l2` uses simplex projected accelerated gradient; `l1` uses a sparse linear program. Requires untied parameters. |
| `deterministic` | Enumerated one-row discrete response updates for squared Euclidean loss, alternating with convex source-weight updates. Responses remain deterministic. |
| `lbfgs` | Joint softmax optimization, including tied symmetry modes. |
| `least-squares` | Trust-region least squares with the full analytical output Jacobian; `l2` only. |
| `adam` | Joint softmax Adam; default learning rate 0.08. |
| `alternating` | Short L-BFGS-B solves over softmax blocks, including tied symmetry modes. |
| `em` | Posterior-count updates preserving independent sources; requires untied parameters and `kl`. |
| `neural` | Experimental two-hidden-layer ReLU response networks of width 24 on independent uniform grids, trained with NumPy Adam. Produces ordinary finite tables. Requires untied parameters and `l2`. |

A block subproblem is convex because its factor enters linearly when the other
five factors are fixed. L2 block solves stop at an inner iteration cap and are
not claimed to be exact global block solutions. Updates are accepted only when
the actual model objective decreases. The deterministic method evaluates every single-row output choice using
floating-point costs and accepts improvements above its numerical tolerance.
It does not optimize the entire discrete party table globally. `--fixed-weights` fixes weights in convex/deterministic updates;
joint L-BFGS-B in `hybrid` still optimizes them.

`--inner-steps` controls the projected-gradient cap, `--polish-steps` controls
hybrid polishing sweeps, and `--hops` controls further perturbation searches.
Perturbed candidates are deduplicated by their output distribution rounded to
seven decimal places for diagnostics. Distinct latent representations are not
claimed equivalent, and deduplication never removes a better candidate.
`--optimizer-tolerance` applies to the numerical objective; the separate
`--tolerance` is an exact L1 certificate tolerance.

Losses are `l1` (sum absolute error), `l2` (sum squared error), and `kl`
(target-to-model relative entropy with a small numerical floor). Benchmark
Euclidean distance is the square root of the L2 objective, following the 2020
paper's Eq. (3).

Symmetry options share model parameters before fitting:

| Mode | Construction |
|---|---|
| `none` | Independent unrestricted source vectors and party tables; appropriate for Renou's cyclic family. |
| `exchangeable` | Equal iid sources and one shared response symmetric in its inputs; guarantees S3 party symmetry. |
| `outcome-equivariant` | Uniform sources on ordered distinct output-label pairs, a shared cyclic response, and stabilizer-fixed output choices. Sizes must be `m(m-1)`. Includes the note's witness. |
| `pair-equivariant` | Extends the previous family with auxiliary source bins and shared optimized bin weights. Sizes are multiples of `m(m-1)`. |
| `color-equivariant` | Sources carry an output color and an auxiliary bin. A shared response chooses incident colors or uniformly distributes an orbit mass over unused colors. Sizes are multiples of `m`, with `m >= 3`. |

In symmetry modes the three source sizes must agree, and the three output
counts must agree. The pair modes support
three through five outcomes. Simultaneous outcome relabeling plus cyclic party
symmetry makes the three-party output tensor depend only on its equality pattern,
which gives full S3 party symmetry. These are restricted model classes, not
complete descriptions of the symmetric local set. New orbit modes rationalize
orbit masses before dividing them among equivalent labels, preserving symmetry
exactly. The benchmark finds the best EJM fits outside the tied families.

## Certificates and verifier guarantee

A `triangle-local-bundle/v1` contains:

- `target`: output sizes and exact rational probabilities;
- `model`: a `triangle-local-model/v1` with `weights.alpha/beta/gamma` and
  `responses.A/B/C`, stored as rational strings in the axes above;
- `certificate`: a `triangle-local-certificate/v1` with model/target hashes,
  exact normalization, exact L1, squared L2, maximum error, total variation,
  all 64 probabilities for four-output models, and exact party-symmetry status;
- `search`: optimizer settings, seeds, evaluated distances, and run diagnostics;
- `source_input`, for CLI results: the input hash, normalization choice and
  available family metadata.

On disk, rational integers longer than twelve digits use underscore digit
groups. The rational parser and bundle loader remove the group marks before
`Fraction` parsing, canonical hashing and certificate comparison, including on
Python 3.10. This
preserves every exact value and the original model/target hashes. Numerical
search diagnostics and CSV distances are displayed to ten significant digits;
the full exact distances remain in the rational certificates. CSV SHA256
digests use eight-character hexadecimal groups separated by colons; the summary
reader removes these separators before using the digest as an identifier.

The verifier checks every simplex, lifts rational factors to common integer
denominators, and recomputes the entire finite contraction with Python integers.
It independently checks party symmetry and does not trust symmetry metadata.
Hashes bind the certificate to the canonical rational model and target. The
checker is tested software; it is not a formally verified implementation.

| Certificate status | Meaning |
|---|---|
| `exact_match` | The finite rational model exactly reproduces the stored rational target. |
| `nearby_local_model` | The exact L1 distance is within the stated tolerance. It certifies that nearby model. |
| `approximate_fit` | The candidate is valid and its certified distance exceeds the tolerance. |

A positive residual gives an upper bound on distance to the unrestricted local
set. It does not prove target nonlocality, a locality threshold, or optimizer
completeness. Small numerical residuals alone do not establish exact membership.
`--round` generates a separate deterministic candidate with a fresh certificate.
Rounding and rationalization can change both distance and symmetry.

Ordinary rows first use bounded-denominator continued fractions, then
largest-remainder quantization when normalization would otherwise fail.
Orbit modes quantize orbit masses and divide them among equivalent outputs;
entry denominators can therefore be small integer multiples of the requested
cap. Changing a cap can change distances non-monotonically.

## Reproduce the measurements

The sequential benchmark executes all component comparisons, forward and
reverse scans, boundary polishing, exact controls, and the Fritz reference
construction. Run one optimizer process with one numerical-library thread. The archived
measurements used two threads; thread counts and library versions can change
the optimization trajectory, so reproduction need not select an identical model:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/run.py --out-dir benchmarks \
  --seed 20261010 --starts 2 --steps 1800 --polish 200
.venv/bin/python benchmarks/verify.py
.venv/bin/python benchmarks/independent_check.py --out benchmarks/independent-verification.json
.venv/bin/python benchmarks/plot.py
.venv/bin/python benchmarks/write_round2_results.py
.venv/bin/python benchmarks/write_note.py
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

Every run checkpoints `all-fits.csv` and its complete candidate bundle in
`benchmarks/runs/`. These intermediate files are ignored by Git and regenerable.
The refinement passes add symmetric fallbacks, further perturbations,
adaptive larger alphabets at the detector boundary, and continuation from
the fitted Renou critical-region point, followed by continuation across the noiseless scan in both directions. A small retained round-1 ideal EJM
seed in `examples/` prevents regression at the noiseless endpoint.
The final selection promotes only the best target candidates and controls to
`benchmarks/models/` as compressed JSON. `best-fits.csv` identifies the selected
file, including reuse across identical endpoints; `ablation.csv` isolates the
component comparisons. `environment.json` records settings, wall time, thread
limits, and process peak RSS. `verification.json` records independent checks.
Timing excludes the exploratory probes and does not compare hardware-matched
training with the original neural-network code.

[benchmarks/round1/](benchmarks/round1/) retains the earlier numeric summaries.
The old sweep can be regenerated into a separate directory with
`python -m triangle_local.benchmarks --out-dir tmp/round1`; its saved round-1
numbers were measured with the earlier implementation.

The 2020 reference curves evaluate Eq. (4) at the reported transition/angle
pairs: EJM source `(0.80, 50 degrees)`, EJM detector `(0.86, 60 degrees)`,
Renou source `(0.89, 6 degrees)`, and Renou detector `(0.91, 6 degrees)`.
They are approximate analytic curves, not digitized neural-network measurements.
The two independently reported Renou distances at `u² = 0.85`, about 0.0034 and
0.0055, are compared explicitly in RESULTS.md. The Fritz CSV visibility labels are rounded to eight significant digits; the
`0.70710678` row represents a numerical evaluation of `1/sqrt(2)`. Bundles
record the exact rational correlator used to define each stored target.
Fritz's clipped-CHSH candidate
has distance `max(v/sqrt(2) - 1/2, 0)/4`; it is an analytical comparator.

Targets include `note`, `parity`, `fritz`, `elegant-source`, `elegant-detector`,
`elegant-global`, `renou`, `renou-source`, and `renou-detector`. `--parameter`
is visibility/efficiency, `u²` for `renou`, and CHSH correlator magnitude
`t = v/sqrt(2)` for `fritz`. `--u-squared` fixes the Renou shape in noisy scans.
The EJM physical noise families are distinct exact rational polynomials.
Renou square-root terms and its six-qubit noisy contraction require rational
approximations at general parameters; the denominator cap is one trillion.
Certificates refer to those stored targets. The generator was cross-checked
against the authors' pinned implementation; receipts and conventions are in
LITERATURE.md.

The exact symmetric witness has `P(A=B=C) = 41/144`, with entry counts
`123/1728`, `19/1728`, and `23/1728`. Its distribution and exact certificate are independently reproducible.

Build the explicit public source archive with `python benchmarks/package_public.py`. It includes source, tests, small selected data, receipts, figures, and the note; local task briefs, logs, environments, intermediate models and downloaded research full texts are omitted.
