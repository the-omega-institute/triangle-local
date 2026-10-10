# Matched CPU comparison

This is the frozen Round 4 experiment: **three solvers × three common targets × ten cold starts**, with 10- and 60-second checkpoints. It supersedes the earlier two-run pilot for performance reporting. Selected best development witnesses remain separate in [RESULTS.md](RESULTS.md).

## Results and success rule

| Common target | Method | 10 s L2 median [Q1, Q3] | Success | 60 s L2 median [Q1, Q3] | Success |
|---|---|---:|---:|---:|---:|
| EJM source, v=0.80 | triangle-local | 0.0109 [0.0063, 0.0125] | 0/10 | 0.00544 [0.00431, 0.00658] | 0/10 |
| EJM source, v=0.80 | LHV-Net-2020 (small CPU configuration) | 0.0722 [0.0669, 0.0767] | 0/10 | 0.0511 [0.0506, 0.053] | 0/10 |
| EJM source, v=0.80 | localmodels | 0.11 [0.103, 0.118] | 0/10 | 0.0518 [0.0494, 0.0538] | 0/10 |
| EJM detector, v=0.86 | triangle-local | 0.017 [0.014, 0.0233] | 0/10 | 0.00674 [0.00508, 0.0075] | 0/10 |
| EJM detector, v=0.86 | LHV-Net-2020 (small CPU configuration) | 0.0703 [0.0668, 0.0745] | 0/10 | 0.0499 [0.0488, 0.0505] | 0/10 |
| EJM detector, v=0.86 | localmodels | 0.103 [0.102, 0.105] | 0/10 | 0.0496 [0.0475, 0.0512] | 0/10 |
| Renou detector, u²=0.85, v=0.93 | triangle-local | 2.86e-05 [1.22e-05, 7.64e-05] | 10/10 | 3.91e-06 [1.67e-06, 5.59e-06] | 10/10 |
| Renou detector, u²=0.85, v=0.93 | LHV-Net-2020 (small CPU configuration) | 0.0279 [0.0268, 0.0295] | 0/10 | 0.0172 [0.0154, 0.0188] | 0/10 |
| Renou detector, u²=0.85, v=0.93 | localmodels | 0.122 [0.0258, 0.216] | 0/10 | 0.00792 [0.00715, 0.00919] | 0/10 |

Cells are median [Q1,Q3]. Success means L2 ≤ **0.001**, declared before any counted run. A neural success additionally requires the upper endpoint of its approximate 95% held-out interval to be at or below the threshold. There were 0 threshold-straddling neural intervals. Ten trials per cell describe this deployment regime; they do not give a stable population ranking. Checkpoints within one run are paired, not additional independent runs.

## One target artifact per problem

Each solver loads the same little-endian float64 `.npy` array with 64 entries. The flattened index is **16a+4b+c**, with a,b,c in {0,1,2,3} and c varying fastest. Companion JSON declares the exact physical parameters, ordering, dtype, file hash and raw array-byte hash. Decimal parameters mean exact rationals. EJM source uses v=4/5; EJM detector uses v=43/50; Renou detector uses u²=17/20 and v=93/100. Detector noise is the independent channel `K_v ⊗ K_v ⊗ K_v`, with `K_v = vI + (1−v)J/4`, not a global white-noise mixture. Renou source noise uses `ρ_v = v|φ⁺⟩⟨φ⁺| + (1−v)I₄/4` on each independent edge, with `|φ⁺⟩ = (|00⟩+|11⟩)/√2` and network state `ρ_v ⊗ ρ_v ⊗ ρ_v`. This is the convention for the selected v=0.89 source-noise witness; detector noise acts on the outputs instead.

| Target | SHA-256 of supplied array file |
|---|---|
| EJM source, v=0.80 | `b006ed64fab04b9bcbb6df6fc39fd58e6e3f848676e63fa13779ad7f665254f1` |
| EJM detector, v=0.86 | `3ed71e63efb2776ca2ade99a6f3df16020547f839641549a2955e0f72e6062e7` |
| Renou detector, u²=0.85, v=0.93 | `33670e65cb80be3c2fa4ddd25f3029a1176323285f51d82d3b475f921268c5f4` |

The public target generator is independently cross-checked against these actual arrays. Its `np.np.sqrt` typo correction is archived as an [explicit downstream patch](benchmarks/round4/patches/lhv-net-target-spelling.patch), applied to a temporary copy by [check_reference.py](benchmarks/round4/check_reference.py). These point comparisons support identity checks; the rigorous original-target enclosures come from separate interval arithmetic.

## Metric and checkpoint evaluation

Every reported L2 is recomputed from a saved 64-entry output distribution in float64. For all finite outputs, the full contraction is independently repeated from saved source vectors and response tables. Localmodels' internal error is RMSE: `sqrt(SSE/64)`. The outcome audit explicitly checks **L2 = 8 RMSE** on every localmodels and triangle-local checkpoint. It never treats the upstream error field as an L2 result. Native training losses may differ.

Finite outputs retain the best feasible candidate whose score completed before each budget. Localmodels may evaluate infeasible trust-region trial points; these are excluded, with counts recorded. All source and response entries must be nonnegative and row sums within 10⁻¹² of one. Neural checkpoints minimize L2 on a fixed 4,096-sample validation set, evaluated every 50 training steps (and after the first). Validation and copying selected weights are charged to search. A checkpoint completed after a deadline cannot improve the corresponding score.

After both neural checkpoint choices are frozen, each is evaluated with **1,048,576 fresh held-out samples**, in 32 blocks of 32,768. Evaluation samples are independent of training and validation; no held-out score is used for selection. The reported and plotted score is the **raw norm `s = ||Q̂_N−T||₂`**, without variance correction or truncation. For B=32 block means `X̄_b`, the saved variance estimate is `V̂ = Σ_b ||X̄_b−Q̂_N||₂² / [B(B−1)]` (equivalently the sum of entrywise sample variances with `ddof=1`, divided by B). The separate diagnostic `sqrt(max(0, s²−V̂))` truncates the corrected squared residual at zero; it is **not** the table or figure score. The estimated RMS Monte Carlo error is `sqrt(V̂)`.

Each of 2,000 block-bootstrap resamples draws 32 block means with replacement and recomputes the raw norm `s*` of their average minus T. With `q_p` the empirical p-quantile of these norms (NumPy's default linear interpolation), the approximate 95% **basic** interval is `[max(0, 2s−q_0.975), max(0, 2s−q_0.025)]`. Both endpoints are clipped at zero. These intervals quantify sampling uncertainty, not rigorous locality bounds. Training, validation, restart, and held-out streams use domain-separated SeedSequence keys. Budgets are analyzed separately; the two checkpoints are paired observations and are not counted as independent replications.

Median held-out evaluation time was 0.87 s per checkpoint; RMS Monte Carlo error ranged from 0.000419 to 0.000733, with median confidence-interval half-width 0.000226. Finite distributions are contracted fully rather than artificially sampled. [Raw outcomes](benchmarks/round4/outcomes.csv) expose all point estimates, intervals, seeds and threshold flags.

## Cold starts, architectures and stopping rules

- **triangle-local:** unrestricted 7×7×7 finite model; random normal softmax initialization; direct hybrid routine bypasses target recognition. One initial start, 400 L-BFGS iteration cap, 120 convex sweeps with 120 inner steps, two perturbation searches; restart from a fresh random seed when a cycle completes. Zero target-residual stopping tolerance. Best feasible candidate retained across that run.
- **LHV-Net-2020:** [public source at the measured revision](https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks/tree/9460fe5340016e9a2f954613a2c07e633449d455); three tanh hidden layers of width 16 per party, softmax outputs, uniform independent continuous sources, batch size 256, Adadelta learning rate 2, native KL objective. TensorFlow 2.16.2 on CPU, float32 network operations and float64 final contraction. The eager compatibility wrapper replaces removed Keras generator methods and retains the public response routing and architecture. This is a declared small CPU configuration, not the larger published EJM experiment or a GPU benchmark.
- **localmodels:** [public source at the measured revision](https://github.com/mariofilho281/localmodels/tree/28fd13ec30761d81eaa434a8545e02db5fe51ed7); unrestricted 7×7×7 model, public random Dirichlet initialization and trust-constr optimizer, one trial per call, outer tolerance zero, restart if the call returns. Native sum-of-squares objective. Numerical optimizer stopping rules remain those of the pinned public implementation.

There are no target-specific fitted or analytic seeds, endpoint recognizers, neighboring-target starts, or development witnesses in any arm. Perturbing the current run's candidate is charged to that run. Native model dimensions differ between neural and finite approaches and are stated above; common targets, budgets, start policy and evaluation metric are matched.

## Hardware and costs

Apple M2 Pro, 16 GiB memory, macOS, Python 3.11.15, NumPy 1.26.4 and SciPy 1.17.1. The exact OS record is in the protocol. One solver process runs at a time; OMP, OpenBLAS, Accelerate, MKL, NumExpr and TensorFlow numerical thread settings are all 1; GPUs are disabled. The orchestrator is idle while a solver runs. Jobs have a shuffled frozen order.

Timing includes initialization, optimizer setup, optimization, wrapper scoring/feasibility checks and checkpoint selection after target/common-library loading. Instrumentation costs are included, not separately subtracted. Small method-specific lazy imports occur inside the finite initialization timer. Final held-out testing, serialization, rationalization and certification are additional recorded costs. Only candidates fully scored before the budget are eligible; measured process time can exceed 60 s slightly while the last operation returns. The serial campaign occupied **1.56 h** including startup and held-out testing. The excluded preflight seed audit took about 13.5 minutes in addition. Three targets preserve ten repetitions per cell and leave room for exact checking and the interoperability experiment within the approximately three-hour compute cap. No 300 s arm was requested or extrapolated, and no optional fourth target was added.

## Certificate production and interoperability

Search scores are operational L2 to the common floating-point artifact. After timing, every finite checkpoint from both finite solvers is rationalized at denominator cap 10⁹ and checked exactly against the corresponding rational target. The float64 conversion of that rational target is checked for entrywise equality with the common array. Target enclosure δ and upward-rounded U are then added. The outcome table reports conversion time, checking time, stored-target residual, δ and U separately from search.

| Object | Accuracy | Additional measured cost |
|---|---:|---:|
| Continuous neural checkpoint | held-out L2 0.0165, 95% interval [0.0163, 0.0167] | 60.02 s training/selection; 0.87 s test |
| Extracted 4³ finite model | U ≤ 0.03136 | 0.288 s extraction |
| Extracted model + polishing | U ≤ 0.0006423 | 2.004 s polishing |
| Random 4³ controls + same polishing, n=10 | median U 0.001602; range [0.0005051, 0.008027] | 2 s budget each |

This conversion uses the predeclared repetition-zero Renou checkpoint, verified by saved-weight hash, and full independent Cartesian supports. It is a new finite model. Ten random finite starts receive the same two-second polishing budget. [Interoperability records](benchmarks/round4/interoperability.json) retain all random controls and exact bundles; no general neural-initialization advantage is inferred from one example.

## Reproduction and provenance

The [frozen protocol](benchmarks/round4/protocol.json), [exact executed runner](benchmarks/round4/frozen_runner.py), [90 run receipts](benchmarks/round4/runs/), [180 outcome rows](benchmarks/round4/outcomes.csv), saved factors/weights, and [verification receipt](benchmarks/round4/verification.json) are retained. The frozen runner's content hash equals the one in the protocol. Each receipt includes the actual target and protocol hashes, run seed, checkpoint completion time, package versions and process peak RSS. Smoke checks use repetition 99 and are excluded from all counted results. Preliminary runs with the earlier sequential restart-seed rule were also excluded in full; the seed-audit summary records their additional compute cost. The final protocol uses domain-separated streams and retains the original targets, settings, threshold and budgets.

Install the package's test, plot and bounds extras, obtain the two external repositories at the pinned revisions, and set `LHV_NET_SOURCE` and `LOCALMODELS_SOURCE` to their roots. Use a Python environment with TensorFlow 2.16.2. The runner's `prepare` action freezes fresh target artifacts and settings; `run` launches the sequential jobs and resumes existing completed records without replacing them. Use an isolated copy for a fresh campaign, remove only that copy's Round 4 target/run/protocol output files, and run `prepare` there. Run the audit and report commands documented in the README after completion.

The prior two-run records under `benchmarks/external/` are historical pilot data and are excluded from all Round 4 tables. The 2026 adaptive neural toolbox is literature context, not a measured arm. The local archive and manifest identify all unpushed results; no new published revision or release is claimed.
