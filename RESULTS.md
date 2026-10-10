# Results: certificates and a matched comparison

The exactly symmetric construction has **P(A=B=C)=41/144 > 1/4** and equal probabilities within all 4 all-equal, 36 exactly-two-equal, and 24 all-distinct cells. It directly addresses the exact-symmetry question posed by Bäumer et al.; the point is exact symmetry, not surpassing their approximately symmetric numerical examples.

## Three-way cold-start comparison

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

Each cell has ten independent starts. The same runs supply the 10- and 60-second checkpoints. The declared success threshold is L2 ≤ 10⁻³; for neural outputs success requires the upper end of the held-out approximate 95% interval to meet it. All scores are recomputed from saved outputs against the **same hashed target array** for each problem. No target-specific seeds or continuation from other targets are used. Localmodels' internal RMSE is explicitly converted and checked: **L2 = 8 × RMSE** for 64 entries. The Renou shape is exactly 17/20. Neural point estimates are the raw norm `s = ||Q̂_N−T||₂`, without variance correction or truncation. The approximate 95% basic block-bootstrap interval is `[max(0, 2s−q_0.975), max(0, 2s−q_0.025)]`, where q denotes quantiles (linear interpolation) of 2,000 raw norms obtained by resampling the 32 held-out block means. The separate zero-truncated variance-corrected diagnostic is not plotted; [BENCHMARK.md](BENCHMARK.md#metric-and-checkpoint-evaluation) gives its formula.

![Three-method comparison with individual runs](docs/figures/baseline-time.png)

**What the tool does better, and what it does not.** The tool's demonstrated advantage is the production of portable rational witnesses with exactly checked locality, rigorous error bounds to the intended target, and a compact independent checking path. In this one-thread CPU configuration, its 60-second median L2 values were 0.00544, 0.00674, 3.91e-06 on EJM source, EJM detector, and Renou detector respectively, with 10/30 runs meeting the declared accuracy threshold; the measured counts were 0/30 for the small LHV-Net-2020 configuration and 0/30 for localmodels. These counts describe three selected targets and ten starts each. They do not establish a universal optimizer ranking, a comparison with the larger published neural configurations or the 2026 adaptive toolbox, exact locality of positive-error targets, or optimality of the reported fits. The neural extraction example demonstrates interoperability; a single checkpoint does not establish that neural initialization usually helps.

The table is descriptive and is not a hypothesis test or global solver ranking. Raw values, model factors, checkpoint times, neural weights, evaluation uncertainty, and success flags are in [outcomes.csv](benchmarks/round4/outcomes.csv) and the [run records](benchmarks/round4/runs/). The complete contract and resource scope are in [BENCHMARK.md](BENCHMARK.md).

## Selected development witnesses, outside the timing comparison

| Target | Stored-target residual √E | Target enclosure δ | Certified U |
|---|---:|---:|---:|
| Exactly symmetric construction | 0 | 0 | 0 |
| Renou noiseless, u²=0.50 | 0 | 0 | 0 |
| EJM source, v=0.80 | 5.77e-11 | 0 | 5.774e-11 |
| EJM detector, v=0.86 | 1.51e-09 | 0 | 1.51e-9 |
| Renou source, v=0.89 | 3.27e-08 | 5.227e-13 | 3.27e-8 |
| Renou detector, v=0.93 | 8.21e-09 | 2.238e-14 | 8.208e-9 |
| EJM noiseless | 0.0483 | 0 | 0.04835 |
| Renou noiseless, u²=0.85 | 0.00224 | 2.782e-14 | 0.002241 |

The checker establishes `Q ∈ L_triangle` and exact `E = ||P_stored-Q||₂²`. A validated interval enclosure supplies `δ >= ||P-P_stored||₂`; the principal bound is the upward-rounded `U >= sqrt(E)+δ`. Positive U proves a nearby local model, not exact target locality, nonlocality, or optimality. Exact stored-target equality establishes intended-target locality only when target identity is also exact. EJM's rational polynomials have δ=0. The Renou bounds use outward-rounded intervals, not high-precision point discrepancies. Renou source noise uses `ρ_v = v|φ⁺⟩⟨φ⁺| + (1−v)I₄/4` on each independent edge, with `|φ⁺⟩ = (|00⟩+|11⟩)/√2` and network state `ρ_v ⊗ ρ_v ⊗ ρ_v`. This is the convention for the selected v=0.89 source-noise witness; detector noise acts on the outputs instead. [All original-target bounds](benchmarks/target-bounds.json) retain the exact target identity and the source of every selected witness.

![Certified local approximations](docs/figures/headline.png)

At visibility v=1, source and detector noise lead to one noiseless target for each quantum family. The headline shows the best eligible witness once; different continuation-path residuals are outcomes for that same target. Changing Renou u² to 1/2 or 1 is a change of measurement parameter, not the same operation as setting v=1 at u²=0.85.

## Compact witness

The Renou detector witness at u²=0.85, v=0.93 trims exactly from (12,12,12) to **(8,9,2)**. Its original-target bound is **U ≤ 8.208e-9**, its compressed bundle is **6,007 bytes**, and its median exact check takes **3.63 ms**. The timer starts with decompressed JSON and parsed target already in memory and includes rational model parsing, simplex checks, integer contraction, distances, symmetry and hashes. Cardinality minimality is not asserted.

![Compact witness summary](docs/figures/local-model.png)

For 0 ≤ v_L ≤ v_H with v_H>0, applying K_(v_L/v_H) to all response rows produces another exactly local model Q_L. The detector-noise channel gives `||Q_L-P(v_L)||₂ ≤ ||Q_H-P(v_H)||₂ ≤ U_H`. Exact high-visibility models therefore propagate exact equality; positive-error witnesses propagate controlled approximations. This does not extend automatically to independent source noise.

## Native neural checkpoint → finite certificate → polishing

| Object | Accuracy | Additional measured cost |
|---|---:|---:|
| Continuous neural checkpoint | held-out L2 0.0165, 95% interval [0.0163, 0.0167] | 60.02 s training/selection; 0.87 s test |
| Extracted 4³ finite model | U ≤ 0.03136 | 0.288 s extraction |
| Extracted model + polishing | U ≤ 0.0006423 | 2.004 s polishing |
| Random 4³ controls + same polishing, n=10 | median U 0.001602; range [0.0005051, 0.008027] | 2 s budget each |

The neural checkpoint was fixed in advance as repetition zero at 60 seconds on the Renou detector target. Its source revision, saved-weight hash, native held-out estimate, extraction supports, costs and exact before/after bundles are in [interoperability.json](benchmarks/round4/interoperability.json). Each source has its own uniform four-point support; the full Cartesian contraction enforces independence. The extracted finite model is distinct from the continuous network. The polished extraction lies below the random-control median in this example. Ten random controls receive the same two-second polishing budget; this one neural checkpoint does not establish typical initialization benefit.

All 120 finite timed outputs from **both** triangle-local and localmodels were also rationalized, exactly checked, and independently audited after timing. Their conversion and checking costs, stored-target residuals, δ and U are separate columns in the outcome table. A native floating-point score is never labeled a certificate.

## Illustrative earlier configurations

| Configuration | Stored-target residual | Additional seconds | Total seconds |
|---|---:|---:|---:|
| Blind L-BFGS, 12 symbols | 0.0198 | — | 4.57 |
| Seeded L-BFGS | 0.00543 | — | 3.76 |
| Seeded L-BFGS + fixed-weight polish | 0.00163 | 5.80 | 9.56 |
| Seeded L-BFGS + free-weight polish | 0.000407 | 8.23 | 11.99 |
| Seeded hybrid + perturbations | 5.06e-10 | — | 27.98 |

Both polishing rows branch from the same seeded L-BFGS model. This is a descriptive earlier configuration study, not an equal-budget ablation. The complete historical records remain in [all-fits.csv](benchmarks/all-fits.csv), [best-fits.csv](benchmarks/best-fits.csv), and [ablation.csv](benchmarks/ablation.csv). Historical fitted 2020 curves are not measured baselines and are omitted from the main figures.

## Validation and artifact scope

68 tests passed. The native-output audit checked 180 outcomes and their deadline/target hashes; the independent rational audit checked 120 new finite checkpoint certificates and the selected development collection. The content-hash manifest, pinned external revisions, source snapshot, patch, and local archive provide reproducibility without claiming that these unpushed changes have been published. No email was sent and no repository was pushed.
