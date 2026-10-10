**Round 2 combines local-model seeds, convex factor updates, and perturbation searches with exact rational certificates.** The benchmark now fits all four reported noise-transition targets with certified Euclidean distances below four times ten to the minus eight. The tables below report the final candidates for all four reported noise transitions and preserve the remaining gaps. Small residuals certify nearby local distributions; exact target locality requires `exact_match`.

| Family | 2020 reported transition | Round 1 Euclidean error | Round 2 certified Euclidean error | Round 2 certified L1 |
|---|---:|---:|---:|---:|
| EJM source | 0.80 | 0.00537513 | 5.77334e-11 | 3.6955e-10 |
| EJM detector | 0.86 | 0.010883 | 1.50971e-09 | 8.68419e-09 |
| Renou source | 0.89 | 0.00107392 | 3.2699e-08 | 2.04649e-07 |
| Renou detector | 0.91 | 0.000828942 | 5.90947e-09 | 3.86994e-08 |

The 2020 references are Tamás Kriváchy et al., “A neural network oracle for quantum nonlocality problems in networks,” *npj Quantum Information* **6**, 70 (2020), [doi:10.1038/s41534-020-00305-x](https://doi.org/10.1038/s41534-020-00305-x). Equation (3), p. 3, defines Euclidean distance. Equation (4), p. 3, gives the approximate reference curve. Figure 4 and p. 5 give EJM source/detector estimates `(0.80, 50°)` and `(0.86, 60°)`. Figure 5 and p. 5 give Renou estimates `(0.89, 6°)` and `(0.91, 6°)` at `u²=0.85`, and independently reported noiseless errors around **0.0034** and **0.0055**. The paper explicitly describes the Renou estimates as cruder than the EJM estimates, with local optima and variation between independent runs. Reference columns evaluate Eq. (4) here; they are not digitized neural-network measurements.

The base sweep uses seed 20261010, two starts, at most 1800 joint iterations, and 200 convex sweeps with 150 inner projected-gradient steps. It pools single fitted models as warm starts, runs every family in both directions, and polishes the four boundary targets for up to 1200 sweeps. Component comparisons use separate recorded caps. The follow-up adds symmetric fallbacks, four perturbation searches at difficult targets, and numerical Fritz checks. The final pass adds the 24-symbol detector-boundary fit, continuation from the fitted Renou critical-region point, and retention/refinement of the earlier ideal EJM seed. Complete settings and diagnostics are in the bundles and `all-fits.csv`.

At squared-loss block updates, one source or party factor is linear while the other five are fixed. The resulting simplex quadratic subproblem is convex, but its iterative solve can hit its cap. L1 blocks use a sparse LP. Deterministic descent enumerates single-row output choices using floating-point costs and a numerical acceptance tolerance. The hybrid pipeline retains the best evaluated model and records duplicate output distributions; it never constructs a global mixture flag.

**Component comparison at EJM source visibility 0.80.** These are descriptive experiments with method-specific restart/iteration caps, not equal-budget statistical estimates. Both polishing variants start from the same seeded L-BFGS candidate and report additional runtime.

| Configuration | Sizes | Certified Euclidean error | Search seconds |
|---|---|---:|---:|
| blind-lbfgs-none-12 | 12x12x12 | 0.0198499 | 4.57 |
| blind-lbfgs-none-24 | 24x24x24 | 0.0121058 | 32.02 |
| blind-least-squares-none-12 | 12x12x12 | 0.00805933 | 3.21 |
| blind-deterministic-none-12 | 12x12x12 | 0.011553 | 1.31 |
| blind-deterministic-none-24 | 24x24x24 | 0.00835333 | 11.19 |
| blind-lbfgs-outcome-equivariant-12 | 12x12x12 | 0.00537632 | 0.50 |
| blind-lbfgs-pair-equivariant-24 | 24x24x24 | 0.00537512 | 2.84 |
| blind-lbfgs-color-equivariant-24 | 24x24x24 | 0.0342312 | 2.27 |
| blind-neural-none-24 | 24x24x24 | 0.0170738 | 23.83 |
| seeded-lbfgs-12 | 12x12x12 | 0.00543094 | 3.76 |
| polish-fixed-weights | 12x12x12 | 0.00162916 | 5.80 |
| polish-free-weights | 12x12x12 | 0.000407216 | 8.23 |
| seeded-hybrid-hops-12 | 12x12x12 | 5.06378e-10 | 27.98 |

Seed lifting, convex polishing with free source weights, and perturbations supply the strongest improvement at this target. Increasing cardinality alone and the tested symmetric extensions do not remove the gap. Deterministic descent and least squares improve some blind starts but remain weaker than the seeded hybrid result. The small NumPy response network is also weaker in these trials. It uses two hidden ReLU layers of width 24 on finite grids, so this is not a rerun of the paper’s depth-5, width-30 EJM architecture or its Monte Carlo training.

**EJM source and independent detector noise.** Rational visibility gives exact target polynomials. A missing table cell means that family was not sampled at that visibility.

| v | Source certified distance | Detector certified distance | 2020 source Eq. (4) | 2020 detector Eq. (4) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0.5 | 1.77649e-10 | 1.30388e-10 | 0 | 0 |
| 0.7 | 3.12665e-09 | 2.16273e-10 | 0 | 0 |
| 0.75 | 1.6322e-11 | - | 0 | - |
| 0.78 | 7.65001e-11 | - | 0 | - |
| 0.79 | 8.22094e-11 | - | 0 | - |
| 0.8 | 5.77334e-11 | 6.11106e-11 | 0 | 0 |
| 0.81 | 0.000615834 | - | 0.00211196 | - |
| 0.84 | 0.00763466 | 9.86652e-11 | 0.00862839 | 0 |
| 0.85 | - | 4.7672e-10 | - | 0 |
| 0.86 | 0.0128866 | 1.50971e-09 | 0.0131252 | 0 |
| 0.87 | - | 0.00473399 | - | 0.00329088 |
| 0.9 | 0.0221296 | 0.012609 | 0.0224923 | 0.0135838 |
| 1 | 0.0483442 | 0.0483442 | 0.0481638 | 0.052681 |

The reference estimates are approximate fitted curves. A candidate below an Eq. (4) value beats that estimate at that point, not necessarily the authors’ best saved model. Positive errors above a transition can still include optimization or finite-alphabet error. The ideal EJM distance is 0.0483442, about 0.37% above the source Eq. (4) estimate 0.0481638; the detector estimate is 0.052681. These compare approximate analytic curves, not unavailable raw neural-network fits. The empirical comparison does not establish a new locality threshold. The EJM ideal endpoint is the same exact target in both noise families, so a common best candidate is reused by target hash.

**Noiseless Renou scan.**

| u² | Round 1 distance, when sampled | Round 2 certified distance | Certified L1 | Exact stored target |
|---:|---:|---:|---:|:---:|
| 0.5 | 0 | 0 | 0 | yes |
| 0.6 | - | 0.00732373 | 0.027957 | no |
| 0.63 | 0.0374563 | 0.00801052 | 0.0319322 | no |
| 0.7 | - | 0.00159344 | 0.00569924 | no |
| 0.75 | - | 2.40227e-08 | 1.46559e-07 | no |
| 0.785 | 0.0239539 | 5.51535e-09 | 2.50721e-08 | no |
| 0.8 | - | 0.000763773 | 0.00365829 | no |
| 0.85 | 0.00620112 | 0.00224085 | 0.0107432 | no |
| 0.9 | - | 0.00227656 | 0.0107802 | no |
| 0.95 | - | 0.00124114 | 0.00584299 | no |
| 1 | 0 | 0 | 0 | yes |

At `u²=0.85`, the certified distance **0.00224085** is below both reported values. The decimal sample `0.785` is not the exact critical parameter of the original theorem. A remaining residual there cannot be interpreted as an exclusion. Blind L-BFGS still returns a nonzero residual at the known local half endpoint; the explicit token/parity control and the default hybrid recognizer recover its exact rational model. This separates a corrected default behavior from the persistent limitation of blind search.

**Renou independent noise, fixed u²=0.85.** General square-root probabilities are quantized rational targets; detector noise is exact convolution of the stored base target. Independent 70-digit evaluations give Euclidean target-approximation errors of 5.23e-13 at source v=0.89, 2.10e-14 at detector v=0.91, and 2.78e-14 at noiseless u²=0.85. These numerical target checks are separate from exact rational certificate verification.

| Family | v | Certified Euclidean distance | Certified L1 | 2020 Eq. (4) estimate |
|---|---:|---:|---:|---:|
| renou-source | 0 | 0 | 0 | 0 |
| renou-source | 0.8 | 1.49682e-09 | 8.18793e-09 | 0 |
| renou-source | 0.87 | 3.66785e-08 | 7.66878e-08 | 0 |
| renou-source | 0.89 | 3.2699e-08 | 2.04649e-07 | 0 |
| renou-source | 0.9 | 8.80961e-06 | 5.20197e-05 | 0.000473959 |
| renou-source | 0.95 | 3.03795e-05 | 0.000155102 | 0.00295237 |
| renou-source | 1 | 0.00224085 | 0.0107432 | 0.0056189 |
| renou-detector | 0 | 0 | 0 | 0 |
| renou-detector | 0.8 | 2.6735e-09 | 1.49427e-08 | 0 |
| renou-detector | 0.89 | 4.995e-09 | 2.92849e-08 | 0 |
| renou-detector | 0.91 | 5.90947e-09 | 3.86994e-08 | 0 |
| renou-detector | 0.93 | 8.20762e-09 | 3.84859e-08 | 0.00126003 |
| renou-detector | 0.95 | 5.582e-08 | 2.27133e-07 | 0.00256868 |
| renou-detector | 1 | 0.00224085 | 0.0107432 | 0.00605951 |

**Fritz.** The clipped-CHSH construction and seeded numerical checks are recorded separately. The reference is `max(v/sqrt(2)-1/2,0)/4`, with the reported transition at `1/sqrt(2)` (Figure 3, pp. 4–5). The visibility-derived irrational correlator is stored as an explicit decimal rational; exact controls at rational `t=0,1/4,1/2` establish locality separately. The CSV visibility parameter is a display label rounded to eight significant digits. In particular, the row labeled `0.70710678` was generated at the numerical value of `1/sqrt(2)`, not at that exact truncated decimal; its bundle records the exact stored correlator.

| v | Certified Euclidean distance | Clipped-CHSH reference | Exact stored target |
|---:|---:|---:|:---:|
| 0 | 0 | 0 | yes |
| 0.5 | 2.34794e-19 | 0 | no |
| 1/sqrt(2) | 1.5e-17 | 0 | no |
| 0.8 | 0.0164214 | 0.0164214 | no |
| 1 | 0.0517767 | 0.0517767 | no |

The base sweep and targeted refinements took **1678.51 seconds** with a **104.12 MB** largest recorded optimizer-process high-water mark. One optimizer ran at a time, with `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, and `VECLIB_MAXIMUM_THREADS=2`. The memory figure is a process high-water mark, not whole-machine memory. The first refinement was sampled below 83 MB; its high-water mark was not saved after a promotion error. Exploratory probes, documentation, plotting, tests, and certificate-only verification are excluded from this timing. There was no controlled hardware comparison with the original neural-network training.

There are **167 candidate rows**, **52 target summaries**, and **50 selected bundles**. Every selected certificate was independently recomputed and matched exactly; **8** selected bundles have exact target equality. Intermediate bundles are regenerable in the ignored `benchmarks/runs/` directory; selected gzip bundles keep the public data small. See [all-fits.csv](benchmarks/all-fits.csv), [best-fits.csv](benchmarks/best-fits.csv), [ablation.csv](benchmarks/ablation.csv), [environment.json](benchmarks/environment.json), and [verification.json](benchmarks/verification.json).

The note has authors **Haobo Ma** (ChronoAI Pte Ltd; The Omega Institute), followed by **Wenlin Zhang** (National University of Singapore; The Omega Institute). The symmetric witness has exact all-equal probability **41/144**. MIT licensing, installable package metadata, input/certificate documentation, and regeneration commands are supplied. Final test and document-render receipts are in [validation.json](benchmarks/validation.json).

The exact verifier certifies the reported finite rational model and its distances. It does not certify global minima, completeness, or nonlocality from failed searches.
