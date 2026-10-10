"""Build readable result tables and manuscript inserts from audited receipts."""
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tool'))
import numpy as np
from triangle_local.bounds import upward_decimal

LABELS={'ejm-source-080':'EJM source, v=0.80','ejm-detector-086':'EJM detector, v=0.86',
        'renou-detector-093':'Renou detector, u²=0.85, v=0.93'}
SHORT={'ejm-source-080':'EJM source','ejm-detector-086':'EJM detector','renou-detector-093':'Renou detector'}


def num(x,upper=False):
    return upward_decimal(Decimal(str(x)),places=4) if upper else f'{float(x):.3g}'


def tex(x,upper=False):
    x=num(x,upper)
    if 'e' in x.lower():
        a,b=x.lower().split('e');return f'{a}\\times10^{{{int(b)}}}'
    return x


def put(name,s):
    (HERE/name).write_text(s+'\n')


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    summary=json.loads((HERE/'summary.json').read_text())
    outcomes=list(csv.DictReader((HERE/'outcomes.csv').open()))
    interop=json.loads((HERE/'interoperability.json').read_text())
    compact=json.loads((HERE/'compact-verification.json').read_text())
    validation=json.loads((HERE/'validation.json').read_text())
    completion=json.loads((HERE/'completion.json').read_text())
    preliminary=json.loads((HERE/'preflight-summary.json').read_text())
    bounds=json.loads((ROOT/'benchmarks/target-bounds.json').read_text())['records']
    lookup={(s['target'],s['method'],s['budget_seconds']):s for s in summary}
    methods=protocol['methods'];ids=[t['id'] for t in protocol['targets']]
    def cell(ident,method,budget,latex=False):
        s=lookup[ident,method,budget];f=tex if latex else num
        if latex and 0<s['median_l2']<1e-3:
            exponent=int(np.floor(np.log10(s['median_l2'])))
            values=[s[k]/10**exponent for k in ['median_l2','q1_l2','q3_l2']]
            return f"({num(values[0])} [{num(values[1])}, {num(values[2])}])\\times10^{{{exponent}}}"
        return f"{f(s['median_l2'])} [{f(s['q1_l2'])}, {f(s['q3_l2'])}]"
    comparison=['| Common target | Method | 10 s L2 median [Q1, Q3] | Success | 60 s L2 median [Q1, Q3] | Success |',
                '|---|---|---:|---:|---:|---:|']
    tt=[r'\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llccrr@{}}\toprule',
        r'Target & Method & 10 s $L_2$ & 60 s $L_2$ & \multicolumn{2}{c}{Successes}\\',
        r' & & & & 10 s & 60 s\\\midrule']
    for ident in ids:
        for j,m in enumerate(methods):
            a,b=lookup[ident,m,10],lookup[ident,m,60]
            comparison.append(f"| {LABELS[ident]} | {m + ' (small CPU configuration)' if m == 'LHV-Net-2020' else m} | {cell(ident,m,10)} | {a['successes']}/10 | {cell(ident,m,60)} | {b['successes']}/10 |")
            tt.append(f"{SHORT[ident] if j==0 else ''} & {m} & ${cell(ident,m,10,True)}$ & ${cell(ident,m,60,True)}$ & {a['successes']}/10 & {b['successes']}/10"+r'\\')
        if ident!=ids[-1]: tt.append(r'\midrule')
    tt.append(r'\bottomrule\end{tabular}');put('comparison-table.tex','\n'.join(tt))
    comparison='\n'.join(comparison)
    successes={m:sum(lookup[i,m,60]['successes'] for i in ids) for m in methods}
    medians={m:[lookup[i,m,60]['median_l2'] for i in ids] for m in methods}
    rankings=[]
    for ident in ids:
        order=sorted(methods,key=lambda m:lookup[ident,m,60]['median_l2'])
        rankings.append(f"{SHORT[ident]}: {' < '.join(order)}")
    honest=(f"The tool's demonstrated advantage is the production of portable rational witnesses with exactly checked locality, rigorous error bounds to the intended target, and a compact independent checking path. "
        f"In this one-thread CPU configuration, its 60-second median L2 values were {', '.join(num(x) for x in medians['triangle-local'])} on EJM source, EJM detector, and Renou detector respectively, with {successes['triangle-local']}/30 runs meeting the declared accuracy threshold; "
        f"the measured counts were {successes['LHV-Net-2020']}/30 for the small LHV-Net-2020 configuration and {successes['localmodels']}/30 for localmodels. "
        "These counts describe three selected targets and ten starts each. They do not establish a universal optimizer ranking, a comparison with the larger published neural configurations or the 2026 adaptive toolbox, exact locality of positive-error targets, or optimality of the reported fits. "
        "The neural extraction example demonstrates interoperability; a single checkpoint does not establish that neural initialization usually helps.")
    put('comparison-prose.tex',f"At 60 seconds the declared threshold was met by {successes['triangle-local']}/30 triangle-local runs, {successes['LHV-Net-2020']}/30 neural runs, and {successes['localmodels']}/30 localmodels runs. These counts summarize the specified targets and settings; ten trials per cell do not establish population-level reliability or a universal ranking.")
    featured=[('Exactly symmetric construction',None,None),('Renou noiseless, u²=0.50','renou-u2','1/2'),
      ('EJM source, v=0.80','elegant-source','4/5'),('EJM detector, v=0.86','elegant-detector','43/50'),
      ('Renou source, v=0.89','renou-source','89/100'),('Renou detector, v=0.93','renou-detector','93/100'),
      ('EJM noiseless','elegant-source','1'),('Renou noiseless, u²=0.85','renou-u2','17/20')]
    best=['| Target | Stored-target residual √E | Target enclosure δ | Certified U |','|---|---:|---:|---:|']
    bt=[r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Target & $\sqrt E$ & $\delta_{\rm target}$ & $U$\\\midrule']
    for label,f,p in featured:
        if f is None:e=d=u='0'
        else:
            r=next(r for r in bounds if r['family']==f and r['parameter']==p)
            e=r['stored_model_target_L2'];d=r['delta_target_L2_upper'];u=r['U_sqrtE_plus_delta_target_upper']
        best.append(f'| {label} | {num(e)} | {num(d,True)} | {num(u,True)} |')
        lab=label.replace('u²=','$u^2=').replace('v=','$v=')
        if '$' in lab:lab+='$'
        bt.append(f'{lab} & ${tex(e)}$ & ${tex(d,True)}$ & ${tex(u,True)}$'+r'\\')
    bt.append(r'\bottomrule\end{tabular}');put('best-table.tex','\n'.join(bt));best='\n'.join(best)
    put('compact-box.tex',f"The original-target bound is $U\\le 8.208\\times10^{{-9}}$. The compressed bundle occupies {compact['bundle_bytes']:,} bytes; median exact checking time is {1000*compact['median_check_seconds']:.2f} ms over eleven checks. ")
    neural=[r for r in outcomes if r['method']=='LHV-Net-2020']
    evaltimes=[float(r['evaluation_seconds']) for r in neural]
    rms=[float(r['mc_rms_l2']) for r in neural]
    widths=[(float(r['ci95_high'])-float(r['ci95_low']))/2 for r in neural]
    unresolved=sum(r['uncertain_success']=='True' for r in neural)
    put('evaluation-prose.tex',f"Held-out evaluation took a median {np.median(evaltimes):.2f} seconds per checkpoint. Estimated Monte Carlo $L_2$ RMS errors ranged from ${tex(min(rms))}$ to ${tex(max(rms))}$; the median interval half-width was ${tex(np.median(widths))}$. There were {unresolved} threshold-straddling neural intervals.")
    random=[float(r['polished']['U']) for r in interop['random_controls']]
    native=interop['native_heldout'];before=interop['extracted'];after=interop['polished']
    extraction=['| Object | Accuracy | Additional measured cost |', '|---|---:|---:|',
        f"| Continuous neural checkpoint | held-out L2 {num(native['l2'])}, 95% interval [{num(native['ci95'][0])}, {num(native['ci95'][1])}] | {interop['training_seconds']:.2f} s training/selection; {native['evaluation_seconds']:.2f} s test |",
        f"| Extracted 4³ finite model | U ≤ {num(before['U'],True)} | {interop['extraction_seconds']:.3f} s extraction |",
        f"| Extracted model + polishing | U ≤ {num(after['U'],True)} | {after['polish_seconds']:.3f} s polishing |",
        f"| Random 4³ controls + same polishing, n=10 | median U {num(np.median(random),True)}; range [{num(min(random),True)}, {num(max(random),True)}] | 2 s budget each |"]
    et=[r'\begin{tabular}{@{}lrl@{}}\toprule',r'Object & Accuracy & Additional cost\\\midrule',
       f"Continuous neural checkpoint & $L_2={tex(native['l2'])}$ & {interop['training_seconds']:.1f} s training/selection"+r'\\',
       f"Extracted finite model & $U\\le {tex(before['U'],True)}$ & {interop['extraction_seconds']:.2f} s extraction"+r'\\',
       f"Extracted model + polish & $U\\le {tex(after['U'],True)}$ & {after['polish_seconds']:.2f} s polish"+r'\\',
       f"Random controls + polish & ${tex(np.median(random),True)}$ median $U$ & 2 s each; ten controls"+r'\\',
       r'\bottomrule\end{tabular}']
    put('extraction-table.tex','\n'.join(et));extraction='\n'.join(extraction)
    relative='below' if float(after['U'])<np.median(random) else 'above'
    put('extraction-prose.tex',f"The polished extraction lies {relative} the median of the ten random controls, whose bounds range from ${tex(min(random),True)}$ to ${tex(max(random),True)}$. This is one preselected neural checkpoint, so it establishes a reproducible conversion example rather than a general initialization advantage. Rationalization and verification took {before['rationalization_seconds']+before['verification_seconds']:.3f} seconds before polishing and {after['rationalization_seconds']+after['verification_seconds']:.3f} seconds afterwards.")
    ablation=list(csv.DictReader((ROOT/'benchmarks/ablation.csv').open()))
    ab={r['configuration']:r for r in ablation if r['family']=='elegant-source'}
    base=float(ab['seeded-lbfgs-12']['search_seconds'])
    configs=[('blind-lbfgs-none-12','Blind L-BFGS, 12 symbols'),('seeded-lbfgs-12','Seeded L-BFGS'),
             ('polish-fixed-weights','Seeded L-BFGS + fixed-weight polish'),('polish-free-weights','Seeded L-BFGS + free-weight polish'),
             ('seeded-hybrid-hops-12','Seeded hybrid + perturbations')]
    ct=[r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Configuration & $\sqrt E$ & Added time (s) & Total (s)\\\midrule']
    cm=['| Configuration | Stored-target residual | Additional seconds | Total seconds |','|---|---:|---:|---:|']
    for key,label in configs:
        r=ab[key];seconds=float(r['search_seconds']);added=key.startswith('polish-')
        total=base+seconds if added else seconds;extra=f'{seconds:.2f}' if added else '—'
        ct.append(f"{label} & ${tex(r['L2_rational'])}$ & {extra.replace('—','--')} & {total:.2f}"+r'\\')
        cm.append(f"| {label} | {num(r['L2_rational'])} | {extra} | {total:.2f} |")
    ct.append(r'\bottomrule\end{tabular}');put('configuration-table.tex','\n'.join(ct));configuration='\n'.join(cm)
    put('validation-prose.tex',f"The final suite passed {validation['tests_passed']} tests. The Round 4 audit recomputed all 180 native outcomes and checked all 120 finite checkpoint certificates; the earlier selected witness collection was also rechecked. The serial comparison took {completion['wall_seconds']/3600:.2f} hours including process startup and held-out evaluation.")
    results=f'''# Results: certificates and a matched comparison

The exactly symmetric construction has **P(A=B=C)=41/144 > 1/4** and equal probabilities within all 4 all-equal, 36 exactly-two-equal, and 24 all-distinct cells. It directly addresses the exact-symmetry question posed by Bäumer et al.; the point is exact symmetry, not surpassing their approximately symmetric numerical examples.

## Three-way cold-start comparison

{comparison}

Each cell has ten independent starts. The same runs supply the 10- and 60-second checkpoints. The declared success threshold is L2 ≤ 10⁻³; for neural outputs success requires the upper end of the held-out approximate 95% interval to meet it. All scores are recomputed from saved outputs against the **same hashed target array** for each problem. No target-specific seeds or continuation from other targets are used. Localmodels' internal RMSE is explicitly converted and checked: **L2 = 8 × RMSE** for 64 entries. The Renou shape is exactly 17/20. Neural point estimates are the raw norm `s = ||Q̂_N−T||₂`, without variance correction or truncation. The approximate 95% basic block-bootstrap interval is `[max(0, 2s−q_0.975), max(0, 2s−q_0.025)]`, where q denotes quantiles (linear interpolation) of 2,000 raw norms obtained by resampling the 32 held-out block means. The separate zero-truncated variance-corrected diagnostic is not plotted; [BENCHMARK.md](BENCHMARK.md#metric-and-checkpoint-evaluation) gives its formula.

![Three-method comparison with individual runs](docs/figures/baseline-time.png)

**What the tool does better, and what it does not.** {honest}

The table is descriptive and is not a hypothesis test or global solver ranking. Raw values, model factors, checkpoint times, neural weights, evaluation uncertainty, and success flags are in [outcomes.csv](benchmarks/round4/outcomes.csv) and the [run records](benchmarks/round4/runs/). The complete contract and resource scope are in [BENCHMARK.md](BENCHMARK.md).

## Selected development witnesses, outside the timing comparison

{best}

The checker establishes `Q ∈ L_triangle` and exact `E = ||P_stored-Q||₂²`. A validated interval enclosure supplies `δ >= ||P-P_stored||₂`; the principal bound is the upward-rounded `U >= sqrt(E)+δ`. Positive U proves a nearby local model, not exact target locality, nonlocality, or optimality. Exact stored-target equality establishes intended-target locality only when target identity is also exact. EJM's rational polynomials have δ=0. The Renou bounds use outward-rounded intervals, not high-precision point discrepancies. Renou source noise uses `ρ_v = v|φ⁺⟩⟨φ⁺| + (1−v)I₄/4` on each independent edge, with `|φ⁺⟩ = (|00⟩+|11⟩)/√2` and network state `ρ_v ⊗ ρ_v ⊗ ρ_v`. This is the convention for the selected v=0.89 source-noise witness; detector noise acts on the outputs instead. [All original-target bounds](benchmarks/target-bounds.json) retain the exact target identity and the source of every selected witness.

![Certified local approximations](docs/figures/headline.png)

At visibility v=1, source and detector noise lead to one noiseless target for each quantum family. The headline shows the best eligible witness once; different continuation-path residuals are outcomes for that same target. Changing Renou u² to 1/2 or 1 is a change of measurement parameter, not the same operation as setting v=1 at u²=0.85.

## Compact witness

The Renou detector witness at u²=0.85, v=0.93 trims exactly from (12,12,12) to **(8,9,2)**. Its original-target bound is **U ≤ 8.208e-9**, its compressed bundle is **{compact['bundle_bytes']:,} bytes**, and its median exact check takes **{1000*compact['median_check_seconds']:.2f} ms**. The timer starts with decompressed JSON and parsed target already in memory and includes rational model parsing, simplex checks, integer contraction, distances, symmetry and hashes. Cardinality minimality is not asserted.

![Compact witness summary](docs/figures/local-model.png)

For 0 ≤ v_L ≤ v_H with v_H>0, applying K_(v_L/v_H) to all response rows produces another exactly local model Q_L. The detector-noise channel gives `||Q_L-P(v_L)||₂ ≤ ||Q_H-P(v_H)||₂ ≤ U_H`. Exact high-visibility models therefore propagate exact equality; positive-error witnesses propagate controlled approximations. This does not extend automatically to independent source noise.

## Native neural checkpoint → finite certificate → polishing

{extraction}

The neural checkpoint was fixed in advance as repetition zero at 60 seconds on the Renou detector target. Its source revision, saved-weight hash, native held-out estimate, extraction supports, costs and exact before/after bundles are in [interoperability.json](benchmarks/round4/interoperability.json). Each source has its own uniform four-point support; the full Cartesian contraction enforces independence. The extracted finite model is distinct from the continuous network. The polished extraction lies {relative} the random-control median in this example. Ten random controls receive the same two-second polishing budget; this one neural checkpoint does not establish typical initialization benefit.

All 120 finite timed outputs from **both** triangle-local and localmodels were also rationalized, exactly checked, and independently audited after timing. Their conversion and checking costs, stored-target residuals, δ and U are separate columns in the outcome table. A native floating-point score is never labeled a certificate.

## Illustrative earlier configurations

{configuration}

Both polishing rows branch from the same seeded L-BFGS model. This is a descriptive earlier configuration study, not an equal-budget ablation. The complete historical records remain in [all-fits.csv](benchmarks/all-fits.csv), [best-fits.csv](benchmarks/best-fits.csv), and [ablation.csv](benchmarks/ablation.csv). Historical fitted 2020 curves are not measured baselines and are omitted from the main figures.

## Validation and artifact scope

{validation['tests_passed']} tests passed. The native-output audit checked 180 outcomes and their deadline/target hashes; the independent rational audit checked 120 new finite checkpoint certificates and the selected development collection. The content-hash manifest, pinned external revisions, source snapshot, patch, and local archive provide reproducibility without claiming that these unpushed changes have been published. No email was sent and no repository was pushed.
'''
    (ROOT/'RESULTS.md').write_text(results)
    hashes='\n'.join(f"| {LABELS[t['id']]} | `{t['sha256']}` |" for t in protocol['targets'])
    benchmark=f'''# Matched CPU comparison

This is the frozen Round 4 experiment: **three solvers × three common targets × ten cold starts**, with 10- and 60-second checkpoints. It supersedes the earlier two-run pilot for performance reporting. Selected best development witnesses remain separate in [RESULTS.md](RESULTS.md).

## Results and success rule

{comparison}

Cells are median [Q1,Q3]. Success means L2 ≤ **0.001**, declared before any counted run. A neural success additionally requires the upper endpoint of its approximate 95% held-out interval to be at or below the threshold. There were {unresolved} threshold-straddling neural intervals. Ten trials per cell describe this deployment regime; they do not give a stable population ranking. Checkpoints within one run are paired, not additional independent runs.

## One target artifact per problem

Each solver loads the same little-endian float64 `.npy` array with 64 entries. The flattened index is **16a+4b+c**, with a,b,c in {{0,1,2,3}} and c varying fastest. Companion JSON declares the exact physical parameters, ordering, dtype, file hash and raw array-byte hash. Decimal parameters mean exact rationals. EJM source uses v=4/5; EJM detector uses v=43/50; Renou detector uses u²=17/20 and v=93/100. Detector noise is the independent channel `K_v ⊗ K_v ⊗ K_v`, with `K_v = vI + (1−v)J/4`, not a global white-noise mixture. Renou source noise uses `ρ_v = v|φ⁺⟩⟨φ⁺| + (1−v)I₄/4` on each independent edge, with `|φ⁺⟩ = (|00⟩+|11⟩)/√2` and network state `ρ_v ⊗ ρ_v ⊗ ρ_v`. This is the convention for the selected v=0.89 source-noise witness; detector noise acts on the outputs instead.

| Target | SHA-256 of supplied array file |
|---|---|
{hashes}

The public target generator is independently cross-checked against these actual arrays. Its `np.np.sqrt` typo correction is archived as an [explicit downstream patch](benchmarks/round4/patches/lhv-net-target-spelling.patch), applied to a temporary copy by [check_reference.py](benchmarks/round4/check_reference.py). These point comparisons support identity checks; the rigorous original-target enclosures come from separate interval arithmetic.

## Metric and checkpoint evaluation

Every reported L2 is recomputed from a saved 64-entry output distribution in float64. For all finite outputs, the full contraction is independently repeated from saved source vectors and response tables. Localmodels' internal error is RMSE: `sqrt(SSE/64)`. The outcome audit explicitly checks **L2 = 8 RMSE** on every localmodels and triangle-local checkpoint. It never treats the upstream error field as an L2 result. Native training losses may differ.

Finite outputs retain the best feasible candidate whose score completed before each budget. Localmodels may evaluate infeasible trust-region trial points; these are excluded, with counts recorded. All source and response entries must be nonnegative and row sums within 10⁻¹² of one. Neural checkpoints minimize L2 on a fixed 4,096-sample validation set, evaluated every 50 training steps (and after the first). Validation and copying selected weights are charged to search. A checkpoint completed after a deadline cannot improve the corresponding score.

After both neural checkpoint choices are frozen, each is evaluated with **1,048,576 fresh held-out samples**, in 32 blocks of 32,768. Evaluation samples are independent of training and validation; no held-out score is used for selection. The reported and plotted score is the **raw norm `s = ||Q̂_N−T||₂`**, without variance correction or truncation. For B=32 block means `X̄_b`, the saved variance estimate is `V̂ = Σ_b ||X̄_b−Q̂_N||₂² / [B(B−1)]` (equivalently the sum of entrywise sample variances with `ddof=1`, divided by B). The separate diagnostic `sqrt(max(0, s²−V̂))` truncates the corrected squared residual at zero; it is **not** the table or figure score. The estimated RMS Monte Carlo error is `sqrt(V̂)`.

Each of 2,000 block-bootstrap resamples draws 32 block means with replacement and recomputes the raw norm `s*` of their average minus T. With `q_p` the empirical p-quantile of these norms (NumPy's default linear interpolation), the approximate 95% **basic** interval is `[max(0, 2s−q_0.975), max(0, 2s−q_0.025)]`. Both endpoints are clipped at zero. These intervals quantify sampling uncertainty, not rigorous locality bounds. Training, validation, restart, and held-out streams use domain-separated SeedSequence keys. Budgets are analyzed separately; the two checkpoints are paired observations and are not counted as independent replications.

Median held-out evaluation time was {np.median(evaltimes):.2f} s per checkpoint; RMS Monte Carlo error ranged from {num(min(rms))} to {num(max(rms))}, with median confidence-interval half-width {num(np.median(widths))}. Finite distributions are contracted fully rather than artificially sampled. [Raw outcomes](benchmarks/round4/outcomes.csv) expose all point estimates, intervals, seeds and threshold flags.

## Cold starts, architectures and stopping rules

- **triangle-local:** unrestricted 7×7×7 finite model; random normal softmax initialization; direct hybrid routine bypasses target recognition. One initial start, 400 L-BFGS iteration cap, 120 convex sweeps with 120 inner steps, two perturbation searches; restart from a fresh random seed when a cycle completes. Zero target-residual stopping tolerance. Best feasible candidate retained across that run.
- **LHV-Net-2020:** [public source at the measured revision](https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks/tree/{protocol['revisions']['LHV-Net-2020']}); three tanh hidden layers of width 16 per party, softmax outputs, uniform independent continuous sources, batch size 256, Adadelta learning rate 2, native KL objective. TensorFlow {json.loads(next((HERE/'runs').glob('*__LHV-Net-2020__00.json')).read_text())['versions']['tensorflow']} on CPU, float32 network operations and float64 final contraction. The eager compatibility wrapper replaces removed Keras generator methods and retains the public response routing and architecture. This is a declared small CPU configuration, not the larger published EJM experiment or a GPU benchmark.
- **localmodels:** [public source at the measured revision](https://github.com/mariofilho281/localmodels/tree/{protocol['revisions']['localmodels']}); unrestricted 7×7×7 model, public random Dirichlet initialization and trust-constr optimizer, one trial per call, outer tolerance zero, restart if the call returns. Native sum-of-squares objective. Numerical optimizer stopping rules remain those of the pinned public implementation.

There are no target-specific fitted or analytic seeds, endpoint recognizers, neighboring-target starts, or development witnesses in any arm. Perturbing the current run's candidate is charged to that run. Native model dimensions differ between neural and finite approaches and are stated above; common targets, budgets, start policy and evaluation metric are matched.

## Hardware and costs

Apple M2 Pro, 16 GiB memory, macOS, Python 3.11.15, NumPy 1.26.4 and SciPy 1.17.1. The exact OS record is in the protocol. One solver process runs at a time; OMP, OpenBLAS, Accelerate, MKL, NumExpr and TensorFlow numerical thread settings are all 1; GPUs are disabled. The orchestrator is idle while a solver runs. Jobs have a shuffled frozen order.

Timing includes initialization, optimizer setup, optimization, wrapper scoring/feasibility checks and checkpoint selection after target/common-library loading. Instrumentation costs are included, not separately subtracted. Small method-specific lazy imports occur inside the finite initialization timer. Final held-out testing, serialization, rationalization and certification are additional recorded costs. Only candidates fully scored before the budget are eligible; measured process time can exceed 60 s slightly while the last operation returns. The serial campaign occupied **{completion['wall_seconds']/3600:.2f} h** including startup and held-out testing. The excluded preflight seed audit took about {preliminary['wall_seconds']/60:.1f} minutes in addition. Three targets preserve ten repetitions per cell and leave room for exact checking and the interoperability experiment within the approximately three-hour compute cap. No 300 s arm was requested or extrapolated, and no optional fourth target was added.

## Certificate production and interoperability

Search scores are operational L2 to the common floating-point artifact. After timing, every finite checkpoint from both finite solvers is rationalized at denominator cap 10⁹ and checked exactly against the corresponding rational target. The float64 conversion of that rational target is checked for entrywise equality with the common array. Target enclosure δ and upward-rounded U are then added. The outcome table reports conversion time, checking time, stored-target residual, δ and U separately from search.

{extraction}

This conversion uses the predeclared repetition-zero Renou checkpoint, verified by saved-weight hash, and full independent Cartesian supports. It is a new finite model. Ten random finite starts receive the same two-second polishing budget. [Interoperability records](benchmarks/round4/interoperability.json) retain all random controls and exact bundles; no general neural-initialization advantage is inferred from one example.

## Reproduction and provenance

The [frozen protocol](benchmarks/round4/protocol.json), [exact executed runner](benchmarks/round4/frozen_runner.py), [90 run receipts](benchmarks/round4/runs/), [180 outcome rows](benchmarks/round4/outcomes.csv), saved factors/weights, and [verification receipt](benchmarks/round4/verification.json) are retained. The frozen runner's content hash equals the one in the protocol. Each receipt includes the actual target and protocol hashes, run seed, checkpoint completion time, package versions and process peak RSS. Smoke checks use repetition 99 and are excluded from all counted results. Preliminary runs with the earlier sequential restart-seed rule were also excluded in full; the seed-audit summary records their additional compute cost. The final protocol uses domain-separated streams and retains the original targets, settings, threshold and budgets.

Install the package's test, plot and bounds extras, obtain the two external repositories at the pinned revisions, and set `LHV_NET_SOURCE` and `LOCALMODELS_SOURCE` to their roots. Use a Python environment with TensorFlow 2.16.2. The runner's `prepare` action freezes fresh target artifacts and settings; `run` launches the sequential jobs and resumes existing completed records without replacing them. Use an isolated copy for a fresh campaign, remove only that copy's Round 4 target/run/protocol output files, and run `prepare` there. Run the audit and report commands documented in the README after completion.

The prior two-run records under `benchmarks/external/` are historical pilot data and are excluded from all Round 4 tables. The 2026 adaptive neural toolbox is literature context, not a measured arm. The local archive and manifest identify all unpushed results; no new published revision or release is claimed.
'''
    (ROOT/'BENCHMARK.md').write_text(benchmark)
    put('showcase.md',honest+'\n\n'+comparison)
    print('Wrote RESULTS.md, BENCHMARK.md and manuscript tables from checked raw data.')


if __name__=='__main__': main()
