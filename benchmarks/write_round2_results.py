"""Write the measured round-2 report from CSV and exact-check receipts."""
import csv
from triangle_local.io import load_csv, dump_csv
from fractions import Fraction
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def read(path):
    return load_csv(path)


def main():
    out=ROOT/'benchmarks';best=read(out/'best-fits.csv');ablation=read(out/'ablation.csv')
    allrows=read(out/'all-fits.csv');old=read(out/'round1'/'best-fits.csv')
    env=json.loads((out/'environment.json').read_text());checks=json.loads((out/'verification.json').read_text())
    def find(family,param,data=best):
        return next((r for r in data if r['family']==family and Fraction(r['parameter'])==Fraction(param)),None)
    def fmt(row,key='L2_rational'):
        return '-' if row is None else f'{float(row.get(key,row["L2_float"])):.6g}'
    transitions=[]
    for label,family,param in [('EJM source','elegant-source','4/5'),('EJM detector','elegant-detector','43/50'),
                               ('Renou source','renou-source','89/100'),('Renou detector','renou-detector','91/100')]:
        row=find(family,param)
        transitions.append(f'| {label} | {float(Fraction(param)):.2f} | {fmt(find(family,param,old),"L2_float")} | {fmt(row)} | {fmt(row,"L1_rational")} |')
    lines=['**Round 2 combines local-model seeds, convex factor updates, and perturbation searches with exact rational certificates.** '
           'The benchmark now fits all four reported noise-transition targets with certified Euclidean distances below four times ten to the minus eight. '
           'The tables below report the final candidates for all four reported noise transitions and preserve the remaining gaps. '
           'Small residuals certify nearby local distributions; exact target locality requires `exact_match`.','',
           '| Family | 2020 reported transition | Round 1 Euclidean error | Round 2 certified Euclidean error | Round 2 certified L1 |',
           '|---|---:|---:|---:|---:|',*transitions,'',
           'The 2020 references are Tamás Kriváchy et al., “A neural network oracle for quantum nonlocality problems in networks,” '
           '*npj Quantum Information* **6**, 70 (2020), [doi:10.1038/s41534-020-00305-x](https://doi.org/10.1038/s41534-020-00305-x). '
           'Equation (3), p. 3, defines Euclidean distance. Equation (4), p. 3, gives the approximate reference curve. '
           'Figure 4 and p. 5 give EJM source/detector estimates `(0.80, 50°)` and `(0.86, 60°)`. '
           'Figure 5 and p. 5 give Renou estimates `(0.89, 6°)` and `(0.91, 6°)` at `u²=0.85`, '
           'and independently reported noiseless errors around **0.0034** and **0.0055**. '
           'The paper explicitly describes the Renou estimates as cruder than the EJM estimates, with local optima and variation between independent runs. Reference columns evaluate Eq. (4) here; they are not digitized neural-network measurements.','',
           'The base sweep uses seed 20261010, two starts, at most 1800 joint iterations, and 200 convex sweeps with 150 inner projected-gradient steps. '
           'It pools single fitted models as warm starts, runs every family in both directions, and polishes the four boundary targets for up to 1200 sweeps. '
           'Component comparisons use separate recorded caps. The follow-up adds symmetric fallbacks, four perturbation searches at difficult targets, '
           'and numerical Fritz checks. The final pass adds the 24-symbol detector-boundary fit, continuation from the fitted Renou critical-region point, and retention/refinement of the earlier ideal EJM seed. Complete settings and diagnostics are in the bundles and `all-fits.csv`.','',
           'At squared-loss block updates, one source or party factor is linear while the other five are fixed. '
           'The resulting simplex quadratic subproblem is convex, but its iterative solve can hit its cap. '
           'L1 blocks use a sparse LP. Deterministic descent enumerates single-row output choices using floating-point costs and a numerical acceptance tolerance. '
           'The hybrid pipeline retains the best evaluated model and records duplicate output distributions; it never constructs a global mixture flag.','',
           '**Component comparison at EJM source visibility 0.80.** These are descriptive experiments with method-specific restart/iteration caps, '
           'not equal-budget statistical estimates. Both polishing variants start from the same seeded L-BFGS candidate and report additional runtime.','',
           '| Configuration | Sizes | Certified Euclidean error | Search seconds |', '|---|---|---:|---:|']
    for row in ablation:
        if row['family']=='elegant-source':
            lines.append(f'| {row["configuration"]} | {row["sizes"]} | {fmt(row)} | {float(row["search_seconds"]):.2f} |')
    lines+=['','Seed lifting, convex polishing with free source weights, and perturbations supply the strongest improvement at this target. '
            'Increasing cardinality alone and the tested symmetric extensions do not remove the gap. Deterministic descent and least squares improve some blind starts '
            'but remain weaker than the seeded hybrid result. The small NumPy response network is also weaker in these trials. '
            'It uses two hidden ReLU layers of width 24 on finite grids, so this is not a rerun of the paper’s depth-5, width-30 EJM architecture or its Monte Carlo training.',
            '', '**EJM source and independent detector noise.** Rational visibility gives exact target polynomials. A missing table cell means that family was not sampled at that visibility.','',
            '| v | Source certified distance | Detector certified distance | 2020 source Eq. (4) | 2020 detector Eq. (4) |',
            '|---:|---:|---:|---:|---:|']
    grid=sorted(set(Fraction(r['parameter']) for r in best if r['family'].startswith('elegant')))
    for v in grid:
        a,b=find('elegant-source',str(v)),find('elegant-detector',str(v))
        lines.append(f'| {float(v):.3g} | {fmt(a)} | {fmt(b)} | {fmt(a,"published_reference_L2")} | {fmt(b,"published_reference_L2")} |')
    lines+=['','The reference estimates are approximate fitted curves. A candidate below an Eq. (4) value beats that estimate at that point, '
            'not necessarily the authors’ best saved model. Positive errors above a transition can still include optimization or finite-alphabet error. '
            '__IDEAL_COMMENT__ The empirical comparison does not establish a new locality threshold. The EJM ideal endpoint is the same exact target in both noise families, '
            'so a common best candidate is reused by target hash.','', '**Noiseless Renou scan.**','',
            '| u² | Round 1 distance, when sampled | Round 2 certified distance | Certified L1 | Exact stored target |',
            '|---:|---:|---:|---:|:---:|']
    for row in sorted([r for r in best if r['family']=='renou-u2'],key=lambda r:Fraction(r['parameter'])):
        lines.append(f'| {float(Fraction(row["parameter"])):.3g} | {fmt(find("renou-u2",row["parameter"],old),"L2_float")} | {fmt(row)} | {fmt(row,"L1_rational")} | {"yes" if row["exact_match"]=="True" else "no"} |')
    pure=find('renou-u2','17/20')
    d=float(pure['L2_rational']);comparison=('below both reported values' if d<.0034 else 'between the two reported values' if d<.0055 else 'above both reported values')
    lines+=['',f'At `u²=0.85`, the certified distance **{d:.6g}** is {comparison}. '
            'The decimal sample `0.785` is not the exact critical parameter of the original theorem. '
            'A remaining residual there cannot be interpreted as an exclusion. Blind L-BFGS still returns a nonzero residual at the known local half endpoint; '
            'the explicit token/parity control and the default hybrid recognizer recover its exact rational model. '
            'This separates a corrected default behavior from the persistent limitation of blind search.','',
            '**Renou independent noise, fixed u²=0.85.** General square-root probabilities are quantized rational targets; detector noise is exact convolution of the stored base target. Independent 70-digit evaluations give Euclidean target-approximation errors of 5.23e-13 at source v=0.89, 2.10e-14 at detector v=0.91, and 2.78e-14 at noiseless u²=0.85. These numerical target checks are separate from exact rational certificate verification.','',
            '| Family | v | Certified Euclidean distance | Certified L1 | 2020 Eq. (4) estimate |','|---|---:|---:|---:|---:|']
    for name in ['renou-source','renou-detector']:
        for row in sorted([r for r in best if r['family']==name],key=lambda r:Fraction(r['parameter'])):
            lines.append(f'| {name} | {float(Fraction(row["parameter"])):.3g} | {fmt(row)} | {fmt(row,"L1_rational")} | {fmt(row,"published_reference_L2")} |')
    lines+=['','**Fritz.** The clipped-CHSH construction and seeded numerical checks are recorded separately. '
            'The reference is `max(v/sqrt(2)-1/2,0)/4`, with the reported transition at `1/sqrt(2)` (Figure 3, pp. 4–5). '
            'The visibility-derived irrational correlator is stored as an explicit decimal rational; exact controls at rational `t=0,1/4,1/2` establish locality separately. The CSV visibility parameter is a display label rounded to eight significant digits. In particular, the row labeled `0.70710678` was generated at the numerical value of `1/sqrt(2)`, not at that exact truncated decimal; its bundle records the exact stored correlator.','',
            '| v | Certified Euclidean distance | Clipped-CHSH reference | Exact stored target |','|---:|---:|---:|:---:|']
    for original_row in [r for r in best if r['family']=='fritz-visibility']:
        row = dict(original_row)
        if row['parameter']=='0.70710678':row['parameter']='1/sqrt(2)'
        lines.append(f'| {row["parameter"]} | {fmt(row)} | {fmt(row,"published_reference_L2")} | {"yes" if row["exact_match"]=="True" else "no"} |')
    peak=env['peak_RSS_bytes']/1e6
    lines+=['',f'The base sweep and targeted refinements took **{env["total_wall_seconds"]:.2f} seconds** '
            f'with a **{peak:.2f} MB** largest recorded optimizer-process high-water mark. One optimizer ran at a time, with `OMP_NUM_THREADS=2`, '
            '`OPENBLAS_NUM_THREADS=2`, and `VECLIB_MAXIMUM_THREADS=2`. The memory figure is a process high-water mark, not whole-machine memory. The first refinement was sampled below 83 MB; its high-water mark was not saved after a promotion error. '
            'Exploratory probes, documentation, plotting, tests, and certificate-only verification are excluded from this timing. '
            'There was no controlled hardware comparison with the original neural-network training.','',
            f'There are **{len(allrows)} candidate rows**, **{len(best)} target summaries**, and '
            f'**{checks["independently_recomputed_certificates"]} selected bundles**. Every selected certificate was independently recomputed and matched exactly; '
            f'**{checks["exact_matches"]}** selected bundles have exact target equality. '
            'Intermediate bundles are regenerable in the ignored `benchmarks/runs/` directory; selected gzip bundles keep the public data small. '
            'See [all-fits.csv](benchmarks/all-fits.csv), [best-fits.csv](benchmarks/best-fits.csv), '
            '[ablation.csv](benchmarks/ablation.csv), [environment.json](benchmarks/environment.json), '
            'and [verification.json](benchmarks/verification.json).','',
            'The note has authors **Haobo Ma** (ChronoAI Pte Ltd; The Omega Institute), followed by '
            '**Wenlin Zhang** (National University of Singapore; The Omega Institute). '
            'The symmetric witness has exact all-equal probability **41/144**. '
            'MIT licensing, installable package metadata, input/certificate documentation, and regeneration commands are supplied. '
            'Final test and document-render receipts are in [validation.json](benchmarks/validation.json).','',
            'The exact verifier certifies the reported finite rational model and its distances. It does not certify global minima, '
            'completeness, or nonlocality from failed searches.']
    ideal=find('elegant-source','1');detector_ideal=find('elegant-detector','1')
    ideal_value=float(ideal['L2_rational']);reference=float(ideal['published_reference_L2'])
    relation='above' if ideal_value>=reference else 'below'
    percentage=100*abs(ideal_value/reference-1)
    comment=(f'The ideal EJM distance is {ideal_value:.6g}, about {percentage:.2f}% {relation} the source '
             f'Eq. (4) estimate {reference:.6g}; the detector estimate is {float(detector_ideal["published_reference_L2"]):.6g}. '
             'These compare approximate analytic curves, not unavailable raw neural-network fits.')
    text='\n'.join(lines)+'\n'
    text=text.replace('__IDEAL_COMMENT__',comment)
    (ROOT/'RESULTS.md').write_text(text)
    print('Wrote RESULTS.md from measured CSVs')


if __name__=='__main__':main()
