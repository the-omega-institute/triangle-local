"""Generate the short academic note from certified benchmark summaries."""
import csv
from triangle_local.io import load_csv, dump_csv
from fractions import Fraction
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def rows(name):
    return load_csv(ROOT/'benchmarks'/name)


def number(x):
    v=float(x)
    if v==0:return '0'
    exponent=int(__import__('math').floor(__import__('math').log10(abs(v))))
    if exponent<-2:return f'{v/10**exponent:.3g}\\times 10^{{{exponent}}}'
    return f'{v:.4g}'


def main():
    best=rows('best-fits.csv');ablation=rows('ablation.csv')
    env=json.loads((ROOT/'benchmarks'/'environment.json').read_text())
    verification=json.loads((ROOT/'benchmarks'/'verification.json').read_text())
    def find(family,param):return next(r for r in best if r['family']==family and Fraction(r['parameter'])==Fraction(param))
    noise=[]
    for label,family,param,prior in [('EJM source','elegant-source','4/5','.00537513'),
                                  ('EJM detector','elegant-detector','43/50','.010883'),
                                  ('Renou source','renou-source','89/100','.00107392'),
                                  ('Renou detector','renou-detector','91/100','.000828942')]:
        r=find(family,param)
        noise.append(f'{label} & {float(Fraction(param)):.2f} & ${number(prior)}$ & ${number(r["L2_rational"])}$ & ${number(r["L1_rational"])}$ \\\\')
    scan=[]
    for param in ['1/2','63/100','157/200','17/20','1']:
        r=find('renou-u2',param)
        scan.append(f'{float(Fraction(param)):.3g} & ${number(r["L2_rational"])}$ & {"yes" if r["exact_match"]=="True" else "no"} \\\\')
    labels={'blind-lbfgs-none-12':'Blind L-BFGS, 12 symbols',
            'blind-lbfgs-none-24':'Blind L-BFGS, 24 symbols',
            'blind-least-squares-none-12':'Least squares, 12 symbols',
            'blind-deterministic-none-24':'Deterministic descent, 24 symbols',
            'blind-lbfgs-outcome-equivariant-12':'Pair symmetry, 12 symbols',
            'blind-lbfgs-pair-equivariant-24':'Pair symmetry, 24 symbols',
            'blind-neural-none-24':'NumPy response network, 24 symbols',
            'seeded-lbfgs-12':'Lifted witness + L-BFGS',
            'polish-fixed-weights':'Then convex polish, fixed weights',
            'polish-free-weights':'Then convex polish, free weights',
            'seeded-hybrid-hops-12':'Seeded hybrid + two perturbations'}
    component=[]
    for r in ablation:
        if r['family']=='elegant-source' and r['configuration'] in labels:
            component.append(f'{labels[r["configuration"]]} & ${number(r["L2_rational"])}$ & {float(r["search_seconds"]):.1f} \\\\')
    pure=find('renou-u2','17/20')
    ideal=find('elegant-source','1')
    detector_ideal=find('elegant-detector','1')
    pure_value=float(pure['L2_rational'])
    pure_relation=('below both' if pure_value<.0034 else 'between the two' if pure_value<.0055 else 'above both')
    ideal_value=float(ideal['L2_rational'])
    ideal_reference=float(ideal['published_reference_L2'])
    ideal_percent=100*abs(ideal_value/ideal_reference-1)
    ideal_relation='above' if ideal_value>=ideal_reference else 'below'
    import math
    noise_max=max(float(find(f,p)['L2_rational']) for f,p in [('elegant-source','4/5'),('elegant-detector','43/50'),('renou-source','89/100'),('renou-detector','91/100')])
    scale=10**math.floor(math.log10(max(noise_max,1e-30)))
    noise_bound=(math.floor(noise_max/scale)+1)*scale
    text=r'''\documentclass[10pt]{article}
\pdfinfoomitdate=1
\pdftrailerid{}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,microtype}
\usepackage[hidelinks]{hyperref}\hypersetup{pdftitle={Finite triangle-local search with exact rational certificates},pdfauthor={Haobo Ma and Wenlin Zhang}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{Finite triangle-local search with exact rational certificates}
\author{\begin{tabular}{cc}
Haobo Ma & Wenlin Zhang\\
\small ChronoAI Pte Ltd & \small National University of Singapore\\
\small The Omega Institute & \small The Omega Institute
\end{tabular}}
\date{10 October 2026}
\begin{document}
\maketitle
\begin{abstract}
We implement a finite-table search for local models in the triangle network,
with independent source weights, stochastic or deterministic local responses,
and exact rational verification. Joint optimization is combined with convex
factor updates, independent source-symbol splitting, and perturbed searches.
The benchmark compares the Elegant Joint Measurement, Renou, and Fritz families
with the numerical estimates of Kriv\'achy et al. (2020). At the four reported
noise transitions, our rational candidates have Euclidean errors below
$__NOISE_BOUND__$. At noiseless Renou $u^2=0.85$, the certified error is
$__PURE__$, __PURE_RELATION__ independently reported 2020 values. The certificate checker
recomputes every probability and model-to-target distance with integer arithmetic.
It also verifies an explicit symmetric local witness with all-equal probability
$41/144$. Numerical fits and exact target witnesses are distinguished throughout:
a positive search residual is an upper bound on distance to the local set.
\end{abstract}

\section{Finite models and exact verification}
The triangle has sources $\alpha,\beta,\gamma$ shared respectively by $BC$, $CA$,
and $AB$. A finite local model consists of probability vectors $q_\alpha,q_\beta,
q_\gamma$ and row-stochastic response tables
$A_{jk}^a$, $B_{ki}^b$, and $C_{ij}^c$. Its output law is
\begin{equation}
 P_M(a,b,c)=\sum_{i,j,k}q_\alpha(i)q_\beta(j)q_\gamma(k)
 A_{jk}^a B_{ki}^b C_{ij}^c.\label{eq:model}
\end{equation}
Independence is imposed by the product of the three source laws. This is the
finite form of the network-locality definition of Fritz~\cite{fritz}; finite
cardinality bounds are established by Rosset et al.~\cite{rosset}.

A certificate stores rational source weights, rational response rows, the
rational target, and exact distances. Checking nonnegativity and row sums gives
three independent random sources and valid local response kernels. Equation
\eqref{eq:model} therefore defines a triangle-local distribution. Clearing the
factor denominators reduces its evaluation to integer multiplication and addition.
Equality of every output entry with the target gives an exact local witness.
Otherwise the same computation gives exact L1, squared Euclidean, and maximum
errors, and hence a certified nearby local distribution. The Euclidean error
bounds the target's distance to the unrestricted local set from above.

The implementation uses Python integer arithmetic and checks all simplexes,
normalization, party permutations, and model/target hashes independently of the
optimizer. It is tested software, rather than a formally verified checker.
Finite alphabets and nonconvex optimization do not give a complete membership
test. General Renou targets contain square roots and are stored as explicit
rational approximations; their certificates refer to those stored targets.

\section{Search and known local seeds}
Joint L-BFGS optimization uses softmax coordinates and analytical derivatives
of the full finite contraction. For fixed other factors, each individual source
vector or party response table enters linearly in \eqref{eq:model}. Squared
Euclidean fitting is thus a convex quadratic problem on a simplex or a product
of simplexes. We solve it by projected accelerated gradient and accept only
updates that decrease the evaluated objective. Inner iteration caps remain
part of the numerical method. For L1 loss, each block is a sparse linear program.
Deterministic descent instead minimizes over one output choice at a time;
all single-row choices are evaluated with floating-point costs and a numerical
acceptance tolerance, rather than exact arithmetic. This does not give a global
discrete optimum.

The hybrid method alternates joint optimization and convex polishing, then
perturbs the best candidate and repeats. Output distributions rounded to seven
decimal places identify duplicate fits for diagnostics. Every evaluated candidate
remains eligible to improve the best value. Forward and reverse continuation
and reuse of nearby fitted models reduce sensitivity to individual starts.

A local seed with larger requested alphabets is embedded by splitting each
source symbol's weight among copies and pulling back its adjacent response
tables. Summing over copies in \eqref{eq:model} recovers the original model.
This operation preserves source independence. Response perturbations then open
additional directions for optimization. Recognized controls are used as seeds
and pass through the ordinary rational certificate checker.

The supplied symmetric witness uses uniform independent sources on
$S=\{(i,j)\in\{0,1,2,3\}^2:i\ne j\}$ and
\[
 f(x,y)=\begin{cases}x_2,&x_2\in\{y_1,y_2\},\\x_1,&\text{otherwise},\end{cases}
 \qquad (A,B,C)=(f(\beta,\gamma),f(\gamma,\alpha),f(\alpha,\beta)).
\]
Enumeration of the $12^3$ source triples gives counts $123$, $19$, and $23$ for
each output triple with one, two, or three distinct labels, respectively.
Consequently $P(A=B=C)=4(123)/1728=41/144>1/4$, with full party and joint-outcome
symmetry. This supplies an exact witness relevant to B\"aumer et
al.~\cite{baumer}. The restricted-class enumeration and a revision-pinned link to the Lean source
are retained in the supporting witness note; this benchmark runs the independent
integer checker. The four-symbol token/parity seed for Renou's $u^2=1/2$ uses
independent token bits and independent edge parity bits and reproduces that
endpoint exactly. Fritz's CHSH-local embeddings and uniform noise are additional
exact controls.

\section{Published targets and benchmark comparison}
Kriv\'achy et al.~\cite{krivachy} use Euclidean distance in Eq.~(3). Their Eq.~(4)
approximates a noisy-family distance by zero below a fitted $v_*$ and by
$\|P(v)-P(v_*)\|_2\sin\theta$ above it. Figure~4 and the discussion on page~5
report EJM source/detector pairs $(v_*,\theta)\simeq(0.80,50^\circ)$ and
$(0.86,60^\circ)$. Figure~5 and page~5 report Renou pairs
$(0.89,6^\circ)$ and $(0.91,6^\circ)$ at $u^2=0.85$. These are heuristic numerical
estimates; the paper explicitly describes the Renou estimates as cruder and
sensitive to local optima and variation between independent runs. Our reference columns evaluate that equation; they are not digitized
training data. Gitton and Renner give a computer-assisted nonclassicality proof for
the ideal EJM~\cite{gitton}, Section 6.2, independently of the present search.
Their Appendix C.2 uses $vP_{\rm pure}+(1-v)/64$ with equality-pattern entries
$(1/8,0,1/48)$ for $P_{\rm pure}$; the ideal EJM occurs at $v=3/4$. That
parameter is distinct from our independent source and detector visibilities.

For EJM, the probabilities of triples with one, two, and three distinct labels
are the entries of
\[
 \frac1{256}\bigl(4+9v+9v^2+3v^3,\ 4+v-3v^2-v^3,\ 4-3v+3v^2+v^3\bigr)
\]
under independent source noise, and
\[
 \frac1{256}\bigl(4+9v^2+12v^3,\ 4+v^2-4v^3,\ 4-3v^2+4v^3\bigr)
\]
under independent detector noise. These distinct physical families are not
replaced by a global white-noise mixture. Renou's family~\cite{renou} uses three
$\lvert\phi^+\rangle$ sources and the measurement basis
$\lvert01\rangle,\lvert10\rangle,u\lvert00\rangle+\sqrt{1-u^2}\lvert11\rangle,
\sqrt{1-u^2}\lvert00\rangle-u\lvert11\rangle$ in cyclic order. Source noise uses
a six-qubit Born-rule contraction; detector noise applies independent uniform
replacement channels. The generators were checked against the 2020 authors'
\href{https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks/blob/9460fe5340016e9a2f954613a2c07e633449d455/sample_code/targets.py}{revision-pinned target code};
the largest entry discrepancy in 24 initial checks was $4.21\times10^{-13}$.
A further 60 Renou source/detector checks, including the high-visibility fits,
agreed within $6.74\times10^{-13}$ after one spelling correction in the
reference detector branch, applied in memory.

Table~\ref{tab:noise} compares the first sweep with the final selected rational
models at the four reported noise transitions. Small certified distances establish
nearby local models; none of these small residuals is substituted for exact target
membership. Certificates for Renou refer to the stored rational targets. Independent
70-digit evaluations of the authors' source-noise polynomial and the detector
channel found Euclidean target-approximation errors of $5.23\times10^{-13}$ at
source $v=0.89$, $2.10\times10^{-14}$ at detector $v=0.91$, and
$2.78\times10^{-14}$ at noiseless $u^2=0.85$. These are numerical
cross-checks of target generation, separate from exact certificate arithmetic.
The full scans, including both directions and unsuccessful fits,
are retained in CSV form. Current-model decimal distances are rounded from
exact rational squared-distance certificates.
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}lrrrr@{}}\toprule
Family & $v$ & First-sweep $d$ & Current certified $d$ & Certified L1\\\midrule
__NOISE__
\bottomrule\end{tabular}
\caption{Distance upper bounds at the reported 2020 transitions.}\label{tab:noise}
\end{table}

The noiseless Renou scan includes the known local endpoints and the difficult
region near $u^2\simeq0.785$ (Table~\ref{tab:scan}). At $u^2=0.85$, the final
certified distance is $__PURE__$; the 2020 paper separately reports about
$0.0034$ and $0.0055$ in two runs (Figure~5, page~5). Our candidate is __PURE_RELATION__ reported values. The ideal EJM error
$__IDEAL__$ is about $__IDEAL_PERCENT__\%$ __IDEAL_RELATION__ the source Eq.~(4) estimate $__IDEAL_REFERENCE__$;
the detector estimate is $__DETECTOR_REFERENCE__$; these approximate curves do not
supply the authors' raw saved models. These comparisons describe candidate
quality, not lower bounds on the true distances. Blind L-BFGS still
misses $u^2=1/2$ in the component test; explicit or recognized token/parity
seeding produces its exact certificate. Fritz's clipped-CHSH construction
matches the reference $\max(v/\sqrt2-1/2,0)/4$, with exact rational local controls
at correlator magnitudes $0$, $1/4$, and $1/2$.
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}rrc@{}}\toprule
$u^2$ & Certified Euclidean distance & Exact stored target\\\midrule
__SCAN__
\bottomrule\end{tabular}
\caption{Selected noiseless Renou points. The decimal $0.785$ is a sampled
parameter, not the exact critical parameter of the original theorem.}\label{tab:scan}
\end{table}

\section{Which components help}
Table~\ref{tab:components} compares configurations at EJM source visibility
$0.80$. Larger alphabets, deterministic descent, least squares, and the small
response network are useful alternative starts, but do not by themselves recover
this target in the measured trials. The strongest improvement comes from a
lifted local seed, free source-weight updates, and perturbed hybrid searches.
The symmetric families are restricted constructions; their persistent residuals
are not nonlocality certificates. The neural component has two hidden ReLU layers
of width 24 on finite source grids and uses NumPy Adam. It is distinct from the
2020 EJM architecture (depth 5, width 30, Monte Carlo batches around 8,000;
page~6 of~\cite{krivachy}).
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}lrr@{}}\toprule
Configuration & Certified distance & Search seconds\\\midrule
__COMPONENTS__
\bottomrule\end{tabular}
\caption{Component comparisons use the same pseudorandom seed, with recorded
method-specific iteration and restart caps. They are descriptive comparisons,
not equal-budget statistical estimates. Polishing rows start from the same
seeded L-BFGS model and report their additional runtime.}\label{tab:components}
\end{table}

\section{Reproduction}
The NumPy/SciPy package provides JSON/CSV input, a command line, MIT licensing,
and gzip-compressed rational bundles. The base sweep uses two starts, at most
1,800 joint iterations, 200 convex sweeps, and seed 20261010. Boundary refinements
and component tests record their separate caps and seeds. One optimizer runs at a
time with two numerical-library threads. The measured base sweep and targeted
refinements took __SECONDS__ seconds and had largest recorded process peak RSS __MEMORY__ MB;
exploratory probes are excluded. All __CERTIFICATES__ selected bundles were
independently recomputed successfully. These timings are not a hardware-matched
comparison with the original neural-network training.

The repository README gives installation, complete reproduction commands,
certificate schemas, and target arithmetic conventions. The tests exercise analytical Jacobians, convex descent, seed lifting, exact
controls, malformed and altered certificates, and command-line round trips.

\begin{thebibliography}{9}
\bibitem{fritz} T. Fritz, Beyond Bell's theorem: correlation scenarios,
\emph{New J. Phys.} \textbf{14}, 103001 (2012),
\href{https://doi.org/10.1088/1367-2630/14/10/103001}{doi:10.1088/1367-2630/14/10/103001}.
\bibitem{rosset} D. Rosset, N. Gisin, and E. Wolfe, Universal bound on the
cardinality of local hidden variables in networks,
\emph{Quantum Inf. Comput.} \textbf{18}, 910--926 (2018),
\href{https://doi.org/10.26421/QIC18.11-12-2}{doi:10.26421/QIC18.11-12-2}.
\bibitem{krivachy} T. Kriv\'achy, Y. Cai, D. Cavalcanti, A. Tavakoli, N. Gisin,
and N. Brunner, A neural network oracle for quantum nonlocality problems in
networks, \emph{npj Quantum Inf.} \textbf{6}, 70 (2020),
\href{https://doi.org/10.1038/s41534-020-00305-x}{doi:10.1038/s41534-020-00305-x}.
\bibitem{baumer} E. B\"aumer, V. Gitton, T. Kriv\'achy, N. Gisin, and R. Renner,
Exploring the local landscape in the triangle network,
\emph{Phys. Rev. A} \textbf{111}, 052453 (2025),
\href{https://doi.org/10.1103/PhysRevA.111.052453}{doi:10.1103/PhysRevA.111.052453}.
\bibitem{renou} M.-O. Renou, E. B\"aumer, S. Boreiri, N. Brunner, N. Gisin,
and S. Beigi, Genuine Quantum Nonlocality in the Triangle Network,
\emph{Phys. Rev. Lett.} \textbf{123}, 140401 (2019),
\href{https://doi.org/10.1103/PhysRevLett.123.140401}{doi:10.1103/PhysRevLett.123.140401}.
\bibitem{gitton} V. Gitton and R. Renner, The Elegant Joint Measurement is
Non-Classical in the Triangle Network,
\href{https://arxiv.org/abs/2510.15143v1}{arXiv:2510.15143v1} (2025).
\end{thebibliography}
\end{document}
'''
    replacements={'__NOISE__':'\n'.join(noise),'__SCAN__':'\n'.join(scan),
                  '__COMPONENTS__':'\n'.join(component),'__PURE__':number(pure['L2_rational']),
                  '__PURE_RELATION__':pure_relation,'__NOISE_BOUND__':number(noise_bound),
                  '__IDEAL__':number(ideal_value),'__IDEAL_PERCENT__':f'{ideal_percent:.2f}',
                  '__IDEAL_RELATION__':ideal_relation,'__IDEAL_REFERENCE__':number(ideal_reference),
                  '__DETECTOR_REFERENCE__':number(detector_ideal['published_reference_L2']),
                  '__SECONDS__':f'{env["total_wall_seconds"]:.1f}',
                  '__MEMORY__':f'{env["peak_RSS_bytes"]/1e6:.1f}',
                  '__CERTIFICATES__':str(verification['independently_recomputed_certificates'])}
    for key,value in replacements.items():text=text.replace(key,value)
    import re
    if re.search(r'\d{13,}',text):raise ValueError('Long digit string in note')
    if '__' in text:raise ValueError('Unfilled note template')
    (ROOT/'note.tex').write_text(text)
    print('Wrote note.tex from certified benchmark summaries')


if __name__=='__main__':main()
