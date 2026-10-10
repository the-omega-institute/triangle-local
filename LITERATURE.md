This review was checked against publisher/Crossref metadata, arXiv records, and selected full texts on 9–10 October 2026. The tool constructs **inner approximations** of the triangle-local set: a verified finite model proves that its own distribution is local. A failed numerical search does not prove that the requested target is nonlocal. Inflation supplies complementary outer approximations and can produce incompatibility certificates. This prototype implements the inner search and exact model verification, without an inflation solver.

1. **Tobias Fritz, “Beyond Bell's theorem: correlation scenarios.”** *New Journal of Physics* **14**, 103001 (2012). [DOI: 10.1088/1367-2630/14/10/103001](https://doi.org/10.1088/1367-2630/14/10/103001); [arXiv:1206.5115](https://arxiv.org/abs/1206.5115). Establishes the independent-source correlation-scenario framework and embeds Bell tests into the triangle. The Fritz control here embeds a CHSH distribution with two copied setting bits, using four outputs at each party. At or below correlator magnitude $t=1/2$, we explicitly construct a local model. The quantum-source visibility used in the 2020 paper is (v=\sqrt{2}t), so the boundary is (v=1/\sqrt{2}).

2. **Marc-Olivier Renou, Elisa Bäumer, Sadra Boreiri, Nicolas Brunner, Nicolas Gisin, and Salman Beigi, “Genuine Quantum Nonlocality in the Triangle Network.”** *Physical Review Letters* **123**, 140401 (2019). [DOI: 10.1103/PhysRevLett.123.140401](https://doi.org/10.1103/PhysRevLett.123.140401); [arXiv:1905.04902](https://arxiv.org/abs/1905.04902). Introduces genuine network examples involving entangled sources and joint measurements, with an odd-cycle generalization. We implement its four-outcome ring family in the ordering used in the 2020 paper: 
   
   \[
   |01\rangle,\quad |10\rangle,\quad u|00\rangle+\sqrt{1-u^2}|11\rangle,\quad
   \sqrt{1-u^2}|00\rangle-u|11\rangle.
   \]
   
   The sources are three (|\phi^+\rangle) pairs. The response argument order is cyclic. This distribution generally has cyclic party symmetry, and should not be forced into the more restrictive S3 search family. Our two analytically constructed local controls are (u^2=1/2) and (u^2=1).

3. **Tamás Kriváchy, Yu Cai, Daniel Cavalcanti, Arash Tavakoli, Nicolas Gisin, and Nicolas Brunner, “A neural network oracle for quantum nonlocality problems in networks.”** *npj Quantum Information* **6**, 70 (2020), published 21 August 2020. [DOI: 10.1038/s41534-020-00305-x](https://doi.org/10.1038/s41534-020-00305-x). The network architecture routes each independent source only to its two adjacent parties. Neural response functions approximate continuous hidden-variable functions; the joint distribution is estimated with Monte Carlo batches. This prototype replaces those functions with explicit finite tables and computes their full finite contraction, including exact integer arithmetic after rationalization.
   
   The paper's Eq. (3) is the KL objective; the Euclidean distance used for the plots is defined in the Results section. Our `l2` optimizer minimizes its square, and `L2_float` is its Euclidean square root. The paper reports Fritz's transition at (1/\sqrt{2}); EJM transitions at approximately 0.80 for source visibility and 0.86 for independent detector efficiency; Renou's (u^2=0.85) transitions at approximately 0.89 and 0.91 respectively. Figure 5 gives (d_M(1)\approx0.0034) and approximately 0.0055 in two separate runs. These are numerical observations, not certified locality thresholds.
   
   The authors' [public implementation](https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks) was inspected at revision `9460fe5340016e9a2f954613a2c07e633449d455`. The [target generator at that revision](https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks/blob/9460fe5340016e9a2f954613a2c07e633449d455/sample_code/targets.py) defines the noise conventions used here. We ran 24 comparisons for EJM source and detector noise, Fritz source visibility, and Renou source noise. The maximum entry discrepancy was (4.21\times10^{-13}), from the explicit rational approximation of an irrational Renou target. Receipts are in [family-crosschecks.json](research/family-crosschecks.json). The archived reference code contains an `np.np.sqrt` typo in its Renou detector branch; our detector channel was checked independently using exact channel convolution.

4. **José Mário da Silva and Fernando Parisio, “Numerically assisted determination of local models in network scenarios.”** *Physical Review A* **108**, 052602 (2023). [DOI: 10.1103/PhysRevA.108.052602](https://doi.org/10.1103/PhysRevA.108.052602); [arXiv:2303.09954v3](https://arxiv.org/abs/2303.09954). Provides the public finite-model `localmodels` implementation and compares it with the Kriváchy neural code on matched global-noise EJM runs. Finite-model search is therefore an established approach; this package emphasizes exact rational certification, compact witnesses, source/detector target conventions, and reproducible downstream validation. The code was located at [github.com/mariofilho281/localmodels](https://github.com/mariofilho281/localmodels), revision `28fd13ec30761d81eaa434a8545e02db5fe51ed7`.

5. **Antoine Girardin, Mohammad Massi Rashidi, Géraldine Haack, Nicolas Brunner, and Alejandro Pozas-Kerstjens, “A versatile neural-network toolbox for testing Bell locality in networks.”** *Quantum Science and Technology* **11**, 035056 (2026). [DOI: 10.1088/2058-9565/ae8df7](https://doi.org/10.1088/2058-9565/ae8df7). A current PyTorch toolbox extending the 2020 neural approach to arbitrary networks with adaptive sampling and optimization improvements. It is literature context for [BENCHMARK.md](BENCHMARK.md), not a measured arm of the three-way CPU experiment; no present-day state-of-the-art ranking is claimed.

6. **Denis Rosset, Nicolas Gisin, and Elie Wolfe, “Universal bound on the cardinality of local hidden variables in networks.”** *Quantum Information and Computation* **18**, 910–926 (2018). [DOI: 10.26421/QIC18.11-12-2](https://doi.org/10.26421/QIC18.11-12-2). Supplies finite-cardinality foundations for discrete network searches. The existence of finite bounds does not make small-cardinality numerical searches complete. The historical development campaign uses source cardinalities up to 24; the matched Round 4 comparison uses seven symbols per source for both finite solvers; neither all response tables nor a sufficient unrestricted cardinality class are exhaustively searched.

7. **Elie Wolfe, Robert W. Spekkens, and Tobias Fritz, “The Inflation Technique for Causal Inference with Latent Variables.”** *Journal of Causal Inference* **7**(2), 20170020 (2019). [DOI: 10.1515/jci-2017-0020](https://doi.org/10.1515/jci-2017-0020). Introduces replicated-source consistency tests. Feasibility at one inflation level is an outer-approximation condition; infeasibility can certify incompatibility.

8. **Miguel Navascués and Elie Wolfe, “The Inflation Technique Completely Solves the Causal Compatibility Problem.”** *Journal of Causal Inference* **8**(1), 70–91 (2020). [DOI: 10.1515/jci-2018-0008](https://doi.org/10.1515/jci-2018-0008). Establishes completeness of the classical inflation hierarchy in the limit. This does not confer completeness on a fixed level or on our optimizer.

9. **Elie Wolfe, Alejandro Pozas-Kerstjens, Matan Grinberg, Denis Rosset, Antonio Acín, and Miguel Navascués, “Quantum Inflation: A General Approach to Quantum Causal Compatibility.”** *Physical Review X* **11**, 021043 (2021). [DOI: 10.1103/PhysRevX.11.021043](https://doi.org/10.1103/PhysRevX.11.021043). Quantum causal-compatibility relaxations complement classical model searches. Quantum inflation and classical triangle locality address different compatibility questions.

10. **Sadra Boreiri, Antoine Girardin, Bora Ulu, Patryk Lipka-Bartosik, Nicolas Brunner, and Pavel Sekatski, “Towards a minimal example of quantum nonlocality without inputs.”** *Physical Review A* **107**, 062413 (2023). [DOI: 10.1103/PhysRevA.107.062413](https://doi.org/10.1103/PhysRevA.107.062413); [arXiv:2207.08532v2](https://arxiv.org/abs/2207.08532v2). Studies smaller output alphabets, including 3–3–3 and 3–3–2, and binary-output prospects. The January 2024 arXiv revision explicitly corrects Theorem 2; that revision is the appropriate reference when using the proof.

11. **Sadra Boreiri, Bora Ulu, Nicolas Brunner, and Pavel Sekatski, “Noise-robust proofs of quantum network nonlocality.”** *Quantum* **9**, 1830 (2025). [DOI: 10.22331/q-2025-08-27-1830](https://doi.org/10.22331/q-2025-08-27-1830). Develops approximate parity-token rigidity for robust nonlocality proofs. Its guarantees concern a specified family and noise models; they should not be substituted for the 2020 paper's heuristic EJM thresholds. Publisher metadata and its abstract were verified through Crossref; the publisher web page timed out during this review.

12. **Elisa Bäumer, Victor Gitton, Tamás Kriváchy, Nicolas Gisin, and Renato Renner, “Exploring the local landscape in the triangle network.”** *Physical Review A* **111**, 052453 (2025). [DOI: 10.1103/PhysRevA.111.052453](https://doi.org/10.1103/PhysRevA.111.052453); [arXiv:2405.08939v1](https://arxiv.org/abs/2405.08939v1). Combines analytical local constructions with LHV-Net exploration of distributions invariant under party permutations and simultaneous output relabelings. Appendix D.3 of the supplied v1 discusses approximately symmetric models with all-equal probability around 0.29 and asks for an exactly symmetric example exceeding 1/4. The accompanying `note.tex` gives such a finite witness, with exact all-equal probability **41/144**. Its restricted 4096-table ceiling is not a bound for the whole local set. Our verifier checks all 64 entries of the supplied witness; it does not use or rerun Lean. The [supporting witness note](research/witness-note-round1.tex) links [the Lean source at revision 59c09d64d1050c396df54b58b4f410f09f0c25d0](https://github.com/the-omega-institute/trureturing/blob/59c09d64d1050c396df54b58b4f410f09f0c25d0/D5/S3/Quantum/Entanglement/TriangleSymmetricLocalRefutation.lean).

13. **José Mário da Silva, Alejandro Pozas-Kerstjens, and Fernando Parisio, “Local models and Bell inequalities for the minimal triangle network.”** *Physical Review A* **112**, L030403 (2025). [DOI: 10.1103/cg9v-wv2t](https://doi.org/10.1103/cg9v-wv2t); [arXiv:2503.16654v2](https://arxiv.org/abs/2503.16654v2). A particularly relevant recent comparator: SciPy-based finite local-model searches for about 40,000 party-symmetric binary distributions, testing cardinalities up to 6–6–6, followed by analytical boundary models. Its proposed completeness/tightness statements should retain the paper's numerical/conjectural qualifications. Our generic discrete search belongs to this established line of work; finite-table optimization itself is not a novelty claim.

14. **Victor Gitton and Renato Renner, “The Elegant Joint Measurement is Non-Classical in the Triangle Network.”** [arXiv:2510.15143v1](https://arxiv.org/abs/2510.15143v1) (2025). Combines inflation, symmetry reduction, Frank–Wolfe optimization, and exact arithmetic to provide a computer-assisted proof of EJM nonclassicality. Its full text was inspected. **The noiseless EJM's nonlocality is no longer merely the 2020 conjecture.** Section 6.2 proves incompatibility of the ideal EJM. Appendix C.2, Eqs. (C.3)--(C.5), uses $vP_{\rm pure}+(1-v)/64$, where $P_{\rm pure}$ has entries $1/8$, $0$, $1/48$ for one, two and three distinct outcomes. The ideal EJM occurs at $v=3/4$ in that family. This visibility is distinct from either physical-noise parameter benchmarked here and from `elegant-global`, whose ideal EJM endpoint is at one. We did not execute or independently audit its full proof computation.

15. **Tamás Kriváchy and Martin Kerschbaumer, “Closing the detection loophole in the triangle network with high-dimensional photonic states.”** [arXiv:2503.24213v3](https://arxiv.org/abs/2503.24213v3) (2025). Uses rigorous loss bounds and neural-network heuristics for realistic photonic noise, making noise models and postselection particularly relevant to current local-model searches. Its reported heuristic robustness is distinct from a proof. Its N00N-state family is not part of the benchmark suite implemented here.

16. **Erwan Don, Jessica Bavaresco, Patryk Lipka-Bartosik, Nicolas Gisin, Nicolas Brunner, and Alejandro Pozas-Kerstjens, “The minimal example of quantum network Bell nonlocality.”** [arXiv:2605.00981v1](https://arxiv.org/abs/2605.00981v1) (2026). Reports binary-output nonlocal target distributions and a quantum tester parameterization reproducing them to machine precision. The full text distinguishes rigorous nonlocality certificates for selected targets from the numerical accuracy of the quantum realization. This updates earlier binary-output open-problem discussions. Our parity control is local; binary outputs alone do not imply locality.

17. **Jean Cortés, Luciano Pereira, and Aldo Delgado, “Self-guided certification of nonlocality in quantum networks.”** [arXiv:2609.11451v1](https://arxiv.org/abs/2609.11451v1) (2026). Uses classical shadows and complex simultaneous-perturbation optimization to search measurement settings, followed by an observed-statistics inequality evaluation. This is a measurement/witness search, rather than a hidden-variable local-model fitter. The title, authors, date, and abstract were verified; its full proof was not audited.

18. **Isaac D. Smith, Elie Wolfe, and Robert W. Spekkens, “Fully Quantum Inflation: Quantum Marginal Problem Constraints in the Service of Causal Inference.”** *PRX Quantum* **7**, 010351 (2026). [DOI: 10.1103/244y-nh5n](https://doi.org/10.1103/244y-nh5n). A recent quantum-compatibility development, included to distinguish stronger outer-approximation work from our finite classical inner search. Bibliographic metadata was verified; no implementation comparison was run.

The EJM targets used by this package are exact rational polynomials for rational (v). If a triple has one, two, or three distinct outcome labels, its source-noise probability is respectively

\[
\frac{4+9v+9v^2+3v^3}{256},\qquad
\frac{4+v-3v^2-v^3}{256},\qquad
\frac{4-3v+3v^2+v^3}{256}.
\]

Independent detector noise gives instead

\[
\frac{4+9v^2+12v^3}{256},\qquad
\frac{4+v^2-4v^3}{256},\qquad
\frac{4-3v^2+4v^3}{256}.
\]

At (v=1), both give the ideal (25/256,1/256,5/256) pattern. The separate `elegant-global` family is (vP+(1-v)/64). That is a target-distribution definition; mixing two triangle-local distributions is not automatically triangle-local, since a shared mixture flag would violate source independence. The global family is not substituted for either published physical noise model.

The 2025–2026 survey used Crossref title searches with a publication-date filter through 9 October 2026 and an arXiv API search for “triangle network” in that period. The latter returned 11 records, with relevant records expanded by identifier. It is a targeted review, not a claim of exhaustive coverage. [verified-metadata.json](research/verified-metadata.json), [verified-arxiv.xml](research/verified-arxiv.xml), and the source-code cross-check receipts preserve the public bibliographic and target evidence. Third-party full texts are available from the cited sources and are excluded from the public archive. Publisher page access failed for the 2025 minimal-triangle article, so its arXiv full text and Crossref publication record were used.

The archived quantitative comparison is anchored to the 2020 paper's full text,
not inferred from its title or later citations. The Euclidean distance is defined in the Results
section; Eq. (3), page 3, is KL divergence. Equation (4), also page 3, gives the approximate
piecewise curve used in our reference columns. Figure 4 and its discussion on
page 5 report EJM source visibility near 0.80 and detector efficiency near
0.86, with exit angles near 50 and 60 degrees. Figure 5 and its discussion on
page 5 report Renou `u² = 0.85` source/detector transitions near 0.89/0.91,
exit angles near 6 degrees, and two independent noiseless distances around
0.0034 and 0.0055. Figure 3 and pages 4–5 give Fritz's CHSH transition at
`1/sqrt(2)`. Page 6 reports depth 5 and width 30 per party for the EJM neural
network and batches around 8,000. The small NumPy network tested here has two
hidden layers of width 24 and uses finite quadrature; it is a separate component
experiment, not a replication of that training setup.

The public comparison retains the 2020 authors' qualification: these are
numerical estimates. Evaluating their Equation (4) is not recovering their raw
training data, and the equation is retained only as historical context. A
finite certified candidate is reported against the original quantum target
through an upward-rounded `U >= sqrt(E) + delta_target`; a near-zero residual at a reported
transition does not prove the stored target exactly local or certify a threshold.

A second target cross-check evaluates **60** Renou cases: `u²` equal to
`0.5, 0.63, 0.785, 0.85, 1`, each at six source/detector noise values including
`0.89, 0.91, 0.95`. The authors' detector branch was executed after correcting
one `np.np.sqrt` spelling error **in memory**, leaving the archived source
unchanged. The maximum entry error was **6.74 × 10⁻¹³**. This directly checks the
noise convention at the selected high-visibility target points. [round2-family-crosschecks.json](research/round2-family-crosschecks.json)
records the revision, source hash, adjustment, and all comparisons.

An independent review on 10 October 2026 rechecked all twelve journal DOIs
against Crossref and the six pinned arXiv records above. The public
`verified-metadata.json` contains compact title, author, publication and source
receipts; the original bulk Crossref search response is excluded. The 2020
publisher text was read directly, and the Gitton--Renner v1 full text was
downloaded and checked at Section 6.2, Appendix C.2 and its exact-arithmetic proof statements.
The paper's Renou transition estimates are explicitly described as cruder than
the EJM estimates because of local optima and variation between independent runs.

A further independent check covers 74 target-generator cases. It obtains the
authors' original source from the pinned revision, corrects only `np.np.sqrt`
in memory, and confirms the four noise-transition conventions and the noiseless
Renou target. Seventy-digit evaluations of the authors' Renou source polynomial
and the independent detector channel measure the stored-target approximation
errors separately. [independent-target-checks.json](research/independent-target-checks.json)
records these checks; certificates continue to refer to the exact stored
rational distributions.

The Round 4 common arrays are also compared directly against the pinned public generator. The spelling correction is now archived as an [explicit patch](benchmarks/round4/patches/lhv-net-target-spelling.patch), applied to a temporary downstream copy by the [reference check](benchmarks/round4/check_reference.py). Point discrepancies are cross-checks; [outward-rounded interval bounds](benchmarks/target-bounds.json), exact endpoint arithmetic and upward rounding justify δ and U. The current LHV-Net arm uses the public architecture with a declared small CPU configuration (depth three, width sixteen, batch 256); it is separate from both the historical paper's larger experiment and the internal NumPy network.
