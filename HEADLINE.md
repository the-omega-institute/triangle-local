# Certified local approximations to triangle-network correlations

Triangle-local produces portable finite triangle-local witnesses whose locality is checked exactly, with a rigorous error bound U to the intended target, without trusting the search.

The leading mathematical example is the exactly symmetric independent-source construction with P(A=B=C)=41/144 > 1/4. Its equality-pattern probabilities prove both party symmetry and simultaneous outcome-relabeling symmetry. This directly addresses the exact-symmetry question posed by Bäumer et al.; the comparison is not against their approximate numerical value.

The headline figure shows selected development witnesses with original-target bounds U, explicit exact zeros, and a single point for each noiseless target. The matched performance figure separately shows all three methods, all ten cold starts per target, and only the completed 10- and 60-second checkpoints, with fresh held-out uncertainty for the neural arm. The compact witness is presented as a box rather than a chart.

For the measured comparison, raw outcomes, and the assessment of strengths and limits, read [RESULTS.md](RESULTS.md). For initialization, target hashes, timing and evaluation details, read [BENCHMARK.md](BENCHMARK.md). The note includes the actual figures and comparison table. Positive U establishes a nearby local model, not exact target membership or optimizer optimality.
