The tool's demonstrated advantage is the production of portable rational witnesses with exactly checked locality, rigorous error bounds to the intended target, and a compact independent checking path. In this one-thread CPU configuration, its 60-second median L2 values were 0.00544, 0.00674, 3.91e-06 on EJM source, EJM detector, and Renou detector respectively, with 10/30 runs meeting the declared accuracy threshold; the measured counts were 0/30 for the small LHV-Net-2020 configuration and 0/30 for localmodels. These counts describe three selected targets and ten starts each. They do not establish a universal optimizer ranking, a comparison with the larger published neural configurations or the 2026 adaptive toolbox, exact locality of positive-error targets, or optimality of the reported fits. The neural extraction example demonstrates interoperability; a single checkpoint does not establish that neural initialization usually helps.

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
