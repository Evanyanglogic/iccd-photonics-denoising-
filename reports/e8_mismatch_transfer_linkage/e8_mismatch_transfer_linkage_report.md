# E8 Mismatch-to-Transfer Linkage Audit

## Statistical design

- Independent units: 10 folders.
- Within-folder repeats: 24 (4 variants x 3 seeds x 2 references).
- Analysis: preregistered single-metric folder-level Spearman associations; no fitted composite or predictor.
- Each variant is analyzed over the same ten folders; the primary statistic is the unweighted mean of four variant-specific correlations.
- Exact permutation permutes the ten folder labels; seed/reference/variant rows are never treated as independent.

## Reliability gate

| Metric | A/B rho | bootstrap 95% CI | LOO positive | Pass |
|---|---:|---:|---:|---|
| strength | 1.000 | [1.000, 1.000] | 40/40 | True |
| tail | 1.000 | [1.000, 1.000] | 40/40 | True |
| spatial | 0.933 | [0.716, 1.000] | 40/40 | True |
| signal_nonstationarity | 1.000 | [1.000, 1.000] | 40/40 | False |

Retained after reliability and collinearity gates: `strength, spatial`.
Tail mismatch was removed when its maximum within-variant correlation with strength exceeded 0.8. Signal/nonstationarity mismatch failed operational validity because only 12.5% of synthetic quantile bins were populated. Spatial mismatch remained eligible but is evaluated without changing its expected negative direction.

## Main associations

| Metric | mean variant rho | variant rhos | exact p | BH q | bootstrap CI | LOO nonpositive | seed/ref nonpositive | partial rho controlling gradient |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| strength | -0.606 | P-L:-0.842424;P-H:-0.600000;H-L:-0.248485;H-H:-0.733333 | 0.0029 | 0.0059 | [-0.500, 0.000] | 10/10 | 6/6 | -0.508 |
| spatial | 0.345 | P-L:-0.333333;P-H:0.672727;H-L:0.381818;H-H:0.660606 | 0.1890 | 0.1890 | [-0.500, 1.000] | 0/10 | 0/6 | 0.219 |

## Negative control

Deterministic random folder-rank control: rho=0.012, exact p=0.9489.

## Decision

**NARROW_GO_STABLE_STRENGTH_ASSOCIATION**

freeze route 2 evidence chain and perform manuscript-level claim/support audit

Strength mismatch is the only GO metric. Its four variant-specific correlations are all negative, all ten leave-one-folder-out statistics remain negative, and all six seed-reference summaries remain negative. The folder bootstrap interval reaches zero, so this remains a narrow descriptive result rather than a predictive model.

## Claim boundary

E8 is an observational association audit with ten independent folders. It cannot establish causality, a performance predictor, a physical noise mechanism, or cross-camera generalization.

Temporal means are surrogate references, not clean ground truth. All failed reliability, collinearity, influence, and sensitivity results are retained in CSV outputs.
