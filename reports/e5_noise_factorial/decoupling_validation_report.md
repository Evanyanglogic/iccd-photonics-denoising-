# E5 Factorial Noise Decoupling Validation

Decision: **GO_TO_TRAIN**

| Variant | Residual mean | Residual std | Skew | Kurtosis | Tail >3sigma | Signal-residual corr | Row energy | Column energy | Clip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H-H | 8.4635e-10 | 0.00470248 | -1.7695 | 75.1176 | 0.00648304 | -0.1366 | 0.00205271 | 0.00559465 | 0.00168579 |
| H-L | -2.88284e-10 | 0.00168231 | -1.5904 | 76.5881 | 0.00652115 | -0.1296 | 0.00204879 | 0.00528724 | 0.0015242 |
| P-H | 1.34618e-09 | 0.00470248 | -5.3861 | 264.9469 | 0.0073315 | -0.1969 | 0.00211656 | 0.00969672 | 0.00171497 |
| P-L | -1.53361e-10 | 0.00168231 | -4.4483 | 266.3285 | 0.00733589 | -0.1800 | 0.00210111 | 0.00859316 | 0.00156437 |

## Checks

- PASS: `low_strength_std_matched`
- PASS: `high_strength_std_matched`
- PASS: `low_is_lower_than_high`
- PASS: `target_std_realized`
- PASS: `residual_mean_controlled`
- PASS: `clipping_controlled`
- PASS: `p99_scale_only_psd`
- PASS: `p99_scale_only_autocorr`
- PASS: `physical_scale_only_psd`
- PASS: `physical_scale_only_autocorr`
- PASS: `low_structure_remains_distinct`
- PASS: `high_structure_remains_distinct`

## Construction Boundary

- Structure: A zero-mean unit-standard-deviation residual field generated with clipping disabled by the frozen ICCD prior on either the p99-scaled source clean or the physical-scale source clean, using the same pair index and RNG seed.
- Strength: Source residual standard deviation divided by source clean standard deviation, transferred to the shared p99 clean standard deviation.
- A common 1024-DN pedestal is applied to all four shared-clean variants; it is not factor-dependent.
- Training is prohibited when the decision is `STOP_AND_REPAIR`.