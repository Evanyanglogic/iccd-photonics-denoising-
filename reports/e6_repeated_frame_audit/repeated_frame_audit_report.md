# Repeated-Frame Supervision Audit

Decision: **STOP_REACQUIRE**

## Folder Evidence

| folder | temporal std | drift/std | local drift/std | shift p95 | residual corr | highpass corr | row corr | col corr | fixed-map corr | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 70.603 | 0.036 | 0.189 | 0.000 | 0.1324 | -0.0037 | 0.5022 | 0.5609 | 0.97717 | no |
| 2 | 43.644 | 0.075 | 0.094 | 0.000 | 0.0007 | -0.0052 | 0.0238 | 0.0851 | 0.97221 | yes |
| 4 | 112.340 | 0.074 | 0.304 | 0.000 | 0.0369 | -0.0014 | 0.1721 | 0.6255 | 0.99869 | no |
| 5 | 235.926 | 0.166 | 0.684 | 0.000 | 0.1288 | -0.0038 | 0.5256 | 0.7884 | 0.99897 | no |
| 7 | 149.738 | 0.013 | 0.420 | 0.000 | 0.0481 | -0.0052 | 0.2546 | 0.7157 | 0.99910 | no |
| 8 | 135.686 | 0.021 | 0.333 | 0.000 | 0.0233 | -0.0062 | 0.2227 | 0.5443 | 0.99937 | no |
| 9 | 87.578 | 0.017 | 0.117 | 0.000 | 0.0035 | -0.0052 | 0.0212 | 0.3998 | 0.99924 | no |
| 10 | 71.342 | 0.001 | 0.093 | 0.000 | -0.0024 | -0.0064 | 0.0173 | 0.2494 | 0.99899 | no |
| 11 | 58.134 | 0.114 | 0.145 | 0.000 | -0.0020 | -0.0058 | 0.0069 | 0.0684 | 0.99848 | yes |
| 13 | 38.546 | 0.056 | 0.074 | 0.000 | -0.0036 | -0.0044 | 0.0019 | -0.0002 | 0.91621 | no |

## Target Evidence

- Median 8-frame target noise reduction: 2.627x
- Representative low/mid/high folders: [13, 10, 5]
- Selected protocol: `E_REACQUIRE`

## Interpretation Boundary

- Temporal means are surrogate expectations, not clean ground truth.
- Repeated-frame supervision can suppress conditionally independent temporal noise but cannot identify static fixed-pattern bias shared by input and target.
- A stable split-half fixed map can be learned or preserved as scene content; it is not evidence that fixed-pattern noise has been removed.
- Registration is estimated from 8-frame means and may be anchored partly by fixed-pattern structure.
- Synthetic images are not content-paired with real temporal means and therefore cannot form a valid direct supervision pair.

## Preregistered Thresholds

```json
{
  "max_registration_shift_p95_px": 0.25,
  "max_first_last_mean_shift_over_temporal_std": 0.25,
  "max_local_drift_over_temporal_std": 0.35,
  "max_abs_residual_correlation": 0.05,
  "max_abs_highpass_correlation": 0.05,
  "max_abs_row_or_column_correlation": 0.1,
  "min_fixed_map_half_correlation": 0.95,
  "min_target_noise_reduction_8frame": 2.0,
  "min_passing_folders": 8
}
```
