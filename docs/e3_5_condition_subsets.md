# E3.5-B Low/High Condition Subset Validation

## Purpose

Separate real ICCD surrogate pairs into low- and high-condition folders using an
E1 noise statistic, then evaluate whether the p99 and physical-scale checkpoints
behave differently across those subsets.

This addresses the route question:

```text
Is the current denoising effect condition-aware ICCD noise reduction, or generic
low-light detail restoration?
```

## Script

```powershell
python scripts\evaluate_condition_subsets.py `
  --physical-eval-csv reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --p99-eval-csv reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --condition-summary-csv reports\e3_condition_gain_analysis\condition_gain_summary.csv `
  --condition-model-label physical `
  --split-metric fano_temporal `
  --quantile 0.4 `
  --output-dir reports\e3_5_condition_subsets
```

Outputs:

- `reports\e3_5_condition_subsets\condition_subset_pair_metrics.csv`
- `reports\e3_5_condition_subsets\condition_subset_summary.csv`
- `reports\e3_5_condition_subsets\condition_subset_folder_summary.csv`
- `reports\e3_5_condition_subsets\condition_subset_report.md`

## Split

- Split metric: `fano_temporal`
- q40 threshold: `4.862652`
- Low-condition folders: `1 2 11 13`
- High-condition folders: `4 5 7 8 9 10`

## Main Result

| Strategy | Mean folder PSNR gain | Positive folders | Negative folders | Positive pair fraction |
|---|---:|---:|---:|---:|
| noisy | 0.0000 dB | 0/10 | 0 | 0.000 |
| p99 checkpoint | 0.0392 dB | 10/10 | 0 | 0.938 |
| physical checkpoint | 0.3431 dB | 6/10 | 4 | 0.650 |
| condition gate, low=noisy high=physical | 0.3669 dB | 6/10 | 0 | 0.562 |
| condition hybrid, low=p99 high=physical | 0.3788 dB | 10/10 | 0 | 0.900 |

## Subset Behavior

| Subset | p99 gain | physical gain | Interpretation |
|---|---:|---:|---|
| Low-condition folders `1 2 11 13` | 0.0300 dB | -0.0595 dB | physical-scale model overcorrects or mismatches low-noise conditions; p99 is safer |
| High-condition folders `4 5 7 8 9 10` | 0.0454 dB | 0.6114 dB | physical-scale model is useful when real ICCD noise/fixed-pattern statistics are stronger |

Boundary folders:

| Boundary | Folder | Fano | physical gain | p99 gain |
|---|---:|---:|---:|---:|
| Low side | 1 | 4.4415 | -0.0077 dB | 0.0189 dB |
| High side | 10 | 5.1434 | 0.0088 dB | 0.0272 dB |

Folder 10 is a warning case: it enters the high-condition side by Fano, but p99
still slightly beats physical. Therefore the q40 split is useful as evidence of
condition dependence, not yet a final optimized deployment rule.

## Interpretation

The result supports a stronger condition-aware story than E3.5-A:

```text
Use conservative p99-like denoising in low-condition ICCD folders and stronger
physical-scale denoising in high-condition folders.
```

This is more defensible than claiming a single denoiser works uniformly. It also
shows that the paper should emphasize ICCD condition statistics and controlled
validation rather than generic low-light detail restoration.

## Claim Boundary

Supported:

```text
Condition-aware checkpoint selection reduces condition-specific degradation in
the current repeated-frame surrogate evaluation.
```

Not yet supported:

```text
The q40 Fano threshold is a final deployable decision rule.
The hybrid strategy generalizes to unseen ICCD devices or acquisition settings.
```

## Next Step

E3.5-C should inspect visual samples and residual statistics for:

- low-condition folder 2, where physical is worst and p99 is positive;
- high-condition folder 5, where physical is best;
- boundary folders 1 and 10.

This will check whether the PSNR gain reflects actual noise reduction or only a
brightness/statistical artifact.
