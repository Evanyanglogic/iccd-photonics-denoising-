# E3.5-D Multi-Metric Condition Score

## Purpose

Replace the single-metric `Fano q40` split with a multi-metric ICCD condition
score, then evaluate whether it improves checkpoint selection between:

```text
p99-like conservative denoising
physical-scale stronger denoising
```

This directly addresses the boundary-case issue found in E3.5-C: folder 10 was
assigned to the high-condition side by Fano q40, but p99 slightly outperformed
physical on the selected visual sample.

## Script

```powershell
python scripts\evaluate_condition_score.py `
  --physical-eval-csv reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --p99-eval-csv reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --condition-summary-csv reports\e3_condition_gain_analysis\condition_gain_summary.csv `
  --condition-model-label physical `
  --output-dir reports\e3_5_condition_score
```

Outputs:

- `reports\e3_5_condition_score\condition_score_folders.csv`
- `reports\e3_5_condition_score\condition_score_pair_metrics.csv`
- `reports\e3_5_condition_score\condition_score_summary.csv`
- `reports\e3_5_condition_score\condition_score_report.md`

## Score Definition

The condition score is the mean z-score of:

- `mean_signal`
- `temporal_std_mean`
- `fano_temporal`
- `fixed_map_std`
- `fixed_to_temporal_std_ratio`

Higher score means stronger ICCD noise/fixed-pattern condition.

## Folder Ranking

| Rank | Folder | Score | Fano | p99 gain | physical gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | -1.0269 | 1.6958 | 0.0013 | -0.0354 |
| 2 | 2 | -0.9640 | 2.0823 | 0.0577 | -0.1395 |
| 3 | 11 | -0.7216 | 3.5146 | 0.0419 | -0.0553 |
| 4 | 1 | -0.5896 | 4.4415 | 0.0189 | -0.0077 |
| 5 | 10 | -0.4986 | 5.1434 | 0.0272 | 0.0088 |
| 6 | 9 | -0.2171 | 6.9575 | 0.0325 | 0.0655 |
| 7 | 4 | 0.2141 | 9.1699 | 0.0490 | 0.3745 |
| 8 | 8 | 0.6335 | 10.8110 | 0.0478 | 0.8227 |
| 9 | 7 | 0.8540 | 11.5785 | 0.0442 | 0.9774 |
| 10 | 5 | 2.3162 | 14.4564 | 0.0718 | 1.4198 |

## Strategy Results

| Strategy | Mean folder PSNR gain | Positive folders | Negative folders | Positive pair fraction |
|---|---:|---:|---:|---:|
| always noisy | 0.0000 dB | 0/10 | 0 | 0.000 |
| always p99 | 0.0392 dB | 10/10 | 0 | 0.938 |
| always physical | 0.3431 dB | 6/10 | 4 | 0.650 |
| Fano q40 hybrid | 0.3788 dB | 10/10 | 0 | 0.900 |
| score q40 hybrid | 0.3788 dB | 10/10 | 0 | 0.900 |
| score q50 hybrid | 0.3807 dB | 10/10 | 0 | 0.938 |
| score q60 hybrid | 0.3774 dB | 10/10 | 0 | 0.938 |

Best diagnostic rule:

```text
score_q50_hybrid_p99_physical
```

It selects:

```text
p99: folders 13, 2, 11, 1, 10
physical: folders 9, 4, 8, 7, 5
```

## Interpretation

The gain over Fano q40 is small:

```text
0.3807 dB vs 0.3788 dB
```

but the boundary behavior is better. In particular, folder 10 moves from the
physical side to the p99 side, matching the E3.5-C visual/residual inspection.

This supports a more defensible condition-aware claim:

```text
Multi-metric ICCD condition scoring can reduce condition-specific checkpoint
selection errors compared with a single Fano threshold.
```

## Claim Boundary

Supported:

```text
The multi-metric score is a better diagnostic condition ranking than Fano alone
for the current ten-folder surrogate evaluation.
```

Not supported:

```text
The score is a validated deployable classifier.
The q50 threshold will generalize to new devices, scenes, gains, or gate widths.
```

## Next Step

Before any larger model:

1. generate all-folder visual/residual panels under `score_q50_hybrid`;
2. consider a three-zone rule:
   - low score: p99;
   - middle score: p99 or uncertain;
   - high score: physical;
3. then decide whether E3.6 should generate condition-scaled synthetic noise.
