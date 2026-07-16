# E3.8 LOFO Condition Protocol Validation

This report evaluates condition-aware p99/physical selection with
leave-one-folder-out validation. For each held-out folder, hard thresholds
and linear blend intervals are selected only on the other folders.

## Inputs

- Pair manifest: `reports/gated_iccd_20260319_surrogate_pairs/pairs.csv`
- Condition score CSV: `reports/e3_5_condition_score/condition_score_folders.csv`
- p99 checkpoint: `reports/e3_manifest_baseline_smallcnn_100ep/checkpoints/best.pth`
- physical checkpoint: `reports/e3_manifest_baseline_physical_scale_100ep/checkpoints/best.pth`

## Strategy Summary

| Strategy | Mean folder gain | Positive folders | Negative folders | Positive pairs | Mean grad/noisy | Warnings |
|---|---:|---:|---:|---:|---:|---:|
| lofo_best_linear | 0.380355 | 10/10 | 0 | 0.938 | 0.9650 | 24 |
| lofo_best_hard | 0.375555 | 10/10 | 0 | 0.900 | 0.9654 | 24 |
| always_physical | 0.343069 | 6/10 | 4 | 0.650 | 0.9674 | 24 |
| always_p99 | 0.039237 | 10/10 | 0 | 0.938 | 0.9949 | 0 |
| always_noisy | 0.000000 | 0/10 | 0 | 0.000 | 1.0000 | 0 |

## Fold Selection

| Held-out folder | Strategy | Selected candidate | Train mean gain | Train negative folders |
|---:|---|---|---:|---:|
| 1 | lofo_best_hard | hard_q40 | 0.420892 | 0 |
| 1 | lofo_best_linear | linear_q40_q60 | 0.419787 | 0 |
| 2 | lofo_best_hard | hard_q40 | 0.416583 | 0 |
| 2 | lofo_best_linear | linear_q40_q60 | 0.415479 | 0 |
| 4 | lofo_best_hard | hard_q60 | 0.381378 | 0 |
| 4 | lofo_best_linear | linear_q40_q60 | 0.382114 | 0 |
| 5 | lofo_best_hard | hard_q60 | 0.265242 | 0 |
| 5 | lofo_best_linear | linear_q40_q60 | 0.265978 | 0 |
| 7 | lofo_best_hard | hard_q60 | 0.314394 | 0 |
| 7 | lofo_best_linear | linear_q40_q60 | 0.315130 | 0 |
| 8 | lofo_best_hard | hard_q60 | 0.331584 | 0 |
| 8 | lofo_best_linear | linear_q40_q60 | 0.332320 | 0 |
| 9 | lofo_best_hard | hard_q60 | 0.415721 | 0 |
| 9 | lofo_best_linear | linear_q40_q60 | 0.416162 | 0 |
| 10 | lofo_best_hard | hard_q40 | 0.419969 | 0 |
| 10 | lofo_best_linear | linear_q40_q60 | 0.419161 | 0 |
| 11 | lofo_best_hard | hard_q40 | 0.418340 | 0 |
| 11 | lofo_best_linear | linear_q40_q60 | 0.417236 | 0 |
| 13 | lofo_best_hard | hard_q40 | 0.422850 | 0 |
| 13 | lofo_best_linear | linear_q40_q60 | 0.421746 | 0 |

## Interpretation

The best LOFO strategy by mean folder gain is `lofo_best_linear` with
0.380355 dB. This is a stronger
generalization check than the earlier same-folder diagnostic q50/q40-q60
rules because each threshold is selected without the held-out folder.

## Claim Boundary

Supported if used carefully:

- condition-aware selection can be evaluated without held-out threshold leakage;
- LOFO summaries should replace same-folder q50/q40-q60 results in the main evidence table.

Still not supported:

- a deployable universal threshold;
- missing-detail restoration;
- model novelty claims based only on small-CNN checkpoints.
