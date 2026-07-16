# E3.8 LOFO Condition Protocol Validation

## Purpose

E3.8 tests whether the condition-aware p99/physical selection result survives a
leave-one-folder-out validation. For each held-out folder, hard thresholds and
linear blend intervals are selected only from the other nine folders.

This directly addresses the E3.7 risk that q50 or q40-q60 could be a diagnostic
same-batch rule rather than a generalizable condition rule.

## Command

```powershell
python scripts\evaluate_lofo_condition_protocol.py
```

Outputs:

- `reports/e3_8_lofo_condition_protocol/lofo_pair_metrics.csv`
- `reports/e3_8_lofo_condition_protocol/lofo_fold_selection.csv`
- `reports/e3_8_lofo_condition_protocol/lofo_strategy_summary.csv`
- `reports/e3_8_lofo_condition_protocol/lofo_folder_summary.csv`
- `reports/e3_8_lofo_condition_protocol/lofo_config.json`
- `reports/e3_8_lofo_condition_protocol/lofo_report.md`

## Result

| Strategy | Mean folder PSNR gain | Positive folders | Positive pairs | Mean grad/noisy | Warnings |
|---|---:|---:|---:|---:|---:|
| LOFO best linear | 0.380355 dB | 10/10 | 75/80 | 0.9650 | 24 |
| LOFO best hard | 0.375555 dB | 10/10 | 72/80 | 0.9654 | 24 |
| always physical | 0.343069 dB | 6/10 | 52/80 | 0.9674 | 24 |
| always p99 | 0.039237 dB | 10/10 | 75/80 | 0.9949 | 0 |

The LOFO result supports the condition-aware strategy more strongly than the
earlier same-folder q50/q40-q60 diagnostic result. Both LOFO strategies keep
all ten folders positive. The linear variant is slightly higher than hard
selection, but the margin over hard selection is about 0.0048 dB, so it should
still be described cautiously.

## Fold-Level Behavior

The selected hard thresholds vary by held-out fold:

- low-condition held-out folders tend to select `hard_q40`;
- high-condition held-out folders tend to select `hard_q60`;
- linear selection consistently chooses `linear_q40_q60`.

This means the method is not yet a fixed deployable classifier. It is a
validated protocol showing that condition-aware selection remains beneficial
when threshold selection excludes the held-out folder.

## Risk

The main risk from E3.7 remains. LOFO physical-style outputs still trigger 24
gradient-ratio warnings under the `grad/noisy < 0.95` rule. High-condition
folders gain PSNR through stronger residual suppression, but the paper must not
call this missing-detail recovery.

## Claim Boundary

Supported:

- condition-aware p99/physical selection survives folder-level LOFO validation;
- selection/blending avoids the four negative folders seen in always-physical;
- LOFO summaries can replace same-batch q50/q40-q60 numbers in the main
  evidence table.

Not supported:

- a universal deployed threshold;
- low-light detail restoration;
- model novelty based on the current small-CNN checkpoints.

## Next Step

Move to E4 formal baselines. The immediate next implementation target is to
extend the manifest trainer with DnCNN and Light U-Net model options, then train
DnCNN on p99 synthetic and physical synthetic data.
