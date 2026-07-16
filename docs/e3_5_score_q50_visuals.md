# E3.5-E Score Q50 Visual and Residual Inspection

## Purpose

Generate all-folder visual/residual evidence for the best diagnostic
multi-metric condition rule found in E3.5-D:

```text
score_q50_hybrid_p99_physical
```

This checks whether the improved folder-level PSNR result is consistent with
visible residual suppression and whether the rule fixes the folder 10 boundary
case found in E3.5-C.

## Script

`scripts/inspect_condition_visuals.py` was extended with:

- `--selection-policy median_physical_gain`
- `--hybrid-physical-folders`

The default behavior still reproduces the original E3.5-C diagnostic selection.
The q50 run uses median physical-gain samples for all ten folders and assigns
the physical checkpoint only to folders selected by the score q50 rule.

Command:

```powershell
python scripts\inspect_condition_visuals.py `
  --pairs-csv reports\gated_iccd_20260319_surrogate_pairs\pairs.csv `
  --p99-checkpoint reports\e3_manifest_baseline_smallcnn_100ep\checkpoints\best.pth `
  --physical-checkpoint reports\e3_manifest_baseline_physical_scale_100ep\checkpoints\best.pth `
  --p99-metrics-csv reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --physical-metrics-csv reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --selection-policy median_physical_gain `
  --hybrid-physical-folders 9 4 8 7 5 `
  --output-dir reports\e3_5_score_q50_visuals `
  --device cpu
```

Outputs:

- `reports/e3_5_score_q50_visuals/condition_visual_metrics.csv`
- `reports/e3_5_score_q50_visuals/condition_visual_report.md`
- `reports/e3_5_score_q50_visuals/panels/*_image_panel.png`
- `reports/e3_5_score_q50_visuals/panels/*_residual_panel.png`

## Folder Decisions

| Folder | Selected Strategy | Representative Pair | p99 gain | physical gain | Hybrid gain |
|---:|---|---|---:|---:|---:|
| 1 | p99 | folder_1_frame_151 | 0.0153 | -0.0079 | 0.0153 |
| 2 | p99 | folder_2_frame_121 | 0.0366 | -0.0924 | 0.0366 |
| 4 | physical | folder_4_frame_151 | 0.0392 | 0.3740 | 0.3740 |
| 5 | physical | folder_5_frame_131 | 0.0741 | 1.4434 | 1.4434 |
| 7 | physical | folder_7_frame_131 | 0.0550 | 0.9959 | 0.9959 |
| 8 | physical | folder_8_frame_141 | 0.0379 | 0.8348 | 0.8348 |
| 9 | physical | folder_9_frame_171 | 0.0312 | 0.0650 | 0.0650 |
| 10 | p99 | folder_10_frame_171 | 0.0272 | 0.0088 | 0.0272 |
| 11 | p99 | folder_11_frame_111 | 0.0335 | -0.0405 | 0.0335 |
| 13 | p99 | folder_13_frame_171 | -0.0163 | 0.0053 | -0.0163 |

Folder 10 is now assigned to p99, which matches the visual/residual boundary
finding from E3.5-C. This is the main practical improvement over the single
Fano q40 split.

## Residual and Smoothing Observations

Low-score folders 1, 2, 10, 11, and 13 use p99. Their gradient ratios remain
close to the noisy input, about 0.993 for p99/hybrid, so the conservative model
mostly performs weak residual suppression rather than strong smoothing.

High-score folders 4, 5, 7, 8, and 9 use physical. The physical checkpoint gives
larger PSNR gains and lower residual standard deviation, but the strongest
folders show smoothing risk:

| Folder | physical gain | residual std noisy | residual std physical | grad/noisy physical |
|---:|---:|---:|---:|---:|
| 5 | 1.4434 | 0.004573 | 0.003865 | 0.7952 |
| 7 | 0.9959 | 0.002760 | 0.002458 | 0.8823 |
| 8 | 0.8348 | 0.002459 | 0.002234 | 0.9085 |

The result should be described as condition-aware residual suppression. It
should not be described as low-light detail restoration.

## Paper Claim Boundary

Supported:

```text
A multi-metric condition score reduces condition-selection errors in the current
ten-folder gated ICCD surrogate evaluation and can choose between conservative
and stronger denoising checkpoints.
```

Not supported:

```text
The q50 threshold is a deployable classifier.
The current small CNN restores missing low-light detail.
The visual panels prove generalization to new gain, gate width, or scene types.
```

## Next Step

Use this q50 rule as a diagnostic gate for E3.6:

1. produce condition-scaled synthetic ICCD-like noise instead of two fixed global
   synthetic variants;
2. evaluate whether condition-scaled training reduces the low-folder negative
   cases without relying on checkpoint switching;
3. keep smoothing checks in the validation table because high-score folders can
   gain PSNR while losing gradient energy.
