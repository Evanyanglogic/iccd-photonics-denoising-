# E3.5-C Visual and Residual Inspection

## Purpose

Inspect selected low/high/boundary ICCD folders to check whether condition-aware
gains reflect residual noise reduction rather than brightness drift or obvious
oversmoothing.

This step follows E3.5-B:

```text
low-condition: use p99-like conservative denoising
high-condition: use physical-scale stronger denoising
```

## Script

```powershell
python scripts\inspect_condition_visuals.py `
  --pairs-csv reports\gated_iccd_20260319_surrogate_pairs\pairs.csv `
  --p99-checkpoint reports\e3_manifest_baseline_smallcnn_100ep\checkpoints\best.pth `
  --physical-checkpoint reports\e3_manifest_baseline_physical_scale_100ep\checkpoints\best.pth `
  --p99-metrics-csv reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --physical-metrics-csv reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --folders 2 5 1 10 `
  --output-dir reports\e3_5_condition_visuals `
  --device cpu
```

Outputs:

- `reports\e3_5_condition_visuals\condition_visual_metrics.csv`
- `reports\e3_5_condition_visuals\condition_visual_report.md`
- `reports\e3_5_condition_visuals\panels`

## Selected Samples

| Folder | Role | Pair | p99 gain | physical gain | Hybrid choice |
|---:|---|---|---:|---:|---|
| 2 | worst physical low-condition sample | `folder_2_frame_161` | +0.1215 dB | -0.2832 dB | p99 |
| 5 | best physical high-condition sample | `folder_5_frame_101` | +0.0903 dB | +1.7393 dB | physical |
| 1 | low/high boundary low side | `folder_1_frame_151` | +0.0153 dB | -0.0079 dB | p99 |
| 10 | low/high boundary high side | `folder_10_frame_171` | +0.0272 dB | +0.0088 dB | physical by current q40 rule |

## Residual Findings

### Low-condition folder 2

- p99 improves PSNR and slightly reduces residual std.
- physical worsens PSNR and increases residual bias.
- Visual panel shows no credible new detail recovery; physical appears to
  overcorrect rather than denoise.

Interpretation:

```text
Low-condition folders should not use the physical-scale checkpoint.
```

### High-condition folder 5

- physical improves PSNR by +1.7393 dB.
- residual std drops from 0.004253 to 0.003471.
- gradient ratio to noisy drops to 0.7945, so the gain comes with visible
  smoothing risk.

Interpretation:

```text
Physical-scale denoising is useful in high-noise ICCD conditions, but the paper
must describe it as residual suppression, not detail enhancement.
```

### Boundary folders 1 and 10

- Folder 1 supports p99 over physical.
- Folder 10 is a warning case: the q40 Fano rule assigns it to high-condition,
  but p99 has higher PSNR gain than physical on the selected sample.

Interpretation:

```text
The q40 condition split is useful diagnostic evidence but not a final optimized
decision rule.
```

## Claim Boundary

Supported:

```text
Condition-aware checkpoint selection reduces negative-condition failures and
matches visible residual behavior on selected diagnostic samples.
```

Not supported:

```text
The method restores missing weak-light details.
The physical-scale checkpoint should always be used in high-condition folders.
The current q40 Fano threshold is deployment-ready.
```

## Next Step

The next technical experiment should not be a larger network yet. It should be
one of:

1. refine condition scoring using multiple E1 metrics instead of Fano alone;
2. generate condition-scaled synthetic noise for E3.6 training;
3. add visual/residual panels for all ten folders before drafting the paper
   experiment section.
