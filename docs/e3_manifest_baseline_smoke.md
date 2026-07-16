# E3 Manifest Baseline Smoke Run

## Purpose

Verify that the ICCD denoising training path can consume manifest-backed data
and produce reproducible training artifacts before using heavier MIRNet/PNGAN
models.

This is an engineering smoke test, not a paper performance result.

## Script

```powershell
python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e3_manifest_baseline_smoke `
  --output-dir reports\e3_manifest_baseline_smoke `
  --epochs 1 `
  --batch-size 2 `
  --patch-size 128 `
  --val-patch-size 256 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --channels 16 `
  --depth 3 `
  --device cpu
```

Default data source:

- `reports\target_scmos_iccd_like_synthetic_512_p99_0p25\pairs.csv`
- `reports\target_scmos_iccd_like_synthetic_512_p99_0p25\splits.yaml`

## Outputs

- `reports\e3_manifest_baseline_smoke\config.json`
- `reports\e3_manifest_baseline_smoke\metrics.csv`
- `reports\e3_manifest_baseline_smoke\validation_predictions.csv`
- `reports\e3_manifest_baseline_smoke\training_report.md`
- `reports\e3_manifest_baseline_smoke\checkpoints\best.pth`
- `reports\e3_manifest_baseline_smoke\checkpoints\last.pth`
- `reports\e3_manifest_baseline_smoke\samples\*.tif`

## Result

- Train samples: 85.
- Validation samples: 8.
- Smoke training batches: 2.
- Smoke validation batches: 2.
- Model parameters: 2,625.
- Train L1: 0.000410107.
- Validation L1: 0.000455504.
- Validation PSNR/SSIM: 55.7709 dB / 0.999513.
- Noisy-input PSNR/SSIM on the same validation subset: 55.7045 dB / 0.999232.

## Interpretation

- The manifest-driven training entry works end to end.
- The model starts from a noisy-input identity baseline by zero-initializing the
  final residual layer, so smoke training does not begin from arbitrary image
  corruption.
- The small PSNR gain in this smoke run only proves the training loop can reduce
  L1 on the synthetic validation subset; it is not a publishable denoising
  claim.

## Next Gate

Run a full synthetic baseline after committing the trainer:

- train on all 85 train pairs;
- validate on all 8 validation pairs;
- use at least 20 epochs for the small residual baseline;
- compare against the B0 no-model synthetic result;
- then decide whether to replace the small CNN with MIRNet/SMNet.
