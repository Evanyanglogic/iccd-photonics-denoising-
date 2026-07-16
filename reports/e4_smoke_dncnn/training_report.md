# Manifest Denoiser Baseline

## Configuration

- Experiment ID: `e4_smoke_dncnn`
- Git commit: `257c209bb4b1d69c21cdde736ebfb99044494772`
- Train manifest: `reports/target_scmos_iccd_like_synthetic_512_p99_0p25/pairs.csv` / split `train`
- Validation manifest: `reports/target_scmos_iccd_like_synthetic_512_p99_0p25/pairs.csv` / split `val`
- Seed: 20260716
- Device: cpu
- Model parameters: 7265
- Model type: `dncnn`
- Input channels: 1
- Condition column: `none`
- Condition value scale: 1

## Final Result

- Train L1: 0.000237476
- Validation L1: 0.000240591
- Validation PSNR/SSIM: 68.9706 / 0.999542
- Noisy-input PSNR/SSIM on same validation subset: 67.7009 / 0.999314

## Outputs

- Config: `reports\e4_smoke_dncnn\config.json`
- Metrics: `reports\e4_smoke_dncnn\metrics.csv`
- Validation rows: `reports\e4_smoke_dncnn\validation_predictions.csv`
- Best checkpoint: `reports\e4_smoke_dncnn\checkpoints\best.pth`
- Last checkpoint: `reports\e4_smoke_dncnn\checkpoints\last.pth`
- Samples: `reports\e4_smoke_dncnn\samples`

## Claim Boundary

- This is an engineering baseline for manifest/training correctness.
- It does not establish real ICCD denoising performance.
- Do not compare paper models against this run unless they use the same manifest, split, seed policy, and metric implementation.
