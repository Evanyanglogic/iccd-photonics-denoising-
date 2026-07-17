# Manifest Denoiser Baseline

## Configuration

- Experiment ID: `e5_factorial_probe_smoke`
- Git commit: `985d4aa0a409f4707c2d42be026262128a6ff7e1`
- Train manifest: `reports\e5_noise_factorial\P-L\pairs.csv` / split `train`
- Validation manifest: `reports\e5_noise_factorial\P-L\pairs.csv` / split `val`
- Seed: 20260716
- Device: cuda
- Model parameters: 2625
- Model type: `residual_small`
- Input channels: 1
- Condition column: `none`
- Condition value scale: 1

## Final Result

- Train L1: 0.000516493
- Validation L1: 0.000700619
- Validation PSNR/SSIM: 55.4850 / 0.999446
- Noisy-input PSNR/SSIM on same validation subset: 55.8445 / 0.999537

## Outputs

- Config: `reports\e5_noise_factorial_probe_smoke\config.json`
- Metrics: `reports\e5_noise_factorial_probe_smoke\metrics.csv`
- Validation rows: `reports\e5_noise_factorial_probe_smoke\validation_predictions.csv`
- Best checkpoint: `reports\e5_noise_factorial_probe_smoke\checkpoints\best.pth`
- Last checkpoint: `reports\e5_noise_factorial_probe_smoke\checkpoints\last.pth`
- Samples: `reports\e5_noise_factorial_probe_smoke\samples`

## Claim Boundary

- This is an engineering baseline for manifest/training correctness.
- It does not establish real ICCD denoising performance.
- Do not compare paper models against this run unless they use the same manifest, split, seed policy, and metric implementation.
