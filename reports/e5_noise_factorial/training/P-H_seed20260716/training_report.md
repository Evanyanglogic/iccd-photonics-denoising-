# Manifest Denoiser Baseline

## Configuration

- Experiment ID: `e5_P-H_seed20260716`
- Git commit: `985d4aa0a409f4707c2d42be026262128a6ff7e1`
- Train manifest: `reports\e5_noise_factorial\P-H\pairs.csv` / split `train`
- Validation manifest: `reports\e5_noise_factorial\P-H\pairs.csv` / split `val`
- Seed: 20260716
- Device: cuda
- Model parameters: 2625
- Model type: `residual_small`
- Input channels: 1
- Condition column: `none`
- Condition value scale: 1

## Final Result

- Train L1: 0.000436571
- Validation L1: 0.000682698
- Validation PSNR/SSIM: 47.1946 / 0.999767
- Noisy-input PSNR/SSIM on same validation subset: 45.8163 / 0.996647

## Outputs

- Config: `reports\e5_noise_factorial\training\P-H_seed20260716\config.json`
- Metrics: `reports\e5_noise_factorial\training\P-H_seed20260716\metrics.csv`
- Validation rows: `reports\e5_noise_factorial\training\P-H_seed20260716\validation_predictions.csv`
- Best checkpoint: `reports\e5_noise_factorial\training\P-H_seed20260716\checkpoints\best.pth`
- Last checkpoint: `reports\e5_noise_factorial\training\P-H_seed20260716\checkpoints\last.pth`
- Samples: `reports\e5_noise_factorial\training\P-H_seed20260716\samples`

## Claim Boundary

- This is an engineering baseline for manifest/training correctness.
- It does not establish real ICCD denoising performance.
- Do not compare paper models against this run unless they use the same manifest, split, seed policy, and metric implementation.
