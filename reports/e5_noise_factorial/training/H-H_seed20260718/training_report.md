# Manifest Denoiser Baseline

## Configuration

- Experiment ID: `e5_H-H_seed20260718`
- Git commit: `985d4aa0a409f4707c2d42be026262128a6ff7e1`
- Train manifest: `reports\e5_noise_factorial\H-H\pairs.csv` / split `train`
- Validation manifest: `reports\e5_noise_factorial\H-H\pairs.csv` / split `val`
- Seed: 20260718
- Device: cuda
- Model parameters: 2625
- Model type: `residual_small`
- Input channels: 1
- Condition column: `none`
- Condition value scale: 1

## Final Result

- Train L1: 0.000368857
- Validation L1: 0.000541326
- Validation PSNR/SSIM: 48.4897 / 0.999802
- Noisy-input PSNR/SSIM on same validation subset: 46.7211 / 0.987830

## Outputs

- Config: `reports\e5_noise_factorial\training\H-H_seed20260718\config.json`
- Metrics: `reports\e5_noise_factorial\training\H-H_seed20260718\metrics.csv`
- Validation rows: `reports\e5_noise_factorial\training\H-H_seed20260718\validation_predictions.csv`
- Best checkpoint: `reports\e5_noise_factorial\training\H-H_seed20260718\checkpoints\best.pth`
- Last checkpoint: `reports\e5_noise_factorial\training\H-H_seed20260718\checkpoints\last.pth`
- Samples: `reports\e5_noise_factorial\training\H-H_seed20260718\samples`

## Claim Boundary

- This is an engineering baseline for manifest/training correctness.
- It does not establish real ICCD denoising performance.
- Do not compare paper models against this run unless they use the same manifest, split, seed policy, and metric implementation.
