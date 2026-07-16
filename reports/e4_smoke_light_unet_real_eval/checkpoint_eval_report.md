# Denoiser Checkpoint Evaluation

## Inputs

- Label: `e4_smoke_light_unet_real_eval`
- Checkpoint: `reports\e4_smoke_light_unet\checkpoints\best.pth`
- Pair manifest: `reports\gated_iccd_20260319_surrogate_pairs\pairs.csv`
- Model parameters: 29681
- Training experiment ID: `e4_smoke_light_unet`

## Summary

- Pair count: 2
- Model PSNR/SSIM: 58.4628 / 0.998387
- Noisy-input PSNR/SSIM: 58.5293 / 0.998421
- PSNR gain mean/std: -0.0665 / 0.0137
- SSIM gain mean/std: -0.000034 / 0.000006
- Residual mean/std: -0.000149767 / 0.00118406

## Outputs

- Metrics CSV: `reports\e4_smoke_light_unet_real_eval\checkpoint_eval_metrics.csv`
- Summary JSON: `reports\e4_smoke_light_unet_real_eval\checkpoint_eval_summary.json`
- Samples: `reports\e4_smoke_light_unet_real_eval\samples`

## Claim Boundary

- This evaluates cross-domain behavior on repeated-frame ICCD surrogate pairs.
- The surrogate clean image is a repeated-frame mean, not a true clean exposure.
- Use this as a gate before claiming real ICCD denoising performance.
