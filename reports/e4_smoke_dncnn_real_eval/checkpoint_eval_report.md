# Denoiser Checkpoint Evaluation

## Inputs

- Label: `e4_smoke_dncnn_real_eval`
- Checkpoint: `reports\e4_smoke_dncnn\checkpoints\best.pth`
- Pair manifest: `reports\gated_iccd_20260319_surrogate_pairs\pairs.csv`
- Model parameters: 7265
- Training experiment ID: `e4_smoke_dncnn`

## Summary

- Pair count: 2
- Model PSNR/SSIM: 58.2530 / 0.998287
- Noisy-input PSNR/SSIM: 58.5293 / 0.998421
- PSNR gain mean/std: -0.2763 / 0.0303
- SSIM gain mean/std: -0.000135 / 0.000014
- Residual mean/std: -0.00030066 / 0.0011851

## Outputs

- Metrics CSV: `reports\e4_smoke_dncnn_real_eval\checkpoint_eval_metrics.csv`
- Summary JSON: `reports\e4_smoke_dncnn_real_eval\checkpoint_eval_summary.json`
- Samples: `reports\e4_smoke_dncnn_real_eval\samples`

## Claim Boundary

- This evaluates cross-domain behavior on repeated-frame ICCD surrogate pairs.
- The surrogate clean image is a repeated-frame mean, not a true clean exposure.
- Use this as a gate before claiming real ICCD denoising performance.
