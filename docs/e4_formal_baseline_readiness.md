# E4 Formal Baseline Readiness

## Purpose

E4 starts the formal network-baseline stage required before any model-level
claim. The first step is engineering readiness: the manifest trainer and
checkpoint evaluator must support more than the original small residual CNN.

## Code Changes

`scripts/train_manifest_denoiser_baseline.py` now supports:

- `--model-type residual_small`
- `--model-type dncnn`
- `--model-type light_unet`

The checkpoint evaluator and condition-blend loader now read `model_type` from
checkpoint config, while remaining compatible with older checkpoints that do
not have this field.

## Smoke Tests

Commands:

```powershell
python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e4_smoke_dncnn `
  --model-type dncnn `
  --channels 16 `
  --depth 5 `
  --epochs 1 `
  --max-train-batches 1 `
  --max-val-batches 1 `
  --batch-size 2 `
  --patch-size 64 `
  --val-patch-size 64 `
  --output-dir reports\e4_smoke_dncnn `
  --device cpu

python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e4_smoke_light_unet `
  --model-type light_unet `
  --channels 8 `
  --depth 4 `
  --epochs 1 `
  --max-train-batches 1 `
  --max-val-batches 1 `
  --batch-size 2 `
  --patch-size 64 `
  --val-patch-size 64 `
  --output-dir reports\e4_smoke_light_unet `
  --device cpu
```

Smoke results:

| Model | Parameters | Synthetic val PSNR/SSIM | Same-subset noisy PSNR/SSIM |
|---|---:|---:|---:|
| DnCNN smoke | 7,265 | 68.9706 / 0.999542 | 67.7009 / 0.999314 |
| Light U-Net smoke | 29,681 | 68.9383 / 0.999537 | 67.7009 / 0.999314 |

The corresponding two-pair real surrogate smoke evaluations were negative:

| Model | Real surrogate smoke PSNR gain |
|---|---:|
| DnCNN smoke | -0.2763 dB |
| Light U-Net smoke | -0.0665 dB |

These real-surrogate numbers are not performance claims because the runs used
only one train batch and two evaluation pairs. They only verify that checkpoint
loading and evaluation support the new architectures.

## Next Formal Runs

Run the real E4 baselines in this order:

1. DnCNN on p99 synthetic;
2. DnCNN on physical synthetic;
3. Light U-Net on p99 synthetic;
4. Light U-Net on physical synthetic;
5. evaluate all four checkpoints on the 80-pair real gated ICCD surrogate set;
6. run post-hoc hard/linear p99-physical condition selection for each
   architecture pair.

The E3.7 reporting protocol remains mandatory for E4: pair metrics,
folder-level metrics, positive folder counts, residual statistics, and
gradient-ratio smoothing checks.
