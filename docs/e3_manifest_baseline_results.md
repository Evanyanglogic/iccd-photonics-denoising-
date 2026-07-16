# E3 Manifest Baseline Results

## Dataset

Training and validation use the first p99-normalized synthetic ICCD-like
manifest:

- `reports\target_scmos_iccd_like_synthetic_512_p99_0p25\pairs.csv`
- `reports\target_scmos_iccd_like_synthetic_512_p99_0p25\splits.yaml`

Claim boundary:

- sCMOS content/reference frames;
- ICCD-like noise injected from the E1-derived prior;
- not real paired ICCD denoising data.

## Model

Script:

- `scripts\train_manifest_denoiser_baseline.py`

Baseline:

- lightweight residual CNN;
- 2,625 trainable parameters for the current `channels=16`, `depth=3` setting;
- final residual layer is zero-initialized, so the model starts from the
  noisy-input identity baseline;
- L1 loss only;
- float-domain PSNR/SSIM from `src.iccd_eval.metrics`.

## 20-Epoch Run

Command:

```powershell
python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e3_manifest_baseline_smallcnn_20ep `
  --output-dir reports\e3_manifest_baseline_smallcnn_20ep `
  --epochs 20 `
  --batch-size 4 `
  --patch-size 128 `
  --val-patch-size 256 `
  --channels 16 `
  --depth 3 `
  --device cpu
```

Result:

- Train L1: 0.000195643.
- Validation L1: 0.000275575.
- Validation PSNR/SSIM: 54.1043 dB / 0.999952.
- Noisy-input PSNR/SSIM on same validation subset: 53.8926 dB / 0.999294.
- PSNR gain over noisy input: about 0.2117 dB.

Interpretation:

- Training works, but the 20-epoch run does not meet the provisional 0.3 dB
  improvement threshold.

## 100-Epoch Run

Command:

```powershell
python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e3_manifest_baseline_smallcnn_100ep `
  --output-dir reports\e3_manifest_baseline_smallcnn_100ep `
  --epochs 100 `
  --batch-size 4 `
  --patch-size 128 `
  --val-patch-size 256 `
  --channels 16 `
  --depth 3 `
  --device cpu
```

Result:

- Train L1: 0.000143928.
- Validation L1: 0.000253492.
- Validation PSNR/SSIM: 54.1966 dB / 0.999948.
- Noisy-input PSNR/SSIM on same validation subset: 53.8926 dB / 0.999294.
- PSNR gain over noisy input: about 0.3040 dB.
- Runtime on CPU: about 58 seconds.

Interpretation:

- The small residual CNN barely passes the provisional synthetic-validation
  PSNR threshold.
- The gain is modest and should be treated as a sanity baseline, not a model
  contribution.
- Since the data are synthetic and SSIM is already near saturation, stronger
  claims require either a harder validation set or real ICCD paired/surrogate
  validation.

## Strict Physical-Scale 100-Epoch Run

Command:

```powershell
python scripts\train_manifest_denoiser_baseline.py `
  --experiment-id e3_manifest_baseline_physical_scale_100ep `
  --train-pairs reports\target_scmos_iccd_like_synthetic_512\pairs.csv `
  --train-splits reports\target_scmos_iccd_like_synthetic_512\splits.yaml `
  --output-dir reports\e3_manifest_baseline_physical_scale_100ep `
  --epochs 100 `
  --batch-size 4 `
  --patch-size 128 `
  --val-patch-size 256 `
  --channels 16 `
  --depth 3 `
  --device cpu
```

Final result:

- Train L1: 0.000286588.
- Validation L1: 0.000338010.
- Validation PSNR/SSIM: 60.1930 dB / 0.999284.
- Noisy-input PSNR/SSIM on same validation subset: 59.5277 dB / 0.999232.
- Final PSNR gain over noisy input: about 0.6653 dB.

Best observed validation row:

- Best validation PSNR: 60.3309 dB at epoch 93.
- Best observed PSNR gain over noisy input: about 0.8032 dB.

Interpretation:

- The strict physical-scale variant is easier because its synthetic noise is
  weaker and its validation PSNR starts much higher.
- Its larger PSNR gain does not mean it is a better proxy for real ICCD data.
- Keep it as an ablation for training-source sensitivity.
- The p99-normalized set remains the first candidate when the goal is matching
  real gated ICCD repeated-frame residual magnitude.

## Next Step

Use the small-CNN results as the minimum supervised baseline. The next useful
comparison is not a larger arbitrary network yet; it is a controlled validation
comparison:

- later real ICCD surrogate validation;
- p99-normalized synthetic training tested on real ICCD surrogate residuals;
- strict physical-scale synthetic training tested on the same real ICCD
  surrogate residuals;
- then MIRNet/SMNet only if the small baseline exposes a clear capacity limit.
