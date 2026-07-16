# E2 Synthetic ICCD-Like Pair Generation

## Purpose

Generate paired training/evaluation manifests from sCMOS content/reference
frames by injecting the E1-derived ICCD prior. These data are synthetic
ICCD-like noisy samples, not real ICCD paired measurements.

## Generator

Script:

```powershell
python scripts\generate_iccd_like_synthetic_pairs.py `
  --pairs-csv reports\target_scmos_15ms_500ms_manifest\pairs.csv `
  --source-splits reports\target_scmos_15ms_500ms_manifest\splits.yaml `
  --config configs\iccd_prior_20260319.yaml `
  --output-dir reports\target_scmos_iccd_like_synthetic_512 `
  --range-max 65535 `
  --crop-size 512
```

The script:

- reads the sCMOS tail-index manifest;
- uses the `clean_path` column as the content/reference source;
- applies crop-level dark-offset correction;
- masks and fills bad pixels before noise synthesis;
- injects `ICCDNoiseModel` using `configs/iccd_prior_20260319.yaml`;
- writes `clean/`, `noisy/`, `pairs.csv`, `splits.yaml`, and metrics.

## Variant A: Preserve Offset-Corrected Physical Scale

Command output:

- `reports\target_scmos_iccd_like_synthetic_512\pairs.csv`
- `reports\target_scmos_iccd_like_synthetic_512\splits.yaml`
- `reports\target_scmos_iccd_like_synthetic_512\synthetic_pair_report.md`

Generation summary:

- Pair count: 100.
- Valid fraction mean: 0.998749.
- Clean mean/std: 0.00143434 / 0.014181.
- Noisy mean/std: 0.00166468 / 0.0141772.
- Residual mean/std: 0.000230341 / 0.000718479.
- Clean p99 / noisy p99: 0.0506657 / 0.0504966.

B0 result:

- PSNR mean/std: 62.4931 / 0.8690 dB.
- SSIM mean/std: 0.999266 / 0.000007.
- Residual mean/std mean: 0.000230336 / 0.00071814.

Interpretation:

- This version is reproducible and dataloader-compatible.
- It is probably too easy for model training because the sCMOS content remains
  very dark after offset correction.

## Variant B: Content p99 Normalized to 0.25

Command:

```powershell
python scripts\generate_iccd_like_synthetic_pairs.py `
  --pairs-csv reports\target_scmos_15ms_500ms_manifest\pairs.csv `
  --source-splits reports\target_scmos_15ms_500ms_manifest\splits.yaml `
  --config configs\iccd_prior_20260319.yaml `
  --output-dir reports\target_scmos_iccd_like_synthetic_512_p99_0p25 `
  --range-max 65535 `
  --crop-size 512 `
  --content-p99-target 0.25
```

Generation summary:

- Pair count: 100.
- Valid fraction mean: 0.998749.
- Clean mean/std: 0.00755683 / 0.0754629.
- Noisy mean/std: 0.00776364 / 0.0751726.
- Residual mean/std: 0.00020681 / 0.00146363.
- Clean p99 / noisy p99: 0.25 / 0.249897.

B0 result:

- PSNR mean/std: 56.6277 / 0.5780 dB.
- SSIM mean/std: 0.999304 / 0.000014.
- Residual mean/std mean: 0.000206838 / 0.00146277.

Comparison reference:

- Real gated ICCD repeated-frame surrogate B0:
  - PSNR mean/std: 56.4087 / 5.2795 dB.
  - SSIM mean/std: 0.995732 / 0.004572.
  - Residual mean/std mean: -4.04601e-05 / 0.00182167.

Interpretation:

- The p99-normalized synthetic set has residual standard deviation closer to
  the real gated ICCD repeated-frame surrogate than the physical-scale variant.
- This version is the better first candidate for downstream synthetic-data
  denoising experiments.
- The normalization must be reported as content intensity normalization, not as
  real exposure calibration.

## Current Decision

Use `reports\target_scmos_iccd_like_synthetic_512_p99_0p25` as the first
training-source candidate after the PyTorch training pipeline is audited.

Keep `reports\target_scmos_iccd_like_synthetic_512` as a conservative ablation
that preserves the dark-offset-corrected physical scale.

Do not claim real paired ICCD denoising performance from either synthetic set.
They support controlled synthetic pretraining or augmentation only.
