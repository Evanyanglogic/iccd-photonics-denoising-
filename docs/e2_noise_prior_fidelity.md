# E2 Noise Prior Fidelity

Date: 2026-07-16

## Surrogate Pair Construction

Current gated ICCD data under `D:\iccd\data\20260319` do not contain true clean
ground truth. For E2.2, a repeated-frame surrogate was built:

```powershell
python scripts\build_repeated_frame_surrogate_pairs.py `
  --root D:\iccd\data\20260319 `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --output-dir reports\gated_iccd_20260319_surrogate_pairs `
  --train-frames 100 `
  --max-heldout-frames 8 `
  --heldout-stride 10 `
  --crop-size 512
```

Output:

- Folders: 10
- Pairs: 80
- Clean surrogate: mean of the first 100 repeated frames in each folder
- Real noisy surrogate: held-out frames 101, 111, 121, 131, 141, 151, 161, 171
- Pair manifest: `reports/gated_iccd_20260319_surrogate_pairs/pairs.csv`

Claim boundary:

- These are not true noisy-clean supervised pairs.
- They are suitable for checking whether synthetic noise resembles held-out
  temporal residuals from the same repeated-frame condition.
- They should not be used as final supervised denoising evidence without this
  limitation.

## E2.2-A: All Priors E1-Calibrated

Command:

```powershell
python scripts\compare_noise_priors.py `
  --pairs-csv reports\gated_iccd_20260319_surrogate_pairs\pairs.csv `
  --config configs\iccd_prior_20260319.yaml `
  --output-dir reports\gated_iccd_20260319_surrogate_noise_priors `
  --range-max 65535 `
  --bins 8
```

Summary:

| prior | pairs | PSNR mean | SSIM mean | residual std error | histogram L1 | PSD L1 |
|---|---:|---:|---:|---:|---:|---:|
| iccd_prior | 80 | 50.1749 | 0.987499 | 0.000915476 | 0.0412771 | 0.000895025 |
| poisson_gaussian | 80 | 50.2121 | 0.987652 | 0.000906913 | 0.0415464 | 0.000895055 |
| scmos_like | 80 | 50.2120 | 0.987648 | 0.000907173 | 0.0427792 | 0.000899347 |

Interpretation:

- When all priors are calibrated to the same E1 variance scale, the current
  runnable models are nearly equivalent.
- This does not support a strong claim that the current simplified ICCD model is
  intrinsically better than a re-fitted Poisson-Gaussian model.
- It does support the need for careful calibration before comparing noise priors.

## E2.2-B: Generic Defaults vs E1-Calibrated ICCD Prior

Command:

```powershell
python scripts\compare_noise_priors.py `
  --pairs-csv reports\gated_iccd_20260319_surrogate_pairs\pairs.csv `
  --config configs\iccd_prior_comparison_20260319.yaml `
  --output-dir reports\gated_iccd_20260319_surrogate_noise_priors_comparison `
  --range-max 65535 `
  --bins 8
```

Summary:

| prior | pairs | PSNR mean | SSIM mean | residual std error | histogram L1 | PSD L1 |
|---|---:|---:|---:|---:|---:|---:|
| iccd_prior | 80 | 50.1749 | 0.987499 | 0.000915476 | 0.0412771 | 0.000895025 |
| poisson_gaussian | 80 | 35.6532 | 0.770608 | 0.0149333 | 1.25121 | 0.000859825 |
| scmos_like | 80 | 25.4691 | 0.264679 | 0.0507340 | 1.78609 | 0.000855631 |

Interpretation:

- The E1-calibrated ICCD prior is much closer to held-out repeated-frame
  residuals than generic default Poisson-Gaussian and sCMOS-like priors in PSNR,
  SSIM, residual standard-deviation error, and residual histogram distance.
- PSD L1 is similar across priors and is not the strongest evidence here.
- The current result supports the claim that device-calibrated noise level and
  residual distribution matter.
- It does not yet prove that all ICCD-specific components are necessary, because
  the runnable ICCD model currently does not inject the empirical fixed-pattern
  map and has `phosphor_sigma = 0` due weak measured residual spatial
  correlation.

## Paper-Safe Claim

Safe wording:

```text
基于重复帧统计校准的 ICCD 噪声先验，相比未校准的通用 Poisson-Gaussian
和 sCMOS-like 先验，更接近真实门控 ICCD held-out 重复帧残差的幅度与分布。
```

Do not yet claim:

```text
完整 ICCD 物理链路模型已显著优于所有重新标定的通用噪声模型。
```

That stronger claim requires E4.1 ablation with fixed-pattern injection,
over-dispersion terms, and optionally matching dark/flat calibration.
