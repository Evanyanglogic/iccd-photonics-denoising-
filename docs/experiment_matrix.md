# Experiment Matrix

## E1: Calibration Statistics

Goal: estimate and report real ICCD noise behavior from controlled acquisition.

Required data:

- Dark frames at each gain/exposure/gate setting.
- Flat-field frames at multiple illumination levels.
- Repeated frames for temporal variance estimation.
- Matching sCMOS sequences for platform comparison.

Outputs:

- Mean-variance curves.
- Dark-current/dark-count distribution.
- Spatial fixed-pattern maps.
- Power spectral density.
- Row/column correlation where applicable.
- Autocorrelation maps.

## E2: Synthetic Noise Fidelity

Compare these sources:

- Poisson-Gaussian baseline.
- Existing sCMOS noise model.
- ICCD physical-prior model.
- ICCD-aware PNGAN generated noise.

Metrics:

- Mean error.
- Variance error.
- Histogram distance.
- PSD distance.
- Autocorrelation distance.
- Brightness-bin noise residual error.
- Visual panels: clean, real noisy, synthetic noisy, generated noisy, residual.

## E3: Downstream Denoising

Train the same denoiser under different training data sources:

- Real paired data only, if enough pairs exist.
- Poisson-Gaussian synthetic data.
- sCMOS synthetic data.
- ICCD physical-prior synthetic data.
- ICCD-aware PNGAN generated data.
- Mixed real + generated data.

Test only on held-out real ICCD data.

Metrics:

- PSNR and SSIM if clean reference is available.
- LPIPS or perceptual quality if appropriate.
- Edge preservation.
- Brightness-bin PSNR.
- Noise residual statistics.
- Task-specific metric if the images serve detection/localization.

## E4: Ablation

Ablate:

- No MCP gain fluctuation.
- No phosphor diffusion.
- No dark-count component.
- No adversarial noise-domain alignment.
- No denoiser-domain/content alignment.
- sCMOS prior instead of ICCD prior.

Expected table:

| Variant | Noise-stat fidelity | Real ICCD denoising | Main failure mode |
|---|---:|---:|---|
| Full model | TBD | TBD | TBD |
| No MCP | TBD | TBD | TBD |
| No phosphor | TBD | TBD | TBD |
| No adversarial alignment | TBD | TBD | TBD |

