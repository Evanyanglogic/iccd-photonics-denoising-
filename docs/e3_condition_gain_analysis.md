# E3 Condition-Stratified Gain Analysis

## Purpose

Link real ICCD surrogate denoising gains from E3 to E1 folder-level device
statistics. This checks whether current denoiser behavior is condition-dependent
before any larger MIRNet/SMNet/PNGAN run.

## Script

```powershell
python scripts\analyze_condition_gain.py `
  --eval-csv p99=reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --eval-csv physical=reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --mean-variance-csv reports\gated_iccd_20260319_mean_variance\mean_variance_summary.csv `
  --fixed-pattern-csv reports\gated_iccd_20260319_fixed_pattern\fixed_pattern_correction_summary.csv `
  --noise-summary-csv reports\gated_iccd_20260319_noise_summary\single_condition_noise_summary.csv `
  --output-dir reports\e3_condition_gain_analysis
```

Outputs:

- `reports\e3_condition_gain_analysis\condition_gain_summary.csv`
- `reports\e3_condition_gain_analysis\condition_gain_correlations.csv`
- `reports\e3_condition_gain_analysis\condition_gain_report.md`

## Key Results

### p99 synthetic checkpoint

- Mean folder PSNR gain: 0.0392 dB.
- Positive-gain folders: 10/10.
- Largest Pearson correlations with folder mean PSNR gain:
  - mean signal: 0.6694;
  - temporal std mean: 0.6691;
  - fixed-map std: 0.6685;
  - fixed/temporal std ratio: 0.6601.

Interpretation: stable but tiny improvement; it behaves close to identity and
does not provide strong real-surrogate denoising evidence.

### strict physical-scale checkpoint

- Mean folder PSNR gain: 0.3431 dB.
- Positive-gain folders: 6/10.
- Negative-gain folders: 1, 2, 11, 13.
- Strongest Pearson correlations with folder mean PSNR gain:
  - temporal std mean: 0.9726;
  - fixed/temporal std ratio: 0.9693;
  - fixed-map std: 0.9504;
  - spatial mean std: 0.9500;
  - Fano: 0.9495;
  - mean signal: 0.9478.

Interpretation: the checkpoint helps mostly in higher-noise / stronger
fixed-pattern folders and fails or barely helps in low-Fano / weak-fixed-pattern
folders. This is condition-dependent denoising behavior, not general
low-exposure detail restoration.

## Gate Decision

Do not move directly to a larger generic network.

The next model-side step should be a minimal condition-aware experiment:

1. keep the current manifest training/evaluation loop;
2. add folder/condition statistics to the manifest or config;
3. test a simple condition-gated strategy before architecture changes.

Candidate minimal variants:

- evaluate a folder-statistics gate that applies the physical checkpoint only
  when predicted benefit is positive;
- train/evaluate separate low-noise and high-noise condition subsets;
- add condition-aware synthetic noise scaling before training a larger model.

## Claim Boundary

This result supports:

```text
The current denoising gain is strongly condition-dependent and tracks real ICCD
noise statistics.
```

It does not support:

```text
The model reconstructs missing weak-light details.
The current checkpoint is a uniformly valid real ICCD denoiser.
```
