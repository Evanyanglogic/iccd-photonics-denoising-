# E3 Real ICCD Surrogate Checkpoint Evaluation

## Purpose

Evaluate whether synthetic-trained small-CNN denoisers transfer to real gated
ICCD repeated-frame surrogate pairs.

Surrogate definition:

- clean/reference: mean of the first 100 repeated frames per folder;
- noisy input: held-out repeated frames;
- not a true clean long-exposure target.

## Evaluation Script

```powershell
python scripts\evaluate_manifest_denoiser_checkpoint.py `
  --checkpoint <checkpoint.pth> `
  --pairs-csv reports\gated_iccd_20260319_surrogate_pairs\pairs.csv `
  --output-dir <output_dir> `
  --experiment-label <label> `
  --device cpu
```

The script reports model PSNR/SSIM, noisy-input PSNR/SSIM, per-pair gains, and
best/median/worst-gain sample TIFFs.

## p99-Normalized Synthetic Training

Checkpoint:

- `reports\e3_manifest_baseline_smallcnn_100ep\checkpoints\best.pth`

Output:

- `reports\e3_real_surrogate_eval_p99_smallcnn_100ep`

Result on 80 real ICCD surrogate pairs:

- Model PSNR/SSIM: 56.4479 dB / 0.995780.
- Noisy-input PSNR/SSIM: 56.4087 dB / 0.995732.
- Mean PSNR gain/std: 0.0392 / 0.0287 dB.
- Positive-gain pairs: 75 / 80.
- Negative-gain pairs: 5 / 80.

Folder-level mean PSNR gain:

| folder | mean gain dB |
|---:|---:|
| 1 | 0.0189 |
| 2 | 0.0577 |
| 4 | 0.0490 |
| 5 | 0.0718 |
| 7 | 0.0442 |
| 8 | 0.0478 |
| 9 | 0.0325 |
| 10 | 0.0272 |
| 11 | 0.0419 |
| 13 | 0.0013 |

Interpretation:

- Transfer is stable but extremely small.
- It mostly behaves like a near-identity denoiser on real surrogate data.

## Strict Physical-Scale Synthetic Training

Checkpoint:

- `reports\e3_manifest_baseline_physical_scale_100ep\checkpoints\best.pth`

Output:

- `reports\e3_real_surrogate_eval_physical_smallcnn_100ep`

Result on 80 real ICCD surrogate pairs:

- Model PSNR/SSIM: 56.7517 dB / 0.996620.
- Noisy-input PSNR/SSIM: 56.4087 dB / 0.995732.
- Mean PSNR gain/std: 0.3431 / 0.5216 dB.
- Positive-gain pairs: 52 / 80.
- Negative-gain pairs: 28 / 80.

Folder-level mean PSNR gain:

| folder | mean gain dB |
|---:|---:|
| 1 | -0.0077 |
| 2 | -0.1395 |
| 4 | 0.3745 |
| 5 | 1.4198 |
| 7 | 0.9774 |
| 8 | 0.8227 |
| 9 | 0.0655 |
| 10 | 0.0088 |
| 11 | -0.0553 |
| 13 | -0.0354 |

Interpretation:

- Mean transfer is stronger than the p99-trained model.
- The improvement is condition-dependent and unstable: it helps several
  folders strongly but degrades many individual pairs.
- This suggests the model is learning a brightness/condition-dependent residual
  correction, not a uniformly valid real ICCD denoiser.

## Current Gate Decision

Do not move directly to MIRNet/SMNet as a paper model yet.

The next controlled step should stratify or condition the baseline by real ICCD
folder statistics:

- folder mean signal;
- temporal residual standard deviation;
- fixed-pattern ratio;
- Fano approximation.

Then evaluate whether a condition-aware input or training split reduces the
folder-dependent failures.

## Follow-Up: Condition-Stratified Gain Analysis

Completed follow-up:

- `docs/e3_condition_gain_analysis.md`
- `reports\e3_condition_gain_analysis\condition_gain_report.md`

Key result:

- p99 checkpoint: stable but tiny gain, 0.0392 dB mean folder PSNR gain.
- strict physical-scale checkpoint: stronger average gain, 0.3431 dB, but only
  6/10 folders improve.
- For the physical-scale checkpoint, mean folder gain is highly correlated with
  E1 condition statistics: temporal std mean `r = 0.9726`, fixed/temporal std
  ratio `r = 0.9693`, fixed-map std `r = 0.9504`, and Fano `r = 0.9495`.

Updated decision:

Do not treat the current model as a general detail restoration model. The next
experiment should test a minimal condition-aware strategy before any larger
generic denoising architecture.
