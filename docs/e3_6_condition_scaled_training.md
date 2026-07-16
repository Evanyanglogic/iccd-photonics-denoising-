# E3.6 Condition-Scaled Synthetic Training

## Purpose

Move the E3.5 score-q50 finding from post-hoc checkpoint switching into the
synthetic training data. Instead of training one model on a single global noise
strength, generate ICCD-like synthetic pairs whose residual standard deviation
matches the repeated-frame surrogate statistics of different gated ICCD
conditions.

## Data Generation

Script updated:

```text
scripts/generate_iccd_like_synthetic_pairs.py
```

New options:

- `--condition-score-csv`
- `--condition-scale-mode residual_std`
- `--condition-target-column mean_residual_std`
- `--allow-zero-valid`
- `--zero-mean-residual-before-scale`

Two datasets were generated from the existing p99-normalized synthetic clean
TIFFs:

| Dataset | Output | Change |
|---|---|---|
| E3.6-A | `reports/target_scmos_iccd_like_synthetic_512_condition_scaled_q50` | Scale residual std to condition target |
| E3.6-B | `reports/target_scmos_iccd_like_synthetic_512_condition_scaled_q50_zero_mean` | Subtract residual mean before std scaling |

The condition target is `mean_residual_std` from:

```text
reports/e3_5_condition_score/condition_score_folders.csv
```

## Synthetic Data Check

E3.6-A matched the target residual standard deviation by condition:

| Folder | Target residual std | Actual residual std |
|---:|---:|---:|
| 13 | 0.000609 | 0.000609 |
| 2 | 0.000692 | 0.000692 |
| 11 | 0.000945 | 0.000945 |
| 1 | 0.001291 | 0.001290 |
| 10 | 0.001206 | 0.001206 |
| 9 | 0.001509 | 0.001505 |
| 4 | 0.001951 | 0.001937 |
| 8 | 0.002233 | 0.002213 |
| 7 | 0.002498 | 0.002468 |
| 5 | 0.003956 | 0.003662 |

Folder 5 is slightly below target because the noise scale was clamped at 3.0.

E3.6-A synthetic residual mean remained positive:

```text
mean residual_mean = +0.000308
```

E3.6-B reduced this bias:

```text
mean residual_mean = +0.000176
```

It did not reach exact zero because black background pixels are clipped at 0
after adding negative residuals.

## Training

Both runs used the same small CNN setting as the previous p99 and physical
baselines:

```text
channels = 16
depth = 3
parameters = 2625
epochs = 100
seed = 20260716
device = cpu
```

Training outputs:

- `reports/e3_6_condition_scaled_q50_smallcnn_100ep`
- `reports/e3_6_condition_scaled_q50_zero_mean_smallcnn_100ep`

## Real Surrogate Evaluation

Evaluation manifest:

```text
reports/gated_iccd_20260319_surrogate_pairs/pairs.csv
```

| Model | Mean PSNR gain | Std | Positive pairs | Interpretation |
|---|---:|---:|---:|---|
| p99 baseline | +0.0392 dB | 0.0289 | 75/80 | Stable but weak |
| physical baseline | +0.3431 dB | 0.5249 | 52/80 | Strong but condition-unstable |
| E3.6-A condition-scaled | -0.0321 dB | 0.0636 | 13/80 | Fails; residual bias/domain gap |
| E3.6-B zero-mean condition-scaled | -0.0066 dB | 0.0249 | 24/80 | Bias reduced, still below p99 |

E3.6-B folder-level gains:

| Folder | Mean PSNR gain | Positive pairs |
|---:|---:|---:|
| 1 | -0.0026 | 3/8 |
| 2 | -0.0448 | 1/8 |
| 4 | -0.0040 | 2/8 |
| 5 | -0.0018 | 1/8 |
| 7 | -0.0030 | 0/8 |
| 8 | -0.0026 | 2/8 |
| 9 | -0.0008 | 3/8 |
| 10 | +0.0022 | 5/8 |
| 11 | -0.0151 | 3/8 |
| 13 | +0.0060 | 4/8 |

## Interpretation

E3.6-A/B do not yet support the claim that condition-scaled synthetic training
improves real ICCD surrogate denoising. They do support two useful conclusions:

1. Matching only residual standard deviation is insufficient.
2. Synthetic residual mean and clipping bias materially affect cross-domain
   transfer.

The current small CNN receives no explicit condition input, so it must infer
condition from image content alone. That is weaker than the E3.5 diagnostic
hybrid, which uses folder-level condition statistics directly.

## Claim Boundary

Supported:

```text
Condition-scaled residual std alone is not enough for a single unconditioned
small CNN to beat the conservative p99 baseline on real gated ICCD surrogate
pairs.
```

Not supported:

```text
Condition-aware synthetic training has failed as a direction.
```

The current failed result is specifically for residual-std-only scaling and an
unconditioned 2625-parameter CNN.

## Next Step

E3.6-C should add explicit condition information or a stronger training target:

1. add a constant condition-score channel to the model input; or
2. train three condition bands separately and compare to q50 checkpoint
   switching; or
3. add a residual-mean calibration step that avoids clipping bias before
   training.

The most paper-relevant next experiment is option 1, because it turns the q50
diagnostic result into a proper condition-aware model rather than a post-hoc
checkpoint selector.
