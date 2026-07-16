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
- `reports/e3_6_condition_channel_q50_zero_mean_smallcnn_100ep`
- `reports/e3_6_condition_channel_q50_zero_mean_scale3_smallcnn_100ep`

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
| E3.6-C condition channel raw score | -0.0487 dB | 0.0701 | 12/80 | Worse; condition input over-drives correction |
| E3.6-C condition channel score/3 | -0.0102 dB | 0.0299 | 20/80 | Less biased, still below p99 |

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

E3.6-C with raw condition score made high-condition folders worse, especially
folder 5, because the model learned an overly negative correction from the
condition channel. Scaling the condition channel by 3 reduced this failure but
still did not beat the conservative p99 baseline.

## Condition Blend Diagnostic

Because E3.6-A/B/C did not beat p99 as a single trained model, E3.6-D evaluated
whether the condition strategy itself remains useful by blending the existing
p99 and physical checkpoints at inference time:

```text
scripts/evaluate_condition_blend.py
```

Outputs:

- `reports/e3_6_condition_blend_p99_physical/condition_blend_metrics.csv`
- `reports/e3_6_condition_blend_p99_physical/condition_blend_summary.csv`
- `reports/e3_6_condition_blend_p99_physical/condition_blend_report.md`

| Strategy | Mean PSNR gain | Positive folders | Positive pairs | Mean physical alpha |
|---|---:|---:|---:|---:|
| score q40-q60 linear blend | +0.3808 dB | 10/10 | 75/80 | 0.472 |
| score q50 hard blend | +0.3807 dB | 10/10 | 75/80 | 0.500 |
| score q30-q70 linear blend | +0.3766 dB | 10/10 | 75/80 | 0.447 |
| score q20-q80 linear blend | +0.3686 dB | 10/10 | 75/80 | 0.438 |

The q40-q60 continuous blend is numerically the best, but the improvement over
hard q50 is only about 0.0001 dB. The practical conclusion is that the
condition score selection is useful, but continuous linear blending does not
materially improve over hard q50 on the current ten-folder surrogate set.

## Interpretation

E3.6-A/B/C do not support the claim that condition-scaled synthetic training
improves real ICCD surrogate denoising with the current small CNN. They do
support three useful conclusions:

1. Matching only residual standard deviation is insufficient.
2. Synthetic residual mean and clipping bias materially affect cross-domain
   transfer.
3. A naive condition-score input channel is not enough; it can over-drive
   brightness correction in high-condition folders.

The E3.6-D blend confirms that the condition score is still useful when applied
directly to existing p99/physical outputs. The problem is not the condition
score itself; the problem is transferring residual-std-only synthetic training
into a single unconditioned or weakly conditioned small CNN.

## Claim Boundary

Supported:

```text
Condition-scaled residual std alone is not enough for a single unconditioned
small CNN to beat the conservative p99 baseline on real gated ICCD surrogate
pairs.
```

Also supported:

```text
Hard or near-hard condition selection between conservative and stronger
denoising remains the strongest current controlled validation result.
```

Not supported:

```text
Condition-aware synthetic training has failed as a direction.
```

The current failed training result is specifically for residual-std-only
scaling, zero-mean residual scaling, and simple condition-channel injection in a
small CNN.

## Next Step

The next experiment should stop trying minor variants of the same small CNN and
instead test one of two stronger directions:

1. make the method explicitly condition-gated, with q50 or q40-q60 blending as
   the baseline method and smoothing-risk checks as guardrails; or
2. train condition-band experts and then evaluate whether learned gating can
   reproduce the p99/physical hybrid without post-hoc oracle tuning.

For the current paper route, option 1 is more defensible and closer to the
evidence already available.
