# E3.7 Evaluation Protocol and Smoothing-Risk Audit

## Purpose

E3.7 freezes the evaluation protocol before E3.8 LOFO and E4 formal network
baselines. The goal is to prevent later decisions from being based only on mean
PSNR when a method may gain PSNR by over-smoothing high-condition ICCD samples.

## Command

```powershell
python scripts\summarize_e3_7_protocol.py
```

Outputs:

- `reports/e3_7_evaluation_protocol/e3_7_strategy_summary.csv`
- `reports/e3_7_evaluation_protocol/e3_7_folder_strategy_summary.csv`
- `reports/e3_7_evaluation_protocol/e3_7_smoothing_risk_summary.csv`
- `reports/e3_7_evaluation_protocol/e3_7_protocol_config.json`
- `reports/e3_7_evaluation_protocol/e3_7_evaluation_protocol_report.md`

## Fixed Protocol for E3.8 and E4

Every later strategy should report:

1. pair-level PSNR and SSIM;
2. folder-level mean PSNR gain;
3. positive pair fraction;
4. positive and negative folder counts;
5. residual mean and residual standard deviation;
6. representative best, median, and worst samples;
7. gradient ratio to noisy input for smoothing-risk checks;
8. strategy source decisions for condition-aware methods.

A strategy is not sufficient for a paper claim if it only improves mean PSNR
while creating negative folder-level behavior or repeated gradient-ratio
warnings.

## Current Result

| Strategy | Mean folder PSNR gain | Positive folders | Positive pairs |
|---|---:|---:|---:|
| score q40-q60 linear blend | 0.380769 dB | 10/10 | 75/80 |
| score q50 hard blend | 0.380695 dB | 10/10 | 75/80 |
| always physical | 0.343069 dB | 6/10 | 52/80 |
| always p99 | 0.039237 dB | 10/10 | 75/80 |

The linear blend and hard q50 result are effectively tied. The current evidence
does not justify claiming that continuous blending is meaningfully better than
hard condition switching.

## Smoothing Risk

Representative all-folder visual metrics show:

| Strategy | Mean gain | Mean grad/noisy | Min grad/noisy | Warnings |
|---|---:|---:|---:|---:|
| hybrid | 0.380947 dB | 0.9498 | 0.7952 | 3 |
| physical | 0.358649 dB | 0.9530 | 0.7952 | 3 |
| p99 | 0.033376 dB | 0.9929 | 0.9927 | 0 |
| noisy | 0.000000 dB | 1.0000 | 1.0000 | 0 |

The strongest warning is folder 5, `folder_5_frame_131`, where physical-style
outputs have `grad/noisy = 0.7952`. This supports the current claim boundary:
describe the result as condition-aware residual suppression, not low-light
detail restoration.

## Claim Boundary

Supported:

- condition-aware selection reduces current diagnostic condition failures;
- physical-style stronger denoising is useful mainly in high-condition folders;
- gradient-ratio checks must accompany PSNR/SSIM because PSNR gain can coincide
  with smoothing.

Not supported:

- q50 or q40-q60 thresholds are deployable classifiers;
- the current small CNN restores missing weak-light details;
- linear blend is a meaningful method contribution over hard q50 switching.

## Next Step

Proceed to E3.8 LOFO. Thresholds and blend intervals must be selected using
only training-fold folders, then evaluated on the held-out folder.
