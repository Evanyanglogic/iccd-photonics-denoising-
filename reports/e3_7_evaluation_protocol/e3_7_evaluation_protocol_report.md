# E3.7 Evaluation Protocol and Smoothing-Risk Audit

## Purpose

E3.7 freezes the evaluation criteria before LOFO validation and formal
network baselines. It prevents later experiments from being judged only by
mean PSNR when a strategy may gain PSNR by smoothing useful gradients.

## Inputs

- Blend metrics: `reports/e3_6_condition_blend_p99_physical/condition_blend_metrics.csv`
- Visual metrics: `reports/e3_5_score_q50_visuals/condition_visual_metrics.csv`
- Warning gradient ratio threshold: `0.95`
- High-risk gradient ratio threshold: `0.85`

## Fixed Evaluation Protocol

For E3.8 and E4.*, every strategy should report:

1. pair-level PSNR and SSIM;
2. folder-level mean PSNR gain;
3. positive pair fraction;
4. positive and negative folder counts;
5. residual mean and residual standard deviation;
6. representative best, median, and worst samples;
7. gradient ratio to noisy input for smoothing-risk checks;
8. strategy source decisions for condition-aware methods.

A candidate strategy is not acceptable as a paper claim if it improves
mean PSNR but introduces negative folder-level behavior or repeated
gradient-ratio warnings without visual qualification.

## Current Strategy Summary

| Strategy | Mean folder gain | Positive folders | Positive pairs | Mean residual std reduction |
|---|---:|---:|---:|---:|
| score_q40_q60_linear_blend | 0.380769 | 10/10 | 0.938 | 0.00013424 |
| score_q50_hard_blend | 0.380695 | 10/10 | 0.938 | 0.00013443 |
| score_q30_q70_linear_blend | 0.376573 | 10/10 | 0.938 | 0.00013314 |
| score_q20_q80_linear_blend | 0.368598 | 10/10 | 0.938 | 0.00013106 |
| always_physical | 0.343069 | 6/10 | 0.650 | 0.00013273 |
| always_p99 | 0.039237 | 10/10 | 0.938 | 0.00000988 |
| always_noisy | 0.000000 | 0/10 | 0.000 | 0.00000000 |

## Smoothing-Risk Summary

| Strategy | Mean gain | Mean grad/noisy | Min grad/noisy | Warnings | High risk | Worst sample |
|---|---:|---:|---:|---:|---:|---|
| hybrid | 0.380947 | 0.9498 | 0.7952 | 3 | 1 | folder 5 `folder_5_frame_131` |
| physical | 0.358649 | 0.9530 | 0.7952 | 3 | 1 | folder 5 `folder_5_frame_131` |
| p99 | 0.033376 | 0.9929 | 0.9927 | 0 | 0 | folder 7 `folder_7_frame_131` |
| noisy | 0.000000 | 1.0000 | 1.0000 | 0 | 0 | folder 1 `folder_1_frame_151` |

## Interpretation

The strongest current diagnostic strategy is `score_q40_q60_linear_blend` with
0.380769 dB mean folder PSNR gain.
This does not yet prove deployable generalization because thresholds were
estimated on the same ten folders. LOFO is therefore mandatory next.

`always_physical` has negative folder-level behavior in folders 1, 2, 11, 13.
This supports condition-aware selection over a globally stronger denoiser.

The representative-sample audit flags smoothing risk for physical-style outputs; the worst physical sample is folder 5 `folder_5_frame_131` with grad/noisy 0.7952.

## Claim Boundary

Supported now:

- condition-aware selection reduces current diagnostic condition failures;
- strong physical-style denoising is useful mainly in high-condition folders;
- gradient-ratio checks are necessary because PSNR gain can coincide with smoothing.

Not supported now:

- q50 or q40-q60 thresholds are deployable classifiers;
- the current small CNN restores missing weak-light details;
- linear blend is meaningfully better than hard q50 switching.

## Next Step

Run E3.8 LOFO. Thresholds and blend intervals must be selected only from
training folders, then evaluated on the held-out folder. The E3.7 metrics
above are the required reporting fields for that LOFO report.
