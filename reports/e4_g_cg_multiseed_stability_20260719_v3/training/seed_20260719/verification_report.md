# E2 G/CG Scaled Training

Status: `TRAINING-RUN-VALID-WITH-LIMITATIONS`

Scale preflight: 417/417 passed. Pair preflight: 717/717 passed.

| experiment   |   pair_count |   noisy_psnr |   noisy_ssim |   output_psnr |   output_ssim |   output_mae |   output_rmse |   max_abs_output_mean_shift_DN |   max_output_zero_ratio |   max_output_one_ratio |   condition_psnr_variance |   first_train_l1 |   final_train_l1 |   train_validation_l1_gap |   best_epoch |   best_validation_psnr |   best_validation_l1 |   elapsed_seconds |   peak_gpu_bytes |   parameter_count |
|:-------------|-------------:|-------------:|-------------:|--------------:|--------------:|-------------:|--------------:|-------------------------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|--------------------------:|-------------:|-----------------------:|---------------------:|------------------:|-----------------:|------------------:|
| CG_NC        |          117 |      57.2765 |     0.997787 |       57.3428 |      0.997846 |   0.00117282 |    0.00146948 |                        7.109   |              0.00765228 |                      0 |                   12.0405 |       0.00263229 |      0.000925514 |               0.000247336 |            2 |                57.3428 |           0.00117285 |           118.478 |        271566848 |            481745 |
| G            |          117 |      57.2765 |     0.997787 |       57.3401 |      0.997836 |   0.00117389 |    0.00147075 |                        1.64102 |              0.00367355 |                      0 |                   12.1322 |       0.00252121 |      0.000847127 |               0.000326774 |            2 |                57.3401 |           0.0011739  |           118.034 |        265765376 |            481745 |

This is independent-content synthetic validation only. It is not real ICCD denoising evidence or cross-camera radiometric calibration.
