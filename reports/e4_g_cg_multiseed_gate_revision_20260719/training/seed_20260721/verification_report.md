# E2 G/CG Scaled Training

Status: `TRAINING-RUN-VALID-WITH-LIMITATIONS`

Scale preflight: 417/417 passed. Pair preflight: 717/717 passed.

| experiment   |   pair_count |   noisy_psnr |   noisy_ssim |   output_psnr |   output_ssim |   output_mae |   output_rmse |   max_abs_output_mean_shift_DN |   max_output_zero_ratio |   max_output_one_ratio |   condition_psnr_variance |   first_train_l1 |   final_train_l1 |   train_validation_l1_gap |   best_epoch |   best_validation_psnr |   best_validation_l1 |   elapsed_seconds |   peak_gpu_bytes |   parameter_count |
|:-------------|-------------:|-------------:|-------------:|--------------:|--------------:|-------------:|--------------:|-------------------------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|--------------------------:|-------------:|-----------------------:|---------------------:|------------------:|-----------------:|------------------:|
| CG_NC        |          117 |      57.2769 |     0.997788 |       57.1761 |      0.997756 |   0.00119403 |    0.0014977  |                        8.85681 |             0.000545502 |                      0 |                   12.008  |       0.00823338 |      0.00115823  |               3.58202e-05 |           15 |                57.1761 |           0.00119405 |           118.716 |        271566848 |            481745 |
| G            |          117 |      57.2769 |     0.997788 |       57.1171 |      0.997773 |   0.00119826 |    0.00151067 |                       24.2142  |             0.00672531  |                      0 |                   12.2094 |       0.00806602 |      0.000991968 |               0.000206324 |            6 |                57.1171 |           0.00119829 |           117.785 |        265765376 |            481745 |

This is independent-content synthetic validation only. It is not real ICCD denoising evidence or cross-camera radiometric calibration.
