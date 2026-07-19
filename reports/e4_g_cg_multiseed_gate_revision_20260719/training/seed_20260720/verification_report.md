# E2 G/CG Scaled Training

Status: `TRAINING-RUN-VALID-WITH-LIMITATIONS`

Scale preflight: 417/417 passed. Pair preflight: 717/717 passed.

| experiment   |   pair_count |   noisy_psnr |   noisy_ssim |   output_psnr |   output_ssim |   output_mae |   output_rmse |   max_abs_output_mean_shift_DN |   max_output_zero_ratio |   max_output_one_ratio |   condition_psnr_variance |   first_train_l1 |   final_train_l1 |   train_validation_l1_gap |   best_epoch |   best_validation_psnr |   best_validation_l1 |   elapsed_seconds |   peak_gpu_bytes |   parameter_count |
|:-------------|-------------:|-------------:|-------------:|--------------:|--------------:|-------------:|--------------:|-------------------------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|--------------------------:|-------------:|-----------------------:|---------------------:|------------------:|-----------------:|------------------:|
| CG_NC        |          117 |      57.2757 |     0.997786 |       57.3931 |      0.997866 |   0.00116591 |    0.00146124 |                        4.04433 |              0.00771332 |                      0 |                   12.0711 |        0.0126208 |       0.00109975 |               6.61801e-05 |           22 |                57.3931 |           0.00116593 |           123.863 |        271566848 |            481745 |
| G            |          117 |      57.2757 |     0.997786 |       57.2796 |      0.997825 |   0.00118071 |    0.0014813  |                        9.85999 |              0.00872803 |                      0 |                   12.1544 |        0.0126788 |       0.0010333  |               0.000147569 |           17 |                57.2796 |           0.00118087 |           123.831 |        265765376 |            481745 |

This is independent-content synthetic validation only. It is not real ICCD denoising evidence or cross-camera radiometric calibration.
