# Experiments

Planned entry points:

1. `fit_iccd_calibration.py`
   Fit ICCD prior parameters from dark/flat sequences.
2. `compare_noise_statistics.py`
   Compare Poisson-Gaussian, sCMOS prior, ICCD prior, and ICCD-aware PNGAN.
3. `train_iccd_pngan.py`
   Adapt the existing PNGAN training loop to ICCD physical-prior input.
4. `evaluate_real_iccd_denoising.py`
   Evaluate denoisers only on held-out real ICCD measurements.

Keep raw data outside git. Commit config files, manifests, scripts, and summary
tables/figures.

