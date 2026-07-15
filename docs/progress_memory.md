# Progress Memory

Date: 2026-07-15

## Current Direction

Target journal: 光子学报.

Chosen route: ICCD-aware physical-prior noise modeling plus PNGAN-style
generative refinement for low-light denoising data augmentation.

The paper should be framed as an optical imaging / device-noise / calibration
paper, not as a generic deep denoising network paper.

## User Constraints

- Real device data can already be collected.
- Both ICCD and sCMOS data are available.
- Calibration data include dark/flat and device-condition sequences.
- The target is 光子学报.

## Current Repository Status

Original `E:/PNGAN-main` contains an sCMOS-oriented PNGAN prototype:

- `scmos_noise_model.py`: physical prior with signal-dependent, read, and row
  noise.
- `networks/smnet_grayscale.py`: residual U-Net style generator.
- `networks/discriminator_grayscale.py`: patch and multiscale discriminators.
- `train_pngan.py`: PNGAN training loop with adversarial, domain, perceptual,
  and identity losses.
- `loss_functions_grayscale.py`: denoising losses including edge, SSIM, and
  intensity-adaptive loss.

## Most Valuable Transfer Points

1. Residual generator: keep the `output = input + residual` idea for ICCD noise
   refinement.
2. Dr content-domain constraint: keep `L1(Dr(fake), Dr(real))`, but train or
   adapt Dr for ICCD conditions.
3. Patch discriminator: use as the first stable noise-domain discriminator.
4. Multiscale discriminator: reserve for ICCD spatial correlation and phosphor
   diffusion once the base version is stable.
5. Physical-prior input: replace `sCMOSNoiseModel` with `ICCDNoiseModel` rather
   than rewriting the whole PNGAN framework.

## Key Changes Needed for ICCD

- Add ICCD imaging-chain prior:
  photon statistics, photocathode response, MCP gain fluctuation, dark counts,
  phosphor diffusion, and readout noise.
- Add condition-aware metadata:
  gain, gate width, exposure, illumination level, device type.
- Add noise-statistical validation:
  mean-variance, histogram, PSD, spatial autocorrelation, dark-field
  distribution.
- Recalibrate intensity-adaptive loss:
  ICCD weak-light emphasis should likely prioritize dark/low-photon regions
  rather than bright regions.

## Immediate Engineering Plan

Updated after the `$pytorch-patterns` training-code audit:

1. Data and metric gates come before model changes.
2. Run `scripts/audit_iccd_dataset.py` to verify strict pairing, TIFF numeric
   range, metadata coverage, dark/flat availability, and split readiness.
3. Use `src/iccd_eval/metrics.py` for float-domain PSNR/SSIM and residual
   statistics.
4. Build an ICCD dataloader with metadata and condition split after the audit
   manifest is stable.
5. Run the first controlled comparison only after the audit passes:
   noisy input vs Poisson-Gaussian vs sCMOS prior vs ICCD prior.
6. Connect the ICCD prior to PNGAN after baseline data, metrics, and splits are
   reproducible.

## Current Implementation Progress

Added on 2026-07-15:

- `configs/dataset_iccd.yaml`: dataset audit configuration template.
- `configs/noise_prior_baselines.yaml`: first-pass synthetic prior comparison
  parameters.
- `configs/pmrid7_exposure_lists.yaml`: known local PMRID7 list paths and
  checked manual pair paths.
- `scripts/audit_iccd_dataset.py`: TIFF pair, range, calibration, manifest, and
  split audit utility.
- `scripts/audit_single_exposure_iccd.py`: single-exposure gated ICCD sequence
  audit for unpaired repeated frames.
- `scripts/inventory_gated_iccd_batch.py`: batch-level gated ICCD folder
  inventory for multiple downloaded exposure/gate folders.
- `scripts/convert_exposure_lists.py`: legacy PMRID exposure-list to
  `pairs.csv`/`splits.yaml` converter using exposure duration to orient
  clean/noisy columns.
- `scripts/evaluate_pair_baseline.py`: no-model B0 noisy-input baseline
  evaluator using the pair manifest.
- `scripts/compare_noise_priors.py`: E2 synthetic-noise fidelity comparison for
  Poisson-Gaussian, sCMOS-like, and ICCD-chain priors.
- `scripts/check_manifest_dataloader.py`: split-aware dataloader smoke test.
- `src/iccd_data/`: manifest-backed pair records and PyTorch-compatible paired
  TIFF dataset.
- `src/iccd_noise/baselines.py`: generic Poisson-Gaussian and simplified
  sCMOS-like noise priors.
- `src/iccd_eval/metrics.py`: float-domain PSNR/SSIM, residual statistics, and
  brightness-bin PSNR.
- `docs/pmrid7_data_inventory.md`: current PMRID7 path inventory and first B0
  observations.
- `docs/gated_iccd_data_inventory.md`: downloaded gated ICCD subset inventory
  and first single-exposure audit.

Smoke test status:

- AST syntax check passed.
- Metric smoke test passed.
- Audit script generated a sample report, pair manifest, and split manifest on
  a tiny synthetic 16-bit TIFF pair.
- B0 baseline evaluator and noise-prior comparison script ran successfully on
  the same smoke-test pair.
- Manifest dataloader check passed and PyTorch DataLoader produced BCHW
  batches on the smoke-test pair.
- PMRID7 `train_lists1` converted to 480 train and 120 val pairs.
- PMRID7 `train_lists2` converted to 320 train and 80 val pairs.
- Both converted PMRID7 manifests passed train/val dataloader checks.
- Gated ICCD subset `F:/20260319/1` audited: 200 TIFF frames, 5120x5120 uint16,
  exposure width 900 ms, Sync A/B width 4 us, gain 60, no saturation in sampled
  frames.
- Gated ICCD batch inventory for `F:/20260319` currently detects one folder:
  `1`, with 200 TIFF frames and 200 metadata rows.
- Full gated ICCD download at `D:/iccd/data/20260319` inventoried: complete
  200-frame folders are `1,2,4,5,7,8,9,10,11,13`; partial folders are `3` and
  `6`; `12` is empty/incomplete. All complete folders share exposure width
  900 ms, Sync A/B width 4 us, and gain 60.
- Single-condition noise summary over 10 complete folders shows mean signal
  range about 936 to 4717 DN, temporal std mean about 37.7 to 217.9 DN, and
  Fano approximation about 1.64 to 14.0 on 512x512 center crops.
- Workflow routing documented in `docs/iccd_research_workflow.md`: use
  `iccd-denoising-optimizer` for data/noise/metric gates, `pytorch-patterns`
  for PyTorch pipeline audits, arXiv MCP for paper discovery, Brave Search MCP
  for web/venue/device sources, GitHub MCP for remote repository context,
  `academic-research-suite` for literature/experiment planning, and
  `paper-spine` only after contribution and evidence gates are mature.
- Detailed experiment roadmap drafted in `docs/experiment_roadmap.md`. It
  breaks the work into device noise characterization, noise-model fidelity,
  training baselines, ablations, and paper evidence, with explicit gates and
  success criteria.
- External evidence check revised the roadmap: mean-variance fitting should
  estimate the linear regime and effective gain instead of assuming unit Poisson
  slope in raw DN; temporal variance should be separated from fixed-pattern
  nonuniformity; provisional thresholds should later be backed by uncertainty
  intervals.
- Tool installation plan saved in `docs/tooling_install_plan.md`: install
  FiftyOne and MLflow first for visual dataset review and experiment tracking,
  add Kornia/PIQ/TorchMetrics/LPIPS with training code, defer DVC until a data
  remote is chosen, and defer broad extra skills or community OpenCV MCP until a
  specific gap appears.
- Added `requirements-analysis.txt` for offline analysis dependencies and
  `scripts/fit_mean_variance_curve.py` for repeated-frame brightness-bin
  mean-variance analysis. First run on the ten complete gated ICCD folders used
  32 frames, 512x512 center crops, 16 bins, and showed temporal Fano increasing
  from about 1.70 to 14.46 with brightness.
- Added `scripts/audit_scmos_target_data.py` and audited
  `F:/目标传感器噪声参数估计/data` as sCMOS data. The audit generated crop-level
  dark offset, dark std, and bad-pixel mask artifacts under
  `reports/target_scmos_risk_audit`; documented the source in
  `docs/target_scmos_data_inventory.md`. This dataset can be used as sCMOS
  baseline/content source for ICCD-like synthetic noise, but not as real ICCD
  paired data.
- Added `scripts/convert_scmos_tail_pairs.py` for E2.5 and generated a
  `15ms -> 500ms` sCMOS tail-index manifest under
  `reports/target_scmos_15ms_500ms_manifest`: 100 pairs, train/val/test =
  85/8/7. Dataloader check passed. B0 no-model baseline on the manifest gives
  PSNR 13.5869 dB and SSIM 0.191758, but residual mean is about 0.18, so
  brightness/offset mismatch must be handled before using it as supervised
  training reference.
- Added `scripts/evaluate_fixed_pattern_correction.py` for E1.4 and ran it on
  the ten complete gated ICCD folders under `D:/iccd/data/20260319`. The run
  used 100 calibration frames, 100 held-out test frames, and 512x512 center
  crops. Median held-out spatial fixed-pattern reduction is about 95.1%, with
  temporal standard deviation effectively unchanged. This supports a
  fixed-pattern correction baseline, but not a true flat-field claim without
  matching flat-field frames.

## Skill Setup

Installed/created on 2026-07-15:

- `pytorch-patterns`: installed from `affaan-m/ECC`, path
  `skills/pytorch-patterns`. Use for PyTorch model, data pipeline, training loop,
  reproducibility, memory, and speed audits.
- `iccd-denoising-optimizer`: custom Codex skill created under
  `C:/Users/Yjia/.codex/skills/iccd-denoising-optimizer`. Use for ICCD-specific
  data diagnostics, experiment planning, metric checks, and Photonics Journal
  evidence discipline.
