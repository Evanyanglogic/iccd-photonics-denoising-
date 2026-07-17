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
- Added `scripts/evaluate_noise_robustness.py` for E1.2 and ran crop/frame-count
  robustness on the same ten gated ICCD folders. Comparing 512x512/32-frame
  statistics with 1024x1024/128-frame statistics gives median absolute relative
  changes of about 5.8% in mean signal, 9.8% in temporal standard deviation,
  6.4% in temporal Fano, 2.9% in spatial fixed-pattern standard deviation, and
  16.2% in fixed/temporal ratio. The main trend is stable, but larger crops
  reveal meaningful spatial nonuniformity.
- Added `scripts/analyze_iccd_spatial_correlation.py` for E1.5 and ran it on 64
  frames from each complete gated ICCD folder. After subtracting each folder's
  per-pixel temporal mean and each frame's residual mean, median row/column
  lag-1 correlations are about 0.032/0.030 and radial autocorrelation falls
  below 0.1 by 1 px. Current evidence supports strong fixed-pattern/spatial
  mean nonuniformity, but only weak short-range correlation in temporal
  residual noise.
- Completed the E1.6 local dark/flat gate check by scanning `D:/iccd/data`.
  Only the `20260319` ICCD batch is present locally, with folders `1` to `13`
  and an empty helper folder. No local ICCD folder is identified as dark, flat,
  background, bias, or no-light calibration data. Dark/flat calibration claims
  remain blocked until matching calibration folders are provided.
- Audited `F:/ICCD_pir/2025.07.09/CDM-A4000-UM90_DH09131AAK00007` as an
  auxiliary ICCD calibration-candidate sequence. It contains 305 continuous
  2048x2048 uint8 TIFFs, indices 57 to 361, without local metadata files.
  Brightness segmentation suggests low-background, mid, and heavily saturated
  regimes. It is useful as auxiliary ICCD dark/background or brightness-regime
  evidence, but not as matching dark/flat correction for the 5120x5120 uint16
  `D:/iccd/data/20260319` gated ICCD batch.
- Added `scripts/audit_iccd_pir_background.py` and audited the low-saturation
  `ICCD_pir` segment 57-187. The 512x512 crop summary over 131 frames gives
  mean signal 112.07 DN, frame mean std 1.99 DN, temporal std mean 17.60 DN,
  spatial mean std 25.76 DN, fixed/temporal ratio 1.46, mean saturation
  fraction 0.00234, and p99.9 hot-pixel fraction 0.00181. This is useful
  auxiliary 8-bit background evidence but not matching correction for the main
  16-bit gated ICCD batch.
- Added `docs/literature_matrix.md` to position the planned paper against
  low-light RAW denoising, physics-based noise synthesis, real-camera noise
  datasets, self-supervised denoising, and ICCD/MCP characterization literature.
  Current differentiation: real gated-ICCD repeated-frame statistics -> device
  prior -> synthetic paired data -> real-device statistical validation.
- Added `scripts/build_iccd_prior_config.py` for E2.1 and generated
  `configs/iccd_prior_20260319.yaml` plus
  `reports/e2_1_iccd_prior/prior_parameter_report.md`. The first config is
  derived from E1 reports: effective raw-domain linear slope about 17.41
  variance DN per DN, effective photon scale about 3764, temporal Fano range
  about 1.70 to 14.46 with median about 6.05, median normalized fixed-pattern
  sigma about 0.00614, and weak residual lag-1 spatial correlation. It is an
  empirical repeated-frame prior, not strict dark/flat calibration.
- Added `scripts/build_repeated_frame_surrogate_pairs.py` and generated
  `reports/gated_iccd_20260319_surrogate_pairs`: 80 surrogate pairs from ten
  complete folders. Each clean surrogate is the mean of the first 100 repeated
  frames, and held-out frames 101, 111, 121, 131, 141, 151, 161, 171 are used as
  noisy residual samples. Added `configs/iccd_prior_comparison_20260319.yaml`
  and `docs/e2_noise_prior_fidelity.md`. E2.2 result: if all priors are
  E1-calibrated, the runnable models are nearly equivalent; if generic
  Poisson-Gaussian and sCMOS-like defaults are compared against the E1-calibrated
  ICCD prior, ICCD is much closer to held-out residuals in PSNR/SSIM, residual
  std error, and histogram L1. Safe claim: E1 device calibration matters; do not
  yet claim full ICCD physics beats every re-fitted generic prior.
- Added `scripts/evaluate_masked_offset_pairs.py` and ran the masked,
  dark-offset-corrected E2.5 check on the sCMOS `15ms -> 500ms` manifest. The
  run used 100 tail-index pairs, 1024x1024 center crops, the derived crop-level
  dark offset, and the bad-pixel mask from `reports/target_scmos_risk_audit`.
  Valid-pixel fraction is about 0.9985, full PSNR/SSIM after correction is
  about 24.9660 dB / 0.071947, masked PSNR is about 24.9716 dB, and the mean
  ratio `noisy / clean` remains about 8.26. Conclusion: bad pixels are not the
  primary blocker; the exposure/brightness relationship is not a valid
  supervised clean/noisy pair. The data can still be used as sCMOS
  content/reference source for ICCD-like synthetic noise generation with strict
  labeling.
- Added `scripts/generate_iccd_like_synthetic_pairs.py` for E2.6 and generated
  two 100-pair synthetic ICCD-like datasets from the sCMOS content manifest:
  `reports/target_scmos_iccd_like_synthetic_512` and
  `reports/target_scmos_iccd_like_synthetic_512_p99_0p25`. Both preserve the
  source split manifest and pass the existing dataloader smoke check. The
  strict physical-scale version has B0 PSNR/SSIM about 62.49 dB / 0.99927 and
  residual std about 0.000718, so it is probably too easy for training. The
  p99-normalized version has B0 PSNR/SSIM about 56.63 dB / 0.99930 and residual
  std about 0.001463, closer to the real gated ICCD repeated-frame surrogate B0
  residual std of about 0.001822. Current decision: use the p99-normalized set
  as the first synthetic training-source candidate, with the explicit claim
  boundary that it is content-normalized synthetic ICCD-like data, not real
  paired ICCD data.
- Completed an E3 PyTorch training-code audit of the parent `E:/PNGAN-main`
  scripts using `pytorch-patterns`. The legacy scripts are not safe as direct
  paper-experiment entry points because they use directory-sorted pairing,
  hardcoded data paths, incomplete seed control, an sCMOS-only online noise
  model, and SSIM that quantizes low-light data to uint8. The model/loss code
  can still be reused. Current decision: build a manifest-driven supervised
  denoiser baseline first, using `src/iccd_data.ICCDPairDataset` and
  `src/iccd_eval.metrics`, before modifying SMNet/MIRNet/PNGAN architecture.
- Added `scripts/train_manifest_denoiser_baseline.py`, a manifest-driven
  supervised denoising baseline with explicit seed control, config capture,
  git commit recording, metrics CSV, validation rows, best/last checkpoints,
  and best/median/worst sample TIFFs. A CPU smoke run on the p99-normalized
  synthetic ICCD-like manifest used one epoch, two train batches, and two
  validation batches. It produced train L1 0.000410107, validation L1
  0.000455504, validation PSNR/SSIM 55.7709 dB / 0.999513, and noisy-input
  PSNR/SSIM 55.7045 dB / 0.999232 on the same validation subset. This confirms
  the E3 training loop and artifact writing, but is not a paper performance
  claim.
- Ran first full E3 small-CNN synthetic baselines after committing the trainer.
  The 20-epoch CPU run on all 85 train and 8 validation pairs reached
  validation PSNR/SSIM 54.1043 dB / 0.999952 versus noisy-input 53.8926 dB /
  0.999294, a gain of about 0.2117 dB. The 100-epoch run reached 54.1966 dB /
  0.999948, a gain of about 0.3040 dB over noisy input. This barely passes the
  provisional synthetic-validation threshold and should be treated as the
  minimum supervised sanity baseline, not a real ICCD performance claim.
- Ran the same 100-epoch small-CNN baseline on the strict physical-scale
  synthetic set. Final validation PSNR/SSIM was 60.1930 dB / 0.999284 versus
  noisy-input 59.5277 dB / 0.999232, and best observed validation PSNR was
  60.3309 dB at epoch 93. This confirms the strict physical-scale set is an
  easier synthetic ablation; it does not replace the p99-normalized set as the
  better real-ICCD-residual-magnitude proxy.
- Added `scripts/evaluate_manifest_denoiser_checkpoint.py` and evaluated both
  synthetic-trained small-CNN checkpoints on 80 real gated ICCD surrogate pairs.
  The p99-trained checkpoint gives mean PSNR/SSIM 56.4479 dB / 0.995780 versus
  noisy-input 56.4087 dB / 0.995732, a stable but tiny 0.0392 dB mean gain. The
  strict physical-scale checkpoint gives 56.7517 dB / 0.996620, a 0.3431 dB mean
  gain, but with high variance and 28/80 negative-gain pairs. Folder-level
  results are strongly condition-dependent, so the next gate should be
  condition-stratified analysis before any larger MIRNet/SMNet/PNGAN run.
- Updated the research route after the user's challenge about denoising versus
  detail restoration. Current boundary: do not frame the paper as ultra-low-light
  detail restoration or "first ICCD denoising." Literature search found direct
  ICCD clustered-noise denoising work (Yang et al., Sensors 2017), plus stronger
  adjacent low-light raw noise synthesis and dark-frame noise modeling work
  (Wei et al. CVPR 2020, Feng et al. PNNP/dark-frame modeling, Cao et al. general
  low-light raw noise synthesis). Safe framing is now gated ICCD repeated-frame
  noise characterization, condition-aware/device-aware synthetic noise, and
  fidelity-controlled denoising validation.
- Diagnosed the Brave Search MCP issue: the configured Node MCP server still
  returns `fetch failed`, while direct Python access to the Brave Search API
  works with approved network access. Added `scripts/brave_search_direct.py` as
  the fallback and fixed its Windows console UTF-8 handling after a search result
  caused a GBK `UnicodeEncodeError`.
- Added `scripts/analyze_condition_gain.py` and ran E3 condition-stratified
  analysis on the p99 and strict physical-scale checkpoints. Outputs are under
  `reports/e3_condition_gain_analysis`, with the summary in
  `docs/e3_condition_gain_analysis.md`. The p99 checkpoint is stable but tiny
  (0.0392 dB mean folder gain, 10/10 positive folders). The physical-scale
  checkpoint has larger mean gain (0.3431 dB) but only 6/10 positive folders and
  its gain is strongly correlated with E1 condition statistics: temporal std
  mean r=0.9726, fixed/temporal ratio r=0.9693, fixed-map std r=0.9504, Fano
  r=0.9495, and mean signal r=0.9478. This supports a condition-aware denoising
  or noise-scaling experiment before any larger generic architecture.
- Upgraded `docs/literature_matrix.md` into a layered reference strategy. High
  standard anchors are now CVPR/TPAMI low-light raw noise modeling, Noise2Noise
  / Noise2Void, Noise Flow, SIDD/SID, and EMVA/photon-transfer style detector
  characterization. Direct ICCD denoising papers are kept as predecessor/gap
  definitions, not as the writing standard.
- Added `scripts/evaluate_condition_gate.py` and ran E3.5-A condition gate.
  Outputs are under `reports/e3_5_condition_gate`, with the summary in
  `docs/e3_5_condition_gate.md`. The best non-oracle q40 condition gate selects
  physical checkpoint outputs for 48 high-condition pairs and noisy inputs for
  32 low-condition pairs. It reaches 0.3669 dB mean folder PSNR gain with zero
  negative-gain folders, compared with 0.3431 dB and four negative folders for
  always applying the physical checkpoint. Treat this as diagnostic evidence
  because only ten folders are available.
- Added `scripts/evaluate_condition_subsets.py` and ran E3.5-B low/high subset
  validation using Fano q40. Outputs are under
  `reports/e3_5_condition_subsets`, with the summary in
  `docs/e3_5_condition_subsets.md`. Low-condition folders are 1, 2, 11, and 13;
  high-condition folders are 4, 5, 7, 8, 9, and 10. Physical checkpoint gain is
  -0.0595 dB on low-condition folders and +0.6114 dB on high-condition folders.
  p99 is weak but positive in both subsets. A condition hybrid using p99 for low
  conditions and physical for high conditions reaches 0.3788 dB mean folder gain
  with 10/10 positive folders. This is the current best minimal condition-aware
  validation result, but it still needs visual/residual inspection and more
  conditions before being treated as a final deployable rule.
- Added `scripts/inspect_condition_visuals.py` and ran E3.5-C visual/residual
  inspection for folders 2, 5, 1, and 10. Outputs are under
  `reports/e3_5_condition_visuals`, with the summary in
  `docs/e3_5_condition_visuals.md`. Folder 2 confirms physical overcorrection
  in a low-condition case (p99 +0.1215 dB, physical -0.2832 dB). Folder 5
  confirms strong physical residual suppression (+1.7393 dB, residual std
  0.004253 -> 0.003471), but gradient/noisy drops to 0.7945, so this must be
  described as noise suppression with smoothing risk, not detail restoration.
  Folder 10 shows q40 Fano is not a final decision rule because p99 slightly
  outperforms physical on the selected boundary sample.
- Added `scripts/evaluate_condition_score.py` and ran E3.5-D multi-metric
  condition score. Outputs are under `reports/e3_5_condition_score`, with the
  summary in `docs/e3_5_condition_score.md`. The score averages z-scored
  `mean_signal`, `temporal_std_mean`, `fano_temporal`, `fixed_map_std`, and
  `fixed_to_temporal_std_ratio`. Best diagnostic rule is
  `score_q50_hybrid_p99_physical`: p99 for folders 13, 2, 11, 1, and 10;
  physical for folders 9, 4, 8, 7, and 5. It reaches 0.3807 dB mean folder gain
  with 10/10 positive folders, slightly above Fano q40 hybrid at 0.3788 dB, and
  fixes the folder 10 boundary decision identified in E3.5-C. Still diagnostic
  only because thresholds are derived from the same ten folders.
- Extended `scripts/inspect_condition_visuals.py` with configurable sample
  selection and hybrid physical-folder assignment, then ran E3.5-E all-folder
  score q50 visual/residual inspection. Outputs are under
  `reports/e3_5_score_q50_visuals`, with the summary in
  `docs/e3_5_score_q50_visuals.md`. The run uses median physical-gain samples
  for all ten folders and the score q50 rule: p99 for folders 13, 2, 11, 1, and
  10; physical for folders 9, 4, 8, 7, and 5. Folder 10 is now correctly kept on
  the p99 side. High-score folders show stronger residual suppression, but
  folder 5 drops grad/noisy to about 0.795, so the claim must remain
  condition-aware noise suppression with smoothing risk, not detail restoration.
  Next step is E3.6 condition-scaled ICCD-like synthetic noise training, using
  q50 checkpoint switching only as a diagnostic/simple baseline.
- Started E3.6 and ran two 100-epoch small-CNN trainings. Updated
  `scripts/generate_iccd_like_synthetic_pairs.py` to support condition-score CSV
  input, residual-std scaling, zero-valued valid pixels for regenerated
  synthetic clean TIFFs, and optional residual mean removal before scaling.
  E3.6-A generated `reports/target_scmos_iccd_like_synthetic_512_condition_scaled_q50`
  and trained `reports/e3_6_condition_scaled_q50_smallcnn_100ep`; real surrogate
  transfer was negative at -0.0321 dB mean PSNR gain. E3.6-B generated
  `reports/target_scmos_iccd_like_synthetic_512_condition_scaled_q50_zero_mean`
  and trained `reports/e3_6_condition_scaled_q50_zero_mean_smallcnn_100ep`;
  transfer improved to -0.0066 dB but still underperformed the p99 baseline
  (+0.0392 dB). Summary is in `docs/e3_6_condition_scaled_training.md`. Current
  conclusion: residual-std-only scaling is insufficient for an unconditioned
  small CNN; next should be E3.6-C with explicit condition-score input or
  condition-band training.
- Ran E3.6-C and E3.6-D. Updated `scripts/train_manifest_denoiser_baseline.py`
  and `scripts/evaluate_manifest_denoiser_checkpoint.py` for optional two-channel
  condition-score input. Raw condition-score input trained on the zero-mean
  condition-scaled synthetic set transferred poorly to real surrogate pairs
  (-0.0487 dB). Scaling the score channel by 3 reduced the bias but remained
  negative (-0.0102 dB). Added `scripts/evaluate_condition_blend.py` and ran
  `reports/e3_6_condition_blend_p99_physical`: q40-q60 linear blend reached
  +0.3808 dB and q50 hard blend reached +0.3807 dB, both with 10/10 positive
  folders and 75/80 positive pairs. Current decision: do not keep tuning minor
  residual-std synthetic small-CNN variants. The strongest evidence is explicit
  condition selection/blending between conservative p99 and stronger physical
  denoising, with smoothing-risk checks.

Added after reviewing `E:/Google_Download/deep-research-report.md` on
2026-07-16:

- The external report agrees with the current direction: frame the work as
  gated-ICCD noise characterization, condition scoring, and condition-aware
  controlled denoising validation, not ultra-low-light detail restoration.
- The next stage should not continue small-CNN residual-std variants. It should
  first run LOFO folder-level validation so q40/q50/q60 thresholds and blend
  intervals are estimated only from training folders, then tested on held-out
  folders.
- Formal network baselines are now required before model-ablation language is
  justified. Minimum baselines: DnCNN on p99 synthetic, DnCNN on physical
  synthetic, Light U-Net on p99 synthetic, and Light U-Net on physical
  synthetic, followed by post-hoc hard-gate and linear-blend comparisons.
- The report's MLflow, FiftyOne, and DVC requirements are useful but should not
  block the next experiments. Keep using the current reproducible artifact style
  under `reports/` first; add MLflow/FiftyOne once the formal baseline scripts
  are stable. Defer DVC until a real data remote is chosen.
- The q40-q60 linear blend and q50 hard blend are nearly tied in current
  evidence, so the paper should not overclaim a continuous-blend contribution
  unless LOFO or cross-architecture baselines show a real advantage.
- Completed E3.7 evaluation-protocol and smoothing-risk audit. Added
  `scripts/summarize_e3_7_protocol.py` and generated
  `reports/e3_7_evaluation_protocol`. The protocol freezes required reporting
  fields for E3.8 and E4: pair PSNR/SSIM, folder-level gain, positive
  pair/folder rates, residual mean/std, representative samples, gradient ratio,
  and strategy source decisions. Current results keep the same boundary:
  q40-q60 linear blend and q50 hard blend are effectively tied
  (0.380769 dB vs 0.380695 dB mean folder gain), while physical/hybrid outputs
  carry smoothing risk in high-condition folders, especially folder 5 with
  grad/noisy about 0.7952. Next stage is E3.8 LOFO with thresholds and blend
  intervals selected only from training folders.
- Completed E3.8 LOFO condition-protocol validation. Added
  `scripts/evaluate_lofo_condition_protocol.py` and generated
  `reports/e3_8_lofo_condition_protocol`. For each held-out folder, hard
  thresholds and linear blend intervals are selected only from the other nine
  folders. Results: LOFO best linear reaches +0.380355 dB mean folder PSNR gain
  with 10/10 positive folders and 75/80 positive pairs; LOFO best hard reaches
  +0.375555 dB with 10/10 positive folders and 72/80 positive pairs. This
  reduces the threshold-leakage concern from E3.5/E3.6. However, both LOFO
  condition strategies still show 24 `grad/noisy < 0.95` warnings, so the
  smoothing-risk claim boundary remains. Next stage is E4 formal baselines:
  DnCNN and Light U-Net on p99 and physical synthetic training data.
- Started E4 formal baseline readiness. Updated
  `scripts/train_manifest_denoiser_baseline.py` with `--model-type`
  `residual_small`, `dncnn`, and `light_unet`; updated checkpoint evaluation
  and condition-blend loading to restore `model_type` from checkpoint config
  while keeping old small-CNN checkpoints compatible. Smoke tests ran on CPU:
  DnCNN smoke has 7,265 parameters and Light U-Net smoke has 29,681 parameters.
  Both one-batch synthetic smoke trainings and two-pair real-surrogate
  checkpoint evaluations completed. The real-surrogate smoke gains are negative
  because these are not trained baselines; they only verify the E4 code path.
- Completed the E4 surrogate-reference reliability audit before formal backbone
  training. Added a preregistered odd/even split-reference experiment using two
  disjoint 50-frame temporal means and the existing frozen p99/physical
  checkpoints plus E3.8 LOFO selections. The automatic result is `GO` for the
  narrow evaluation-stability hypothesis: LOFO-linear exceeds the best fixed
  model by +0.038255 dB on reference A and +0.035602 dB on reference B; paired
  folder bootstrap intervals are [0.007836, 0.078854] and
  [0.003657, 0.080092]. Folder-sign agreement is 0.90. This does not establish
  clean-ground-truth restoration or a deployable gate: linear is better than
  physical in five folders, equal in four, and slightly worse in one. The next
  unique experiment is a scale-matched synthetic-noise distribution audit to
  separate residual strength from distribution/structure before training a
  stronger backbone.
- Completed E5, a controlled 2x2 synthetic noise structure-by-strength
  experiment with four datasets (P-L, P-H, H-L, H-H), one 2,625-parameter
  residual small CNN, three seeds, and both odd/even temporal-mean references.
  The original p99/physical absolute residual std values were not directly
  comparable because their clean normalizations differ, so source
  residual-to-clean ratios were transferred to one shared clean domain. A
  common 1024-DN pedestal was required after an initial smoke test exposed 91%
  lower-bound clipping at exact-zero clean pixels. The final 100-pair datasets
  passed strength matching, zero-mean, clipping, PSD/autocorrelation, and
  distribution-structure checks. Real transfer was negative for all cells:
  P-L -0.496 dB, P-H -4.315 dB, H-L -1.892 dB, and H-H -4.891 dB. Factor
  effects were strength -3.409 dB, structure -0.986 dB, and interaction
  +0.820 dB, all with folder bootstrap intervals excluding zero. The decision
  is `C_INTERACTION`, but maximum seed variation (0.668 dB) is much larger than
  the previous 0.036 dB condition gain, so the gate result is downgraded to an
  exploratory finding. The next unique experiment is conservative P-L
  synthetic-real gap repair with the model and strength held fixed.
- Completed E6 repeated-frame supervision feasibility audit before any real
  training. Only 2/10 folders passed all preregistered stability and
  independence checks. Eight-frame targets reduced median random target noise
  by 2.627x and high-frequency residuals were nearly independent, but several
  folders retained strong lagged row/column correlations; folders 5 and 7 also
  showed local brightness drift. Split-half fixed maps were stable and can be
  learned as scene content, while the current data cannot separate true scene
  signal from fixed-pattern bias without dark/flat or independent high-SNR
  calibration. Decision: `STOP_REACQUIRE`, protocol E. Generated all ten
  blocked LOFO role manifests (176 train, 22 validation, 8 test inputs per
  fold); every fold passed source-frame leakage checks. No target
  materialization or small-CNN training was performed. The unique next step is
  minimum calibrated reacquisition, not synthetic-prior modification or model
  scaling.

## Skill Setup

Installed/created on 2026-07-15:

- `pytorch-patterns`: installed from `affaan-m/ECC`, path
  `skills/pytorch-patterns`. Use for PyTorch model, data pipeline, training loop,
  reproducibility, memory, and speed audits.
- `iccd-denoising-optimizer`: custom Codex skill created under
  `C:/Users/Yjia/.codex/skills/iccd-denoising-optimizer`. Use for ICCD-specific
  data diagnostics, experiment planning, metric checks, and Photonics Journal
  evidence discipline.

# E7 Data-to-Route Eligibility and Literature Audit (2026-07-17)

- Added `scripts/audit_data_route_eligibility.py` and audited all 2,000 complete-batch TIFF center crops.
- Integrity: 10/10 PASS; all have 200 continuous `uint16` 5120x5120 frames and 200 metadata rows, with no sampled zeros/saturation/duplicates/corruption.
- Device metadata do not vary across complete folders: 900 ms exposure width, Sync A/B 4 us, gain 60.
- Characterization: 9/10 PASS, folder 13 WARN because split-half stable-map correlation is 0.916.
- Repeated-frame supervision remains 2/10 PASS; no real-domain training was started.
- Condition audit: max feature correlation 0.99996, PC1 96.9%, max VIF 54,165; LOFO ridge RMSE 0.300 dB versus null 0.558 dB, but maximum E5 seed SD is 0.668 dB. The score is an image-statistical state descriptor confounded with folder/scene, not a verified acquisition-condition variable.
- Added 25/50/100-frame surrogate audit. Overall strategy ranking is stable, but maximum folder-level gain range is 0.715 dB (folder 5), so references support only cautious relative comparison.
- Brave MCP returned `fetch failed`; arXiv MCP returned `Transport closed`. Literature verification used official publisher/CVF/PMLR/NeurIPS/DOI pages instead.
- Updated `docs/literature_matrix.md` with a 19-paper verified matrix and corrected Noise Flow arXiv ID to `1908.08453`.
- Unique route decision: route 2, `gated ICCD characterization + conditional noise mismatch analysis + surrogate-based denoising applicability/failure-boundary validation`.
- Full condition-aware generator, deployable gate, clean recovery, and naive repeated-frame supervision are not supported.
- Next unique experiment is preregistered in `configs/e8_mismatch_transfer_linkage.yaml`; no new model training is allowed before its Go/No-Go result.

## E8 folder-blocked mismatch-to-transfer linkage

- Replaced the original seven-feature ridge/LOFO proposal because ten folders do
  not support fitted multivariable prediction. The independent unit is the
  folder; four variants, three seeds, and two surrogate references are repeated
  conditions, not additional independent samples.
- Froze scheme A: at most four single mismatch dimensions, exact permutation of
  the ten folder profiles, folder bootstrap, leave-one-folder influence,
  seed-reference sensitivity, and a deterministic random-rank negative control.
- Corrected a preliminary aggregation artifact that averaged low/high mismatch
  before analysis. Final inference computes n=10 Spearman correlations within
  each E5 variant and uses their unweighted mean; folder labels are permuted as
  one block across all variants.
- Strength, tail, and spatial mismatch passed A/B construction reliability.
  Tail was removed for `|rho| > 0.8` collinearity with strength in a high-strength
  variant. Signal/nonstationarity mismatch was removed because only 12.5% of its
  synthetic quantile bins were populated. Strength and spatial remained in the
  frozen main analysis.
- Strength mismatch gave mean variant rho `-0.6061`, exact folder-block
  permutation `p=0.00293`, exploratory BH `q=0.00587`, 10/10 negative LOO
  results, 6/6 negative seed-reference summaries, and partial rank rho `-0.5077`
  after controlling gradient ratio. Variant rhos were P-L `-0.8424`, P-H
  `-0.6000`, H-L `-0.2485`, and H-H `-0.7333`. The conservative folder
  bootstrap interval reached zero, so the result is a narrow descriptive GO.
- Spatial mismatch did not support the preregistered direction: mean variant rho
  `+0.3455`, exact `p=0.1890`, with three of four variants positive. The random
  negative control gave rho `+0.0121`, exact `p=0.9489`.
- E8 therefore supports only this claim: residual-strength mismatch has a stable
  folder-level association with E5 real-domain transfer under the present
  device/data protocol. It does not establish causality, a general predictor, a
  physical mechanism, or cross-camera validity.
- Unique next step: freeze route 2's evidence chain and perform a manuscript
  claim-support audit; do not train a stronger backbone or construct a mismatch
  predictor from these ten folders.
