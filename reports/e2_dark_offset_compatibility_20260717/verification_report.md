# E2 Dark-Offset Compatibility Audit

- Audit status: **PARTIAL-RUN**
- Compatibility decision: **NO-DARK-OPERATIONAL**
- Synthetic pairs generated: no
- Models trained: no
- Dark artifact: `E:\PNGAN-main\iccd-photonics-denoising\reports\target_scmos_risk_audit\dark_offset_center_crop.npy`
- Dark SHA256: `519a57949cfa613ae1eeb6ecd4251601edd4a2d7e116eff55018324bd196e0f2`
- Dark effective 512 mean/median: 27404.988281 / 27395.421875 DN

## Finding

The dark array is a raw-DN mean of 64 named dark-folder frames, but its calibration conditions were not recorded.
It is therefore an UNTRACED CALIBRATION ARTIFACT for compatibility purposes.
Historical subtraction produced zero ratios of 97.7322%..98.8529%.
This is direct numerical evidence that the artifact is not compatible with the current 500 ms content under the historical subtraction pipeline.

## Candidate Ratings

| candidate | rating | condition |
|---|---|---|
| A_no_dark | ACCEPTABLE | operational content source only; unresolved pedestal and sCMOS noise retained |
| B_matched_dark_mean | INVALID | dark acquisition conditions are unrecorded and correction clips 97%+ pixels |
| C_scalar_pedestal | INVALID | no camera black-level metadata or matched-dark scalar is available |
| D_abandon_source | CONDITIONAL | required if source files or metadata cannot be recovered for the next formal no-dark audit |

## Quality Gates

- PASS `source_file_count`: 100 (saved audit)
- FAIL `source_volume_currently_accessible`: False (direct raw recheck unavailable when false)
- PASS `raw_zero_ratio_max`: 0.0 (no-dark)
- PASS `raw_saturation_ratio_max`: 0.00048422813415527344 (no-dark)
- PASS `raw_mean_range_dn`: 17036.7..19693 (no per-image normalization)
- PASS `raw_p99_range_dn`: 39463..44785 (no per-image normalization)
- FAIL `dark_subtracted_zero_ratio_median`: 0.9880847930908203 (historical dark subtraction)
- PASS `bad_pixel_mask_ratio`: 0.001251220703125 (mask only; no correction)
- PASS `per_image_p99_scaling_disabled`: True (training data rule)
- FAIL `metadata_proves_raw_or_corrected_state`: False (TIFF tags unavailable while source volume is offline)

## Decision Boundary

Raw sCMOS TIFF values divided by 65535 may be retained only as an operational content source. They are not clean ground truth, retain an unresolved pedestal and sCMOS noise, and do not establish physical ICCD signal scale.
The source volume was unavailable during this run, so raw pixel-level spatial alignment and TIFF metadata checks remain explicitly incomplete.
Historical clipped outputs were used only for a labeled censored diagnostic; they were not used to validate raw alignment or select the processing decision.
