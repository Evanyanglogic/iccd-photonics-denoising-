# ICCD PIR 2025-07-09 Data Inventory

Date checked: 2026-07-15

## Path

```text
F:\ICCD_pir\2025.07.09\CDM-A4000-UM90_DH09131AAK00007
```

## File Inventory

- TIFF files: 305
- Index range parsed from filenames: 57 to 361
- Missing indices in that range: 0
- Subdirectories: none detected
- Metadata text files: none detected in this folder
- File size: about 4.19 MB per TIFF

## Sample TIFF Properties

Sampled TIFFs are:

- Shape: 2048x2048
- dtype: uint8
- Max possible DN: 255

This differs from the gated ICCD batch under `D:\iccd\data\20260319`, which is
5120x5120 uint16 with `PictureInfo.txt` metadata.

## Full-Sequence Brightness Segments

The 305 frames appear to contain multiple brightness regimes. Segments below
were split by large frame-mean or saturation-fraction changes.

| index range | frames | mean DN | p50 DN | p99 DN | std DN | saturated fraction |
|---|---:|---:|---:|---:|---:|---:|
| 57-187 | 131 | 101.960 | 100.962 | 186.580 | 32.277 | 0.00212 |
| 188-235 | 48 | 159.498 | 165.500 | 255.000 | 58.556 | 0.06652 |
| 236-248 | 13 | 202.987 | 246.769 | 255.000 | 68.533 | 0.46798 |
| 249-249 | 1 | 206.298 | 255.000 | 255.000 | 67.192 | 0.50481 |
| 250-251 | 2 | 201.954 | 243.000 | 255.000 | 68.878 | 0.45655 |
| 252-282 | 31 | 203.463 | 247.258 | 255.000 | 68.337 | 0.47309 |
| 283-327 | 45 | 94.204 | 91.467 | 173.844 | 27.743 | 0.00202 |
| 328-361 | 34 | 177.144 | 185.588 | 255.000 | 62.509 | 0.17710 |

## Assessment

This dataset is useful, but with a bounded role.

Suitable uses:

- 8-bit ICCD dark/background candidate analysis.
- Brightness-regime and saturation-behavior analysis for the `ICCD_pir` data
  family.
- Possible low/mid/bright flat-field candidate screening if acquisition notes
  confirm the scene is uniform.
- Supplementary evidence that ICCD data may have multiple operating regimes and
  saturation risks.

Not yet suitable for:

- Direct dark correction of `D:\iccd\data\20260319`.
- Direct flat-field correction of `D:\iccd\data\20260319`.
- Strict gain/gate/exposure-matched calibration claims.
- 16-bit raw-domain model calibration unless the export pipeline from the
  original camera data is known.

Main blockers:

- No local metadata file was found in this folder.
- The data are uint8, while the main gated ICCD batch is uint16.
- The image size is 2048x2048, while the main gated ICCD batch is 5120x5120.
- Several segments are heavily saturated, especially indices 236-282 and
  328-361.

## Current Decision

Use this path as an auxiliary ICCD calibration-candidate dataset, not as a
matching dark/flat correction set for `D:\iccd\data\20260319`.

The best immediate use is to audit the first low-saturation segment
`57-187` as a dark/background candidate and compare its temporal distribution
against `F:\ICCD_pir\dark`.

## Background Candidate Audit: Indices 57-187

Computed with:

```powershell
python scripts\audit_iccd_pir_background.py `
  --root F:\ICCD_pir\2025.07.09\CDM-A4000-UM90_DH09131AAK00007 `
  --start-index 57 `
  --end-index 187 `
  --output-dir reports\iccd_pir_20250709_background_57_187 `
  --crop-size 512 `
  --save-maps
```

This run uses 512x512 center crops from the low-saturation segment.

| metric | value |
|---|---:|
| frames | 131 |
| mean_signal | 112.069 |
| frame_mean_std | 1.99475 |
| frame_mean_min | 107.781 |
| frame_mean_max | 117.206 |
| temporal_std_mean | 17.5962 |
| temporal_std_p50 | 17.8577 |
| temporal_std_p99 | 24.8166 |
| spatial_mean_std | 25.7620 |
| fixed_to_temporal_std_ratio | 1.46407 |
| saturated_fraction_mean | 0.00234452 |
| saturated_fraction_max | 0.00259781 |
| hot_pixel_fraction_p999 | 0.00181198 |
| unstable_pixel_fraction_p999 | 0.00100327 |

Interpretation:

- Frame means are fairly stable, so this segment is credible as an auxiliary
  8-bit background/dark candidate.
- Saturation is low but nonzero, and the p99.9 mean-map threshold reaches 255,
  so hot/saturated pixels must be masked if this data family is used.
- The fixed/temporal ratio is about 1.46, showing structured background
  nonuniformity even in the low-saturation segment.
- This result strengthens auxiliary background evidence, but it still does not
  provide matching dark correction for the 5120x5120 uint16 `20260319` batch.
