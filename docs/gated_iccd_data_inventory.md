# Gated ICCD Data Inventory

## Downloaded Subset

Local path:

```text
F:\20260319\1
```

Current contents:

- TIFF frames: 200
- Sidecar metadata: `PictureInfo.txt`
- Filename pattern: `<frame>-Camera1[20600555].tif`
- Frame index range: 1 to 200, no missing indices detected.

Batch inventory command:

```powershell
python scripts\inventory_gated_iccd_batch.py `
  --root F:\20260319 `
  --output-dir reports\gated_iccd_20260319_inventory
```

Current batch inventory:

| folder | TIFFs | metadata rows | exposure width ms | Sync A width us | Sync B width us | gain |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 |

## Acquisition Metadata

Parsed from `PictureInfo.txt`:

| Field | Value |
|---|---:|
| exposure_delay_ms | 0.000000 |
| exposure_width_ms | 900.000000 |
| sync_a_delay_ns | 4.000000 |
| sync_a_width_us | 4.000000 |
| sync_b_delay_ns | 4.000000 |
| sync_b_width_us | 4.000000 |
| gain | 60 |

Interpretation: this subset is a single exposure/gate/gain condition. It is
useful for noise stability and fixed-pattern analysis, but not enough for
supervised clean/noisy denoising by itself.

## TIFF Properties

Sampled with:

```powershell
python scripts\audit_single_exposure_iccd.py `
  --input-dir F:\20260319\1 `
  --output-dir reports\single_exposure_20260319_1 `
  --max-files 32 `
  --max-temporal-frames 32 `
  --crop-size 512
```

Sampled frame statistics:

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| shape | 5120x5120 | - | - | - |
| dtype | uint16 | - | - | - |
| minimum | 395 | 40.8779 | 320 | 464 |
| maximum | 42306.5 | 4377.1 | 39072 | 53296 |
| p001 | 800 | 0 | 800 | 800 |
| p01 | 869.5 | 7.59934 | 864 | 880 |
| p50 | 990.5 | 4.66369 | 976 | 992 |
| p99 | 2288.5 | 2.78388 | 2288 | 2304 |
| p999 | 2532.5 | 7.19375 | 2528 | 2544 |
| mean | 1124.2 | 1.69817 | 1120.63 | 1127.32 |
| std | 353.187 | 0.34103 | 352.432 | 353.886 |
| saturated_fraction | 0 | 0 | 0 | 0 |

## Temporal Crop Statistics

Computed on a 512x512 center crop over 32 frames:

| Metric | Value |
|---|---:|
| frame_mean_mean | 1157.76 |
| frame_mean_std | 2.30129 |
| per_pixel_mean_std_spatial | 239.786 |
| per_pixel_temporal_std_mean | 64.8298 |
| per_pixel_temporal_std_p50 | 55.0795 |
| per_pixel_temporal_std_p99 | 148.539 |
| temporal_var_mean | 4981.52 |
| temporal_fano_approx | 4.30272 |

## Current Assessment

Strengths:

- Real gated ICCD metadata is present.
- 200 repeated frames are enough for first-pass temporal noise and fixed-pattern
  analysis.
- Dynamic range is not saturated at 65535.
- The data are much more relevant to the intended ICCD paper than PMRID7.

Limitations:

- Only one exposure/gate/gain condition is currently downloaded.
- No paired longer/shorter exposure sequence is available locally yet.
- No dark/flat sequence has been identified in this downloaded subset.
- Cannot support supervised denoising training or exposure-normalized clean/noisy
  claims by itself.

## Next Data Needed

For the next download, prioritize one of these:

1. Matching longer/shorter exposure sequence with the same scene and frame index
   pattern.
2. Dark frames at the same gain/gate/exposure condition.
3. Flat-field frames at the same gain/gate/exposure condition.

The most useful next condition is another folder from `F:\20260319\...` whose
`PictureInfo.txt` has different `exposure_width_ms` or gate width but matching
frame count and scene.

## Full Downloaded Batch

Local path:

```text
D:\iccd\data\20260319
```

Inventory command:

```powershell
python scripts\inventory_gated_iccd_batch.py `
  --root D:\iccd\data\20260319 `
  --output-dir reports\gated_iccd_20260319_full_inventory
```

Folder completeness:

| folder | TIFFs | PictureInfo length | status |
|---|---:|---:|---|
| 1 | 200 | 35692 | complete |
| 2 | 200 | 35692 | complete |
| 3 | 4 | 708 | partial |
| 4 | 200 | 35692 | complete |
| 5 | 200 | 35692 | complete |
| 6 | 5 | 885 | partial |
| 7 | 200 | 35692 | complete |
| 8 | 200 | 35692 | complete |
| 9 | 200 | 35692 | complete |
| 10 | 200 | 35692 | complete |
| 11 | 200 | 35692 | complete |
| 12 | 0 | 0 | empty/incomplete |
| 13 | 200 | 35692 | complete |
| 1_20260715_143749 | 0 | - | empty helper folder |

Batch inventory summary:

| folder | TIFFs | metadata rows | exposure width ms | Sync A width us | Sync B width us | gain | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 992 | 2288 |
| 2 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 960 | 1184 |
| 3 | 4 | 4 | 900.000000 | 4.000000 | 4.000000 | 60 | 1056 | 4352 |
| 4 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 1056 | 4336 |
| 5 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 1456 | 15248 |
| 6 | 5 | 5 | 900.000000 | 4.000000 | 4.000000 | 60 | 1440 | 15136 |
| 7 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 1168 | 7696 |
| 8 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 1120 | 6608 |
| 9 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 1008 | 3056 |
| 10 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 992 | 2224 |
| 11 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 976 | 1680 |
| 13 | 200 | 200 | 900.000000 | 4.000000 | 4.000000 | 60 | 944 | 1056 |

Current interpretation:

- The batch currently appears to contain one exposure/gate/gain setting:
  exposure width 900 ms, Sync A/B width 4 us, gain 60.
- Folder-to-folder brightness changes are substantial, but they are not
  explained by exposure/gate/gain metadata in `PictureInfo.txt`.
- These folders are suitable for single-condition temporal noise and
  fixed-pattern analysis across several brightness levels.
- They do not yet form supervised clean/noisy pairs by exposure. A paired
  denoising dataset still needs another exposure/gate/gain condition or a
  clearly identified reference acquisition.

## Single-Condition Noise Summary

Computed with:

```powershell
python scripts\summarize_single_condition_noise.py `
  --root D:\iccd\data\20260319 `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --output-dir reports\gated_iccd_20260319_noise_summary `
  --max-frames 32 `
  --crop-size 512
```

These values use 512x512 center crops from the first 32 frames of each complete
folder.

| folder | mean signal | frame mean std | spatial fixed std | temporal std mean | Fano approx | fixed/temporal |
|---|---:|---:|---:|---:|---:|---:|
| 13 | 936.109 | 2.2052 | 14.4721 | 37.7137 | 1.64282 | 0.383735 |
| 2 | 966.032 | 2.77462 | 35.5336 | 42.5688 | 2.01727 | 0.834734 |
| 11 | 1083.35 | 2.85444 | 163.789 | 56.784 | 3.40482 | 2.88441 |
| 1 | 1157.76 | 2.30129 | 239.786 | 64.8298 | 4.30272 | 3.6987 |
| 10 | 1202.05 | 2.87118 | 296.421 | 69.7951 | 4.98268 | 4.24702 |
| 9 | 1395.74 | 3.06875 | 507.677 | 85.5111 | 6.74011 | 5.93697 |
| 4 | 1787.63 | 2.66886 | 923.57 | 108.358 | 8.88335 | 8.52332 |
| 8 | 2243.32 | 3.46062 | 1420.08 | 131.337 | 10.4732 | 10.8125 |
| 7 | 2510.35 | 2.80575 | 1706.33 | 143.278 | 11.2167 | 11.9092 |
| 5 | 4716.93 | 6.79848 | 4030.13 | 217.86 | 14.0046 | 18.4987 |

Preliminary interpretation:

- Temporal noise increases with signal level, which supports a signal-dependent
  noise component.
- Approximate Fano factor is greater than 1 and grows with brightness, which is
  consistent with multiplicative/intensifier behavior rather than simple
  Poisson noise alone.
- Spatial fixed-pattern variation grows faster than temporal noise in brighter
  folders, making flat-field/fixed-pattern correction important before final
  denoising claims.
