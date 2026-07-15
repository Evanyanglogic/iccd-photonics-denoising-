# Target sCMOS Data Inventory

## Local Path

```text
F:\目标传感器噪声参数估计\data
```

This dataset is treated as sCMOS data. It must not be described as real ICCD
paired denoising data. It can be used as:

- a sCMOS multi-exposure baseline source;
- a clean/content source for ICCD-like synthetic noisy images;
- a comparison dataset for ICCD-vs-sCMOS noise behavior if acquisition
  conditions can be justified.

## Audit Command

```powershell
python scripts\audit_scmos_target_data.py `
  --root "F:\目标传感器噪声参数估计\data" `
  --output-dir reports\target_scmos_risk_audit `
  --max-sample-frames 16 `
  --max-dark-frames 64 `
  --mask-crop-size 1024
```

Generated artifacts:

- `reports\target_scmos_risk_audit\folder_summary.csv`
- `reports\target_scmos_risk_audit\pair_candidates.csv`
- `reports\target_scmos_risk_audit\dark_offset_center_crop.npy`
- `reports\target_scmos_risk_audit\dark_std_center_crop.npy`
- `reports\target_scmos_risk_audit\bad_pixel_mask_center_crop.npy`
- `reports\target_scmos_risk_audit\scmos_target_data_audit.md`

These are derived preprocessing/audit artifacts. Raw TIFF files are not
modified.

## Folder Summary

| folder | TIFFs | unique tail indices | missing tail indices | sample p50 | sample p99 | saturation frac |
|---|---:|---:|---:|---:|---:|---:|
| dark_Background | 100 | 100 | 0 | 27940 | 40529.8 | 0.000484228 |
| 1ms | 500 | 500 | 0 | 29840.9 | 40876.4 | 0.000484228 |
| 5ms | 500 | 500 | 0 | 29878.6 | 41133.6 | 0.000484228 |
| 10ms | 500 | 500 | 0 | 29319.9 | 40660.1 | 0.000484228 |
| 15ms | 500 | 500 | 0 | 29090.1 | 40822.6 | 0.000484228 |
| 25ms | 300 | 300 | 0 | 28298.7 | 40798 | 0.000484228 |
| 50ms | 100 | 100 | 0 | 27586.6 | 43088.5 | 0.000484228 |
| 125ms | 100 | 100 | 0 | 24891.8 | 46042.9 | 0.000484228 |
| 250ms | 100 | 100 | 0 | 20136.8 | 43038.3 | 0.000484228 |
| 500ms | 100 | 100 | 0 | 16223.8 | 42733 | 0.000484228 |
| 1s | 100 | 19 | 81 | 12277.1 | 37722.2 | 0.000486076 |

## Risk Handling

### Dark / Offset

Observed:

- Dark offset median on 1024x1024 center crop: 27627.5 DN.
- Dark offset mean on 1024x1024 center crop: 27712.1 DN.
- Dark temporal std median: 3788.01 DN.

Action:

- Do not subtract a scalar blindly from all raw images.
- Use the saved `dark_offset_center_crop.npy` for crop-level corrected analysis.
- For full-frame processing, estimate full-frame dark offset before correction.

### Saturated / Bad Pixels

Observed:

- Sample saturation fraction is about `4.84e-4` in most folders.
- A crop-level bad-pixel mask was generated from dark frames.
- Bad-pixel mask fraction on 1024x1024 crop: 0.00148296.

Action:

- Use `bad_pixel_mask_center_crop.npy` for crop-level statistics and loss/mask
  experiments.
- Do not treat saturated or hot pixels as valid signal in PSNR/SSIM or
  mean-variance fitting.

### Exposure Pairing

Observed:

- Tail-index pairing works for many exposure combinations.
- `15ms -> 500ms` has 100 common indices.
- `1ms/5ms/10ms/15ms` share 500 indices.
- `1s` has only 19 unique tail indices and 81 missing tail indices.

Action:

- Generate pair manifests by tail index, not full filename.
- Exclude `1s` from primary clean-reference experiments unless manually
  reviewed.
- Do not assume exposure folders are clean/noisy pairs until brightness,
  alignment, and scene consistency are checked.

## Recommended Use

Use these data in three stages:

1. sCMOS audit baseline:
   - dark/offset statistics;
   - bad-pixel and saturation mask;
   - tail-index pair integrity.
2. sCMOS denoising/proxy baseline:
   - candidate pairs such as `15ms -> 500ms`;
   - report clearly as sCMOS data.
3. ICCD-like synthetic training source:
   - use longer-exposure sCMOS frames as content/reference;
   - inject ICCD-like noise calibrated from real ICCD repeated frames;
   - describe as synthetic ICCD-like noisy data, not real ICCD paired data.

## Next Steps

1. Add dark-offset and bad-pixel mask support to pair evaluation scripts.
2. Visually/statistically inspect `15ms -> 500ms` for brightness mismatch,
   alignment, and residual structure.
3. Use real ICCD mean-variance and fixed-pattern statistics to synthesize
   ICCD-like noisy images from selected sCMOS content frames.

## Tail-Index Pair Manifest: 15ms -> 500ms

Generated with:

```powershell
python scripts\convert_scmos_tail_pairs.py `
  --root "F:\目标传感器噪声参数估计\data" `
  --noisy-exposure 15ms `
  --clean-exposure 500ms `
  --output-dir reports\target_scmos_15ms_500ms_manifest `
  --dark-offset-path reports\target_scmos_risk_audit\dark_offset_center_crop.npy `
  --bad-pixel-mask-path reports\target_scmos_risk_audit\bad_pixel_mask_center_crop.npy
```

Outputs:

- `reports\target_scmos_15ms_500ms_manifest\pairs.csv`
- `reports\target_scmos_15ms_500ms_manifest\splits.yaml`
- `reports\target_scmos_15ms_500ms_manifest\manifest_report.md`

Pairing result:

- Common tail-index pairs: 100.
- Tail-index range: `000..099`.
- Splits: train 85, val 8, test 7.
- Metadata records `source_device=sCMOS` and claim boundary:
  `sCMOS proxy/content source, not real ICCD paired data`.

No-model baseline command:

```powershell
python scripts\evaluate_pair_baseline.py `
  --pairs-csv reports\target_scmos_15ms_500ms_manifest\pairs.csv `
  --output-dir reports\target_scmos_15ms_500ms_b0 `
  --range-max 65535 `
  --bins 8
```

B0 result:

- Pair count: 100.
- PSNR mean/std: 13.5869 / 0.1699 dB.
- SSIM mean/std: 0.191758 / 0.001216.
- Residual mean/std mean: 0.179934 / 0.106725.

Interpretation:

- The manifest is valid for dataloader and metric scripts.
- The large residual mean indicates brightness/offset mismatch, so this pair set
  is not yet a clean supervised training target.
- Before training or ICCD-like synthesis, run mask-aware and offset-corrected
  pair checks.
