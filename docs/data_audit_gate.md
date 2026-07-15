# Data Audit Gate

Run this gate before changing models or starting new training runs.

## Inputs

- Paired long-exposure/reference TIFF directory.
- Paired short-exposure/noisy TIFF directory.
- Optional dark-frame directory.
- Optional flat-field directory.
- Optional metadata CSV with one row per file.

Recommended metadata columns:

```text
filename,stem,scene_id,frame_id,device,gain,gate_width,exposure_ms,illumination_level,bit_depth,temperature
```

## Command

```powershell
python scripts\audit_iccd_dataset.py `
  --config configs\dataset_iccd.yaml `
  --output-dir reports `
  --pairs-out data_manifest\pairs.csv `
  --splits-out data_manifest\splits.yaml
```

## Pass Criteria

- Clean and noisy TIFF counts are explainable.
- Missing clean/noisy pair counts are zero or deliberately documented.
- Sampled clean/noisy shapes match.
- TIFF dtype and percentile range match the camera export expectation.
- Saturated pixel fraction is acceptable for the reference images.
- Metadata is available for scene, device, gain, gate width, exposure, and
  illumination level before paper-facing experiments.
- Dark and flat calibration sequences are available before ICCD noise modeling
  claims are made.
- Train/validation/test split is based on scene and condition fields when
  metadata is available.

## Blocking Findings

- Pair keys do not match between clean and noisy directories.
- Shape mismatch between paired frames.
- Unknown bit depth or unexplained numeric range compression.
- Training and validation use the same scene/condition without a documented
  reason.
- PSNR/SSIM are computed after uint8 conversion of normalized data.
