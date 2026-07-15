# PMRID7 Data Inventory

Known local root:

```text
E:\PMRID\PMRID7\data
```

Detected exposure directories:

```text
1ms, 5ms, 10ms, 15ms, 25ms, 50ms, 125ms, 250ms, 500ms, 1s
```

Each exposure directory currently contains 100 TIFF files with matching stems
such as `scene0001_0001.tif`.

Calibration directory detected:

```text
E:\PMRID\PMRID7\data\dark_Background
```

## Old Model Path Usage

The original training scripts repeatedly point to:

```text
E:/PMRID/PMRID7/data
```

The legacy dataloader then chooses `1s/` as clean/reference and `50ms/` as
noisy/input when those directories exist.

## Checked Pairs

### 1s Reference / 50ms Input

- Paired TIFFs: 100 / 100.
- Pair keys: all matched by filename stem.
- B0 noisy-input baseline:
  - PSNR mean/std: 12.9606 / 0.2807 dB.
  - SSIM mean/std: 0.174743 / 0.005663.

### 500ms Reference / 15ms Input

- Paired TIFFs: 100 / 100.
- Pair keys: all matched by filename stem.
- B0 noisy-input baseline:
  - PSNR mean/std: 13.5869 / 0.1699 dB.
  - SSIM mean/std: 0.191758 / 0.001216.

## Important Warning

In both checked pairs, the shorter-exposure input has a higher median intensity
than the longer-exposure reference. This suggests background offset, dark-field
bias, exposure normalization, or acquisition-condition differences. Treat this
as a data/normalization gate before training.

## Legacy List Files

Useful pair lists exist under:

```text
E:\PMRID\PMRID7\data\train_lists1
E:\PMRID\PMRID7\data\train_lists2
```

Convert them with:

```powershell
python scripts\convert_exposure_lists.py `
  --train-list E:\PMRID\PMRID7\data\train_lists1\train_list_exposure_mapping_FIXED.txt `
  --val-list E:\PMRID\PMRID7\data\train_lists1\val_list_exposure_mapping_FIXED.txt `
  --path-root E:\PMRID\PMRID7 `
  --pairs-out reports\pmrid_list_exposure_mapping\pairs.csv `
  --splits-out reports\pmrid_list_exposure_mapping\splits.yaml
```

The converter assigns the longer exposure as `clean_path` and the shorter
exposure as `noisy_path`.
