# Manifest Dataloader

Use the manifest dataloader after the data audit gate has generated:

- `data_manifest/pairs.csv`
- `data_manifest/splits.yaml`

Legacy exposure-pair text lists can be converted with
`scripts/convert_exposure_lists.py`.

The dataloader prevents training scripts from silently using the same directory
for train and validation.

## Check Command

```powershell
python scripts\check_manifest_dataloader.py `
  --pairs-csv data_manifest\pairs.csv `
  --splits-yaml data_manifest\splits.yaml `
  --split train `
  --range-max 65535 `
  --patch-size 128 `
  --crop-mode random
```

## Training Usage

```python
from torch.utils.data import DataLoader

from src.iccd_data import make_iccd_dataset

train_dataset = make_iccd_dataset(
    pairs_csv="data_manifest/pairs.csv",
    splits_yaml="data_manifest/splits.yaml",
    split="train",
    range_max=65535.0,
    patch_size=128,
    crop_mode="random",
    augment=True,
    seed=20260715,
)

val_dataset = make_iccd_dataset(
    pairs_csv="data_manifest/pairs.csv",
    splits_yaml="data_manifest/splits.yaml",
    split="val",
    range_max=65535.0,
    patch_size=128,
    crop_mode="center",
    augment=False,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
```

Each sample is a dictionary:

```text
{
  "pair_key": str,
  "noisy": Tensor or ndarray,  # CHW float32 in [0, 1]
  "clean": Tensor or ndarray,  # CHW float32 in [0, 1]
  "metadata": dict
}
```

## Rules

- Training split uses random crop and optional paired augmentation.
- Validation/test split uses center crop or full image.
- Do not re-sort directories inside training scripts.
- Do not create validation data from the training directory unless the manifest
  explicitly assigns those pair keys to validation.
