"""PyTorch Dataset for ICCD paired denoising data."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .manifest import PairRecord, resolve_split_records


class ICCDPairDataset:
    """Manifest-backed clean/noisy TIFF dataset.

    Returns a dictionary with `noisy`, `clean`, `pair_key`, and `metadata`.
    Image arrays are CHW float32 tensors when torch is available; otherwise
    they are CHW float32 numpy arrays.
    """

    def __init__(
        self,
        pairs_csv: str | Path,
        splits_yaml: str | Path,
        split: str,
        range_max: float = 65535.0,
        patch_size: int | None = None,
        crop_mode: Literal["random", "center", "none"] = "none",
        augment: bool = False,
        seed: int | None = None,
        base_dir: str | Path | None = None,
        return_tensors: bool = True,
    ) -> None:
        self.records = resolve_split_records(pairs_csv, splits_yaml, split, base_dir=base_dir)
        self.split = split
        self.range_max = float(range_max)
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.augment = augment
        self.return_tensors = return_tensors
        self.rng = random.Random(seed)
        if self.range_max <= 0:
            raise ValueError("range_max must be positive")
        if self.crop_mode not in {"random", "center", "none"}:
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        clean = load_tiff_normalized(record.clean_path, self.range_max)
        noisy = load_tiff_normalized(record.noisy_path, self.range_max)
        if clean.shape != noisy.shape:
            raise ValueError(f"Shape mismatch for {record.pair_key}: clean {clean.shape}, noisy {noisy.shape}")

        clean, noisy = crop_pair(clean, noisy, self.patch_size, self.crop_mode, self.rng)
        if self.augment:
            clean, noisy = augment_pair(clean, noisy, self.rng)

        clean_chw = to_chw(clean)
        noisy_chw = to_chw(noisy)
        if self.return_tensors:
            clean_chw = to_tensor(clean_chw)
            noisy_chw = to_tensor(noisy_chw)

        return {
            "pair_key": record.pair_key,
            "noisy": noisy_chw,
            "clean": clean_chw,
            "metadata": record.metadata,
        }


def make_iccd_dataset(
    pairs_csv: str | Path = "data_manifest/pairs.csv",
    splits_yaml: str | Path = "data_manifest/splits.yaml",
    split: str = "train",
    **kwargs: Any,
) -> ICCDPairDataset:
    """Convenience factory for training scripts."""

    return ICCDPairDataset(pairs_csv=pairs_csv, splits_yaml=splits_yaml, split=split, **kwargs)


def load_tiff_normalized(path: Path, range_max: float) -> np.ndarray:
    try:
        import tifffile

        arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc
    return np.clip(arr / range_max, 0.0, 1.0).astype(np.float32)


def crop_pair(
    clean: np.ndarray,
    noisy: np.ndarray,
    patch_size: int | None,
    crop_mode: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size is None or crop_mode == "none":
        return clean, noisy
    h, w = spatial_shape(clean)
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size {patch_size} exceeds image shape {clean.shape}")
    if crop_mode == "center":
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2
    else:
        top = rng.randint(0, h - patch_size)
        left = rng.randint(0, w - patch_size)
    return crop_image(clean, top, left, patch_size), crop_image(noisy, top, left, patch_size)


def augment_pair(clean: np.ndarray, noisy: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        clean = np.flip(clean, axis=-2 if is_chw(clean) else 0).copy()
        noisy = np.flip(noisy, axis=-2 if is_chw(noisy) else 0).copy()
    if rng.random() < 0.5:
        clean = np.flip(clean, axis=-1 if is_chw(clean) else 1).copy()
        noisy = np.flip(noisy, axis=-1 if is_chw(noisy) else 1).copy()
    k = rng.randint(0, 3)
    if k:
        axes = (-2, -1) if is_chw(clean) else (0, 1)
        clean = np.rot90(clean, k=k, axes=axes).copy()
        noisy = np.rot90(noisy, k=k, axes=axes).copy()
    return clean, noisy


def spatial_shape(image: np.ndarray) -> tuple[int, int]:
    if image.ndim == 2:
        return int(image.shape[0]), int(image.shape[1])
    if image.ndim == 3 and is_chw(image):
        return int(image.shape[1]), int(image.shape[2])
    if image.ndim == 3:
        return int(image.shape[0]), int(image.shape[1])
    raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def crop_image(image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    if image.ndim == 2:
        return image[top : top + size, left : left + size]
    if is_chw(image):
        return image[:, top : top + size, left : left + size]
    return image[top : top + size, left : left + size, :]


def to_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, :, :].astype(np.float32)
    if image.ndim == 3 and is_chw(image):
        return image.astype(np.float32)
    if image.ndim == 3:
        return np.moveaxis(image, -1, 0).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def is_chw(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[0] <= 4 and image.shape[1] > 4 and image.shape[2] > 4


def to_tensor(image: np.ndarray) -> Any:
    try:
        import torch

        return torch.from_numpy(np.ascontiguousarray(image))
    except Exception:
        return image
