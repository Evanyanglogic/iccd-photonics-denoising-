"""Shared, auditable utilities for the E1 formal ICCD rerun."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FRAME_NUMBER = re.compile(r"^(\d+)")
TIFF_SUFFIXES = {".tif", ".tiff"}
COMMON_FIELDS = [
    "folder",
    "frame_count",
    "crop_size",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
    "metric_definition",
    "value",
    "valid_sample_count",
    "warning_flags",
]


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def indexed_tiffs(folder: Path) -> list[tuple[int, Path]]:
    indexed: dict[int, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        match = FRAME_NUMBER.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate frame index {index} in {folder}")
        indexed[index] = path
    return sorted(indexed.items())


def crop_coordinates(shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    height, width = shape
    actual = min(int(size), height, width)
    top = (height - actual) // 2
    left = (width - actual) // 2
    return top, left, actual, actual


def read_crop(path: Path, crop_size: int) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    import tifffile

    try:
        image = tifffile.memmap(path)
    except Exception:
        image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale TIFF at {path}, got {image.shape}")
    coords = crop_coordinates(tuple(image.shape), crop_size)
    top, left, height, width = coords
    crop = np.asarray(image[top : top + height, left : left + width])
    return crop, coords, str(image.dtype)


def read_stack(paths: list[Path], crop_size: int, frame_count: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    selected = paths[: int(frame_count)]
    if len(selected) < int(frame_count):
        raise ValueError(f"Requested {frame_count} frames but only {len(selected)} are available")
    crops = []
    coords = None
    for path in selected:
        crop, current, _ = read_crop(path, crop_size)
        if coords is None:
            coords = current
        elif coords != current:
            raise ValueError(f"Inconsistent crop coordinates: {coords} != {current}")
        crops.append(crop.astype(np.float32, copy=False))
    assert coords is not None
    return np.stack(crops), coords


def selected_paths(config: dict[str, Any], folder: int) -> list[Path]:
    folder_path = Path(config["data_root"]) / str(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(folder_path)
    indexed = indexed_tiffs(folder_path)
    return [path for _, path in indexed[: int(config["max_frames"])]]


def common_metadata(
    folder: int,
    frame_count: int,
    crop_size: int,
    coords: tuple[int, int, int, int],
    definition: str,
    value: float,
    valid_count: int,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    top, left, height, width = coords
    return {
        "folder": folder,
        "frame_count": int(frame_count),
        "crop_size": int(crop_size),
        "crop_top": int(top),
        "crop_left": int(left),
        "crop_height": int(height),
        "crop_width": int(width),
        "metric_definition": definition,
        "value": float(value),
        "valid_sample_count": int(valid_count),
        "warning_flags": ";".join(sorted(set(warnings))),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows produced for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def correlation(first: np.ndarray, second: np.ndarray, stride: int = 1) -> float:
    a = np.asarray(first, dtype=np.float64)[::stride, ::stride].ravel()
    b = np.asarray(second, dtype=np.float64)[::stride, ::stride].ravel()
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def vector_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def linear_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(1, len(y) + 1, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def warnings_for_values(values: Iterable[float]) -> list[str]:
    return ["non_finite_metric"] if any(not math.isfinite(float(value)) for value in values) else []
