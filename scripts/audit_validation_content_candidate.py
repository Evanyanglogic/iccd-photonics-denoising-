"""Read-only candidate loaders and integrity summaries for E2 validation-source audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from compare_content_source_independence import perceptual_hash, sha256, thumbnail

GENERIC_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def evenly_spaced(items: list, maximum: int) -> list:
    if len(items) <= maximum:
        return items
    indexes = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[index] for index in indexes]


def load_generic(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image).copy()


def pmrid_records(root: Path) -> list[dict]:
    records = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    return [{**record, "input_path": root / record["input"], "gt_path": root / record["gt"]} for record in records]


def load_pmrid_raw(path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"Unexpected PMRID RAW size: {path}")
    return data.reshape(shape)


def image_statistics(candidate_id: str, path: Path, array: np.ndarray, role: str, group: str, audit_scope: str) -> dict:
    values = np.asarray(array)
    numeric = values.astype(np.float64)
    maximum = float(np.iinfo(values.dtype).max) if np.issubdtype(values.dtype, np.integer) else float(np.nanmax(numeric))
    return {
        "candidate_id": candidate_id,
        "path": str(path),
        "sha256": sha256(path),
        "file_size": path.stat().st_size,
        "modified_time": path.stat().st_mtime_ns,
        "role": role,
        "group": group,
        "audit_scope": audit_scope,
        "read_success": True,
        "dtype": str(values.dtype),
        "shape": "x".join(map(str, values.shape)),
        "page_count": 1,
        "bit_depth": values.dtype.itemsize * 8,
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "zero_ratio": float(np.mean(numeric == 0)),
        "saturation_ratio": float(np.mean(numeric >= maximum)),
    }


def prepare_representation(path: Path, array: np.ndarray, size: int, content_id: str | None = None) -> dict:
    thumb = thumbnail(array, size)
    return {"path": str(path), "content_id": content_id or path.stem, "sha256": sha256(path), "phash": perceptual_hash(thumb), "thumbnail": thumb}


def audit_pmrid(root: Path, size: int, maximum: int) -> tuple[list[dict], list[dict], dict]:
    records = pmrid_records(root)
    stats, reps = [], []
    selected = []
    for scene in sorted({record["meta"]["scene_id"] for record in records}):
        selected.extend([record for record in records if record["meta"]["scene_id"] == scene][:4])
    selected_paths = {record["gt_path"] for record in selected[:maximum]}
    failures = []
    for record in records:
        try:
            shape = tuple(record["meta"]["shape"])
            array = load_pmrid_raw(record["gt_path"], shape)
            stats.append(image_statistics("pmrid_official_benchmark", record["gt_path"], array, "gt_raw_content", record["meta"]["scene_id"], "all_gt_files"))
            if record["gt_path"] in selected_paths:
                reps.append(prepare_representation(record["gt_path"], array, size, record["meta"]["name"]))
        except Exception as error:
            failures.append({"path": str(record["gt_path"]), "error": repr(error)})
    summary = {
        "total_records": len(records), "gt_files_expected": len(records), "gt_files_read": len(stats), "failures": failures,
        "scene_ids": sorted({record["meta"]["scene_id"] for record in records}),
        "light_conditions": sorted({record["meta"]["light"] for record in records}),
        "iso_values": sorted({record["meta"]["ISO"] for record in records}),
        "exposure_values": sorted({record["meta"]["exp_time"] for record in records}),
        "shape_values": sorted({tuple(record["meta"]["shape"]) for record in records}),
        "bayer_patterns": sorted({record["meta"]["bayer_pattern"] for record in records}),
        "representative_count": len(reps),
    }
    return stats, reps, summary


def generic_files(root: Path, exclude_parts: set[str] | None = None) -> list[Path]:
    exclude_parts = {x.lower() for x in (exclude_parts or set())}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in GENERIC_SUFFIXES and not any(part.lower() in exclude_parts for part in p.parts))


def audit_generic(candidate_id: str, root: Path, size: int, maximum: int, exclude_parts: set[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    files = generic_files(root, exclude_parts)
    selected = evenly_spaced(files, maximum)
    stats, reps, failures = [], [], []
    for path in selected:
        try:
            array = load_generic(path)
            group = str(path.parent.relative_to(root))
            stats.append(image_statistics(candidate_id, path, array, "candidate_content", group, "deterministic_representative_sample"))
            reps.append(prepare_representation(path, array, size))
        except Exception as error:
            failures.append({"path": str(path), "error": repr(error)})
    return stats, reps, {"total_image_files": len(files), "sampled_files": len(selected), "successfully_read": len(stats), "failures": failures, "representative_count": len(reps)}


def metadata_rows(candidate_id: str, summary: dict) -> pd.DataFrame:
    return pd.DataFrame([{"candidate_id": candidate_id, "field": key, "value": json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (list, dict)) else str(value)} for key, value in summary.items()])
