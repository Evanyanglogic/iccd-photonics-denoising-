"""Audit multi-exposure sCMOS data for pairing, offset, and bad-pixel risks."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
TRAILING_INDEX = re.compile(r"(?P<index>\d{3})$")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = [path for path in sorted(root.iterdir(), key=lambda path: exposure_sort_key(path.name)) if path.is_dir()]
    if not folders:
        raise ValueError(f"No subfolders found under {root}")

    folder_rows = [summarize_folder(folder, args.max_sample_frames, args.saturation_value) for folder in folders]
    dark_folder = root / args.dark_folder
    dark_files = list_tiffs(dark_folder)
    if not dark_files:
        raise ValueError(f"No dark TIFF files found under {dark_folder}")

    dark_stack = read_sample_stack(dark_files[: args.max_dark_frames], args.mask_crop_size)
    dark_offset = np.mean(dark_stack, axis=0).astype(np.float32)
    dark_std = np.std(dark_stack, axis=0, ddof=1).astype(np.float32) if dark_stack.shape[0] > 1 else np.zeros_like(dark_offset)
    bad_mask = build_bad_pixel_mask(dark_stack, dark_offset, dark_std, args.saturation_value, args.hot_sigma)

    pair_rows = build_pair_candidates(root, folders, args.dark_folder)

    folder_csv = output_dir / "folder_summary.csv"
    pair_csv = output_dir / "pair_candidates.csv"
    dark_offset_path = output_dir / "dark_offset_center_crop.npy"
    dark_std_path = output_dir / "dark_std_center_crop.npy"
    bad_mask_path = output_dir / "bad_pixel_mask_center_crop.npy"
    report_path = output_dir / "scmos_target_data_audit.md"

    write_csv(folder_rows, folder_csv)
    write_csv(pair_rows, pair_csv)
    np.save(dark_offset_path, dark_offset)
    np.save(dark_std_path, dark_std)
    np.save(bad_mask_path, bad_mask.astype(np.uint8))
    write_report(root, folder_rows, pair_rows, bad_mask, dark_offset, dark_std, folder_csv, pair_csv, dark_offset_path, bad_mask_path, report_path)

    print(f"Wrote folder summary: {folder_csv}")
    print(f"Wrote pair candidates: {pair_csv}")
    print(f"Wrote dark offset: {dark_offset_path}")
    print(f"Wrote bad-pixel mask: {bad_mask_path}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/target_scmos_risk_audit")
    parser.add_argument("--dark-folder", default="dark_Background")
    parser.add_argument("--max-sample-frames", type=int, default=16)
    parser.add_argument("--max-dark-frames", type=int, default=64)
    parser.add_argument("--mask-crop-size", type=int, default=1024)
    parser.add_argument("--saturation-value", type=float, default=65535.0)
    parser.add_argument("--hot-sigma", type=float, default=6.0)
    return parser.parse_args()


def summarize_folder(folder: Path, max_sample_frames: int, saturation_value: float) -> dict[str, Any]:
    files = list_tiffs(folder)
    indices = [trailing_index(path) for path in files]
    indices = [idx for idx in indices if idx is not None]
    unique_indices = set(indices)
    missing = missing_indices(unique_indices)
    sample_rows = [summarize_image(path, saturation_value) for path in files[:max_sample_frames]]
    aggregate = aggregate_image_summaries(sample_rows)
    return {
        "folder": folder.name,
        "tiff_count": len(files),
        "unique_tail_indices": len(unique_indices),
        "min_tail_index": min(unique_indices) if unique_indices else "",
        "max_tail_index": max(unique_indices) if unique_indices else "",
        "missing_tail_index_count": len(missing),
        "first_missing_tail_indices": " ".join(str(item) for item in missing[:12]),
        **aggregate,
    }


def build_pair_candidates(root: Path, folders: list[Path], dark_folder: str) -> list[dict[str, Any]]:
    maps: dict[str, dict[int, Path]] = {}
    for folder in folders:
        if folder.name == dark_folder:
            continue
        current: dict[int, Path] = {}
        for path in list_tiffs(folder):
            idx = trailing_index(path)
            if idx is not None and idx not in current:
                current[idx] = path
        maps[folder.name] = current

    exposure_names = sorted(maps, key=exposure_sort_key)
    rows: list[dict[str, Any]] = []
    for noisy in exposure_names:
        for clean in exposure_names:
            if exposure_sort_key(clean) <= exposure_sort_key(noisy):
                continue
            common = sorted(set(maps[noisy]).intersection(maps[clean]))
            rows.append(
                {
                    "noisy_exposure": noisy,
                    "clean_exposure": clean,
                    "common_tail_indices": len(common),
                    "first_common_indices": " ".join(str(item) for item in common[:12]),
                    "noisy_count": len(maps[noisy]),
                    "clean_count": len(maps[clean]),
                    "candidate_pairs_csv": str(root / noisy) + " -> " + str(root / clean),
                    "recommended": "yes" if len(common) >= 50 else "no",
                }
            )
    return rows


def list_tiffs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES], key=natural_file_key)


def natural_file_key(path: Path) -> tuple[int, str]:
    idx = trailing_index(path)
    return (idx if idx is not None else 10**12, path.name)


def trailing_index(path: Path) -> int | None:
    match = TRAILING_INDEX.search(path.stem)
    if not match:
        return None
    return int(match.group("index"))


def missing_indices(indices: set[int]) -> list[int]:
    if not indices:
        return []
    return [idx for idx in range(min(indices), max(indices) + 1) if idx not in indices]


def summarize_image(path: Path, saturation_value: float) -> dict[str, float]:
    arr = read_tiff(path).astype(np.float64)
    return {
        "mean": float(np.mean(arr)),
        "minimum": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "maximum": float(np.max(arr)),
        "saturated_fraction": float(np.mean(arr >= saturation_value)),
        "zero_fraction": float(np.mean(arr <= 0)),
    }


def aggregate_image_summaries(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_mean": "",
            "sample_p50": "",
            "sample_p99": "",
            "sample_saturated_fraction": "",
            "sample_zero_fraction": "",
        }
    return {
        "sample_mean": float(np.mean([row["mean"] for row in rows])),
        "sample_p50": float(np.mean([row["p50"] for row in rows])),
        "sample_p99": float(np.mean([row["p99"] for row in rows])),
        "sample_saturated_fraction": float(np.mean([row["saturated_fraction"] for row in rows])),
        "sample_zero_fraction": float(np.mean([row["zero_fraction"] for row in rows])),
    }


def read_sample_stack(paths: list[Path], crop_size: int) -> np.ndarray:
    crops = [center_crop(read_tiff(path), crop_size).astype(np.float32) for path in paths]
    return np.stack(crops, axis=0)


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale TIFF, got shape {arr.shape}")
    h, w = arr.shape
    size = min(crop_size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top : top + size, left : left + size]


def build_bad_pixel_mask(dark_stack: np.ndarray, dark_offset: np.ndarray, dark_std: np.ndarray, saturation_value: float, hot_sigma: float) -> np.ndarray:
    saturated = np.any(dark_stack >= saturation_value, axis=0)
    dead = np.any(dark_stack <= 0, axis=0)
    median_offset = float(np.median(dark_offset))
    median_std = float(np.median(dark_std))
    mad_offset = median_absolute_deviation(dark_offset)
    mad_std = median_absolute_deviation(dark_std)
    hot_offset = dark_offset > median_offset + hot_sigma * max(1.4826 * mad_offset, 1.0)
    hot_std = dark_std > median_std + hot_sigma * max(1.4826 * mad_std, 1.0)
    return saturated | dead | hot_offset | hot_std


def median_absolute_deviation(arr: np.ndarray) -> float:
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def exposure_sort_key(name: str) -> float:
    if name == "dark_Background":
        return -1.0
    lower = name.lower()
    try:
        if lower.endswith("ms"):
            return float(lower[:-2])
        if lower.endswith("s"):
            return float(lower[:-1]) * 1000.0
        return float(lower)
    except ValueError:
        return 10**12


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    root: Path,
    folder_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    bad_mask: np.ndarray,
    dark_offset: np.ndarray,
    dark_std: np.ndarray,
    folder_csv: Path,
    pair_csv: Path,
    dark_offset_path: Path,
    bad_mask_path: Path,
    report_path: Path,
) -> None:
    recommended_pairs = [row for row in pair_rows if row["recommended"] == "yes"]
    lines = [
        "# sCMOS Target Data Risk Audit",
        "",
        f"- Root: `{root}`",
        f"- Folder summary CSV: `{folder_csv}`",
        f"- Pair candidates CSV: `{pair_csv}`",
        f"- Dark offset crop: `{dark_offset_path}`",
        f"- Bad-pixel mask crop: `{bad_mask_path}`",
        "",
        "## Folder Summary",
        "",
        "| folder | TIFFs | unique indices | missing indices | sample p50 | sample p99 | saturation frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folder_rows:
        lines.append(
            "| "
            f"{row['folder']} | {row['tiff_count']} | {row['unique_tail_indices']} | "
            f"{row['missing_tail_index_count']} | {format_value(row['sample_p50'])} | "
            f"{format_value(row['sample_p99'])} | {format_value(row['sample_saturated_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Dark / Bad-Pixel Summary",
            "",
            f"- Dark offset median: {float(np.median(dark_offset)):.6g} DN",
            f"- Dark offset mean: {float(np.mean(dark_offset)):.6g} DN",
            f"- Dark temporal std median: {float(np.median(dark_std)):.6g} DN",
            f"- Bad-pixel mask fraction on crop: {float(np.mean(bad_mask)):.6g}",
            "",
            "## Recommended Pair Candidates",
            "",
            "| noisy | clean | common indices | first indices |",
            "|---|---|---:|---|",
        ]
    )
    for row in recommended_pairs[:20]:
        lines.append(
            "| "
            f"{row['noisy_exposure']} | {row['clean_exposure']} | "
            f"{row['common_tail_indices']} | {row['first_common_indices']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat this dataset as sCMOS data, not real ICCD paired data.",
            "- Do not modify raw TIFF files. Apply dark/offset correction and bad-pixel masks as derived preprocessing artifacts.",
            "- Exposure folders are not automatically clean/noisy pairs just because exposure names differ; use tail-index candidate pairs and visual/statistical checks.",
            "- `1s` has incomplete/duplicated tail-index coverage and should not be a primary clean reference without manual review.",
            "- These data are suitable as sCMOS baselines or clean/content sources for ICCD-like synthetic noise generated from real ICCD statistics.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
