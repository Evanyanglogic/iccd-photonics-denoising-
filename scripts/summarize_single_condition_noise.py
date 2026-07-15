"""Summarize temporal/spatial noise for multiple single-condition ICCD folders."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
LEADING_NUMBER = re.compile(r"^(?P<number>\d+)")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = select_folders(root, args.folders)
    rows = []
    for folder in folders:
        paths = list_tiffs(folder)
        if args.max_frames > 0:
            paths = paths[: args.max_frames]
        if len(paths) < 2:
            continue
        print(f"Processing {folder.name}: {len(paths)} frames")
        rows.append(summarize_folder(folder, paths, crop_size=args.crop_size))

    if not rows:
        raise ValueError("No folder had at least two TIFF frames.")

    csv_path = output_dir / "single_condition_noise_summary.csv"
    report_path = output_dir / "single_condition_noise_summary.md"
    write_csv(rows, csv_path)
    write_report(root, rows, csv_path, report_path)
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/single_condition_noise")
    parser.add_argument("--folders", nargs="*", default=[], help="Folder names to process. Empty means all immediate subfolders.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args()


def select_folders(root: Path, names: list[str]) -> list[Path]:
    if names:
        return [root / name for name in names]
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: natural_folder_key(path.name))


def natural_folder_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_", 1)[0]), name
    except ValueError:
        return 10**12, name


def list_tiffs(root: Path) -> list[Path]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_file_key)


def natural_file_key(path: Path) -> tuple[int, str]:
    match = LEADING_NUMBER.match(path.name)
    if match:
        return int(match.group("number")), path.name
    return 10**12, path.name


def summarize_folder(folder: Path, paths: list[Path], crop_size: int) -> dict[str, Any]:
    crops = [center_crop(read_tiff(path), crop_size).astype(np.float32) for path in paths]
    stack = np.stack(crops, axis=0)
    frame_means = np.mean(stack, axis=(1, 2))
    per_pixel_mean = np.mean(stack, axis=0)
    per_pixel_var = np.var(stack, axis=0)
    per_pixel_std = np.sqrt(np.maximum(per_pixel_var, 0.0))
    residual = stack - per_pixel_mean[None, :, :]

    mean_signal = float(np.mean(per_pixel_mean))
    temporal_var = float(np.mean(per_pixel_var))
    spatial_fixed_std = float(np.std(per_pixel_mean))
    temporal_std = float(np.mean(per_pixel_std))

    return {
        "folder": folder.name,
        "frame_count": len(paths),
        "crop_size": crop_size,
        "frame_mean_mean": float(np.mean(frame_means)),
        "frame_mean_std": float(np.std(frame_means)),
        "frame_mean_min": float(np.min(frame_means)),
        "frame_mean_max": float(np.max(frame_means)),
        "per_pixel_mean_mean": mean_signal,
        "per_pixel_mean_p50": float(np.percentile(per_pixel_mean, 50)),
        "per_pixel_mean_p99": float(np.percentile(per_pixel_mean, 99)),
        "spatial_fixed_std": spatial_fixed_std,
        "temporal_std_mean": temporal_std,
        "temporal_std_p50": float(np.percentile(per_pixel_std, 50)),
        "temporal_std_p99": float(np.percentile(per_pixel_std, 99)),
        "temporal_var_mean": temporal_var,
        "temporal_fano_approx": safe_div(temporal_var, mean_signal),
        "temporal_cv": safe_div(temporal_std, mean_signal),
        "fixed_to_temporal_std_ratio": safe_div(spatial_fixed_std, temporal_std),
        "residual_p01": float(np.percentile(residual, 1)),
        "residual_p50": float(np.percentile(residual, 50)),
        "residual_p99": float(np.percentile(residual, 99)),
    }


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


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, rows: list[dict[str, Any]], csv_path: Path, report_path: Path) -> None:
    lines = [
        "# Single-Condition Noise Summary",
        "",
        f"- Root: `{root}`",
        f"- Folders summarized: {len(rows)}",
        f"- CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
        "| folder | frames | mean signal | frame mean std | spatial fixed std | temporal std mean | Fano approx | fixed/temporal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['folder']} | {row['frame_count']} | {row['per_pixel_mean_mean']:.6g} | "
            f"{row['frame_mean_std']:.6g} | {row['spatial_fixed_std']:.6g} | "
            f"{row['temporal_std_mean']:.6g} | {row['temporal_fano_approx']:.6g} | "
            f"{row['fixed_to_temporal_std_ratio']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are computed on center crops, not full frames.",
            "- `spatial_fixed_std` is the spatial standard deviation of the per-pixel temporal mean.",
            "- `temporal_std_mean` is the mean per-pixel temporal standard deviation.",
            "- `Fano approx` is temporal variance divided by mean signal in raw digital counts.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
