"""Evaluate an empirical fixed-pattern correction baseline for ICCD repeats."""

from __future__ import annotations

import argparse
import csv
import math
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
    maps_dir = output_dir / "fixed_maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for folder in select_folders(root, args.folders):
        paths = list_tiffs(folder)
        train_paths, test_paths = split_paths(paths, args.train_frames, args.test_frames)
        if len(train_paths) < 2 or len(test_paths) < 2:
            print(f"Skipping {folder.name}: need at least 2 train and 2 test frames, found {len(train_paths)}/{len(test_paths)}")
            continue

        print(f"Processing {folder.name}: {len(train_paths)} train frames, {len(test_paths)} test frames")
        train_stack = read_stack(train_paths, args.crop_size)
        test_stack = read_stack(test_paths, args.crop_size)
        fixed_map, row = evaluate_folder(folder.name, train_stack, test_stack)
        rows.append(row)

        if args.save_maps:
            np.save(maps_dir / f"folder_{folder.name}_fixed_map.npy", fixed_map.astype(np.float32))

    if not rows:
        raise ValueError("No valid fixed-pattern rows were produced.")

    csv_path = output_dir / "fixed_pattern_correction_summary.csv"
    report_path = output_dir / "fixed_pattern_correction_report.md"
    write_csv(rows, csv_path)
    write_report(root, rows, csv_path, report_path, args)
    maybe_write_plot(rows, output_dir / "fixed_pattern_correction_reduction.png")
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/fixed_pattern_correction")
    parser.add_argument("--folders", nargs="*", default=[], help="Folder names to process. Empty means all immediate subfolders.")
    parser.add_argument("--train-frames", type=int, default=100, help="Frames used to estimate the fixed pattern.")
    parser.add_argument("--test-frames", type=int, default=100, help="Held-out frames used for evaluation.")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--save-maps", action="store_true", help="Save per-folder zero-mean fixed pattern maps as NPY files.")
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
    if not root.exists():
        return []
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_file_key)


def natural_file_key(path: Path) -> tuple[int, str]:
    match = LEADING_NUMBER.match(path.name)
    if match:
        return int(match.group("number")), path.name
    return 10**12, path.name


def split_paths(paths: list[Path], train_frames: int, test_frames: int) -> tuple[list[Path], list[Path]]:
    if train_frames <= 0 or test_frames <= 0:
        midpoint = len(paths) // 2
        return paths[:midpoint], paths[midpoint:]

    train_paths = paths[:train_frames]
    test_paths = paths[train_frames : train_frames + test_frames]
    if len(test_paths) >= 2:
        return train_paths, test_paths

    midpoint = len(paths) // 2
    return paths[:midpoint], paths[midpoint:]


def read_stack(paths: list[Path], crop_size: int) -> np.ndarray:
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


def evaluate_folder(folder: str, train_stack: np.ndarray, test_stack: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    fixed_map = estimate_additive_fixed_map(train_stack)
    corrected = test_stack - fixed_map[None, :, :]

    before_mean_map = np.mean(test_stack, axis=0)
    after_mean_map = np.mean(corrected, axis=0)
    before_temporal_std = np.std(test_stack, axis=0, ddof=1)
    after_temporal_std = np.std(corrected, axis=0, ddof=1)
    before_frame_means = np.mean(test_stack, axis=(1, 2))
    after_frame_means = np.mean(corrected, axis=(1, 2))
    correction_delta = corrected - test_stack

    spatial_before = sample_std(before_mean_map)
    spatial_after = sample_std(after_mean_map)
    temporal_before = float(np.mean(before_temporal_std))
    temporal_after = float(np.mean(after_temporal_std))

    return fixed_map, {
        "folder": folder,
        "train_frame_count": int(train_stack.shape[0]),
        "test_frame_count": int(test_stack.shape[0]),
        "crop_size": int(test_stack.shape[1]),
        "mean_signal_before": float(np.mean(test_stack)),
        "mean_signal_after": float(np.mean(corrected)),
        "fixed_map_std": sample_std(fixed_map),
        "fixed_map_p01": float(np.percentile(fixed_map, 1)),
        "fixed_map_p50": float(np.percentile(fixed_map, 50)),
        "fixed_map_p99": float(np.percentile(fixed_map, 99)),
        "spatial_mean_std_before": spatial_before,
        "spatial_mean_std_after": spatial_after,
        "spatial_reduction_fraction": safe_div(spatial_before - spatial_after, spatial_before),
        "temporal_std_mean_before": temporal_before,
        "temporal_std_mean_after": temporal_after,
        "temporal_std_change_fraction": safe_div(temporal_after - temporal_before, temporal_before),
        "frame_mean_std_before": sample_std(before_frame_means),
        "frame_mean_std_after": sample_std(after_frame_means),
        "residual_bias_mean": float(np.mean(correction_delta)),
        "residual_bias_abs_mean": float(abs(np.mean(correction_delta))),
        "passes_spatial_threshold": bool((spatial_before - spatial_after) / spatial_before >= 0.5) if spatial_before > 0 else False,
        "passes_temporal_guardrail": bool((temporal_after - temporal_before) / temporal_before <= 0.1) if temporal_before > 0 else False,
    }


def estimate_additive_fixed_map(stack: np.ndarray) -> np.ndarray:
    mean_map = np.mean(stack, axis=0)
    return (mean_map - float(np.mean(mean_map))).astype(np.float32)


def sample_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, rows: list[dict[str, Any]], csv_path: Path, report_path: Path, args: argparse.Namespace) -> None:
    reduction_values = [float(row["spatial_reduction_fraction"]) for row in rows if is_finite(row["spatial_reduction_fraction"])]
    temporal_changes = [float(row["temporal_std_change_fraction"]) for row in rows if is_finite(row["temporal_std_change_fraction"])]
    lines = [
        "# ICCD Fixed-Pattern Correction Report",
        "",
        f"- Root: `{root}`",
        f"- Folders summarized: {len(rows)}",
        f"- Train frames per folder: {args.train_frames}",
        f"- Test frames per folder: {args.test_frames}",
        f"- Crop size: {args.crop_size}",
        f"- CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
        "| folder | train | test | mean signal | fixed map std | spatial std before | spatial std after | reduction | temporal std before | temporal std after | temporal change | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        passed = "yes" if row["passes_spatial_threshold"] and row["passes_temporal_guardrail"] else "no"
        lines.append(
            "| "
            f"{row['folder']} | {row['train_frame_count']} | {row['test_frame_count']} | "
            f"{row['mean_signal_before']:.6g} | {row['fixed_map_std']:.6g} | "
            f"{row['spatial_mean_std_before']:.6g} | {row['spatial_mean_std_after']:.6g} | "
            f"{format_percent(row['spatial_reduction_fraction'])} | "
            f"{row['temporal_std_mean_before']:.6g} | {row['temporal_std_mean_after']:.6g} | "
            f"{format_percent(row['temporal_std_change_fraction'])} | {passed} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Median spatial fixed-pattern reduction: {format_percent(np.median(reduction_values)) if reduction_values else 'nan'}",
            f"- Maximum spatial fixed-pattern reduction: {format_percent(np.max(reduction_values)) if reduction_values else 'nan'}",
            f"- Median temporal standard-deviation change: {format_percent(np.median(temporal_changes)) if temporal_changes else 'nan'}",
            "",
            "## Notes",
            "",
            "- This is an empirical repeated-frame baseline, not a true dark/flat calibration.",
            "- The fixed map is estimated from calibration frames only and evaluated on held-out frames from the same folder.",
            "- The correction is additive and zero-mean, so it preserves each frame's global brightness in expectation.",
            "- This result can support a bounded fixed-pattern claim, but true flat-field claims still require identified flat-field frames.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def format_percent(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number * 100:.3f}%"


def maybe_write_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [str(row["folder"]) for row in rows]
    before = [float(row["spatial_mean_std_before"]) for row in rows]
    after = [float(row["spatial_mean_std_after"]) for row in rows]

    x = np.arange(len(rows))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, before, width=width, label="before")
    plt.bar(x + width / 2, after, width=width, label="after")
    plt.xlabel("Folder")
    plt.ylabel("Spatial std of temporal mean (DN)")
    plt.title("Empirical fixed-pattern correction baseline")
    plt.xticks(x, labels)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
