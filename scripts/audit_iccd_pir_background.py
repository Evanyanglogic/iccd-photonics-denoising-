"""Audit an 8-bit ICCD_pir background/dark candidate sequence."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
TRAILING_INDEX = re.compile(r"_(?P<index>\d+)\.tiff?$", re.IGNORECASE)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    indexed_paths = select_indexed_paths(root, args.start_index, args.end_index)
    if len(indexed_paths) < 2:
        raise ValueError(f"Need at least two TIFFs in index range, found {len(indexed_paths)}")

    print(f"Processing {len(indexed_paths)} frames from {indexed_paths[0][0]} to {indexed_paths[-1][0]}")
    stack, frame_rows = read_stack_and_frame_rows(indexed_paths, args.crop_size)
    summary = summarize_stack(root, stack, frame_rows, args.crop_size)

    frame_csv = output_dir / "background_frame_stats.csv"
    summary_csv = output_dir / "background_summary.csv"
    report_path = output_dir / "background_audit_report.md"
    write_csv(frame_rows, frame_csv)
    write_csv([summary], summary_csv)
    write_report(summary, frame_csv, summary_csv, report_path)
    maybe_write_histogram(frame_rows, output_dir / "frame_mean_histogram.png")

    if args.save_maps:
        np.save(maps_dir / "background_mean_map.npy", np.mean(stack, axis=0).astype(np.float32))
        np.save(maps_dir / "background_std_map.npy", np.std(stack, axis=0, ddof=1).astype(np.float32))

    print(f"Wrote frame CSV: {frame_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/iccd_pir_background")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--save-maps", action="store_true")
    return parser.parse_args()


def select_indexed_paths(root: Path, start_index: int, end_index: int) -> list[tuple[int, Path]]:
    pairs = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        match = TRAILING_INDEX.search(path.name)
        if match is None:
            continue
        index = int(match.group("index"))
        if start_index <= index <= end_index:
            pairs.append((index, path))
    return sorted(pairs, key=lambda pair: pair[0])


def read_stack_and_frame_rows(indexed_paths: list[tuple[int, Path]], crop_size: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    crops = []
    rows = []
    for index, path in indexed_paths:
        arr = read_tiff(path)
        full_stats = summarize_frame(arr, index, path.name, region="full")
        crop = center_crop(arr, crop_size).astype(np.float32)
        crop_stats = summarize_frame(crop, index, path.name, region="crop")
        rows.append({**full_stats, **{f"crop_{key}": value for key, value in crop_stats.items() if key not in {"index", "filename", "region"}}})
        crops.append(crop)
    return np.stack(crops, axis=0).astype(np.float32), rows


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


def summarize_frame(arr: np.ndarray, index: int, filename: str, region: str) -> dict[str, Any]:
    arr_float = np.asarray(arr, dtype=np.float32)
    max_value = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else float(np.max(arr_float))
    return {
        "index": index,
        "filename": filename,
        "region": region,
        "shape": "x".join(str(part) for part in arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr_float)),
        "p01": float(np.percentile(arr_float, 1)),
        "p50": float(np.percentile(arr_float, 50)),
        "p99": float(np.percentile(arr_float, 99)),
        "max": float(np.max(arr_float)),
        "mean": float(np.mean(arr_float)),
        "std": float(np.std(arr_float, ddof=1)),
        "saturated_fraction": float(np.mean(arr_float >= max_value)),
        "zero_fraction": float(np.mean(arr_float <= 0.0)),
    }


def summarize_stack(root: Path, stack: np.ndarray, frame_rows: list[dict[str, Any]], crop_size: int) -> dict[str, Any]:
    temporal_mean = np.mean(stack, axis=0)
    temporal_std = np.std(stack, axis=0, ddof=1)
    frame_means = np.asarray([row["crop_mean"] for row in frame_rows], dtype=np.float64)
    frame_saturation = np.asarray([row["crop_saturated_fraction"] for row in frame_rows], dtype=np.float64)
    mean_signal = float(np.mean(temporal_mean))
    temporal_std_mean = float(np.mean(temporal_std))
    spatial_mean_std = sample_std(temporal_mean)
    hot_pixel_threshold = float(np.percentile(temporal_mean, 99.9))
    unstable_threshold = float(np.percentile(temporal_std, 99.9))
    return {
        "root": str(root),
        "start_index": int(frame_rows[0]["index"]),
        "end_index": int(frame_rows[-1]["index"]),
        "frame_count": len(frame_rows),
        "crop_size": crop_size,
        "shape": frame_rows[0]["shape"],
        "dtype": frame_rows[0]["dtype"],
        "mean_signal": mean_signal,
        "frame_mean_mean": float(np.mean(frame_means)),
        "frame_mean_std": sample_std(frame_means),
        "frame_mean_min": float(np.min(frame_means)),
        "frame_mean_max": float(np.max(frame_means)),
        "temporal_std_mean": temporal_std_mean,
        "temporal_std_p50": float(np.percentile(temporal_std, 50)),
        "temporal_std_p99": float(np.percentile(temporal_std, 99)),
        "spatial_mean_std": spatial_mean_std,
        "fixed_to_temporal_std_ratio": safe_div(spatial_mean_std, temporal_std_mean),
        "saturated_fraction_mean": float(np.mean(frame_saturation)),
        "saturated_fraction_max": float(np.max(frame_saturation)),
        "zero_fraction_mean": float(np.mean([row["crop_zero_fraction"] for row in frame_rows])),
        "hot_pixel_threshold_p999_mean_map": hot_pixel_threshold,
        "hot_pixel_fraction_p999": float(np.mean(temporal_mean >= hot_pixel_threshold)),
        "unstable_pixel_threshold_p999_std_map": unstable_threshold,
        "unstable_pixel_fraction_p999": float(np.mean(temporal_std >= unstable_threshold)),
    }


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


def write_report(summary: dict[str, Any], frame_csv: Path, summary_csv: Path, report_path: Path) -> None:
    lines = [
        "# ICCD_pir Background Candidate Audit",
        "",
        f"- Root: `{summary['root']}`",
        f"- Index range: {summary['start_index']} to {summary['end_index']}",
        f"- Frames: {summary['frame_count']}",
        f"- Shape: {summary['shape']}",
        f"- dtype: {summary['dtype']}",
        f"- Crop size: {summary['crop_size']}",
        f"- Frame CSV: `{frame_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "mean_signal",
        "frame_mean_std",
        "frame_mean_min",
        "frame_mean_max",
        "temporal_std_mean",
        "temporal_std_p50",
        "temporal_std_p99",
        "spatial_mean_std",
        "fixed_to_temporal_std_ratio",
        "saturated_fraction_mean",
        "saturated_fraction_max",
        "zero_fraction_mean",
        "hot_pixel_threshold_p999_mean_map",
        "hot_pixel_fraction_p999",
        "unstable_pixel_threshold_p999_std_map",
        "unstable_pixel_fraction_p999",
    ]:
        lines.append(f"| {key} | {format_float(summary[key])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an auxiliary 8-bit ICCD_pir background candidate audit.",
            "- It should not be used as matching dark correction for the 5120x5120 uint16 gated ICCD batch unless metadata prove matching acquisition conditions.",
            "- Low saturation and stable frame means support use as background evidence; high fixed/temporal ratio would indicate structured background nonuniformity.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:.6g}"


def maybe_write_histogram(frame_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    values = [row["crop_mean"] for row in frame_rows]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=24)
    plt.xlabel("Frame mean DN")
    plt.ylabel("Frame count")
    plt.title("ICCD_pir background candidate frame means")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
