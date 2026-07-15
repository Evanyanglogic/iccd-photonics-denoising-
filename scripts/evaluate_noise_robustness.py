"""Evaluate crop-size and frame-count robustness for ICCD repeated frames."""

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

    crop_sizes = sorted(set(args.crop_sizes))
    frame_counts = sorted(set(args.frame_counts))
    max_crop_size = max(crop_sizes)
    max_frame_count = max(frame_counts)

    rows: list[dict[str, Any]] = []
    for folder in select_folders(root, args.folders):
        paths = list_tiffs(folder)[:max_frame_count]
        if len(paths) < min(frame_counts):
            print(f"Skipping {folder.name}: need at least {min(frame_counts)} frames, found {len(paths)}")
            continue
        print(f"Processing {folder.name}: {len(paths)} frames, max crop {max_crop_size}")
        stack = read_stack(paths, max_crop_size)
        for crop_size in crop_sizes:
            cropped = center_crop_stack(stack, crop_size)
            for frame_count in frame_counts:
                if cropped.shape[0] < frame_count:
                    continue
                rows.append(summarize_stack(folder.name, cropped[:frame_count], crop_size, frame_count))

    if not rows:
        raise ValueError("No robustness rows were produced.")

    csv_path = output_dir / "noise_robustness_summary.csv"
    report_path = output_dir / "noise_robustness_report.md"
    write_csv(rows, csv_path)
    write_report(root, rows, crop_sizes, frame_counts, csv_path, report_path)
    maybe_write_plot(rows, output_dir / "noise_robustness_fano.png")
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/noise_robustness")
    parser.add_argument("--folders", nargs="*", default=[], help="Folder names to process. Empty means all immediate subfolders.")
    parser.add_argument("--crop-sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[16, 32, 64, 128])
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


def center_crop_stack(stack: np.ndarray, crop_size: int) -> np.ndarray:
    _, h, w = stack.shape
    size = min(crop_size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return stack[:, top : top + size, left : left + size]


def summarize_stack(folder: str, stack: np.ndarray, crop_size: int, frame_count: int) -> dict[str, Any]:
    frame_means = np.mean(stack, axis=(1, 2))
    per_pixel_mean = np.mean(stack, axis=0)
    per_pixel_var = np.var(stack, axis=0, ddof=1)
    per_pixel_std = np.sqrt(np.maximum(per_pixel_var, 0.0))
    mean_signal = float(np.mean(per_pixel_mean))
    temporal_var = float(np.mean(per_pixel_var))
    spatial_fixed_std = sample_std(per_pixel_mean)
    temporal_std = float(np.mean(per_pixel_std))

    return {
        "folder": folder,
        "crop_size": crop_size,
        "frame_count": frame_count,
        "mean_signal": mean_signal,
        "frame_mean_std": sample_std(frame_means),
        "spatial_fixed_std": spatial_fixed_std,
        "temporal_std_mean": temporal_std,
        "temporal_var_mean": temporal_var,
        "temporal_fano_approx": safe_div(temporal_var, mean_signal),
        "fixed_to_temporal_std_ratio": safe_div(spatial_fixed_std, temporal_std),
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


def write_report(
    root: Path,
    rows: list[dict[str, Any]],
    crop_sizes: list[int],
    frame_counts: list[int],
    csv_path: Path,
    report_path: Path,
) -> None:
    baseline_rows = [row for row in rows if row["crop_size"] == 512 and row["frame_count"] == 32]
    stable_rows = [row for row in rows if row["crop_size"] == max(crop_sizes) and row["frame_count"] == max(frame_counts)]
    deltas = compare_rows(baseline_rows, stable_rows)

    lines = [
        "# ICCD Noise Robustness Report",
        "",
        f"- Root: `{root}`",
        f"- Folders summarized: {len({row['folder'] for row in rows})}",
        f"- Crop sizes: {', '.join(str(size) for size in crop_sizes)}",
        f"- Frame counts: {', '.join(str(count) for count in frame_counts)}",
        f"- CSV: `{csv_path}`",
        "",
        "## Baseline 512 Crop / 32 Frames",
        "",
        "| folder | mean signal | temporal std | Fano approx | spatial fixed std | fixed/temporal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(baseline_rows, key=lambda item: natural_folder_key(str(item["folder"]))):
        lines.append(format_summary_row(row))

    lines.extend(
        [
            "",
            f"## Largest Setting {max(crop_sizes)} Crop / {max(frame_counts)} Frames",
            "",
            "| folder | mean signal | temporal std | Fano approx | spatial fixed std | fixed/temporal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(stable_rows, key=lambda item: natural_folder_key(str(item["folder"]))):
        lines.append(format_summary_row(row))

    lines.extend(
        [
            "",
            "## Relative Change From Baseline To Largest Setting",
            "",
            "| metric | median abs relative change | max abs relative change |",
            "|---|---:|---:|",
        ]
    )
    for metric, values in deltas.items():
        lines.append(f"| {metric} | {format_percent(np.median(values))} | {format_percent(np.max(values))} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The baseline comparison is against the existing 512x512 center-crop, 32-frame summary.",
            "- The largest setting uses the same center region expanded to the configured maximum crop size and first maximum frame count.",
            "- Large changes here should be treated as spatial nonuniformity or insufficient frame averaging, not as model behavior.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_summary_row(row: dict[str, Any]) -> str:
    return (
        "| "
        f"{row['folder']} | {row['mean_signal']:.6g} | {row['temporal_std_mean']:.6g} | "
        f"{row['temporal_fano_approx']:.6g} | {row['spatial_fixed_std']:.6g} | "
        f"{row['fixed_to_temporal_std_ratio']:.6g} |"
    )


def compare_rows(baseline_rows: list[dict[str, Any]], stable_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    stable_by_folder = {str(row["folder"]): row for row in stable_rows}
    metrics = ["mean_signal", "temporal_std_mean", "temporal_fano_approx", "spatial_fixed_std", "fixed_to_temporal_std_ratio"]
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    for baseline in baseline_rows:
        stable = stable_by_folder.get(str(baseline["folder"]))
        if stable is None:
            continue
        for metric in metrics:
            deltas[metric].append(abs_relative_change(float(baseline[metric]), float(stable[metric])))
    return {metric: values for metric, values in deltas.items() if values}


def abs_relative_change(old: float, new: float) -> float:
    if not math.isfinite(old) or not math.isfinite(new) or abs(old) < 1e-12:
        return float("nan")
    return abs(new - old) / abs(old)


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

    plt.figure(figsize=(9, 5))
    for crop_size in sorted({row["crop_size"] for row in rows}):
        crop_rows = [row for row in rows if row["crop_size"] == crop_size]
        grouped: dict[int, list[float]] = {}
        for row in crop_rows:
            grouped.setdefault(int(row["frame_count"]), []).append(float(row["temporal_fano_approx"]))
        xs = sorted(grouped)
        ys = [float(np.median(grouped[x])) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.2, label=f"crop {crop_size}")
    plt.xlabel("Frame count")
    plt.ylabel("Median temporal Fano approximation")
    plt.title("ICCD noise robustness across crops and frame counts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
