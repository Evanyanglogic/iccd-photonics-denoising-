"""Fit mean-variance statistics from repeated single-condition ICCD frames."""

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

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for folder in select_folders(root, args.folders):
        paths = list_tiffs(folder)
        if args.max_frames > 0:
            paths = paths[: args.max_frames]
        if len(paths) < 3:
            print(f"Skipping {folder.name}: need at least 3 frames, found {len(paths)}")
            continue
        print(f"Processing {folder.name}: {len(paths)} frames")
        stack = read_stack(paths, args.crop_size)
        rows = mean_variance_rows(folder.name, stack, bins=args.bins, min_count=args.min_count)
        all_rows.extend(rows)
        summary_rows.append(summarize_folder(folder.name, stack, rows, min_linear_bins=args.min_linear_bins))

    if not all_rows:
        raise ValueError("No valid mean-variance rows were produced.")

    bins_csv = output_dir / "mean_variance_bins.csv"
    summary_csv = output_dir / "mean_variance_summary.csv"
    report_path = output_dir / "mean_variance_report.md"
    write_csv(all_rows, bins_csv)
    write_csv(summary_rows, summary_csv)
    write_report(root, all_rows, summary_rows, bins_csv, summary_csv, report_path)
    maybe_write_plot(all_rows, output_dir / "mean_variance_curve.png")
    print(f"Wrote bin CSV: {bins_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/mean_variance")
    parser.add_argument("--folders", nargs="*", default=[], help="Folder names to process. Empty means all immediate subfolders.")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--min-count", type=int, default=512)
    parser.add_argument("--min-linear-bins", type=int, default=6)
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


def mean_variance_rows(folder: str, stack: np.ndarray, bins: int, min_count: int) -> list[dict[str, Any]]:
    temporal_mean = np.mean(stack, axis=0)
    temporal_var = np.var(stack, axis=0, ddof=1)
    total_pixel_var = np.var(stack.reshape(-1), ddof=1)
    frame_diff_var = frame_difference_temporal_variance(stack)

    lo = float(np.percentile(temporal_mean, 0.5))
    hi = float(np.percentile(temporal_mean, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(temporal_mean))
        hi = float(np.max(temporal_mean))
    edges = np.linspace(lo, hi, bins + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(bins):
        bin_low = edges[idx]
        bin_high = edges[idx + 1]
        if idx == bins - 1:
            mask = (temporal_mean >= bin_low) & (temporal_mean <= bin_high)
        else:
            mask = (temporal_mean >= bin_low) & (temporal_mean < bin_high)
        count = int(np.count_nonzero(mask))
        if count < min_count:
            continue
        bin_mean_values = temporal_mean[mask]
        bin_temporal_var = temporal_var[mask]
        rows.append(
            {
                "folder": folder,
                "bin_index": idx,
                "bin_low": float(bin_low),
                "bin_high": float(bin_high),
                "pixel_count": count,
                "mean_signal": float(np.mean(bin_mean_values)),
                "temporal_var_mean": float(np.mean(bin_temporal_var)),
                "temporal_var_median": float(np.median(bin_temporal_var)),
                "temporal_std_mean": float(np.mean(np.sqrt(np.maximum(bin_temporal_var, 0.0)))),
                "spatial_mean_std": float(np.std(bin_mean_values, ddof=1)) if count > 1 else 0.0,
                "fano_temporal": safe_div(float(np.mean(bin_temporal_var)), float(np.mean(bin_mean_values))),
                "frame_diff_var_mean": float(np.mean(frame_diff_var[mask])),
                "global_total_pixel_var": float(total_pixel_var),
            }
        )
    return rows


def frame_difference_temporal_variance(stack: np.ndarray) -> np.ndarray:
    diffs = np.diff(stack, axis=0)
    if diffs.shape[0] == 0:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    return (np.var(diffs, axis=0, ddof=1) / 2.0).astype(np.float32) if diffs.shape[0] > 1 else ((diffs[0] ** 2) / 2.0).astype(np.float32)


def summarize_folder(folder: str, stack: np.ndarray, rows: list[dict[str, Any]], min_linear_bins: int) -> dict[str, Any]:
    temporal_mean = np.mean(stack, axis=0)
    temporal_var = np.var(stack, axis=0, ddof=1)
    frame_means = np.mean(stack, axis=(1, 2))
    fitted = fit_linear_regime(rows, min_linear_bins=min_linear_bins)
    return {
        "folder": folder,
        "frame_count": int(stack.shape[0]),
        "crop_size": int(stack.shape[1]),
        "frame_mean_mean": float(np.mean(frame_means)),
        "frame_mean_std": float(np.std(frame_means, ddof=1)) if stack.shape[0] > 1 else 0.0,
        "mean_signal": float(np.mean(temporal_mean)),
        "temporal_var_mean": float(np.mean(temporal_var)),
        "temporal_std_mean": float(np.mean(np.sqrt(np.maximum(temporal_var, 0.0)))),
        "spatial_mean_std": float(np.std(temporal_mean, ddof=1)),
        "fano_temporal": safe_div(float(np.mean(temporal_var)), float(np.mean(temporal_mean))),
        "valid_bins": len(rows),
        "linear_bin_count": fitted["linear_bin_count"],
        "linear_slope_var_per_dn": fitted["slope"],
        "linear_intercept_var": fitted["intercept"],
        "linear_r2": fitted["r2"],
        "effective_gain_dn_per_variance": safe_div(1.0, fitted["slope"]) if fitted["slope"] > 0 else float("nan"),
    }


def fit_linear_regime(rows: list[dict[str, Any]], min_linear_bins: int) -> dict[str, float]:
    valid = [row for row in rows if is_finite(row["mean_signal"]) and is_finite(row["temporal_var_mean"])]
    min_bins = max(3, min_linear_bins)
    if len(valid) < min_bins:
        return {"linear_bin_count": float(len(valid)), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    xs = np.asarray([row["mean_signal"] for row in valid], dtype=np.float64)
    ys = np.asarray([row["temporal_var_mean"] for row in valid], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    best = {"linear_bin_count": 0.0, "slope": float("nan"), "intercept": float("nan"), "r2": -float("inf")}
    for end in range(min_bins, len(xs) + 1):
        x = xs[:end]
        y = ys[:end]
        slope, intercept = np.polyfit(x, y, deg=1)
        predicted = slope * x + intercept
        r2 = coefficient_of_determination(y, predicted)
        if slope > 0 and r2 > best["r2"]:
            best = {"linear_bin_count": float(end), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    if best["r2"] == -float("inf"):
        return {"linear_bin_count": 0.0, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    return best


def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], bins_csv: Path, summary_csv: Path, report_path: Path) -> None:
    lines = [
        "# ICCD Mean-Variance Report",
        "",
        f"- Root: `{root}`",
        f"- Folders summarized: {len(summary_rows)}",
        f"- Bin CSV: `{bins_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## Summary",
        "",
        "| folder | frames | mean signal | temporal var | temporal Fano | spatial mean std | linear bins | slope | intercept | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['folder']} | {row['frame_count']} | {row['mean_signal']:.6g} | "
            f"{row['temporal_var_mean']:.6g} | {row['fano_temporal']:.6g} | "
            f"{row['spatial_mean_std']:.6g} | {row['linear_bin_count']:.0f} | "
            f"{format_float(row['linear_slope_var_per_dn'])} | {format_float(row['linear_intercept_var'])} | "
            f"{format_float(row['linear_r2'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `temporal_var_mean` is computed per pixel across repeated frames and then averaged.",
            "- `spatial_mean_std` is computed from the per-pixel temporal mean map, so it measures fixed spatial nonuniformity rather than temporal noise.",
            "- Linear fits are exploratory and use the best low-signal prefix of brightness bins by R2, requiring the configured minimum number of bins.",
            "- Inspect the bin CSV and curve plot before making paper claims.",
            "- Do not interpret the slope as a raw-DN Poisson unit slope. It is an effective raw-domain variance-vs-mean slope for this processing path.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def maybe_write_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    folders = sorted({row["folder"] for row in rows}, key=natural_folder_key)
    plt.figure(figsize=(8, 5))
    for folder in folders:
        folder_rows = [row for row in rows if row["folder"] == folder]
        xs = [row["mean_signal"] for row in folder_rows]
        ys = [row["temporal_var_mean"] for row in folder_rows]
        plt.plot(xs, ys, marker="o", linewidth=1.0, markersize=3, label=str(folder))
    plt.xlabel("Mean signal (DN)")
    plt.ylabel("Temporal variance (DN^2)")
    plt.title("ICCD mean-variance by brightness bin")
    plt.grid(True, alpha=0.3)
    if len(folders) <= 12:
        plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
