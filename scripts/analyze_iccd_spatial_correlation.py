"""Analyze spatial correlation and PSD of ICCD repeated-frame residuals."""

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

    summary_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    autocorr_rows: list[dict[str, Any]] = []

    for folder in select_folders(root, args.folders):
        paths = list_tiffs(folder)
        if args.max_frames > 0:
            paths = paths[: args.max_frames]
        if len(paths) < 3:
            print(f"Skipping {folder.name}: need at least 3 frames, found {len(paths)}")
            continue

        print(f"Processing {folder.name}: {len(paths)} frames")
        stack = read_stack(paths, args.crop_size)
        residual = residual_after_fixed_pattern(stack)
        summary, radial_psd, radial_autocorr = analyze_folder(folder.name, residual, args.max_radius)
        summary_rows.append(summary)
        psd_rows.extend(radial_psd)
        autocorr_rows.extend(radial_autocorr)

    if not summary_rows:
        raise ValueError("No spatial-correlation rows were produced.")

    summary_csv = output_dir / "spatial_correlation_summary.csv"
    psd_csv = output_dir / "radial_psd.csv"
    autocorr_csv = output_dir / "radial_autocorrelation.csv"
    report_path = output_dir / "spatial_correlation_report.md"
    write_csv(summary_rows, summary_csv)
    write_csv(psd_rows, psd_csv)
    write_csv(autocorr_rows, autocorr_csv)
    write_report(root, summary_rows, summary_csv, psd_csv, autocorr_csv, report_path, args)
    maybe_write_plot(psd_rows, output_dir / "radial_psd_normalized.png", y_key="psd_norm", ylabel="Normalized PSD")
    maybe_write_plot(autocorr_rows, output_dir / "radial_autocorrelation.png", y_key="autocorr_norm", ylabel="Normalized autocorrelation")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote PSD CSV: {psd_csv}")
    print(f"Wrote autocorrelation CSV: {autocorr_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/spatial_correlation")
    parser.add_argument("--folders", nargs="*", default=[], help="Folder names to process. Empty means all immediate subfolders.")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--max-radius", type=int, default=128)
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


def residual_after_fixed_pattern(stack: np.ndarray) -> np.ndarray:
    residual = stack - np.mean(stack, axis=0, keepdims=True)
    residual = residual - np.mean(residual, axis=(1, 2), keepdims=True)
    return residual.astype(np.float32)


def analyze_folder(folder: str, residual: np.ndarray, max_radius: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    residual_std = float(np.std(residual, ddof=1))
    row_corr = adjacent_corr(residual[:, :, :-1], residual[:, :, 1:])
    col_corr = adjacent_corr(residual[:, :-1, :], residual[:, 1:, :])
    diag_corr = adjacent_corr(residual[:, :-1, :-1], residual[:, 1:, 1:])

    power = average_power_spectrum(residual)
    radial_psd = radial_average(power, max_radius)
    autocorr = normalized_autocorrelation_from_power(power)
    radial_autocorr = radial_average(autocorr, max_radius)

    psd_total = float(np.sum(radial_psd["values"]))
    low_fraction = radial_fraction(radial_psd, 1, max(2, max_radius // 16), psd_total)
    mid_fraction = radial_fraction(radial_psd, max(2, max_radius // 16), max(3, max_radius // 4), psd_total)
    high_fraction = radial_fraction(radial_psd, max(3, max_radius // 4), max_radius, psd_total)
    corr_length_1e = first_radius_below(radial_autocorr["values"], threshold=1.0 / math.e, start=1)
    corr_length_01 = first_radius_below(radial_autocorr["values"], threshold=0.1, start=1)

    summary = {
        "folder": folder,
        "frame_count": int(residual.shape[0]),
        "crop_size": int(residual.shape[1]),
        "residual_std": residual_std,
        "lag1_row_corr": row_corr,
        "lag1_col_corr": col_corr,
        "lag1_diag_corr": diag_corr,
        "radial_autocorr_r1": value_at(radial_autocorr["values"], 1),
        "radial_autocorr_r2": value_at(radial_autocorr["values"], 2),
        "corr_length_1e_px": corr_length_1e,
        "corr_length_0p1_px": corr_length_01,
        "psd_low_fraction": low_fraction,
        "psd_mid_fraction": mid_fraction,
        "psd_high_fraction": high_fraction,
    }
    psd_rows = rows_from_radial(folder, radial_psd, "psd")
    autocorr_rows = rows_from_radial(folder, radial_autocorr, "autocorr")
    return summary, psd_rows, autocorr_rows


def adjacent_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def average_power_spectrum(residual: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(residual, axes=(1, 2))
    power = np.mean(np.abs(fft) ** 2, axis=0)
    return np.fft.fftshift(power).astype(np.float64)


def normalized_autocorrelation_from_power(shifted_power: np.ndarray) -> np.ndarray:
    power = np.fft.ifftshift(shifted_power)
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)
    center = autocorr[autocorr.shape[0] // 2, autocorr.shape[1] // 2]
    if abs(center) < 1e-12:
        return autocorr * float("nan")
    return (autocorr / center).astype(np.float64)


def radial_average(image: np.ndarray, max_radius: int) -> dict[str, np.ndarray]:
    h, w = image.shape
    y, x = np.indices((h, w))
    radius = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2).astype(np.int32)
    max_radius = min(max_radius, int(radius.max()))
    mask = radius <= max_radius
    radial_sum = np.bincount(radius[mask].ravel(), weights=image[mask].ravel(), minlength=max_radius + 1)
    radial_count = np.bincount(radius[mask].ravel(), minlength=max_radius + 1)
    values = radial_sum[: max_radius + 1] / np.maximum(radial_count[: max_radius + 1], 1)
    radii = np.arange(max_radius + 1, dtype=np.int32)
    total = float(np.sum(np.abs(values)))
    norm = values / total if total > 1e-12 else values * float("nan")
    return {"radius": radii, "values": values.astype(np.float64), "norm": norm.astype(np.float64)}


def radial_fraction(radial: dict[str, np.ndarray], start: int, end: int, total: float) -> float:
    if total <= 1e-12:
        return float("nan")
    values = radial["values"]
    start = max(0, start)
    end = min(len(values), end)
    if end <= start:
        return float("nan")
    return float(np.sum(values[start:end]) / total)


def first_radius_below(values: np.ndarray, threshold: float, start: int) -> float:
    for idx in range(start, len(values)):
        if values[idx] <= threshold:
            return float(idx)
    return float("nan")


def value_at(values: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])


def rows_from_radial(folder: str, radial: dict[str, np.ndarray], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for radius, value, norm in zip(radial["radius"], radial["values"], radial["norm"]):
        rows.append(
            {
                "folder": folder,
                "radius_px": int(radius),
                f"{prefix}_mean": float(value),
                f"{prefix}_norm": float(norm),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    root: Path,
    rows: list[dict[str, Any]],
    summary_csv: Path,
    psd_csv: Path,
    autocorr_csv: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    row_corrs = finite_values(rows, "lag1_row_corr")
    col_corrs = finite_values(rows, "lag1_col_corr")
    corr_lengths = finite_values(rows, "corr_length_0p1_px")
    lines = [
        "# ICCD Spatial Correlation Report",
        "",
        f"- Root: `{root}`",
        f"- Folders summarized: {len(rows)}",
        f"- Frames per folder: {args.max_frames}",
        f"- Crop size: {args.crop_size}",
        f"- Summary CSV: `{summary_csv}`",
        f"- PSD CSV: `{psd_csv}`",
        f"- Autocorrelation CSV: `{autocorr_csv}`",
        "",
        "## Summary",
        "",
        "| folder | residual std | row lag-1 | col lag-1 | diag lag-1 | autocorr r1 | corr <= 0.1 px | PSD low | PSD mid | PSD high |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['folder']} | {row['residual_std']:.6g} | {row['lag1_row_corr']:.6g} | "
            f"{row['lag1_col_corr']:.6g} | {row['lag1_diag_corr']:.6g} | "
            f"{row['radial_autocorr_r1']:.6g} | {format_float(row['corr_length_0p1_px'])} | "
            f"{format_percent(row['psd_low_fraction'])} | {format_percent(row['psd_mid_fraction'])} | "
            f"{format_percent(row['psd_high_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Median row lag-1 correlation: {format_float(np.median(row_corrs)) if row_corrs else 'nan'}",
            f"- Median column lag-1 correlation: {format_float(np.median(col_corrs)) if col_corrs else 'nan'}",
            f"- Median radius where autocorrelation falls below 0.1: {format_float(np.median(corr_lengths)) if corr_lengths else 'nan'} px",
            "",
            "## Notes",
            "",
            "- Residuals are computed after subtracting each folder's per-pixel temporal mean and each frame's residual mean.",
            "- Positive lag-1 correlation or slow autocorrelation decay indicates spatially correlated residual noise rather than white noise.",
            "- PSD fractions are computed from radially averaged power within the configured maximum radius and are intended for relative comparison.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def format_percent(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number * 100:.3f}%"


def maybe_write_plot(rows: list[dict[str, Any]], output_path: Path, y_key: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    folders = sorted({row["folder"] for row in rows}, key=natural_folder_key)
    plt.figure(figsize=(8, 5))
    for folder in folders:
        folder_rows = [row for row in rows if row["folder"] == folder]
        xs = [row["radius_px"] for row in folder_rows]
        ys = [row[y_key] for row in folder_rows]
        plt.plot(xs, ys, linewidth=1.0, label=str(folder))
    plt.xlabel("Radius (px)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(folders) <= 12:
        plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
