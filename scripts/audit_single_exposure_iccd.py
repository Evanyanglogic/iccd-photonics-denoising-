"""Audit a single-exposure ICCD TIFF sequence.

Use this when only one exposure/gate setting has been downloaded. The script
does not require clean/noisy pairs; it checks TIFF range, acquisition metadata,
frame-to-frame stability, and rough temporal noise on a center crop.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
LEADING_NUMBER = re.compile(r"^(?P<number>\d+)")
PICTURE_INFO_PATTERN = re.compile(
    r"^(?P<filename>.+?\.tiff?)\s+"
    r"Exposure.*?delay:(?P<exposure_delay_ms>[-+0-9.]+)ms,\s*width:(?P<exposure_width_ms>[-+0-9.]+)ms\s+"
    r"Sync\.A.*?delay:(?P<sync_a_delay_ns>[-+0-9.]+)ns,\s*width:(?P<sync_a_width_us>[-+0-9.]+)us\s+"
    r"Sync\.B.*?delay:(?P<sync_b_delay_ns>[-+0-9.]+)ns,\s*width:(?P<sync_b_width_us>[-+0-9.]+)us\s+"
    r"gain.*?(?P<gain>[-+0-9.]+)"
)


@dataclass(frozen=True)
class FrameStats:
    filename: str
    shape: str
    dtype: str
    minimum: float
    maximum: float
    p001: float
    p01: float
    p50: float
    p99: float
    p999: float
    mean: float
    std: float
    zero_fraction: float
    saturated_fraction: float


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list_tiffs(input_dir)
    if not tiff_files:
        raise ValueError(f"No TIFF files found under {input_dir}")
    sampled_files = tiff_files[: args.max_files] if args.max_files > 0 else tiff_files

    metadata = read_picture_info(input_dir / args.picture_info)
    frame_stats = [summarize_frame(path, saturated_value=args.saturated_value) for path in sampled_files]
    temporal = temporal_crop_stats(sampled_files[: args.max_temporal_frames], crop_size=args.crop_size)

    frame_csv = output_dir / "single_exposure_frame_stats.csv"
    report_path = output_dir / "single_exposure_audit.md"
    write_frame_csv(frame_stats, frame_csv)
    write_report(
        input_dir=input_dir,
        all_files=tiff_files,
        sampled_files=sampled_files,
        metadata=metadata,
        frame_stats=frame_stats,
        temporal=temporal,
        frame_csv=frame_csv,
        report_path=report_path,
    )
    print(f"Wrote frame stats: {frame_csv}")
    print(f"Wrote audit report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="reports/single_exposure_iccd")
    parser.add_argument("--picture-info", default="PictureInfo.txt")
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--max-temporal-frames", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--saturated-value", type=float, default=65535.0)
    return parser.parse_args()


def list_tiffs(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_key)


def natural_key(path: Path) -> tuple[int, str]:
    match = LEADING_NUMBER.match(path.name)
    if match:
        return int(match.group("number")), path.name
    return 10**12, path.name


def read_picture_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = PICTURE_INFO_PATTERN.match(line)
            if not match:
                continue
            data = match.groupdict()
            filename = data.pop("filename")
            rows[filename] = data
    return rows


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def summarize_frame(path: Path, saturated_value: float) -> FrameStats:
    arr = read_tiff(path)
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return FrameStats(
        filename=path.name,
        shape="x".join(str(item) for item in arr.shape),
        dtype=str(arr.dtype),
        minimum=float(np.min(flat)),
        maximum=float(np.max(flat)),
        p001=float(np.percentile(flat, 0.1)),
        p01=float(np.percentile(flat, 1)),
        p50=float(np.percentile(flat, 50)),
        p99=float(np.percentile(flat, 99)),
        p999=float(np.percentile(flat, 99.9)),
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        zero_fraction=float(np.mean(flat <= 0)),
        saturated_fraction=float(np.mean(flat >= saturated_value)),
    )


def temporal_crop_stats(paths: list[Path], crop_size: int) -> dict[str, float]:
    if not paths:
        return {}
    crops = []
    for path in paths:
        arr = np.asarray(read_tiff(path), dtype=np.float32)
        crop = center_crop(arr, crop_size)
        crops.append(crop)
    stack = np.stack(crops, axis=0).astype(np.float32)
    frame_means = np.mean(stack, axis=tuple(range(1, stack.ndim)))
    per_pixel_mean = np.mean(stack, axis=0)
    per_pixel_var = np.var(stack, axis=0)
    per_pixel_std = np.sqrt(np.maximum(per_pixel_var, 0.0))
    mean_signal = float(np.mean(per_pixel_mean))
    mean_var = float(np.mean(per_pixel_var))
    return {
        "temporal_frame_count": float(stack.shape[0]),
        "crop_height": float(stack.shape[-2] if stack.ndim == 3 else stack.shape[1]),
        "crop_width": float(stack.shape[-1] if stack.ndim == 3 else stack.shape[2]),
        "frame_mean_mean": float(np.mean(frame_means)),
        "frame_mean_std": float(np.std(frame_means)),
        "per_pixel_mean_mean": mean_signal,
        "per_pixel_mean_std_spatial": float(np.std(per_pixel_mean)),
        "per_pixel_temporal_std_mean": float(np.mean(per_pixel_std)),
        "per_pixel_temporal_std_p50": float(np.percentile(per_pixel_std, 50)),
        "per_pixel_temporal_std_p99": float(np.percentile(per_pixel_std, 99)),
        "temporal_var_mean": mean_var,
        "temporal_fano_approx": safe_div(mean_var, mean_signal),
    }


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


def write_frame_csv(rows: list[FrameStats], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FrameStats.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_report(
    input_dir: Path,
    all_files: list[Path],
    sampled_files: list[Path],
    metadata: dict[str, dict[str, str]],
    frame_stats: list[FrameStats],
    temporal: dict[str, float],
    frame_csv: Path,
    report_path: Path,
) -> None:
    first_meta = next(iter(metadata.values()), {})
    lines = [
        "# Single-Exposure ICCD Audit",
        "",
        "## Summary",
        "",
        f"- Input dir: `{input_dir}`",
        f"- TIFF count: {len(all_files)}",
        f"- Sampled for frame stats: {len(sampled_files)}",
        f"- PictureInfo rows parsed: {len(metadata)}",
        f"- Frame stats CSV: `{frame_csv}`",
        "",
        "## Acquisition Metadata",
        "",
    ]
    if first_meta:
        for key, value in first_meta.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No parseable PictureInfo metadata found.")

    lines.extend(["", "## Sampled Frame Range", ""])
    lines.append("| metric | mean | std | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for field in ["minimum", "maximum", "p001", "p01", "p50", "p99", "p999", "mean", "std", "saturated_fraction"]:
        values = np.asarray([float(getattr(row, field)) for row in frame_stats], dtype=np.float64)
        lines.append(
            f"| {field} | {np.mean(values):.6g} | {np.std(values):.6g} | {np.min(values):.6g} | {np.max(values):.6g} |"
        )

    lines.extend(["", "## Temporal Crop Statistics", ""])
    if temporal:
        for key, value in temporal.items():
            lines.append(f"- {key}: {format_float(value)}")
    else:
        lines.append("- Not computed.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is a single-exposure sequence, so it cannot be used for supervised clean/noisy denoising by itself.",
            "- Repeated frames can support temporal noise, fixed-pattern, dark/offset, and stability analysis.",
            "- To build paired denoising data, download at least one matching longer/shorter exposure sequence with the same frame indices and scene.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
