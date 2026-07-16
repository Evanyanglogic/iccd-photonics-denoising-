"""Evaluate pair manifests with crop-level dark-offset correction and masks."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_eval.metrics import image_quality


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_pairs(Path(args.pairs_csv))
    if not rows:
        raise ValueError(f"No rows found in pair manifest: {args.pairs_csv}")
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]

    metric_rows = []
    for row in rows:
        dark_path = Path(args.dark_offset_path or row.get("dark_offset_path", ""))
        mask_path = Path(args.bad_pixel_mask_path or row.get("bad_pixel_mask_path", ""))
        dark_offset = load_optional_npy(dark_path)
        bad_mask = load_optional_npy(mask_path)

        clean_raw = center_crop(load_tiff(Path(row["clean_path"])), args.crop_size).astype(np.float32)
        noisy_raw = center_crop(load_tiff(Path(row["noisy_path"])), args.crop_size).astype(np.float32)
        if clean_raw.shape != noisy_raw.shape:
            raise ValueError(f"Shape mismatch for {row['pair_key']}: {clean_raw.shape} vs {noisy_raw.shape}")

        dark_crop = align_optional_map(dark_offset, clean_raw.shape)
        mask_crop = align_optional_map(bad_mask, clean_raw.shape)
        good_mask = valid_mask(clean_raw, noisy_raw, mask_crop, args.range_max)

        clean_corr = correct_and_normalize(clean_raw, dark_crop, args.range_max)
        noisy_corr = correct_and_normalize(noisy_raw, dark_crop, args.range_max)

        metric_rows.append(evaluate_one(row, clean_corr, noisy_corr, good_mask))

    metrics_csv = output_dir / "masked_offset_pair_metrics.csv"
    report_path = output_dir / "masked_offset_pair_report.md"
    write_csv(metric_rows, metrics_csv)
    write_report(metric_rows, metrics_csv, report_path, args)
    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", default="reports/masked_offset_pairs")
    parser.add_argument("--dark-offset-path", default="")
    parser.add_argument("--bad-pixel-mask-path", default="")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def load_optional_npy(path: Path) -> np.ndarray | None:
    if not str(path) or not path.exists():
        return None
    return np.load(path)


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {arr.shape}")
    h, w = arr.shape
    size = min(crop_size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top : top + size, left : left + size]


def align_optional_map(arr: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.shape == shape:
        return arr
    return center_crop(np.asarray(arr), min(shape))


def valid_mask(clean_raw: np.ndarray, noisy_raw: np.ndarray, bad_mask: np.ndarray | None, range_max: float) -> np.ndarray:
    mask = np.ones(clean_raw.shape, dtype=bool)
    if bad_mask is not None:
        mask &= ~np.asarray(bad_mask, dtype=bool)
    mask &= clean_raw > 0
    mask &= noisy_raw > 0
    mask &= clean_raw < range_max
    mask &= noisy_raw < range_max
    return mask


def correct_and_normalize(raw: np.ndarray, dark_offset: np.ndarray | None, range_max: float) -> np.ndarray:
    arr = raw.astype(np.float32)
    if dark_offset is not None:
        arr = arr - dark_offset.astype(np.float32)
    return np.clip(arr / float(range_max), 0.0, 1.0).astype(np.float32)


def evaluate_one(row: dict[str, str], clean: np.ndarray, noisy: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    full_quality = image_quality(noisy, clean, data_range=1.0)
    masked_quality = masked_image_quality(noisy, clean, mask)
    residual = noisy - clean
    valid_residual = residual[mask]
    valid_clean = clean[mask]
    valid_noisy = noisy[mask]
    return {
        "pair_key": row["pair_key"],
        "tail_index": row.get("tail_index", ""),
        "valid_fraction": float(np.mean(mask)),
        "psnr_full": full_quality["psnr"],
        "ssim_full": full_quality["ssim"],
        "psnr_masked": masked_quality["psnr"],
        "mae_masked": masked_quality["mae"],
        "clean_mean_masked": float(np.mean(valid_clean)),
        "noisy_mean_masked": float(np.mean(valid_noisy)),
        "mean_ratio_noisy_over_clean": safe_div(float(np.mean(valid_noisy)), float(np.mean(valid_clean))),
        "residual_mean_masked": float(np.mean(valid_residual)),
        "residual_std_masked": float(np.std(valid_residual, ddof=1)),
        "residual_p01_masked": float(np.percentile(valid_residual, 1)),
        "residual_p50_masked": float(np.percentile(valid_residual, 50)),
        "residual_p99_masked": float(np.percentile(valid_residual, 99)),
    }


def masked_image_quality(noisy: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = noisy[mask] - clean[mask]
    if diff.size == 0:
        return {"psnr": float("nan"), "mae": float("nan")}
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
    return {"psnr": psnr, "mae": float(np.mean(np.abs(diff)))}


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], metrics_csv: Path, report_path: Path, args: argparse.Namespace) -> None:
    summary = summarize(rows)
    lines = [
        "# Masked Offset-Corrected Pair Evaluation",
        "",
        f"- Pairs CSV: `{args.pairs_csv}`",
        f"- Crop size: {args.crop_size}",
        f"- Range max: {args.range_max:g}",
        f"- Metrics CSV: `{metrics_csv}`",
        "",
        "## Summary",
        "",
        f"- Pair count: {len(rows)}",
        f"- Valid fraction mean: {summary['valid_fraction_mean']:.6g}",
        f"- Full PSNR mean/std: {summary['psnr_full_mean']:.4f} / {summary['psnr_full_std']:.4f}",
        f"- Masked PSNR mean/std: {summary['psnr_masked_mean']:.4f} / {summary['psnr_masked_std']:.4f}",
        f"- Masked MAE mean: {summary['mae_masked_mean']:.6g}",
        f"- Clean mean / noisy mean: {summary['clean_mean_masked_mean']:.6g} / {summary['noisy_mean_masked_mean']:.6g}",
        f"- Noisy/clean mean ratio: {summary['mean_ratio_noisy_over_clean_mean']:.6g}",
        f"- Residual mean/std: {summary['residual_mean_masked_mean']:.6g} / {summary['residual_std_masked_mean']:.6g}",
        "",
        "## Claim Boundary",
        "",
        "- This evaluates sCMOS proxy/content-source pairs, not real ICCD clean/noisy data.",
        "- Dark-offset correction and bad-pixel masking are applied on center crops using derived audit artifacts.",
        "- Large residual mean or mean ratio far from 1 indicates exposure/brightness mismatch; such pairs should be used as content/reference sources rather than clean/noisy supervised pairs.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "valid_fraction",
        "psnr_full",
        "psnr_masked",
        "mae_masked",
        "clean_mean_masked",
        "noisy_mean_masked",
        "mean_ratio_noisy_over_clean",
        "residual_mean_masked",
        "residual_std_masked",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
