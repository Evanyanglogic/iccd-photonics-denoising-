"""Evaluate noisy-input baseline from a pair manifest.

This is the B0 gate: before training any model, measure how far the noisy input
is from the clean/reference image using float-domain metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_eval.metrics import brightness_bin_psnr, image_quality, residual_statistics


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_pairs(Path(args.pairs_csv))
    if not rows:
        raise ValueError(f"No rows found in pair manifest: {args.pairs_csv}")

    metrics_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = load_tiff_normalized(Path(row["clean_path"]), args.range_max)
        noisy = load_tiff_normalized(Path(row["noisy_path"]), args.range_max)
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Shape mismatch for {row['pair_key']}: clean {clean.shape}, noisy {noisy.shape}"
            )

        quality = image_quality(noisy, clean, data_range=1.0)
        residual = residual_statistics(noisy, clean)
        metric_row = {
            "pair_key": row["pair_key"],
            "clean_path": row["clean_path"],
            "noisy_path": row["noisy_path"],
            **quality,
            "residual_var": residual["var"],
            "residual_p01": residual["p01"],
            "residual_p50": residual["p50"],
            "residual_p99": residual["p99"],
        }
        metrics_rows.append(metric_row)

        for bin_row in brightness_bin_psnr(noisy, clean, bins=args.bins, data_range=1.0):
            bin_rows.append({"pair_key": row["pair_key"], **bin_row})

    metrics_csv = output_dir / "b0_noisy_baseline_metrics.csv"
    bins_csv = output_dir / "b0_brightness_bin_psnr.csv"
    report_path = output_dir / "b0_noisy_baseline_report.md"
    write_csv(metrics_rows, metrics_csv)
    write_csv(bin_rows, bins_csv)
    write_report(metrics_rows, bin_rows, metrics_csv, bins_csv, report_path)
    print(f"Wrote baseline metrics: {metrics_csv}")
    print(f"Wrote brightness-bin PSNR: {bins_csv}")
    print(f"Wrote baseline report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="data_manifest/pairs.csv")
    parser.add_argument("--output-dir", default="reports/b0_noisy_baseline")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--bins", type=int, default=8)
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_tiff_normalized(path: Path, range_max: float) -> np.ndarray:
    try:
        import tifffile

        arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc
    if range_max <= 0:
        raise ValueError("--range-max must be positive")
    arr = arr / float(range_max)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    metrics_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    metrics_csv: Path,
    bins_csv: Path,
    report_path: Path,
) -> None:
    summary = summarize_metrics(metrics_rows)
    lines = [
        "# B0 Noisy-Input Baseline",
        "",
        "This report evaluates noisy TIFF inputs directly against clean/reference TIFFs.",
        "No model is used.",
        "",
        "## Outputs",
        "",
        f"- Per-pair metrics: `{metrics_csv}`",
        f"- Brightness-bin PSNR: `{bins_csv}`",
        "",
        "## Summary",
        "",
        f"- Pair count: {len(metrics_rows)}",
        f"- PSNR mean/std: {summary['psnr_mean']:.4f} / {summary['psnr_std']:.4f}",
        f"- SSIM mean/std: {summary['ssim_mean']:.6f} / {summary['ssim_std']:.6f}",
        f"- Residual mean/std mean: {summary['residual_mean_mean']:.6g} / {summary['residual_std_mean']:.6g}",
        "",
        "## Brightness-Bin Coverage",
        "",
        f"- Bin rows: {len(bin_rows)}",
        "",
        "## Next Gate",
        "",
        "Use this B0 result as the lower bound for later denoisers and synthetic-noise training sources.",
        "Do not compare trained models unless they use the same pair manifest and split policy.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    return {
        "psnr_mean": float(np.mean(values("psnr"))),
        "psnr_std": float(np.std(values("psnr"))),
        "ssim_mean": float(np.mean(values("ssim"))),
        "ssim_std": float(np.std(values("ssim"))),
        "residual_mean_mean": float(np.mean(values("residual_mean"))),
        "residual_std_mean": float(np.mean(values("residual_std"))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
