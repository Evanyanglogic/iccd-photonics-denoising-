"""Compare synthetic noise priors against real paired noisy frames.

This is the E2 gate: after pair integrity is confirmed, compare generic
Poisson-Gaussian, sCMOS-like, and ICCD-chain priors using residual statistics
instead of training a denoiser first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_eval.metrics import image_quality, residual_statistics
from src.iccd_noise import (
    ICCDNoiseConfig,
    ICCDNoiseModel,
    PoissonGaussianConfig,
    PoissonGaussianNoiseModel,
    SCMOSLikeConfig,
    SCMOSLikeNoiseModel,
)
from src.iccd_noise.statistics import mean_variance_by_intensity, radial_power_spectrum


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    range_max = float(args.range_max or config.get("range_max", 65535.0))
    bins = int(args.bins or config.get("bins", 8))
    seed = int(args.seed if args.seed is not None else config.get("seed", 20260715))

    rows = read_pairs(Path(args.pairs_csv))
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    if not rows:
        raise ValueError(f"No rows found in pair manifest: {args.pairs_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    for pair_idx, row in enumerate(rows):
        clean = load_tiff_normalized(Path(row["clean_path"]), range_max=range_max)
        real_noisy = load_tiff_normalized(Path(row["noisy_path"]), range_max=range_max)
        if clean.shape != real_noisy.shape:
            raise ValueError(
                f"Shape mismatch for {row['pair_key']}: clean {clean.shape}, noisy {real_noisy.shape}"
            )

        priors = build_priors(config, seed=seed + pair_idx * 1000)
        for prior_name, prior in priors.items():
            synthetic = prior.add_noise(clean)
            metric_rows.append(compare_one(row["pair_key"], prior_name, clean, real_noisy, synthetic))
            mv_rows.extend(compare_mean_variance(row["pair_key"], prior_name, clean, real_noisy, synthetic, bins=bins))

    metrics_csv = output_dir / "noise_prior_fidelity_metrics.csv"
    mv_csv = output_dir / "noise_prior_mean_variance_bins.csv"
    summary_csv = output_dir / "noise_prior_summary.csv"
    report_path = output_dir / "noise_prior_fidelity_report.md"
    write_csv(metric_rows, metrics_csv)
    write_csv(mv_rows, mv_csv)
    summary_rows = summarize_by_prior(metric_rows)
    write_csv(summary_rows, summary_csv)
    write_report(summary_rows, metric_rows, metrics_csv, mv_csv, summary_csv, report_path, config)

    print(f"Wrote prior metrics: {metrics_csv}")
    print(f"Wrote mean-variance bins: {mv_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="data_manifest/pairs.csv")
    parser.add_argument("--config", default="configs/noise_prior_baselines.yaml")
    parser.add_argument("--output-dir", default="reports/noise_prior_fidelity")
    parser.add_argument("--range-max", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                result[key] = {}
                current = result[key]
            else:
                result[key] = coerce_scalar(value.strip('"').strip("'"))
                current = None
            continue
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = coerce_scalar(value.strip().strip('"').strip("'"))
    return result


def coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


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
        raise ValueError("range_max must be positive")
    return np.clip(arr / range_max, 0.0, 1.0).astype(np.float32)


def build_priors(config: dict[str, Any], seed: int) -> dict[str, Any]:
    pg_config = {**config.get("poisson_gaussian", {}), "seed": seed + 1}
    scmos_config = {**config.get("scmos_like", {}), "seed": seed + 2}
    iccd_config = {**config.get("iccd", {}), "seed": seed + 3}
    return {
        "poisson_gaussian": PoissonGaussianNoiseModel(PoissonGaussianConfig(**pg_config)),
        "scmos_like": SCMOSLikeNoiseModel(SCMOSLikeConfig(**scmos_config)),
        "iccd_prior": ICCDNoiseModel(ICCDNoiseConfig(**iccd_config)),
    }


def compare_one(
    pair_key: str,
    prior_name: str,
    clean: np.ndarray,
    real_noisy: np.ndarray,
    synthetic: np.ndarray,
) -> dict[str, Any]:
    real_residual = real_noisy - clean
    synthetic_residual = synthetic - clean
    real_stats = residual_statistics(real_noisy, clean)
    synth_stats = residual_statistics(synthetic, clean)
    fidelity = image_quality(synthetic, real_noisy, data_range=1.0)
    return {
        "pair_key": pair_key,
        "prior": prior_name,
        "synthetic_vs_real_psnr": fidelity["psnr"],
        "synthetic_vs_real_ssim": fidelity["ssim"],
        "residual_mean_abs_error": abs(synth_stats["mean"] - real_stats["mean"]),
        "residual_std_abs_error": abs(synth_stats["std"] - real_stats["std"]),
        "residual_mae_abs_error": abs(synth_stats["mae"] - real_stats["mae"]),
        "histogram_l1": histogram_l1(real_residual, synthetic_residual),
        "psd_l1": psd_l1(real_residual, synthetic_residual),
    }


def compare_mean_variance(
    pair_key: str,
    prior_name: str,
    clean: np.ndarray,
    real_noisy: np.ndarray,
    synthetic: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    real_rows = mean_variance_by_intensity(clean, real_noisy, bins=bins)
    synth_rows = mean_variance_by_intensity(clean, synthetic, bins=bins)
    output = []
    for real_row, synth_row in zip(real_rows, synth_rows):
        output.append(
            {
                "pair_key": pair_key,
                "prior": prior_name,
                "bin_low": real_row["bin_low"],
                "bin_high": real_row["bin_high"],
                "count": real_row["count"],
                "real_mean": real_row["mean"],
                "synthetic_mean": synth_row["mean"],
                "mean_abs_error": abs_or_nan(real_row["mean"], synth_row["mean"]),
                "real_var": real_row["var"],
                "synthetic_var": synth_row["var"],
                "var_abs_error": abs_or_nan(real_row["var"], synth_row["var"]),
            }
        )
    return output


def histogram_l1(real_residual: np.ndarray, synthetic_residual: np.ndarray) -> float:
    low = float(min(np.percentile(real_residual, 0.5), np.percentile(synthetic_residual, 0.5), -0.25))
    high = float(max(np.percentile(real_residual, 99.5), np.percentile(synthetic_residual, 99.5), 0.25))
    real_hist, edges = np.histogram(real_residual.ravel(), bins=64, range=(low, high), density=False)
    synth_hist, _ = np.histogram(synthetic_residual.ravel(), bins=edges, density=False)
    real_prob = real_hist / max(float(np.sum(real_hist)), 1.0)
    synth_prob = synth_hist / max(float(np.sum(synth_hist)), 1.0)
    return float(np.sum(np.abs(real_prob - synth_prob)))


def psd_l1(real_residual: np.ndarray, synthetic_residual: np.ndarray) -> float:
    real_psd = normalize_vector(np.log1p(radial_power_spectrum(real_residual)))
    synth_psd = normalize_vector(np.log1p(radial_power_spectrum(synthetic_residual)))
    length = min(real_psd.size, synth_psd.size)
    if length == 0:
        return float("nan")
    return float(np.mean(np.abs(real_psd[:length] - synth_psd[:length])))


def normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = float(np.sum(np.abs(arr)))
    if total <= 1e-12:
        return np.zeros_like(arr)
    return arr / total


def abs_or_nan(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return float("nan")
    return abs(float(left) - float(right))


def summarize_by_prior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priors = sorted({row["prior"] for row in rows})
    summary = []
    for prior in priors:
        prior_rows = [row for row in rows if row["prior"] == prior]
        summary.append(
            {
                "prior": prior,
                "pair_count": len(prior_rows),
                "synthetic_vs_real_psnr_mean": mean(prior_rows, "synthetic_vs_real_psnr"),
                "synthetic_vs_real_psnr_std": std(prior_rows, "synthetic_vs_real_psnr"),
                "synthetic_vs_real_ssim_mean": mean(prior_rows, "synthetic_vs_real_ssim"),
                "residual_mean_abs_error_mean": mean(prior_rows, "residual_mean_abs_error"),
                "residual_std_abs_error_mean": mean(prior_rows, "residual_std_abs_error"),
                "histogram_l1_mean": mean(prior_rows, "histogram_l1"),
                "psd_l1_mean": mean(prior_rows, "psd_l1"),
            }
        )
    return summary


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanstd(values))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    summary_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    metrics_csv: Path,
    mv_csv: Path,
    summary_csv: Path,
    report_path: Path,
    config: dict[str, Any],
) -> None:
    config_snapshot = {
        "poisson_gaussian": config.get("poisson_gaussian", asdict(PoissonGaussianConfig())),
        "scmos_like": config.get("scmos_like", asdict(SCMOSLikeConfig())),
        "iccd": config.get("iccd", asdict(ICCDNoiseConfig())),
    }
    lines = [
        "# Noise Prior Fidelity Report",
        "",
        "This report compares synthetic noisy images against paired real noisy ICCD/sCMOS frames.",
        "Lower residual-statistical errors and higher synthetic-vs-real PSNR/SSIM indicate better fidelity.",
        "",
        "## Outputs",
        "",
        f"- Per-pair prior metrics: `{metrics_csv}`",
        f"- Mean-variance bins: `{mv_csv}`",
        f"- Summary table: `{summary_csv}`",
        "",
        "## Summary",
        "",
        "| prior | pairs | PSNR mean | SSIM mean | residual std error | histogram L1 | PSD L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['prior']} | {row['pair_count']} | "
            f"{row['synthetic_vs_real_psnr_mean']:.4f} | "
            f"{row['synthetic_vs_real_ssim_mean']:.6f} | "
            f"{row['residual_std_abs_error_mean']:.6g} | "
            f"{row['histogram_l1_mean']:.6g} | "
            f"{row['psd_l1_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Configuration Snapshot",
            "",
            "```json",
            json.dumps(config_snapshot, indent=2, sort_keys=True),
            "```",
            "",
            "## Claim Boundary",
            "",
            "Use this report to decide which prior is statistically closer to real device noise.",
            "Do not claim downstream denoising improvement until the same priors are used to train denoisers and tested on held-out real ICCD data.",
            "",
            f"Per-pair comparisons: {len(metric_rows)}",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
