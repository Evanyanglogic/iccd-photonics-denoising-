"""Relate denoiser real-surrogate gains to ICCD condition statistics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


CONDITION_COLUMNS = [
    "mean_signal",
    "temporal_std_mean",
    "fano_temporal",
    "spatial_mean_std",
    "fixed_map_std",
    "spatial_reduction_fraction",
    "fixed_to_temporal_std_ratio",
]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = load_condition_rows(
        Path(args.mean_variance_csv),
        Path(args.fixed_pattern_csv),
        Path(args.noise_summary_csv),
    )
    eval_specs = parse_eval_specs(args.eval_csv)
    summary_rows: list[dict[str, Any]] = []
    for label, path in eval_specs:
        summary_rows.extend(summarize_eval(label, path, condition_rows))

    if not summary_rows:
        raise ValueError("No merged condition/gain rows were produced.")

    correlation_rows = compute_correlations(summary_rows)
    summary_csv = output_dir / "condition_gain_summary.csv"
    correlations_csv = output_dir / "condition_gain_correlations.csv"
    report_path = output_dir / "condition_gain_report.md"
    write_csv(summary_rows, summary_csv)
    write_csv(correlation_rows, correlations_csv)
    write_report(report_path, summary_rows, correlation_rows, summary_csv, correlations_csv, args)

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote correlations: {correlations_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-csv",
        action="append",
        required=True,
        help="Evaluation CSV as label=path. Can be repeated.",
    )
    parser.add_argument("--mean-variance-csv", required=True)
    parser.add_argument("--fixed-pattern-csv", required=True)
    parser.add_argument("--noise-summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_eval_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--eval-csv must use label=path, got: {value}")
        label, path = value.split("=", 1)
        specs.append((label.strip(), Path(path.strip())))
    return specs


def load_condition_rows(
    mean_variance_csv: Path,
    fixed_pattern_csv: Path,
    noise_summary_csv: Path,
) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for row in read_csv(mean_variance_csv):
        folder = int(float(row["folder"]))
        rows.setdefault(folder, {})["mean_signal"] = to_float(row.get("mean_signal"))
        rows[folder]["temporal_std_mean"] = to_float(row.get("temporal_std_mean"))
        rows[folder]["fano_temporal"] = to_float(row.get("fano_temporal"))
        rows[folder]["spatial_mean_std"] = to_float(row.get("spatial_mean_std"))

    for row in read_csv(fixed_pattern_csv):
        folder = int(float(row["folder"]))
        rows.setdefault(folder, {})["fixed_map_std"] = to_float(row.get("fixed_map_std"))
        rows[folder]["spatial_reduction_fraction"] = to_float(row.get("spatial_reduction_fraction"))

    for row in read_csv(noise_summary_csv):
        folder = int(float(row["folder"]))
        rows.setdefault(folder, {})["fixed_to_temporal_std_ratio"] = to_float(row.get("fixed_to_temporal_std_ratio"))

    return rows


def summarize_eval(
    label: str,
    eval_csv: Path,
    condition_rows: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    by_folder: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(eval_csv):
        folder = int(float(row["meta_folder"]))
        by_folder.setdefault(folder, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for folder, rows in sorted(by_folder.items()):
        gains = np.asarray([to_float(row["psnr_gain"]) for row in rows], dtype=np.float64)
        ssim_gains = np.asarray([to_float(row["ssim_gain"]) for row in rows], dtype=np.float64)
        residual_stds = np.asarray([to_float(row["residual_std"]) for row in rows], dtype=np.float64)
        noisy_psnr = np.asarray([to_float(row["noisy_psnr"]) for row in rows], dtype=np.float64)
        model_psnr = np.asarray([to_float(row["psnr"]) for row in rows], dtype=np.float64)
        condition = condition_rows.get(folder, {})
        item: dict[str, Any] = {
            "model_label": label,
            "folder": folder,
            "pair_count": len(rows),
            "mean_psnr_gain": float(np.mean(gains)),
            "std_psnr_gain": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
            "min_psnr_gain": float(np.min(gains)),
            "max_psnr_gain": float(np.max(gains)),
            "positive_gain_fraction": float(np.mean(gains > 0.0)),
            "mean_ssim_gain": float(np.mean(ssim_gains)),
            "mean_residual_std": float(np.mean(residual_stds)),
            "mean_noisy_psnr": float(np.mean(noisy_psnr)),
            "mean_model_psnr": float(np.mean(model_psnr)),
        }
        for key in CONDITION_COLUMNS:
            item[key] = condition.get(key, math.nan)
        summary_rows.append(item)
    return summary_rows


def compute_correlations(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({str(row["model_label"]) for row in summary_rows})
    correlation_rows: list[dict[str, Any]] = []
    for label in labels:
        rows = [row for row in summary_rows if row["model_label"] == label]
        y = np.asarray([float(row["mean_psnr_gain"]) for row in rows], dtype=np.float64)
        for column in CONDITION_COLUMNS:
            x = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(mask)) < 3:
                pearson = math.nan
                spearman = math.nan
                n = int(np.sum(mask))
            else:
                pearson = pearsonr(x[mask], y[mask])
                spearman = pearsonr(rankdata(x[mask]), rankdata(y[mask]))
                n = int(np.sum(mask))
            correlation_rows.append(
                {
                    "model_label": label,
                    "condition_metric": column,
                    "n_folders": n,
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "abs_pearson_r": abs(pearson) if math.isfinite(pearson) else math.nan,
                }
            )
    return sorted(correlation_rows, key=lambda row: (str(row["model_label"]), -safe_abs(row["pearson_r"])))


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom == 0.0:
        return math.nan
    return float(np.sum(x_centered * y_centered) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = 0.5 * (index + end - 1)
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def safe_abs(value: Any) -> float:
    value_float = float(value)
    if not math.isfinite(value_float):
        return -1.0
    return abs(value_float)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    summary_csv: Path,
    correlations_csv: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Condition-Stratified Real-Surrogate Gain Analysis",
        "",
        "This report links E3 real-surrogate denoiser gains to E1 ICCD folder-level statistics.",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Correlations CSV: `{correlations_csv}`",
        f"- Mean-variance source: `{args.mean_variance_csv}`",
        f"- Fixed-pattern source: `{args.fixed_pattern_csv}`",
        f"- Noise-summary source: `{args.noise_summary_csv}`",
        "",
        "## Folder-Level Gain Summary",
        "",
    ]
    for label in sorted({str(row["model_label"]) for row in summary_rows}):
        rows = [row for row in summary_rows if row["model_label"] == label]
        gains = np.asarray([float(row["mean_psnr_gain"]) for row in rows], dtype=np.float64)
        positives = sum(1 for row in rows if float(row["mean_psnr_gain"]) > 0.0)
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Folders: {len(rows)}",
                f"- Mean folder PSNR gain: {np.mean(gains):.4f} dB",
                f"- Median folder PSNR gain: {np.median(gains):.4f} dB",
                f"- Positive-gain folders: {positives}/{len(rows)}",
                "",
                "| folder | mean gain dB | positive pair fraction | mean signal | Fano | fixed/temporal ratio | fixed-pattern reduction |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda item: int(item["folder"])):
            lines.append(
                "| {folder} | {gain:.4f} | {positive:.3f} | {signal:.2f} | {fano:.3f} | {ratio:.3f} | {reduction:.3f} |".format(
                    folder=int(row["folder"]),
                    gain=float(row["mean_psnr_gain"]),
                    positive=float(row["positive_gain_fraction"]),
                    signal=float(row["mean_signal"]),
                    fano=float(row["fano_temporal"]),
                    ratio=float(row["fixed_to_temporal_std_ratio"]),
                    reduction=float(row["spatial_reduction_fraction"]),
                )
            )
        lines.append("")

    lines.extend(["## Strongest Condition Correlations", ""])
    for label in sorted({str(row["model_label"]) for row in correlation_rows}):
        rows = [row for row in correlation_rows if row["model_label"] == label]
        lines.extend([f"### {label}", "", "| condition metric | Pearson r | Spearman r | n |", "|---|---:|---:|---:|"])
        for row in rows[:5]:
            lines.append(
                f"| {row['condition_metric']} | {float(row['pearson_r']):.4f} | {float(row['spearman_r']):.4f} | {int(row['n_folders'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "- This analysis is folder-level and uses only ten ICCD conditions, so correlations are diagnostic rather than definitive.",
            "- A strong correlation means the current denoiser behavior is condition-dependent; it does not prove the model recovers missing detail.",
            "- Negative-gain folders should be inspected before any larger architecture run.",
            "- If gains track fixed-pattern or brightness statistics, the next model change should be condition-aware normalization/noise synthesis, not a generic deeper denoiser.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
