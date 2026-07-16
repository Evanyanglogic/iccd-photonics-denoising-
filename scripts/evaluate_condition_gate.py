"""Evaluate condition-gated use of an ICCD denoiser on real surrogate pairs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = [
    "mean_signal",
    "temporal_std_mean",
    "fano_temporal",
    "fixed_map_std",
    "fixed_to_temporal_std_ratio",
]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_csv(Path(args.eval_csv))
    p99_rows = read_optional_csv(args.p99_eval_csv)
    condition_rows = load_condition_rows(Path(args.condition_summary_csv), args.model_label)
    pair_rows, summary_rows = evaluate_gates(eval_rows, p99_rows, condition_rows, args)

    pair_csv = output_dir / "condition_gate_pair_metrics.csv"
    summary_csv = output_dir / "condition_gate_summary.csv"
    report_path = output_dir / "condition_gate_report.md"
    write_csv(pair_rows, pair_csv)
    write_csv(summary_rows, summary_csv)
    write_report(report_path, summary_rows, pair_csv, summary_csv, args)

    print(f"Wrote pair metrics: {pair_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, help="Physical/checkpoint pair-level evaluation CSV.")
    parser.add_argument("--p99-eval-csv", default="", help="Optional p99 checkpoint CSV for fallback comparison.")
    parser.add_argument("--condition-summary-csv", required=True, help="Output from analyze_condition_gain.py.")
    parser.add_argument("--model-label", default="physical", help="Rows in condition summary used for threshold statistics.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=[0.4, 0.5, 0.6],
        help="Condition metric quantiles used as non-oracle thresholds.",
    )
    return parser.parse_args()


def read_optional_csv(path_value: str) -> list[dict[str, str]]:
    if not path_value:
        return []
    return read_csv(Path(path_value))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_condition_rows(path: Path, model_label: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        if row.get("model_label") != model_label:
            continue
        folder = int(float(row["folder"]))
        rows[folder] = {metric: to_float(row.get(metric)) for metric in METRICS}
        rows[folder]["oracle_folder_gain"] = to_float(row.get("mean_psnr_gain"))
    if not rows:
        raise ValueError(f"No condition rows found for model_label={model_label}")
    return rows


def evaluate_gates(
    eval_rows: list[dict[str, str]],
    p99_rows: list[dict[str, str]],
    condition_rows: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p99_by_key = {row["pair_key"]: row for row in p99_rows}
    gate_specs = build_gate_specs(condition_rows, args.quantiles)
    pair_metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    strategies = ["always_noisy", "always_model"]
    if p99_by_key:
        strategies.append("always_p99")
    strategies.extend([spec["name"] for spec in gate_specs])
    strategies.append("oracle_folder_positive")

    for strategy in strategies:
        rows_for_strategy: list[dict[str, Any]] = []
        for row in eval_rows:
            folder = int(float(row["meta_folder"]))
            condition = condition_rows[folder]
            selected = select_metrics(strategy, row, p99_by_key.get(row["pair_key"]), condition, gate_specs)
            out = {
                "strategy": strategy,
                "pair_key": row["pair_key"],
                "folder": folder,
                "selected_source": selected["source"],
                "psnr": selected["psnr"],
                "ssim": selected["ssim"],
                "noisy_psnr": to_float(row["noisy_psnr"]),
                "noisy_ssim": to_float(row["noisy_ssim"]),
                "psnr_gain": selected["psnr"] - to_float(row["noisy_psnr"]),
                "ssim_gain": selected["ssim"] - to_float(row["noisy_ssim"]),
            }
            for metric in METRICS:
                out[metric] = condition[metric]
            rows_for_strategy.append(out)
            pair_metric_rows.append(out)
        summary_rows.append(summarize_strategy(strategy, rows_for_strategy))

    return pair_metric_rows, summary_rows


def build_gate_specs(condition_rows: dict[int, dict[str, float]], quantiles: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in condition_rows.values()], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 3:
            continue
        for quantile in quantiles:
            threshold = float(np.quantile(values, quantile))
            specs.append(
                {
                    "name": f"gate_{metric}_q{int(round(quantile * 100)):02d}",
                    "metric": metric,
                    "threshold": threshold,
                }
            )
    return specs


def select_metrics(
    strategy: str,
    physical_row: dict[str, str],
    p99_row: dict[str, str] | None,
    condition: dict[str, float],
    gate_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    if strategy == "always_noisy":
        return {"source": "noisy", "psnr": to_float(physical_row["noisy_psnr"]), "ssim": to_float(physical_row["noisy_ssim"])}
    if strategy == "always_model":
        return {"source": "physical", "psnr": to_float(physical_row["psnr"]), "ssim": to_float(physical_row["ssim"])}
    if strategy == "always_p99":
        if p99_row is None:
            raise ValueError("always_p99 requested but p99 row is missing")
        return {"source": "p99", "psnr": to_float(p99_row["psnr"]), "ssim": to_float(p99_row["ssim"])}
    if strategy == "oracle_folder_positive":
        if condition["oracle_folder_gain"] > 0.0:
            return {"source": "physical", "psnr": to_float(physical_row["psnr"]), "ssim": to_float(physical_row["ssim"])}
        return {"source": "noisy", "psnr": to_float(physical_row["noisy_psnr"]), "ssim": to_float(physical_row["noisy_ssim"])}

    spec = next((item for item in gate_specs if item["name"] == strategy), None)
    if spec is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    if condition[spec["metric"]] >= spec["threshold"]:
        return {"source": "physical", "psnr": to_float(physical_row["psnr"]), "ssim": to_float(physical_row["ssim"])}
    return {"source": "noisy", "psnr": to_float(physical_row["noisy_psnr"]), "ssim": to_float(physical_row["noisy_ssim"])}


def summarize_strategy(strategy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = np.asarray([float(row["psnr_gain"]) for row in rows], dtype=np.float64)
    ssim_gains = np.asarray([float(row["ssim_gain"]) for row in rows], dtype=np.float64)
    by_folder: dict[int, list[float]] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        by_folder.setdefault(int(row["folder"]), []).append(float(row["psnr_gain"]))
        source_counts[str(row["selected_source"])] = source_counts.get(str(row["selected_source"]), 0) + 1
    folder_gains = np.asarray([float(np.mean(values)) for values in by_folder.values()], dtype=np.float64)
    return {
        "strategy": strategy,
        "pair_count": len(rows),
        "folder_count": len(by_folder),
        "mean_psnr_gain": float(np.mean(gains)),
        "median_psnr_gain": float(np.median(gains)),
        "std_psnr_gain": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
        "positive_pair_fraction": float(np.mean(gains > 0.0)),
        "positive_folder_count": int(np.sum(folder_gains > 0.0)),
        "negative_folder_count": int(np.sum(folder_gains < 0.0)),
        "mean_folder_psnr_gain": float(np.mean(folder_gains)),
        "median_folder_psnr_gain": float(np.median(folder_gains)),
        "mean_ssim_gain": float(np.mean(ssim_gains)),
        "selected_sources": ";".join(f"{key}:{value}" for key, value in sorted(source_counts.items())),
    }


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    pair_csv: Path,
    summary_csv: Path,
    args: argparse.Namespace,
) -> None:
    ranked = sorted(summary_rows, key=lambda row: (int(row["negative_folder_count"]), -float(row["mean_folder_psnr_gain"])))
    lines = [
        "# E3.5-A Condition Gate Evaluation",
        "",
        "This experiment evaluates whether E1 folder-level ICCD condition statistics can gate the physical-scale denoiser on real surrogate pairs.",
        "",
        f"- Physical evaluation CSV: `{args.eval_csv}`",
        f"- p99 evaluation CSV: `{args.p99_eval_csv}`",
        f"- Condition summary CSV: `{args.condition_summary_csv}`",
        f"- Pair metrics: `{pair_csv}`",
        f"- Summary: `{summary_csv}`",
        "",
        "## Strategy Summary",
        "",
        "| strategy | mean folder gain dB | mean pair gain dB | positive folders | negative folders | positive pairs | selected sources |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| {strategy} | {folder_gain:.4f} | {pair_gain:.4f} | {positive_folders}/{folders} | {negative_folders} | {positive_pairs:.3f} | {sources} |".format(
                strategy=row["strategy"],
                folder_gain=float(row["mean_folder_psnr_gain"]),
                pair_gain=float(row["mean_psnr_gain"]),
                positive_folders=int(row["positive_folder_count"]),
                folders=int(row["folder_count"]),
                negative_folders=int(row["negative_folder_count"]),
                positive_pairs=float(row["positive_pair_fraction"]),
                sources=row["selected_sources"],
            )
        )

    best_non_oracle = next(row for row in ranked if row["strategy"] != "oracle_folder_positive")
    oracle = next(row for row in summary_rows if row["strategy"] == "oracle_folder_positive")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best non-oracle strategy by negative-folder count then folder gain: `{best_non_oracle['strategy']}`.",
            f"- Its mean folder gain is {float(best_non_oracle['mean_folder_psnr_gain']):.4f} dB with {int(best_non_oracle['negative_folder_count'])} negative-gain folders.",
            f"- Oracle folder-positive gating reaches {float(oracle['mean_folder_psnr_gain']):.4f} dB and is an upper bound, not a deployable rule.",
            "- A useful gate should reduce negative-gain folders relative to `always_model` without collapsing to a trivial no-denoising rule.",
            "- This is a condition-level sanity check; final claims still require visual inspection and a held-out condition split when more data are available.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
