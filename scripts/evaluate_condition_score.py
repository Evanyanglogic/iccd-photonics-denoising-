"""Evaluate multi-metric ICCD condition scores for checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SCORE_METRICS = [
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

    condition_rows = load_condition_rows(Path(args.condition_summary_csv), args.condition_model_label)
    score_rows = compute_scores(condition_rows, args.score_metrics)
    physical_rows = read_csv(Path(args.physical_eval_csv))
    p99_rows = read_csv(Path(args.p99_eval_csv))
    pair_rows, summary_rows = evaluate_strategies(physical_rows, p99_rows, score_rows, args)

    score_csv = output_dir / "condition_score_folders.csv"
    pair_csv = output_dir / "condition_score_pair_metrics.csv"
    summary_csv = output_dir / "condition_score_summary.csv"
    report_path = output_dir / "condition_score_report.md"
    write_csv(score_rows, score_csv)
    write_csv(pair_rows, pair_csv)
    write_csv(summary_rows, summary_csv)
    write_report(report_path, score_rows, summary_rows, score_csv, pair_csv, summary_csv, args)

    print(f"Wrote folder scores: {score_csv}")
    print(f"Wrote pair metrics: {pair_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-eval-csv", required=True)
    parser.add_argument("--p99-eval-csv", required=True)
    parser.add_argument("--condition-summary-csv", required=True)
    parser.add_argument("--condition-model-label", default="physical")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-metrics", nargs="*", default=DEFAULT_SCORE_METRICS)
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.4, 0.5, 0.6])
    parser.add_argument("--reference-fano-quantile", type=float, default=0.4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_condition_rows(path: Path, model_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_rows = read_csv(path)
    p99_gain_by_folder = {
        int(float(row["folder"])): to_float(row["mean_psnr_gain"])
        for row in all_rows
        if row.get("model_label") == "p99"
    }
    for row in all_rows:
        if row.get("model_label") != model_label:
            continue
        folder = int(float(row["folder"]))
        item: dict[str, Any] = {
            "folder": folder,
            "physical_folder_gain": to_float(row["mean_psnr_gain"]),
            "p99_folder_gain": p99_gain_by_folder.get(folder, math.nan),
        }
        for key, value in row.items():
            if key not in item and key != "model_label":
                item[key] = to_float(value)
        rows.append(item)
    if not rows:
        raise ValueError(f"No condition rows for model_label={model_label}")
    return sorted(rows, key=lambda item: int(item["folder"]))


def compute_scores(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    stats: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[metric] = (mean, std if std > 0.0 else 1.0)

    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        z_values = []
        for metric in metrics:
            mean, std = stats[metric]
            z = (float(row[metric]) - mean) / std
            item[f"z_{metric}"] = z
            z_values.append(z)
        item["condition_score"] = float(np.mean(z_values))
        scored.append(item)
    return sorted(scored, key=lambda item: float(item["condition_score"]))


def evaluate_strategies(
    physical_rows: list[dict[str, str]],
    p99_rows: list[dict[str, str]],
    score_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p99_by_key = {row["pair_key"]: row for row in p99_rows}
    score_by_folder = {int(row["folder"]): row for row in score_rows}
    strategies = build_strategies(score_rows, args.quantiles, args.reference_fano_quantile)
    pair_metric_rows: list[dict[str, Any]] = []
    for physical in physical_rows:
        pair_key = physical["pair_key"]
        p99 = p99_by_key[pair_key]
        folder = int(float(physical["meta_folder"]))
        condition = score_by_folder[folder]
        for strategy in strategies:
            selected = select_strategy(strategy, physical, p99, condition)
            pair_metric_rows.append(
                {
                    "strategy": strategy["name"],
                    "folder": folder,
                    "pair_key": pair_key,
                    "selected_source": selected["source"],
                    "condition_score": condition["condition_score"],
                    "fano_temporal": condition["fano_temporal"],
                    "psnr": selected["psnr"],
                    "ssim": selected["ssim"],
                    "noisy_psnr": to_float(physical["noisy_psnr"]),
                    "noisy_ssim": to_float(physical["noisy_ssim"]),
                    "psnr_gain": selected["psnr"] - to_float(physical["noisy_psnr"]),
                    "ssim_gain": selected["ssim"] - to_float(physical["noisy_ssim"]),
                }
            )
    summary_rows = summarize(pair_metric_rows)
    return pair_metric_rows, summary_rows


def build_strategies(score_rows: list[dict[str, Any]], quantiles: list[float], fano_quantile: float) -> list[dict[str, Any]]:
    score_values = np.asarray([float(row["condition_score"]) for row in score_rows], dtype=np.float64)
    fano_values = np.asarray([float(row["fano_temporal"]) for row in score_rows], dtype=np.float64)
    strategies: list[dict[str, Any]] = [
        {"name": "always_noisy", "kind": "constant", "source": "noisy"},
        {"name": "always_p99", "kind": "constant", "source": "p99"},
        {"name": "always_physical", "kind": "constant", "source": "physical"},
        {
            "name": f"fano_q{int(round(fano_quantile * 100)):02d}_hybrid_p99_physical",
            "kind": "threshold",
            "metric": "fano_temporal",
            "threshold": float(np.quantile(fano_values, fano_quantile)),
            "low_source": "p99",
            "high_source": "physical",
        },
    ]
    for quantile in quantiles:
        threshold = float(np.quantile(score_values, quantile))
        qname = f"q{int(round(quantile * 100)):02d}"
        strategies.append(
            {
                "name": f"score_{qname}_hybrid_p99_physical",
                "kind": "threshold",
                "metric": "condition_score",
                "threshold": threshold,
                "low_source": "p99",
                "high_source": "physical",
            }
        )
        strategies.append(
            {
                "name": f"score_{qname}_gate_noisy_physical",
                "kind": "threshold",
                "metric": "condition_score",
                "threshold": threshold,
                "low_source": "noisy",
                "high_source": "physical",
            }
        )
    return strategies


def select_strategy(strategy: dict[str, Any], physical: dict[str, str], p99: dict[str, str], condition: dict[str, Any]) -> dict[str, Any]:
    if strategy["kind"] == "constant":
        return source_metrics(strategy["source"], physical, p99)
    metric = str(strategy["metric"])
    source = strategy["high_source"] if float(condition[metric]) >= float(strategy["threshold"]) else strategy["low_source"]
    return source_metrics(source, physical, p99)


def source_metrics(source: str, physical: dict[str, str], p99: dict[str, str]) -> dict[str, Any]:
    if source == "noisy":
        return {"source": "noisy", "psnr": to_float(physical["noisy_psnr"]), "ssim": to_float(physical["noisy_ssim"])}
    if source == "p99":
        return {"source": "p99", "psnr": to_float(p99["psnr"]), "ssim": to_float(p99["ssim"])}
    if source == "physical":
        return {"source": "physical", "psnr": to_float(physical["psnr"]), "ssim": to_float(physical["ssim"])}
    raise ValueError(f"Unknown source: {source}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["strategy"]), []).append(row)
    out: list[dict[str, Any]] = []
    for strategy, group_rows in groups.items():
        gains = np.asarray([float(row["psnr_gain"]) for row in group_rows], dtype=np.float64)
        ssim_gains = np.asarray([float(row["ssim_gain"]) for row in group_rows], dtype=np.float64)
        by_folder: dict[int, list[float]] = {}
        source_counts: dict[str, int] = {}
        for row in group_rows:
            by_folder.setdefault(int(row["folder"]), []).append(float(row["psnr_gain"]))
            source = str(row["selected_source"])
            source_counts[source] = source_counts.get(source, 0) + 1
        folder_gains = np.asarray([float(np.mean(values)) for values in by_folder.values()], dtype=np.float64)
        out.append(
            {
                "strategy": strategy,
                "pair_count": len(group_rows),
                "folder_count": len(by_folder),
                "mean_pair_psnr_gain": float(np.mean(gains)),
                "median_pair_psnr_gain": float(np.median(gains)),
                "positive_pair_fraction": float(np.mean(gains > 0.0)),
                "mean_folder_psnr_gain": float(np.mean(folder_gains)),
                "median_folder_psnr_gain": float(np.median(folder_gains)),
                "positive_folder_count": int(np.sum(folder_gains > 0.0)),
                "negative_folder_count": int(np.sum(folder_gains < 0.0)),
                "mean_ssim_gain": float(np.mean(ssim_gains)),
                "selected_sources": ";".join(f"{key}:{value}" for key, value in sorted(source_counts.items())),
            }
        )
    return sorted(out, key=lambda row: (-float(row["mean_folder_psnr_gain"]), int(row["negative_folder_count"])))


def write_report(
    path: Path,
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    score_csv: Path,
    pair_csv: Path,
    summary_csv: Path,
    args: argparse.Namespace,
) -> None:
    best = summary_rows[0]
    lines = [
        "# E3.5-D Multi-Metric Condition Score",
        "",
        "This report evaluates a multi-metric ICCD condition score for selecting between conservative p99 denoising and stronger physical-scale denoising.",
        "",
        f"- Score metrics: `{', '.join(args.score_metrics)}`",
        f"- Folder scores: `{score_csv}`",
        f"- Pair metrics: `{pair_csv}`",
        f"- Summary: `{summary_csv}`",
        "",
        "## Folder Scores",
        "",
        "| rank | folder | score | Fano | p99 gain | physical gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(score_rows, start=1):
        lines.append(
            "| {rank} | {folder} | {score:.4f} | {fano:.4f} | {p99:.4f} | {physical:.4f} |".format(
                rank=rank,
                folder=int(row["folder"]),
                score=float(row["condition_score"]),
                fano=float(row["fano_temporal"]),
                p99=float(row.get("p99_folder_gain", float("nan"))),
                physical=float(row["physical_folder_gain"]),
            )
        )
    lines.extend(
        [
            "",
            "## Strategy Summary",
            "",
            "| strategy | mean folder gain dB | positive folders | negative folders | positive pairs | selected sources |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {strategy} | {gain:.4f} | {pos}/{count} | {neg} | {pairs:.3f} | {sources} |".format(
                strategy=row["strategy"],
                gain=float(row["mean_folder_psnr_gain"]),
                pos=int(row["positive_folder_count"]),
                count=int(row["folder_count"]),
                neg=int(row["negative_folder_count"]),
                pairs=float(row["positive_pair_fraction"]),
                sources=row["selected_sources"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best strategy in this diagnostic run: `{best['strategy']}` with {float(best['mean_folder_psnr_gain']):.4f} dB mean folder gain.",
            "- Multi-metric scoring is intended to reduce boundary mistakes from single-metric Fano gating.",
            "- The score is diagnostic because it is derived from the same ten folders used for evaluation; it is not a final deployable classifier.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
