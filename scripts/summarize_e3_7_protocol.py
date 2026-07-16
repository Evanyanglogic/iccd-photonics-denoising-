"""Summarize E3.7 evaluation protocol and smoothing-risk evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blend_rows = read_csv(Path(args.blend_metrics_csv))
    visual_rows = read_csv(Path(args.visual_metrics_csv))

    strategy_summary = summarize_strategy_metrics(blend_rows)
    folder_summary = summarize_folder_metrics(blend_rows)
    smoothing_rows = summarize_smoothing_risk(visual_rows, args.warning_gradient_ratio, args.high_risk_gradient_ratio)

    strategy_csv = output_dir / "e3_7_strategy_summary.csv"
    folder_csv = output_dir / "e3_7_folder_strategy_summary.csv"
    smoothing_csv = output_dir / "e3_7_smoothing_risk_summary.csv"
    config_json = output_dir / "e3_7_protocol_config.json"
    report_md = output_dir / "e3_7_evaluation_protocol_report.md"

    write_csv(strategy_summary, strategy_csv)
    write_csv(folder_summary, folder_csv)
    write_csv(smoothing_rows, smoothing_csv)
    write_json(
        config_json,
        {
            "experiment_id": "E3.7",
            "blend_metrics_csv": args.blend_metrics_csv,
            "visual_metrics_csv": args.visual_metrics_csv,
            "warning_gradient_ratio": args.warning_gradient_ratio,
            "high_risk_gradient_ratio": args.high_risk_gradient_ratio,
            "git_commit": git_commit(),
            "protocol_status": "fixed_before_lofo",
        },
    )
    write_report(report_md, strategy_summary, folder_summary, smoothing_rows, args)

    print(f"Wrote strategy summary: {strategy_csv}")
    print(f"Wrote folder summary: {folder_csv}")
    print(f"Wrote smoothing summary: {smoothing_csv}")
    print(f"Wrote config: {config_json}")
    print(f"Wrote report: {report_md}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blend-metrics-csv",
        default="reports/e3_6_condition_blend_p99_physical/condition_blend_metrics.csv",
    )
    parser.add_argument(
        "--visual-metrics-csv",
        default="reports/e3_5_score_q50_visuals/condition_visual_metrics.csv",
    )
    parser.add_argument("--output-dir", default="reports/e3_7_evaluation_protocol")
    parser.add_argument("--warning-gradient-ratio", type=float, default=0.95)
    parser.add_argument("--high-risk-gradient-ratio", type=float, default=0.85)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_strategy_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    out: list[dict[str, Any]] = []
    for strategy, strategy_rows in grouped.items():
        gains = [f(row, "psnr_gain") for row in strategy_rows]
        ssim_gains = [f(row, "ssim_gain") for row in strategy_rows]
        residual_stds = [f(row, "residual_std") for row in strategy_rows]
        noisy_residual_stds = [f(row, "residual_std") for row in strategy_rows if row["strategy"] == "always_noisy"]
        if not noisy_residual_stds:
            noisy_lookup = {
                row["pair_key"]: f(row, "residual_std")
                for row in rows
                if row["strategy"] == "always_noisy"
            }
            residual_reductions = [noisy_lookup[row["pair_key"]] - f(row, "residual_std") for row in strategy_rows]
        else:
            residual_reductions = [0.0 for _ in strategy_rows]

        folder_means = mean_by(strategy_rows, "folder", "psnr_gain")
        out.append(
            {
                "strategy": strategy,
                "pair_count": len(strategy_rows),
                "folder_count": len(folder_means),
                "mean_pair_psnr_gain": mean(gains),
                "std_pair_psnr_gain": std(gains),
                "positive_pair_fraction": sum(value > 0 for value in gains) / len(gains),
                "mean_folder_psnr_gain": mean(list(folder_means.values())),
                "positive_folder_count": sum(value > 0 for value in folder_means.values()),
                "negative_folder_count": sum(value < 0 for value in folder_means.values()),
                "mean_ssim_gain": mean(ssim_gains),
                "mean_residual_std": mean(residual_stds),
                "mean_residual_std_reduction_vs_noisy": mean(residual_reductions),
            }
        )
    return sorted(out, key=lambda row: float(row["mean_folder_psnr_gain"]), reverse=True)


def summarize_folder_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["folder"])].append(row)

    out: list[dict[str, Any]] = []
    for (strategy, folder), strategy_rows in grouped.items():
        gains = [f(row, "psnr_gain") for row in strategy_rows]
        residual_stds = [f(row, "residual_std") for row in strategy_rows]
        out.append(
            {
                "strategy": strategy,
                "folder": int(float(folder)),
                "pair_count": len(strategy_rows),
                "mean_psnr_gain": mean(gains),
                "positive_pair_fraction": sum(value > 0 for value in gains) / len(gains),
                "mean_residual_std": mean(residual_stds),
            }
        )
    return sorted(out, key=lambda row: (row["strategy"], row["folder"]))


def summarize_smoothing_risk(
    rows: list[dict[str, str]],
    warning_gradient_ratio: float,
    high_risk_gradient_ratio: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    out: list[dict[str, Any]] = []
    for strategy, strategy_rows in grouped.items():
        grad_ratios = [f(row, "gradient_ratio_to_noisy") for row in strategy_rows]
        gains = [f(row, "psnr_gain_vs_noisy") for row in strategy_rows]
        risky = [
            row
            for row in strategy_rows
            if f(row, "gradient_ratio_to_noisy") < warning_gradient_ratio
        ]
        high_risk = [
            row
            for row in strategy_rows
            if f(row, "gradient_ratio_to_noisy") < high_risk_gradient_ratio
        ]
        gain_with_risk = [
            row
            for row in risky
            if f(row, "psnr_gain_vs_noisy") > 0
        ]
        worst = min(strategy_rows, key=lambda row: f(row, "gradient_ratio_to_noisy"))
        out.append(
            {
                "strategy": strategy,
                "sample_count": len(strategy_rows),
                "mean_psnr_gain": mean(gains),
                "mean_gradient_ratio_to_noisy": mean(grad_ratios),
                "min_gradient_ratio_to_noisy": f(worst, "gradient_ratio_to_noisy"),
                "warning_count_grad_lt_0p95": len(risky),
                "high_risk_count_grad_lt_0p85": len(high_risk),
                "positive_gain_with_warning_count": len(gain_with_risk),
                "worst_folder": int(float(worst["folder"])),
                "worst_pair_key": worst["pair_key"],
            }
        )
    return sorted(out, key=lambda row: float(row["mean_psnr_gain"]), reverse=True)


def mean_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(f(row, value))
    return {group_key: mean(values) for group_key, values in grouped.items()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write_report(
    path: Path,
    strategy_rows: list[dict[str, Any]],
    folder_rows: list[dict[str, Any]],
    smoothing_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    top = strategy_rows[0]
    smoothing_by_strategy = {row["strategy"]: row for row in smoothing_rows}
    high_risk = [
        row
        for row in smoothing_rows
        if int(row["high_risk_count_grad_lt_0p85"]) > 0
        or int(row["warning_count_grad_lt_0p95"]) > 0
    ]
    physical_folders = [
        row
        for row in folder_rows
        if row["strategy"] == "always_physical" and float(row["mean_psnr_gain"]) < 0
    ]

    lines = [
        "# E3.7 Evaluation Protocol and Smoothing-Risk Audit",
        "",
        "## Purpose",
        "",
        "E3.7 freezes the evaluation criteria before LOFO validation and formal",
        "network baselines. It prevents later experiments from being judged only by",
        "mean PSNR when a strategy may gain PSNR by smoothing useful gradients.",
        "",
        "## Inputs",
        "",
        f"- Blend metrics: `{args.blend_metrics_csv}`",
        f"- Visual metrics: `{args.visual_metrics_csv}`",
        f"- Warning gradient ratio threshold: `{args.warning_gradient_ratio}`",
        f"- High-risk gradient ratio threshold: `{args.high_risk_gradient_ratio}`",
        "",
        "## Fixed Evaluation Protocol",
        "",
        "For E3.8 and E4.*, every strategy should report:",
        "",
        "1. pair-level PSNR and SSIM;",
        "2. folder-level mean PSNR gain;",
        "3. positive pair fraction;",
        "4. positive and negative folder counts;",
        "5. residual mean and residual standard deviation;",
        "6. representative best, median, and worst samples;",
        "7. gradient ratio to noisy input for smoothing-risk checks;",
        "8. strategy source decisions for condition-aware methods.",
        "",
        "A candidate strategy is not acceptable as a paper claim if it improves",
        "mean PSNR but introduces negative folder-level behavior or repeated",
        "gradient-ratio warnings without visual qualification.",
        "",
        "## Current Strategy Summary",
        "",
        "| Strategy | Mean folder gain | Positive folders | Positive pairs | Mean residual std reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in strategy_rows:
        lines.append(
            "| {strategy} | {gain:.6f} | {pos}/{folders} | {pairs:.3f} | {resid:.8f} |".format(
                strategy=row["strategy"],
                gain=float(row["mean_folder_psnr_gain"]),
                pos=int(row["positive_folder_count"]),
                folders=int(row["folder_count"]),
                pairs=float(row["positive_pair_fraction"]),
                resid=float(row["mean_residual_std_reduction_vs_noisy"]),
            )
        )

    lines.extend(
        [
            "",
            "## Smoothing-Risk Summary",
            "",
            "| Strategy | Mean gain | Mean grad/noisy | Min grad/noisy | Warnings | High risk | Worst sample |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in smoothing_rows:
        lines.append(
            "| {strategy} | {gain:.6f} | {mean_grad:.4f} | {min_grad:.4f} | {warn} | {high} | folder {folder} `{pair}` |".format(
                strategy=row["strategy"],
                gain=float(row["mean_psnr_gain"]),
                mean_grad=float(row["mean_gradient_ratio_to_noisy"]),
                min_grad=float(row["min_gradient_ratio_to_noisy"]),
                warn=int(row["warning_count_grad_lt_0p95"]),
                high=int(row["high_risk_count_grad_lt_0p85"]),
                folder=int(row["worst_folder"]),
                pair=row["worst_pair_key"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest current diagnostic strategy is `{top['strategy']}` with",
            f"{float(top['mean_folder_psnr_gain']):.6f} dB mean folder PSNR gain.",
            "This does not yet prove deployable generalization because thresholds were",
            "estimated on the same ten folders. LOFO is therefore mandatory next.",
            "",
        ]
    )
    if physical_folders:
        folders = ", ".join(str(row["folder"]) for row in physical_folders)
        lines.extend(
            [
                f"`always_physical` has negative folder-level behavior in folders {folders}.",
                "This supports condition-aware selection over a globally stronger denoiser.",
                "",
            ]
        )
    if high_risk:
        risk_names = ", ".join(row["strategy"] for row in high_risk)
        physical_risk = smoothing_by_strategy.get("physical")
        if physical_risk:
            lines.append(
                "The representative-sample audit flags smoothing risk for physical-style "
                f"outputs; the worst physical sample is folder {physical_risk['worst_folder']} "
                f"`{physical_risk['worst_pair_key']}` with grad/noisy "
                f"{float(physical_risk['min_gradient_ratio_to_noisy']):.4f}."
            )
        else:
            lines.append(f"Strategies with smoothing warnings: {risk_names}.")
        lines.append("")

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "Supported now:",
            "",
            "- condition-aware selection reduces current diagnostic condition failures;",
            "- strong physical-style denoising is useful mainly in high-condition folders;",
            "- gradient-ratio checks are necessary because PSNR gain can coincide with smoothing.",
            "",
            "Not supported now:",
            "",
            "- q50 or q40-q60 thresholds are deployable classifiers;",
            "- the current small CNN restores missing weak-light details;",
            "- linear blend is meaningfully better than hard q50 switching.",
            "",
            "## Next Step",
            "",
            "Run E3.8 LOFO. Thresholds and blend intervals must be selected only from",
            "training folders, then evaluated on the held-out folder. The E3.7 metrics",
            "above are the required reporting fields for that LOFO report.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
