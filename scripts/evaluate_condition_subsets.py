"""Evaluate denoising behavior on low/high ICCD condition subsets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_rows = read_csv(Path(args.physical_eval_csv))
    p99_rows = read_csv(Path(args.p99_eval_csv))
    condition_rows = load_conditions(Path(args.condition_summary_csv), args.condition_model_label)
    threshold = condition_threshold(condition_rows, args.split_metric, args.quantile)
    p99_by_key = {row["pair_key"]: row for row in p99_rows}

    pair_rows = build_pair_rows(physical_rows, p99_by_key, condition_rows, args.split_metric, threshold)
    summary_rows = summarize(pair_rows)
    folder_rows = summarize_folders(pair_rows)
    report_path = output_dir / "condition_subset_report.md"
    pair_csv = output_dir / "condition_subset_pair_metrics.csv"
    summary_csv = output_dir / "condition_subset_summary.csv"
    folder_csv = output_dir / "condition_subset_folder_summary.csv"
    write_csv(pair_rows, pair_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(folder_rows, folder_csv)
    write_report(report_path, pair_csv, summary_csv, folder_csv, summary_rows, folder_rows, threshold, args)

    print(f"Wrote pair metrics: {pair_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote folder summary: {folder_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-eval-csv", required=True)
    parser.add_argument("--p99-eval-csv", required=True)
    parser.add_argument("--condition-summary-csv", required=True)
    parser.add_argument("--condition-model-label", default="physical")
    parser.add_argument("--split-metric", default="fano_temporal")
    parser.add_argument("--quantile", type=float, default=0.4)
    parser.add_argument("--output-dir", required=True)
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


def load_conditions(path: Path, model_label: str) -> dict[int, dict[str, float]]:
    conditions: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        if row.get("model_label") != model_label:
            continue
        folder = int(float(row["folder"]))
        conditions[folder] = {key: to_float(value) for key, value in row.items() if key not in {"model_label"}}
    if not conditions:
        raise ValueError(f"No condition rows for model_label={model_label}")
    return conditions


def condition_threshold(conditions: dict[int, dict[str, float]], metric: str, quantile: float) -> float:
    values = np.asarray([row[metric] for row in conditions.values()], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError(f"Need at least 3 condition values for {metric}, found {values.size}")
    return float(np.quantile(values, quantile))


def build_pair_rows(
    physical_rows: list[dict[str, str]],
    p99_by_key: dict[str, dict[str, str]],
    conditions: dict[int, dict[str, float]],
    split_metric: str,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for physical in physical_rows:
        pair_key = physical["pair_key"]
        p99 = p99_by_key[pair_key]
        folder = int(float(physical["meta_folder"]))
        condition_value = conditions[folder][split_metric]
        subset = "high_condition" if condition_value >= threshold else "low_condition"
        strategies = {
            "noisy": (to_float(physical["noisy_psnr"]), to_float(physical["noisy_ssim"])),
            "p99": (to_float(p99["psnr"]), to_float(p99["ssim"])),
            "physical": (to_float(physical["psnr"]), to_float(physical["ssim"])),
            "condition_gate": (
                to_float(physical["psnr"]) if subset == "high_condition" else to_float(physical["noisy_psnr"]),
                to_float(physical["ssim"]) if subset == "high_condition" else to_float(physical["noisy_ssim"]),
            ),
            "condition_hybrid_p99_physical": (
                to_float(physical["psnr"]) if subset == "high_condition" else to_float(p99["psnr"]),
                to_float(physical["ssim"]) if subset == "high_condition" else to_float(p99["ssim"]),
            ),
        }
        for strategy, (psnr, ssim) in strategies.items():
            rows.append(
                {
                    "subset": subset,
                    "strategy": strategy,
                    "pair_key": pair_key,
                    "folder": folder,
                    "condition_metric": split_metric,
                    "condition_value": condition_value,
                    "threshold": threshold,
                    "psnr": psnr,
                    "ssim": ssim,
                    "noisy_psnr": to_float(physical["noisy_psnr"]),
                    "noisy_ssim": to_float(physical["noisy_ssim"]),
                    "psnr_gain": psnr - to_float(physical["noisy_psnr"]),
                    "ssim_gain": ssim - to_float(physical["noisy_ssim"]),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["subset"]), str(row["strategy"])), []).append(row)
        groups.setdefault(("all_conditions", str(row["strategy"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (subset, strategy), group_rows in sorted(groups.items()):
        gains = np.asarray([float(row["psnr_gain"]) for row in group_rows], dtype=np.float64)
        ssim_gains = np.asarray([float(row["ssim_gain"]) for row in group_rows], dtype=np.float64)
        folders = sorted({int(row["folder"]) for row in group_rows})
        folder_gain_values = []
        for folder in folders:
            folder_gains = [float(row["psnr_gain"]) for row in group_rows if int(row["folder"]) == folder]
            folder_gain_values.append(float(np.mean(folder_gains)))
        folder_gains_arr = np.asarray(folder_gain_values, dtype=np.float64)
        out.append(
            {
                "subset": subset,
                "strategy": strategy,
                "pair_count": len(group_rows),
                "folder_count": len(folders),
                "folders": " ".join(str(folder) for folder in folders),
                "mean_pair_psnr_gain": float(np.mean(gains)),
                "median_pair_psnr_gain": float(np.median(gains)),
                "positive_pair_fraction": float(np.mean(gains > 0.0)),
                "mean_folder_psnr_gain": float(np.mean(folder_gains_arr)),
                "median_folder_psnr_gain": float(np.median(folder_gains_arr)),
                "positive_folder_count": int(np.sum(folder_gains_arr > 0.0)),
                "negative_folder_count": int(np.sum(folder_gains_arr < 0.0)),
                "mean_ssim_gain": float(np.mean(ssim_gains)),
            }
        )
    return out


def summarize_folders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["subset"]), int(row["folder"]), str(row["strategy"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (subset, folder, strategy), group_rows in sorted(groups.items()):
        gains = np.asarray([float(row["psnr_gain"]) for row in group_rows], dtype=np.float64)
        out.append(
            {
                "subset": subset,
                "folder": folder,
                "strategy": strategy,
                "pair_count": len(group_rows),
                "condition_metric": group_rows[0]["condition_metric"],
                "condition_value": group_rows[0]["condition_value"],
                "mean_psnr_gain": float(np.mean(gains)),
                "min_psnr_gain": float(np.min(gains)),
                "max_psnr_gain": float(np.max(gains)),
                "positive_pair_fraction": float(np.mean(gains > 0.0)),
            }
        )
    return out


def write_report(
    path: Path,
    pair_csv: Path,
    summary_csv: Path,
    folder_csv: Path,
    summary_rows: list[dict[str, Any]],
    folder_rows: list[dict[str, Any]],
    threshold: float,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# E3.5-B Low/High Condition Subset Validation",
        "",
        "This report separates real ICCD surrogate pairs by an E1 condition statistic and evaluates each denoising strategy within low- and high-condition folders.",
        "",
        f"- Split metric: `{args.split_metric}`",
        f"- Quantile: `{args.quantile}`",
        f"- Threshold: `{threshold:.6f}`",
        f"- Pair metrics: `{pair_csv}`",
        f"- Summary: `{summary_csv}`",
        f"- Folder summary: `{folder_csv}`",
        "",
        "## Subset Summary",
        "",
            "| subset | strategy | folders | mean folder gain dB | positive folders | negative folders | positive pairs |",
            "|---|---|---|---:|---:|---:|---:|",
    ]
    order = {"all_conditions": 0, "low_condition": 1, "high_condition": 2}
    for row in sorted(summary_rows, key=lambda item: (order[str(item["subset"])], str(item["strategy"]))):
        lines.append(
            "| {subset} | {strategy} | {folders} | {gain:.4f} | {pos}/{count} | {neg} | {pairs:.3f} |".format(
                subset=row["subset"],
                strategy=row["strategy"],
                folders=row["folders"],
                gain=float(row["mean_folder_psnr_gain"]),
                pos=int(row["positive_folder_count"]),
                count=int(row["folder_count"]),
                neg=int(row["negative_folder_count"]),
                pairs=float(row["positive_pair_fraction"]),
            )
        )

    lines.extend(["", "## Boundary Folders", ""])
    boundary_rows = boundary_folder_rows(folder_rows)
    lines.extend(
        [
            "| subset | folder | condition value | physical mean gain dB | p99 mean gain dB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in boundary_rows:
        lines.append(
            "| {subset} | {folder} | {value:.6f} | {physical:.4f} | {p99:.4f} |".format(
                subset=item["subset"],
                folder=item["folder"],
                value=item["condition_value"],
                physical=item["physical_gain"],
                p99=item["p99_gain"],
            )
        )

    low_physical = find_summary(summary_rows, "low_condition", "physical")
    high_physical = find_summary(summary_rows, "high_condition", "physical")
    low_p99 = find_summary(summary_rows, "low_condition", "p99")
    high_p99 = find_summary(summary_rows, "high_condition", "p99")
    low_hybrid = find_summary(summary_rows, "low_condition", "condition_hybrid_p99_physical")
    high_hybrid = find_summary(summary_rows, "high_condition", "condition_hybrid_p99_physical")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- In low-condition folders, the physical checkpoint has {float(low_physical['mean_folder_psnr_gain']):.4f} dB mean folder gain, while p99 has {float(low_p99['mean_folder_psnr_gain']):.4f} dB.",
            f"- In high-condition folders, the physical checkpoint has {float(high_physical['mean_folder_psnr_gain']):.4f} dB mean folder gain, while p99 has {float(high_p99['mean_folder_psnr_gain']):.4f} dB.",
            f"- The hybrid strategy uses p99 in low-condition folders ({float(low_hybrid['mean_folder_psnr_gain']):.4f} dB) and physical in high-condition folders ({float(high_hybrid['mean_folder_psnr_gain']):.4f} dB).",
            "- This supports a condition-aware validation story: the stronger physical-scale denoiser is useful mainly in higher-noise ICCD conditions, while low-condition folders need either identity/p99 behavior or a separate calibration.",
            "- The split is diagnostic because it uses only ten folders. A final claim needs more conditions or a held-out condition split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def boundary_folder_rows(folder_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_folder: dict[int, dict[str, Any]] = {}
    for row in folder_rows:
        folder = int(row["folder"])
        item = by_folder.setdefault(
            folder,
            {
                "subset": row["subset"],
                "folder": folder,
                "condition_value": float(row["condition_value"]),
                "physical_gain": math.nan,
                "p99_gain": math.nan,
            },
        )
        if row["strategy"] == "physical":
            item["physical_gain"] = float(row["mean_psnr_gain"])
        if row["strategy"] == "p99":
            item["p99_gain"] = float(row["mean_psnr_gain"])
    low = [row for row in by_folder.values() if row["subset"] == "low_condition"]
    high = [row for row in by_folder.values() if row["subset"] == "high_condition"]
    result: list[dict[str, Any]] = []
    if low:
        result.append(max(low, key=lambda row: row["condition_value"]))
    if high:
        result.append(min(high, key=lambda row: row["condition_value"]))
    return result


def find_summary(rows: list[dict[str, Any]], subset: str, strategy: str) -> dict[str, Any]:
    for row in rows:
        if row["subset"] == subset and row["strategy"] == strategy:
            return row
    raise KeyError((subset, strategy))


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
