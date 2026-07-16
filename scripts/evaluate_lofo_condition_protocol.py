"""Run LOFO validation for ICCD condition-aware p99/physical selection."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_condition_blend import load_model
from evaluate_manifest_denoiser_checkpoint import load_tiff_tensor
from src.iccd_eval.metrics import image_quality


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = read_csv(Path(args.pairs_csv))
    conditions = load_condition_rows(Path(args.condition_score_csv))
    folders = sorted({int(float(row["folder"])) for row in pairs})

    p99_model = load_model(Path(args.p99_checkpoint), device)
    physical_model = load_model(Path(args.physical_checkpoint), device)

    metric_cache = build_metric_cache(pairs, conditions, p99_model, physical_model, device, args)
    pair_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for heldout_folder in folders:
        train_folders = [folder for folder in folders if folder != heldout_folder]
        candidates = build_candidates(conditions, train_folders, args)
        hard_candidates = [item for item in candidates if item["group"] == "hard"]
        linear_candidates = [item for item in candidates if item["group"] == "linear"]
        selected_hard = select_best_candidate(hard_candidates, metric_cache, train_folders)
        selected_linear = select_best_candidate(linear_candidates, metric_cache, train_folders)
        eval_candidates = [
            constant_candidate("always_noisy", "noisy"),
            constant_candidate("always_p99", "p99"),
            constant_candidate("always_physical", "physical"),
            rename_candidate(selected_hard, "lofo_best_hard"),
            rename_candidate(selected_linear, "lofo_best_linear"),
        ]

        selection_rows.extend(
            [
                selection_row(heldout_folder, "lofo_best_hard", selected_hard, metric_cache, train_folders),
                selection_row(heldout_folder, "lofo_best_linear", selected_linear, metric_cache, train_folders),
            ]
        )
        heldout_pairs = [row for row in pairs if int(float(row["folder"])) == heldout_folder]
        for candidate in eval_candidates:
            for pair in heldout_pairs:
                pair_rows.append(evaluate_pair_candidate(pair, candidate, metric_cache, heldout_folder))

    strategy_summary = summarize_strategy(pair_rows)
    folder_summary = summarize_folder(pair_rows)

    pair_csv = output_dir / "lofo_pair_metrics.csv"
    selection_csv = output_dir / "lofo_fold_selection.csv"
    strategy_csv = output_dir / "lofo_strategy_summary.csv"
    folder_csv = output_dir / "lofo_folder_summary.csv"
    config_json = output_dir / "lofo_config.json"
    report_md = output_dir / "lofo_report.md"

    write_csv(pair_rows, pair_csv)
    write_csv(selection_rows, selection_csv)
    write_csv(strategy_summary, strategy_csv)
    write_csv(folder_summary, folder_csv)
    write_json(
        config_json,
        {
            "experiment_id": "E3.8-LOFO",
            "pairs_csv": args.pairs_csv,
            "condition_score_csv": args.condition_score_csv,
            "p99_checkpoint": args.p99_checkpoint,
            "physical_checkpoint": args.physical_checkpoint,
            "hard_quantiles": args.hard_quantiles,
            "linear_quantile_pairs": args.linear_quantile_pairs,
            "warning_gradient_ratio": args.warning_gradient_ratio,
            "git_commit": git_commit(),
        },
    )
    write_report(report_md, strategy_summary, folder_summary, selection_rows, args)

    print(f"Wrote pair metrics: {pair_csv}")
    print(f"Wrote fold selection: {selection_csv}")
    print(f"Wrote strategy summary: {strategy_csv}")
    print(f"Wrote folder summary: {folder_csv}")
    print(f"Wrote config: {config_json}")
    print(f"Wrote report: {report_md}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="reports/gated_iccd_20260319_surrogate_pairs/pairs.csv")
    parser.add_argument("--condition-score-csv", default="reports/e3_5_condition_score/condition_score_folders.csv")
    parser.add_argument("--p99-checkpoint", default="reports/e3_manifest_baseline_smallcnn_100ep/checkpoints/best.pth")
    parser.add_argument("--physical-checkpoint", default="reports/e3_manifest_baseline_physical_scale_100ep/checkpoints/best.pth")
    parser.add_argument("--output-dir", default="reports/e3_8_lofo_condition_protocol")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--hard-quantiles", nargs="*", default=["0.3", "0.4", "0.5", "0.6", "0.7"])
    parser.add_argument(
        "--linear-quantile-pairs",
        nargs="*",
        default=["0.4,0.6", "0.3,0.7", "0.2,0.8"],
    )
    parser.add_argument("--warning-gradient-ratio", type=float, default=0.95)
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


def load_condition_rows(path: Path) -> dict[int, dict[str, float]]:
    rows = read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        folder = int(float(row["folder"]))
        out[folder] = {key: float(value) for key, value in row.items() if key != "folder"}
    return out


@torch.no_grad()
def build_metric_cache(
    pairs: list[dict[str, str]],
    conditions: dict[int, dict[str, float]],
    p99_model: torch.nn.Module,
    physical_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        clean = load_tiff_tensor(pair["clean_path"], args.range_max).to(device)
        noisy = load_tiff_tensor(pair["noisy_path"], args.range_max).to(device)
        p99 = p99_model(noisy).clamp(0.0, 1.0)
        physical = physical_model(noisy).clamp(0.0, 1.0)
        folder = int(float(pair["folder"]))
        score = conditions[folder]["condition_score"]
        cache[pair["pair_key"]] = {
            "folder": folder,
            "condition_score": score,
            "clean": clean.cpu(),
            "noisy": noisy.cpu(),
            "p99": p99.cpu(),
            "physical": physical.cpu(),
            "noisy_quality": image_quality(noisy, clean, data_range=1.0),
            "noisy_gradient": gradient_mean(noisy.cpu().numpy()),
        }
    return cache


def build_candidates(
    conditions: dict[int, dict[str, float]],
    train_folders: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    train_scores = np.asarray([conditions[folder]["condition_score"] for folder in train_folders], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for quantile_text in args.hard_quantiles:
        quantile = float(quantile_text)
        threshold = float(np.quantile(train_scores, quantile))
        candidates.append(
            {
                "strategy": f"hard_q{int(quantile * 100):02d}",
                "group": "hard",
                "kind": "hard",
                "threshold": threshold,
                "low": "",
                "high": "",
            }
        )
    for item in args.linear_quantile_pairs:
        low_q, high_q = [float(part) for part in item.split(",", 1)]
        low = float(np.quantile(train_scores, low_q))
        high = float(np.quantile(train_scores, high_q))
        candidates.append(
            {
                "strategy": f"linear_q{int(low_q * 100):02d}_q{int(high_q * 100):02d}",
                "group": "linear",
                "kind": "linear",
                "threshold": "",
                "low": low,
                "high": high,
            }
        )
    return candidates


def constant_candidate(name: str, source: str) -> dict[str, Any]:
    return {
        "strategy": name,
        "selected_candidate": name,
        "group": "constant",
        "kind": "constant",
        "source": source,
        "threshold": "",
        "low": "",
        "high": "",
        "train_mean_folder_psnr_gain": "",
        "train_negative_folder_count": "",
    }


def rename_candidate(candidate: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    renamed = dict(candidate)
    renamed["selected_candidate"] = candidate["strategy"]
    renamed["strategy"] = strategy_name
    return renamed


def select_best_candidate(
    candidates: list[dict[str, Any]],
    metric_cache: dict[str, dict[str, Any]],
    train_folders: list[int],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        folder_gains = candidate_folder_gains(candidate, metric_cache, train_folders)
        gains = list(folder_gains.values())
        scored_candidate = dict(candidate)
        scored_candidate["train_mean_folder_psnr_gain"] = mean(gains)
        scored_candidate["train_negative_folder_count"] = sum(value < 0 for value in gains)
        scored_candidate["train_positive_folder_count"] = sum(value > 0 for value in gains)
        scored.append(scored_candidate)
    return sorted(
        scored,
        key=lambda row: (
            -float(row["train_mean_folder_psnr_gain"]),
            int(row["train_negative_folder_count"]),
            row["strategy"],
        ),
    )[0]


def candidate_folder_gains(
    candidate: dict[str, Any],
    metric_cache: dict[str, dict[str, Any]],
    folders: list[int],
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for pair_key, item in metric_cache.items():
        folder = int(item["folder"])
        if folder not in folders:
            continue
        quality, _alpha = candidate_quality(candidate, item)
        grouped[folder].append(quality["psnr"] - item["noisy_quality"]["psnr"])
    return {folder: mean(values) for folder, values in grouped.items()}


def selection_row(
    heldout_folder: int,
    strategy: str,
    candidate: dict[str, Any],
    metric_cache: dict[str, dict[str, Any]],
    train_folders: list[int],
) -> dict[str, Any]:
    folder_gains = candidate_folder_gains(candidate, metric_cache, train_folders)
    return {
        "heldout_folder": heldout_folder,
        "strategy": strategy,
        "selected_candidate": candidate["strategy"],
        "kind": candidate["kind"],
        "threshold": candidate.get("threshold", ""),
        "low": candidate.get("low", ""),
        "high": candidate.get("high", ""),
        "train_mean_folder_psnr_gain": mean(list(folder_gains.values())),
        "train_positive_folder_count": sum(value > 0 for value in folder_gains.values()),
        "train_negative_folder_count": sum(value < 0 for value in folder_gains.values()),
    }


def evaluate_pair_candidate(
    pair: dict[str, str],
    candidate: dict[str, Any],
    metric_cache: dict[str, dict[str, Any]],
    heldout_folder: int,
) -> dict[str, Any]:
    item = metric_cache[pair["pair_key"]]
    quality, alpha = candidate_quality(candidate, item)
    noisy_quality = item["noisy_quality"]
    grad = gradient_mean(candidate_image(candidate, item).numpy())
    grad_ratio = grad / max(float(item["noisy_gradient"]), 1e-12)
    return {
        "heldout_folder": heldout_folder,
        "strategy": candidate["strategy"],
        "selected_candidate": candidate.get("selected_candidate", candidate["strategy"]),
        "pair_key": pair["pair_key"],
        "folder": int(item["folder"]),
        "condition_score": item["condition_score"],
        "alpha_physical": alpha,
        "threshold": candidate.get("threshold", ""),
        "low": candidate.get("low", ""),
        "high": candidate.get("high", ""),
        "psnr": quality["psnr"],
        "ssim": quality["ssim"],
        "residual_mean": quality["residual_mean"],
        "residual_std": quality["residual_std"],
        "noisy_psnr": noisy_quality["psnr"],
        "noisy_ssim": noisy_quality["ssim"],
        "psnr_gain": quality["psnr"] - noisy_quality["psnr"],
        "ssim_gain": quality["ssim"] - noisy_quality["ssim"],
        "gradient_ratio_to_noisy": grad_ratio,
    }


def candidate_quality(candidate: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, float], float]:
    pred = candidate_image(candidate, item)
    quality = image_quality(pred, item["clean"], data_range=1.0)
    return quality, alpha_for_candidate(candidate, item)


def candidate_image(candidate: dict[str, Any], item: dict[str, Any]) -> torch.Tensor:
    kind = candidate["kind"]
    if kind == "constant":
        return item[candidate["source"]]
    alpha = alpha_for_candidate(candidate, item)
    return ((1.0 - alpha) * item["p99"] + alpha * item["physical"]).clamp(0.0, 1.0)


def alpha_for_candidate(candidate: dict[str, Any], item: dict[str, Any]) -> float:
    kind = candidate["kind"]
    score = float(item["condition_score"])
    if kind == "constant":
        source = candidate["source"]
        if source == "physical":
            return 1.0
        return 0.0
    if kind == "hard":
        return 1.0 if score >= float(candidate["threshold"]) else 0.0
    if kind == "linear":
        low = float(candidate["low"])
        high = float(candidate["high"])
        return float(np.clip((score - low) / max(high - low, 1e-12), 0.0, 1.0))
    raise ValueError(f"Unsupported candidate kind: {kind}")


def summarize_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    out: list[dict[str, Any]] = []
    for strategy, strategy_rows in grouped.items():
        gains = [float(row["psnr_gain"]) for row in strategy_rows]
        folder_gains = mean_by(strategy_rows, "folder", "psnr_gain")
        grad_ratios = [float(row["gradient_ratio_to_noisy"]) for row in strategy_rows]
        out.append(
            {
                "strategy": strategy,
                "pair_count": len(strategy_rows),
                "folder_count": len(folder_gains),
                "mean_pair_psnr_gain": mean(gains),
                "std_pair_psnr_gain": std(gains),
                "positive_pair_fraction": sum(value > 0 for value in gains) / len(gains),
                "mean_folder_psnr_gain": mean(list(folder_gains.values())),
                "positive_folder_count": sum(value > 0 for value in folder_gains.values()),
                "negative_folder_count": sum(value < 0 for value in folder_gains.values()),
                "mean_ssim_gain": mean([float(row["ssim_gain"]) for row in strategy_rows]),
                "mean_gradient_ratio_to_noisy": mean(grad_ratios),
                "min_gradient_ratio_to_noisy": min(grad_ratios),
                "warning_count_grad_lt_0p95": sum(value < 0.95 for value in grad_ratios),
            }
        )
    return sorted(out, key=lambda row: (-float(row["mean_folder_psnr_gain"]), int(row["negative_folder_count"])))


def summarize_folder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], int(row["folder"]))].append(row)
    out: list[dict[str, Any]] = []
    for (strategy, folder), folder_rows in grouped.items():
        gains = [float(row["psnr_gain"]) for row in folder_rows]
        grad_ratios = [float(row["gradient_ratio_to_noisy"]) for row in folder_rows]
        selected = sorted({str(row["selected_candidate"]) for row in folder_rows})
        out.append(
            {
                "strategy": strategy,
                "folder": folder,
                "selected_candidates": ";".join(selected),
                "pair_count": len(folder_rows),
                "mean_psnr_gain": mean(gains),
                "positive_pair_fraction": sum(value > 0 for value in gains) / len(gains),
                "mean_gradient_ratio_to_noisy": mean(grad_ratios),
                "min_gradient_ratio_to_noisy": min(grad_ratios),
            }
        )
    return sorted(out, key=lambda row: (row["strategy"], row["folder"]))


def mean_by(rows: list[dict[str, Any]], key: str, value: str) -> dict[Any, float]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(float(row[value]))
    return {group_key: mean(values) for group_key, values in grouped.items()}


def gradient_mean(image: np.ndarray) -> float:
    arr = np.squeeze(image).astype(np.float64)
    gy, gx = np.gradient(arr)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


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
    selection_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# E3.8 LOFO Condition Protocol Validation",
        "",
        "This report evaluates condition-aware p99/physical selection with",
        "leave-one-folder-out validation. For each held-out folder, hard thresholds",
        "and linear blend intervals are selected only on the other folders.",
        "",
        "## Inputs",
        "",
        f"- Pair manifest: `{args.pairs_csv}`",
        f"- Condition score CSV: `{args.condition_score_csv}`",
        f"- p99 checkpoint: `{args.p99_checkpoint}`",
        f"- physical checkpoint: `{args.physical_checkpoint}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Mean folder gain | Positive folders | Negative folders | Positive pairs | Mean grad/noisy | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategy_rows:
        lines.append(
            "| {strategy} | {gain:.6f} | {pos}/{folders} | {neg} | {pairs:.3f} | {grad:.4f} | {warn} |".format(
                strategy=row["strategy"],
                gain=float(row["mean_folder_psnr_gain"]),
                pos=int(row["positive_folder_count"]),
                folders=int(row["folder_count"]),
                neg=int(row["negative_folder_count"]),
                pairs=float(row["positive_pair_fraction"]),
                grad=float(row["mean_gradient_ratio_to_noisy"]),
                warn=int(row["warning_count_grad_lt_0p95"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fold Selection",
            "",
            "| Held-out folder | Strategy | Selected candidate | Train mean gain | Train negative folders |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in selection_rows:
        lines.append(
            "| {folder} | {strategy} | {candidate} | {gain:.6f} | {neg} |".format(
                folder=int(row["heldout_folder"]),
                strategy=row["strategy"],
                candidate=row["selected_candidate"],
                gain=float(row["train_mean_folder_psnr_gain"]),
                neg=int(row["train_negative_folder_count"]),
            )
        )

    best = strategy_rows[0]
    negative = [
        row
        for row in folder_rows
        if row["strategy"] in {"lofo_best_hard", "lofo_best_linear"}
        and float(row["mean_psnr_gain"]) < 0
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best LOFO strategy by mean folder gain is `{best['strategy']}` with",
            f"{float(best['mean_folder_psnr_gain']):.6f} dB. This is a stronger",
            "generalization check than the earlier same-folder diagnostic q50/q40-q60",
            "rules because each threshold is selected without the held-out folder.",
            "",
        ]
    )
    if negative:
        lines.append("Negative held-out folders remain for:")
        lines.append("")
        for row in negative:
            lines.append(
                f"- `{row['strategy']}` folder {row['folder']}: {float(row['mean_psnr_gain']):.6f} dB"
            )
        lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "Supported if used carefully:",
            "",
            "- condition-aware selection can be evaluated without held-out threshold leakage;",
            "- LOFO summaries should replace same-folder q50/q40-q60 results in the main evidence table.",
            "",
            "Still not supported:",
            "",
            "- a deployable universal threshold;",
            "- missing-detail restoration;",
            "- model novelty claims based only on small-CNN checkpoints.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
