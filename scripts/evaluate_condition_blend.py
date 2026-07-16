"""Evaluate condition-score blending between p99 and physical checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

from evaluate_manifest_denoiser_checkpoint import load_checkpoint, load_tiff_tensor
from src.iccd_eval.metrics import image_quality
from train_manifest_denoiser_baseline import build_model, count_parameters


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    p99_model = load_model(Path(args.p99_checkpoint), device)
    physical_model = load_model(Path(args.physical_checkpoint), device)
    pairs = read_csv(Path(args.pairs_csv))
    conditions = load_condition_rows(Path(args.condition_score_csv))
    strategies = build_strategies(conditions, args.quantile_pairs)

    metric_rows = evaluate(pairs, conditions, strategies, p99_model, physical_model, device, args)
    summary_rows = summarize(metric_rows)

    metrics_csv = output_dir / "condition_blend_metrics.csv"
    summary_csv = output_dir / "condition_blend_summary.csv"
    summary_json = output_dir / "condition_blend_summary.json"
    report_path = output_dir / "condition_blend_report.md"
    write_csv(metric_rows, metrics_csv)
    write_csv(summary_rows, summary_csv)
    write_json(summary_json, {"summary": summary_rows})
    write_report(report_path, summary_rows, metrics_csv, summary_csv, args, count_parameters(p99_model), count_parameters(physical_model))

    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--condition-score-csv", required=True)
    parser.add_argument("--p99-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--quantile-pairs",
        nargs="*",
        default=["0.4,0.6", "0.3,0.7", "0.2,0.8"],
        help="Low,high score quantiles for linear p99->physical blending.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, str(device))
    config = checkpoint.get("config", {})
    model_type = str(config.get("model_type", "residual_small"))
    model = build_model(
        model_type=model_type,
        channels=int(config.get("channels", 16)),
        depth=int(config.get("depth", 3)),
        input_channels=int(config.get("input_channels", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_condition_rows(path: Path) -> dict[int, dict[str, float]]:
    rows = read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        folder = int(float(row["folder"]))
        out[folder] = {
            "condition_score": float(row["condition_score"]),
            "fano_temporal": float(row["fano_temporal"]),
        }
    return out


def build_strategies(
    conditions: dict[int, dict[str, float]],
    quantile_pairs: list[str],
) -> list[dict[str, Any]]:
    scores = np.asarray([row["condition_score"] for row in conditions.values()], dtype=np.float64)
    q50 = float(np.quantile(scores, 0.5))
    strategies: list[dict[str, Any]] = [
        {"name": "always_noisy", "kind": "constant", "alpha": 0.0, "source": "noisy"},
        {"name": "always_p99", "kind": "constant", "alpha": 0.0, "source": "p99"},
        {"name": "always_physical", "kind": "constant", "alpha": 1.0, "source": "physical"},
        {"name": "score_q50_hard_blend", "kind": "hard", "threshold": q50},
    ]
    for item in quantile_pairs:
        low_q, high_q = [float(part) for part in item.split(",", 1)]
        low = float(np.quantile(scores, low_q))
        high = float(np.quantile(scores, high_q))
        strategies.append(
            {
                "name": f"score_q{int(low_q * 100):02d}_q{int(high_q * 100):02d}_linear_blend",
                "kind": "linear",
                "low": low,
                "high": high,
            }
        )
    return strategies


@torch.no_grad()
def evaluate(
    pairs: list[dict[str, str]],
    conditions: dict[int, dict[str, float]],
    strategies: list[dict[str, Any]],
    p99_model: torch.nn.Module,
    physical_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        clean = load_tiff_tensor(pair["clean_path"], args.range_max).to(device)
        noisy = load_tiff_tensor(pair["noisy_path"], args.range_max).to(device)
        folder = int(float(pair["folder"]))
        score = conditions[folder]["condition_score"]
        p99 = p99_model(noisy).clamp(0.0, 1.0)
        physical = physical_model(noisy).clamp(0.0, 1.0)
        noisy_quality = image_quality(noisy, clean, data_range=1.0)
        for strategy in strategies:
            pred, alpha = apply_strategy(strategy, noisy, p99, physical, score)
            quality = image_quality(pred, clean, data_range=1.0)
            rows.append(
                {
                    "strategy": strategy["name"],
                    "pair_key": pair["pair_key"],
                    "folder": folder,
                    "condition_score": score,
                    "alpha_physical": alpha,
                    "psnr": quality["psnr"],
                    "ssim": quality["ssim"],
                    "residual_mean": quality["residual_mean"],
                    "residual_std": quality["residual_std"],
                    "noisy_psnr": noisy_quality["psnr"],
                    "noisy_ssim": noisy_quality["ssim"],
                    "psnr_gain": quality["psnr"] - noisy_quality["psnr"],
                    "ssim_gain": quality["ssim"] - noisy_quality["ssim"],
                }
            )
    return rows


def apply_strategy(
    strategy: dict[str, Any],
    noisy: torch.Tensor,
    p99: torch.Tensor,
    physical: torch.Tensor,
    score: float,
) -> tuple[torch.Tensor, float]:
    kind = strategy["kind"]
    if kind == "constant":
        source = strategy["source"]
        if source == "noisy":
            return noisy, 0.0
        if source == "p99":
            return p99, 0.0
        if source == "physical":
            return physical, 1.0
        raise ValueError(f"Unknown source: {source}")
    if kind == "hard":
        alpha = 1.0 if score >= float(strategy["threshold"]) else 0.0
    elif kind == "linear":
        low = float(strategy["low"])
        high = float(strategy["high"])
        alpha = float(np.clip((score - low) / max(high - low, 1e-12), 0.0, 1.0))
    else:
        raise ValueError(f"Unsupported strategy kind: {kind}")
    return (1.0 - alpha) * p99 + alpha * physical, alpha


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["strategy"]), []).append(row)
    out: list[dict[str, Any]] = []
    for strategy, group in groups.items():
        gains = np.asarray([float(row["psnr_gain"]) for row in group], dtype=np.float64)
        ssim_gains = np.asarray([float(row["ssim_gain"]) for row in group], dtype=np.float64)
        alphas = np.asarray([float(row["alpha_physical"]) for row in group], dtype=np.float64)
        by_folder: dict[int, list[float]] = {}
        for row in group:
            by_folder.setdefault(int(row["folder"]), []).append(float(row["psnr_gain"]))
        folder_gains = np.asarray([float(np.mean(values)) for values in by_folder.values()], dtype=np.float64)
        out.append(
            {
                "strategy": strategy,
                "pair_count": len(group),
                "folder_count": len(by_folder),
                "mean_pair_psnr_gain": float(np.mean(gains)),
                "std_pair_psnr_gain": float(np.std(gains)),
                "positive_pair_fraction": float(np.mean(gains > 0.0)),
                "mean_folder_psnr_gain": float(np.mean(folder_gains)),
                "positive_folder_count": int(np.sum(folder_gains > 0.0)),
                "negative_folder_count": int(np.sum(folder_gains < 0.0)),
                "mean_ssim_gain": float(np.mean(ssim_gains)),
                "mean_alpha_physical": float(np.mean(alphas)),
            }
        )
    return sorted(out, key=lambda row: (-float(row["mean_folder_psnr_gain"]), int(row["negative_folder_count"])))


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    metrics_csv: Path,
    summary_csv: Path,
    args: argparse.Namespace,
    p99_params: int,
    physical_params: int,
) -> None:
    lines = [
        "# E3.6-D Condition Blend Evaluation",
        "",
        "This report evaluates hard and continuous condition-score blending between the p99 and physical checkpoints.",
        "",
        f"- Pair manifest: `{args.pairs_csv}`",
        f"- Condition score CSV: `{args.condition_score_csv}`",
        f"- p99 checkpoint: `{args.p99_checkpoint}` ({p99_params} parameters)",
        f"- physical checkpoint: `{args.physical_checkpoint}` ({physical_params} parameters)",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## Summary",
        "",
        "| strategy | mean folder gain | positive folders | negative folders | positive pairs | mean alpha physical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {strategy} | {gain:.4f} | {pos}/{count} | {neg} | {pairs:.3f} | {alpha:.3f} |".format(
                strategy=row["strategy"],
                gain=float(row["mean_folder_psnr_gain"]),
                pos=int(row["positive_folder_count"]),
                count=int(row["folder_count"]),
                neg=int(row["negative_folder_count"]),
                pairs=float(row["positive_pair_fraction"]),
                alpha=float(row["mean_alpha_physical"]),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This is an inference-time diagnostic blend, not a trained deployable single model.",
            "- If blending beats hard gating, a future model should learn continuous condition modulation.",
            "- If hard gating remains best, the current evidence supports explicit condition selection more than residual-std-only synthetic training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
