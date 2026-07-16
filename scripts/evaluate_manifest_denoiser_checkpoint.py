"""Evaluate a manifest-trained denoiser checkpoint on a pair CSV."""

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

from src.iccd_eval.metrics import image_quality
from train_manifest_denoiser_baseline import ResidualDenoiser, count_parameters, save_triplet_tiff


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)

    checkpoint = load_checkpoint(Path(args.checkpoint), args.device)
    ckpt_config = checkpoint.get("config", {})
    channels = int(args.channels or ckpt_config.get("channels", 16))
    depth = int(args.depth or ckpt_config.get("depth", 3))
    input_channels = int(args.input_channels or ckpt_config.get("input_channels", 1))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    condition_map = load_condition_map(Path(args.condition_score_csv), args) if args.condition_score_csv else {}
    args._condition_map = condition_map
    args._checkpoint_condition_column = str(ckpt_config.get("condition_column", ""))
    args._checkpoint_condition_scale = float(ckpt_config.get("condition_value_scale", 1.0) or 1.0)

    model = ResidualDenoiser(channels=channels, depth=depth, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = read_pairs(Path(args.pairs_csv))
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    metric_rows = evaluate_rows(model, rows, device, args)
    metrics_csv = output_dir / "checkpoint_eval_metrics.csv"
    summary_json = output_dir / "checkpoint_eval_summary.json"
    report_path = output_dir / "checkpoint_eval_report.md"
    write_csv(metric_rows, metrics_csv)
    summary = summarize(metric_rows)
    write_json(summary_json, summary)
    save_ranked_samples(model, rows, metric_rows, device, output_dir / "samples", args)
    write_report(report_path, metrics_csv, summary_json, args, summary, ckpt_config, count_parameters(model))

    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote summary: {summary_json}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-label", default="")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--channels", type=int, default=0)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--input-channels", type=int, default=0)
    parser.add_argument("--condition-column", default="")
    parser.add_argument("--condition-score-csv", default="")
    parser.add_argument("--condition-folder-column", default="folder")
    parser.add_argument("--condition-score-column", default="condition_score")
    parser.add_argument("--condition-value-scale", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    return parser.parse_args()


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_condition_map(path: Path, args: argparse.Namespace) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, float] = {}
    for row in rows:
        mapping[str(int(float(row[args.condition_folder_column])))] = float(row[args.condition_score_column])
    return mapping


@torch.no_grad()
def evaluate_rows(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = load_tiff_tensor(row["clean_path"], args.range_max).to(device)
        noisy = load_tiff_tensor(row["noisy_path"], args.range_max).to(device)
        model_input = make_model_input(noisy, row, args, model, device)
        pred = model(model_input).clamp(0.0, 1.0)
        quality = image_quality(pred, clean, data_range=1.0)
        noisy_quality = image_quality(noisy, clean, data_range=1.0)
        metric_rows.append(
            {
                "pair_key": row["pair_key"],
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "residual_mean": quality["residual_mean"],
                "residual_std": quality["residual_std"],
                "noisy_psnr": noisy_quality["psnr"],
                "noisy_ssim": noisy_quality["ssim"],
                "psnr_gain": quality["psnr"] - noisy_quality["psnr"],
                "ssim_gain": quality["ssim"] - noisy_quality["ssim"],
                **{f"meta_{key}": value for key, value in row.items() if key not in {"pair_key", "clean_path", "noisy_path"}},
            }
        )
    return metric_rows


def make_model_input(
    noisy: torch.Tensor,
    row: dict[str, str],
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    input_channels = int(getattr(model, "input_channels", 1))
    if input_channels == 1:
        return noisy
    if input_channels != 2:
        raise ValueError(f"Only input_channels 1 or 2 are supported, got {input_channels}")
    condition = resolve_condition_value(row, args)
    condition_tensor = torch.full(
        (noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1]),
        fill_value=condition,
        dtype=noisy.dtype,
        device=device,
    )
    return torch.cat([noisy, condition_tensor], dim=1)


def resolve_condition_value(row: dict[str, str], args: argparse.Namespace) -> float:
    checkpoint_scale = getattr(args, "_checkpoint_condition_scale", 1.0)
    scale = args.condition_value_scale if args.condition_value_scale else checkpoint_scale
    scale = scale if scale else 1.0
    column = args.condition_column or getattr(args, "_checkpoint_condition_column", "")
    if column and column in row:
        return float(row[column]) / scale
    condition_map = getattr(args, "_condition_map", {})
    folder_column = args.condition_folder_column
    if condition_map and folder_column in row:
        folder = str(int(float(row[folder_column])))
        if folder not in condition_map:
            raise KeyError(f"Folder {folder} missing from condition map")
        return float(condition_map[folder]) / scale
    raise KeyError(
        "Condition input requested but no condition value was found. "
        "Provide --condition-column or --condition-score-csv."
    )


def load_tiff_tensor(path_value: str, range_max: float) -> torch.Tensor:
    import tifffile

    path = resolve_path(path_value)
    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale TIFF, got {arr.shape}: {path}")
    arr = np.clip(arr / float(range_max), 0.0, 1.0)
    return torch.from_numpy(arr[None, None, :, :].copy())


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["psnr", "ssim", "noisy_psnr", "noisy_ssim", "psnr_gain", "ssim_gain", "residual_mean", "residual_std"]
    summary: dict[str, float] = {"pair_count": float(len(rows))}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


@torch.no_grad()
def save_ranked_samples(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    metric_rows: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    for old_file in output_dir.glob("*.tif"):
        old_file.unlink()
    ranked = sorted(metric_rows, key=lambda row: float(row["psnr_gain"]))
    if not ranked:
        return
    selected = [ranked[0], ranked[len(ranked) // 2], ranked[-1]][: args.sample_count]
    labels = ["worst_gain", "median_gain", "best_gain"]
    by_key = {row["pair_key"]: row for row in rows}
    for label, metric in zip(labels, selected):
        row = by_key[metric["pair_key"]]
        clean = load_tiff_tensor(row["clean_path"], args.range_max).to(device)
        noisy = load_tiff_tensor(row["noisy_path"], args.range_max).to(device)
        model_input = make_model_input(noisy, row, args, model, device)
        pred = model(model_input).clamp(0.0, 1.0)
        save_triplet_tiff(output_dir / f"{label}_{metric['pair_key']}.tif", clean.cpu(), noisy.cpu(), pred.cpu())


def write_report(
    path: Path,
    metrics_csv: Path,
    summary_json: Path,
    args: argparse.Namespace,
    summary: dict[str, float],
    ckpt_config: dict[str, Any],
    parameter_count: int,
) -> None:
    lines = [
        "# Denoiser Checkpoint Evaluation",
        "",
        "## Inputs",
        "",
        f"- Label: `{args.experiment_label or Path(args.checkpoint).parent.parent.name}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Pair manifest: `{args.pairs_csv}`",
        f"- Model parameters: {parameter_count}",
        f"- Training experiment ID: `{ckpt_config.get('experiment_id', 'unknown')}`",
        "",
        "## Summary",
        "",
        f"- Pair count: {int(summary['pair_count'])}",
        f"- Model PSNR/SSIM: {summary['psnr_mean']:.4f} / {summary['ssim_mean']:.6f}",
        f"- Noisy-input PSNR/SSIM: {summary['noisy_psnr_mean']:.4f} / {summary['noisy_ssim_mean']:.6f}",
        f"- PSNR gain mean/std: {summary['psnr_gain_mean']:.4f} / {summary['psnr_gain_std']:.4f}",
        f"- SSIM gain mean/std: {summary['ssim_gain_mean']:.6f} / {summary['ssim_gain_std']:.6f}",
        f"- Residual mean/std: {summary['residual_mean_mean']:.6g} / {summary['residual_std_mean']:.6g}",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Summary JSON: `{summary_json}`",
        f"- Samples: `{path.parent / 'samples'}`",
        "",
        "## Claim Boundary",
        "",
        "- This evaluates cross-domain behavior on repeated-frame ICCD surrogate pairs.",
        "- The surrogate clean image is a repeated-frame mean, not a true clean exposure.",
        "- Use this as a gate before claiming real ICCD denoising performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
