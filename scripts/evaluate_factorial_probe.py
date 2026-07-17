"""Evaluate one factorial residual-small-CNN probe on both split references."""

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
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from audit_surrogate_reference_reliability import build_references, gradient_mean
from evaluate_condition_blend import load_model
from evaluate_manifest_denoiser_checkpoint import load_tiff_tensor
from src.iccd_eval.metrics import image_quality


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    eval_cfg = config["real_evaluation"]
    source_cfg = config["source"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = read_csv(resolve_path(eval_cfg["pair_manifest"]))
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    folders = sorted({int(float(row["folder"])) for row in pairs})
    reference_cfg = {
        "reference_frame_start": int(eval_cfg["odd_reference_frames"][0]),
        "reference_frame_end": int(eval_cfg["even_reference_frames"][1]),
        "crop_size": int(eval_cfg["crop_size"]),
        "range_max": float(source_cfg["range_max"]),
    }
    cache_dir = Path(config["output_root"]) / "reference_cache"
    references, reference_summary = load_or_build_references(
        cache_dir, Path(eval_cfg["raw_root"]), folders, reference_cfg
    )
    model = load_model(Path(args.checkpoint), device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("config", {})
    metrics = evaluate(model, pairs, references, device, float(source_cfg["range_max"]), args)
    write_csv(metrics, output_dir / "probe_metrics.csv")
    write_csv(reference_summary, output_dir / "reference_summary.csv")
    summary = summarize(metrics, args, checkpoint_cfg)
    write_json(output_dir / "probe_summary.json", summary)
    save_ranked_panels(model, pairs, references, metrics, device, float(source_cfg["range_max"]), output_dir / "panels")
    write_report(output_dir / "probe_report.md", summary, args)
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5_noise_factorial.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["P-L", "P-H", "H-L", "H-H"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_or_build_references(
    cache_dir: Path,
    raw_root: Path,
    folders: list[int],
    reference_cfg: dict[str, Any],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    summary_path = cache_dir / "reference_summary.csv"
    expected = [cache_dir / f"folder_{folder}_{name}.npy" for folder in folders for name in ("reference_a_odd", "reference_b_even")]
    if summary_path.exists() and all(path.exists() for path in expected):
        references: dict[int, dict[str, torch.Tensor]] = {}
        for folder in folders:
            references[folder] = {
                name: torch.from_numpy(np.load(cache_dir / f"folder_{folder}_{name}.npy"))
                for name in ("reference_a_odd", "reference_b_even")
            }
        return references, read_csv(summary_path)
    references, summary = build_references(raw_root, folders, reference_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for folder, items in references.items():
        for name, tensor in items.items():
            np.save(cache_dir / f"folder_{folder}_{name}.npy", tensor.numpy())
    write_csv(summary, summary_path)
    return references, summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    pairs: list[dict[str, str]],
    references: dict[int, dict[str, torch.Tensor]],
    device: torch.device,
    range_max: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for pair in pairs:
        folder = int(float(pair["folder"]))
        noisy = load_tiff_tensor(pair["noisy_path"], range_max).to(device)
        prediction = model(noisy).clamp(0.0, 1.0)
        noisy_gradient = gradient_mean(noisy)
        prediction_gradient = gradient_mean(prediction)
        for reference_name, reference_cpu in references[folder].items():
            reference = reference_cpu.to(device)
            quality = image_quality(prediction, reference, data_range=1.0)
            noisy_quality = image_quality(noisy, reference, data_range=1.0)
            rows.append(
                {
                    "variant": args.variant,
                    "seed": args.seed,
                    "reference": reference_name,
                    "pair_key": pair["pair_key"],
                    "folder": folder,
                    "psnr": quality["psnr"],
                    "psnr_gain": quality["psnr"] - noisy_quality["psnr"],
                    "ssim": quality["ssim"],
                    "ssim_gain": quality["ssim"] - noisy_quality["ssim"],
                    "gradient_ratio_to_noisy": prediction_gradient / max(noisy_gradient, 1e-12),
                    "residual_mean": quality["residual_mean"],
                    "residual_std": quality["residual_std"],
                    "brightness_bias_to_reference": float(torch.mean(prediction - reference).cpu()),
                    "prediction_mean_minus_noisy_mean": float(torch.mean(prediction - noisy).cpu()),
                    "removed_residual_std": float(torch.std(noisy - prediction, unbiased=False).cpu()),
                    "noisy_psnr": noisy_quality["psnr"],
                    "noisy_ssim": noisy_quality["ssim"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, checkpoint_cfg: dict[str, Any]) -> dict[str, Any]:
    by_reference: dict[str, dict[str, Any]] = {}
    for reference in sorted({row["reference"] for row in rows}):
        group = [row for row in rows if row["reference"] == reference]
        folder_gains: dict[int, list[float]] = {}
        for row in group:
            folder_gains.setdefault(int(row["folder"]), []).append(float(row["psnr_gain"]))
        means = np.asarray([np.mean(values) for values in folder_gains.values()])
        pair_gains = np.asarray([float(row["psnr_gain"]) for row in group])
        by_reference[reference] = {
            "mean_folder_psnr_gain": float(np.mean(means)),
            "worst_folder_psnr_gain": float(np.min(means)),
            "positive_folder_count": int(np.sum(means > 0.0)),
            "mean_pair_psnr_gain": float(np.mean(pair_gains)),
            "positive_pair_fraction": float(np.mean(pair_gains > 0.0)),
            "mean_ssim": mean_field(group, "ssim"),
            "mean_gradient_ratio_to_noisy": mean_field(group, "gradient_ratio_to_noisy"),
            "mean_residual_std": mean_field(group, "residual_std"),
            "mean_brightness_bias": mean_field(group, "brightness_bias_to_reference"),
        }
    return {
        "variant": args.variant,
        "seed": args.seed,
        "experiment_id": checkpoint_cfg.get("experiment_id", "unknown"),
        "checkpoint_selection": "synthetic validation PSNR",
        "pair_count": len(rows) // 2,
        "references": by_reference,
    }


def mean_field(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


@torch.no_grad()
def save_ranked_panels(
    model: torch.nn.Module,
    pairs: list[dict[str, str]],
    references: dict[int, dict[str, torch.Tensor]],
    metrics: list[dict[str, Any]],
    device: torch.device,
    range_max: float,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(exist_ok=True)
    gains: dict[str, list[float]] = {}
    for row in metrics:
        gains.setdefault(row["pair_key"], []).append(float(row["psnr_gain"]))
    ranked = sorted((float(np.mean(values)), key) for key, values in gains.items())
    selected = [ranked[0][1], ranked[len(ranked) // 2][1], ranked[-1][1]]
    by_key = {row["pair_key"]: row for row in pairs}
    for label, key in zip(("worst", "median", "best"), selected):
        pair = by_key[key]
        folder = int(float(pair["folder"]))
        noisy = load_tiff_tensor(pair["noisy_path"], range_max).to(device)
        prediction = model(noisy).clamp(0.0, 1.0)
        ref_a = references[folder]["reference_a_odd"].to(device)
        ref_b = references[folder]["reference_b_even"].to(device)
        arrays = [np.squeeze(item.cpu().numpy()) for item in (noisy, prediction, noisy - prediction, ref_a, prediction - ref_a, ref_b, prediction - ref_b)]
        titles = ["input", "denoised", "predicted residual", "surrogate A", "error A", "surrogate B", "error B"]
        fig, axes = plt.subplots(1, 7, figsize=(21, 3.5), constrained_layout=True)
        lo, hi = np.percentile(arrays[0], [1, 99])
        for axis, title, image in zip(axes, titles, arrays):
            if title in {"input", "denoised", "surrogate A", "surrogate B"}:
                axis.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            else:
                limit = max(float(np.percentile(np.abs(image), 99)), 1e-6)
                axis.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.axis("off")
        fig.savefig(output_dir / f"{label}_{key}.png", dpi=140)
        plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = ["# Factorial Probe Real-Surrogate Evaluation", "", f"- Variant: `{args.variant}`", f"- Seed: `{args.seed}`", ""]
    for reference, row in summary["references"].items():
        lines.extend([
            f"## {reference}", "",
            f"- Mean folder PSNR gain: {row['mean_folder_psnr_gain']:.6f} dB",
            f"- Positive folders: {row['positive_folder_count']}/10",
            f"- Worst folder: {row['worst_folder_psnr_gain']:.6f} dB",
            f"- Gradient/noisy: {row['mean_gradient_ratio_to_noisy']:.4f}", "",
        ])
    lines.extend(["## Claim Boundary", "", "- Both references are temporal-mean surrogates, not clean ground truth.", "- This single run cannot establish a factor effect before three-seed aggregation."])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
