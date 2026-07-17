"""Audit denoiser ranking and gain stability for 25/50/100-frame references."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_surrogate_reference_reliability import (
    evaluate,
    indexed_tiffs,
    load_conditions,
    load_selections,
    resolve_path,
    summarize_folders,
)
from evaluate_condition_blend import load_model


def main() -> int:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    models = cfg["fixed_models"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    pairs = read_csv(resolve_path(data["pair_manifest"]))
    references, reference_rows = build_references(cfg)
    p99 = load_model(resolve_path(models["p99_checkpoint"]), device)
    physical = load_model(resolve_path(models["physical_checkpoint"]), device)
    metrics, _ = evaluate(
        pairs,
        references,
        load_conditions(resolve_path(models["condition_score_csv"])),
        load_selections(resolve_path(models["lofo_selection_csv"])),
        p99,
        physical,
        device,
        float(data["range_max"]),
        str(models["primary_strategy"]),
    )
    folders = summarize_folders(metrics)
    strategies = summarize_strategies(metrics)
    stability = summarize_stability(folders)
    decision = decide(strategies, stability)
    write_csv(reference_rows, output_dir / "reference_definitions.csv")
    write_csv(metrics, output_dir / "reference_pair_metrics.csv")
    write_csv(folders, output_dir / "reference_folder_metrics.csv")
    write_csv(strategies, output_dir / "reference_strategy_summary.csv")
    write_csv(stability, output_dir / "reference_stability_summary.csv")
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_report(output_dir / "surrogate_reference_scaling_report.md", strategies, stability, decision)
    print(json.dumps({"device": str(device), **decision}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e7_surrogate_reference_scaling.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_references(cfg: dict[str, Any]) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    import tifffile

    data = cfg["data"]
    root = Path(data["raw_root"])
    crop_size = int(data["crop_size"])
    range_max = float(data["range_max"])
    groups = {name: tuple(int(value) for value in bounds) for name, bounds in data["reference_groups"].items()}
    boundaries = {0, *(value for bounds in groups.values() for value in (bounds[0] - 1, bounds[1]))}
    maximum_frame = max(end for _, end in groups.values())
    references: dict[int, dict[str, torch.Tensor]] = {}
    rows = []
    for folder in [int(value) for value in data["folders"]]:
        indexed = indexed_tiffs(root / str(folder))
        prefix: dict[int, np.ndarray] = {}
        accumulator: np.ndarray | None = None
        for index in range(1, maximum_frame + 1):
            image = tifffile.memmap(indexed[index])
            size = min(crop_size, image.shape[0], image.shape[1])
            top = (image.shape[0] - size) // 2
            left = (image.shape[1] - size) // 2
            crop = np.asarray(image[top : top + size, left : left + size], dtype=np.float64)
            if accumulator is None:
                accumulator = np.zeros_like(crop, dtype=np.float64)
                prefix[0] = accumulator.copy()
            accumulator += crop
            if index in boundaries:
                prefix[index] = accumulator.copy()
        references[folder] = {}
        for name, (start, end) in groups.items():
            count = end - start + 1
            mean = np.clip((prefix[end] - prefix[start - 1]) / count / range_max, 0.0, 1.0).astype(np.float32)
            references[folder][name] = torch.from_numpy(mean[None, None].copy())
            rows.append({"folder": folder, "reference": name, "frame_start": start, "frame_end": end, "frame_count": count})
    return references, rows


def reference_size(name: str) -> int:
    if name.startswith("ref25"):
        return 25
    if name.startswith("ref50"):
        return 50
    return 100


def summarize_strategies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["reference"]), str(row["strategy"]))].append(row)
    result = []
    for (reference, strategy), group in groups.items():
        folder_groups: dict[int, list[float]] = defaultdict(list)
        for row in group:
            folder_groups[int(row["folder"])].append(float(row["psnr_gain"]))
        folder_gains = np.asarray([np.mean(values) for values in folder_groups.values()])
        result.append({
            "reference": reference,
            "reference_frames": reference_size(reference),
            "strategy": strategy,
            "mean_folder_psnr_gain": float(np.mean(folder_gains)),
            "worst_folder_psnr_gain": float(np.min(folder_gains)),
            "positive_folder_count": int(np.sum(folder_gains > 0)),
            "mean_ssim_gain": float(np.mean([float(row["ssim_gain"]) for row in group])),
            "mean_gradient_ratio_to_noisy": float(np.mean([float(row["gradient_ratio_to_noisy"]) for row in group])),
        })
    for reference in sorted({row["reference"] for row in result}):
        subset = sorted((row for row in result if row["reference"] == reference), key=lambda row: -float(row["mean_folder_psnr_gain"]))
        for rank, row in enumerate(subset, start=1):
            row["gain_rank"] = rank
    return sorted(result, key=lambda row: (reference_size(str(row["reference"])), str(row["reference"]), int(row["gain_rank"])))


def summarize_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["strategy"]), int(row["folder"]))].append(row)
    result = []
    for (strategy, folder), group in groups.items():
        gains = np.asarray([float(row["mean_psnr_gain"]) for row in group])
        signs = np.sign(gains)
        result.append({
            "strategy": strategy,
            "folder": folder,
            "reference_count": len(group),
            "gain_mean_db": float(np.mean(gains)),
            "gain_sd_across_references_db": float(np.std(gains, ddof=1)),
            "gain_range_across_references_db": float(np.ptp(gains)),
            "sign_agreement_fraction": float(max(np.mean(signs >= 0), np.mean(signs <= 0))),
        })
    return sorted(result, key=lambda row: (row["strategy"], row["folder"]))


def decide(strategies: list[dict[str, Any]], stability: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [row for row in stability if row["strategy"] in {"always_p99", "always_physical"}]
    primary = [row for row in stability if row["strategy"] == "lofo_best_linear"]
    rankings: dict[str, list[str]] = defaultdict(list)
    for row in strategies:
        rankings[str(row["reference"])].append(str(row["strategy"]))
    best = {reference: values[0] for reference, values in rankings.items()}
    return {
        "fixed_model_folder_sign_stability_mean": float(np.mean([row["sign_agreement_fraction"] for row in fixed])),
        "primary_folder_sign_stability_mean": float(np.mean([row["sign_agreement_fraction"] for row in primary])),
        "maximum_primary_reference_range_db": float(max(row["gain_range_across_references_db"] for row in primary)),
        "best_strategy_by_reference": best,
        "ranking_top_is_stable": len(set(best.values())) == 1,
        "status": "LIMITED_USE_FOR_RELATIVE_COMPARISON",
        "absolute_ground_truth_status": "NOT_SUPPORTED",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, strategies: list[dict[str, Any]], stability: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    lines = [
        "# E7 Surrogate Reference Scaling Audit",
        "",
        "References use disjoint 25-frame quarters, disjoint 50-frame halves, and the first 100 frames. Test inputs remain frames 101-200 from the existing held-out manifest.",
        "",
        "## Strategy Summary",
        "",
        "| reference | frames | strategy | folder gain dB | worst folder dB | positive folders | rank |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in strategies:
        lines.append(f"| {row['reference']} | {row['reference_frames']} | {row['strategy']} | {row['mean_folder_psnr_gain']:.4f} | {row['worst_folder_psnr_gain']:.4f} | {row['positive_folder_count']} | {row['gain_rank']} |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Fixed-model folder sign stability: {decision['fixed_model_folder_sign_stability_mean']:.3f}",
        f"- LOFO-linear folder sign stability: {decision['primary_folder_sign_stability_mean']:.3f}",
        f"- Maximum LOFO-linear gain range across references: {decision['maximum_primary_reference_range_db']:.4f} dB",
        f"- Top-ranked strategy stable: {decision['ranking_top_is_stable']}",
        "- These references can support cautious relative comparisons. They cannot support absolute clean-image recovery claims because all means retain the repeatable scene-plus-stable-pattern component.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
