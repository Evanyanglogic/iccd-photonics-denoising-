"""Audit whether denoiser conclusions survive independent temporal-mean references."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluate_condition_blend import load_model
from evaluate_manifest_denoiser_checkpoint import load_tiff_tensor
from src.iccd_eval.metrics import image_quality

FRAME_NUMBER = re.compile(r"^(\d+)")


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    data_cfg = config["data"]
    model_cfg = config["fixed_models"]
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panels").mkdir(exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = read_csv(resolve_path(data_cfg["pair_manifest"]))
    allowed_folders = {int(value) for value in data_cfg["folders"]}
    pairs = [row for row in pairs if int(float(row["folder"])) in allowed_folders]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    folders = sorted({int(float(row["folder"])) for row in pairs})

    if args.reuse_pair_metrics:
        metric_rows = read_csv(output_dir / "reference_pair_metrics.csv")
        reference_rows = read_csv(output_dir / "surrogate_reference_summary.csv")
        image_cache: dict[str, dict[str, torch.Tensor]] = {}
    else:
        references, reference_rows = build_references(Path(data_cfg["raw_root"]), folders, data_cfg)
        conditions = load_conditions(resolve_path(model_cfg["condition_score_csv"]))
        selections = load_selections(resolve_path(model_cfg["lofo_selection_csv"]))
        p99_model = load_model(resolve_path(model_cfg["p99_checkpoint"]), device)
        physical_model = load_model(resolve_path(model_cfg["physical_checkpoint"]), device)
        metric_rows, image_cache = evaluate(
            pairs, references, conditions, selections, p99_model, physical_model,
            device, float(data_cfg["range_max"]), str(model_cfg["primary_strategy"]),
        )
    folder_rows = summarize_folders(metric_rows)
    strategy_rows = summarize_strategies(metric_rows, config)
    agreement_rows = summarize_reference_agreement(folder_rows, config)
    contrast_rows = summarize_strategy_contrasts(folder_rows, config)
    decision = decide(strategy_rows, folder_rows, agreement_rows, config)

    write_csv(reference_rows, output_dir / "surrogate_reference_summary.csv")
    write_csv(metric_rows, output_dir / "reference_pair_metrics.csv")
    write_csv(folder_rows, output_dir / "reference_folder_summary.csv")
    write_csv(strategy_rows, output_dir / "reference_strategy_summary.csv")
    write_csv(agreement_rows, output_dir / "reference_agreement.csv")
    write_csv(contrast_rows, output_dir / "reference_strategy_contrasts.csv")
    write_json(output_dir / "decision.json", decision)
    run_manifest_path = output_dir / "run_manifest.json"
    if not args.reuse_pair_metrics or not run_manifest_path.exists():
        write_json(run_manifest_path, build_run_manifest(config, args, device, len(pairs)))
    if image_cache:
        save_panels(image_cache, folder_rows, output_dir / "panels")
    write_report(
        output_dir / "surrogate_reference_reliability_report.md",
        config, decision, strategy_rows, agreement_rows, contrast_rows,
    )
    print(json.dumps(decision, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e4_surrogate_reference_reliability.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--reuse-pair-metrics", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def indexed_tiffs(folder: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = FRAME_NUMBER.match(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale image, got {array.shape}")
    size = min(crop_size, array.shape[0], array.shape[1])
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    return array[top : top + size, left : left + size]


def mean_frames(paths: list[Path], crop_size: int, range_max: float) -> torch.Tensor:
    import tifffile

    accumulator: np.ndarray | None = None
    for path in paths:
        image = center_crop(np.asarray(tifffile.imread(path), dtype=np.float64), crop_size)
        if accumulator is None:
            accumulator = np.zeros_like(image, dtype=np.float64)
        accumulator += image
    if accumulator is None:
        raise ValueError("No reference frames")
    mean = np.clip(accumulator / len(paths) / range_max, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(mean[None, None].copy())


def build_references(
    root: Path, folders: list[int], config: dict[str, Any]
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    start = int(config["reference_frame_start"])
    end = int(config["reference_frame_end"])
    crop_size = int(config["crop_size"])
    range_max = float(config["range_max"])
    references: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for folder in folders:
        indexed = indexed_tiffs(root / str(folder))
        expected = list(range(start, end + 1))
        missing = [index for index in expected if index not in indexed]
        if missing:
            raise ValueError(f"Folder {folder} missing reference frames: {missing[:10]}")
        a_paths = [indexed[index] for index in expected if index % 2 == 1]
        b_paths = [indexed[index] for index in expected if index % 2 == 0]
        ref_a = mean_frames(a_paths, crop_size, range_max)
        ref_b = mean_frames(b_paths, crop_size, range_max)
        refs = {"reference_a_odd": ref_a, "reference_b_even": ref_b}
        references[folder] = refs
        quality = image_quality(ref_a, ref_b, data_range=1.0)
        rows.append(
            {
                "folder": folder,
                "reference_a_frames": len(a_paths),
                "reference_b_frames": len(b_paths),
                "reference_ab_psnr": quality["psnr"],
                "reference_ab_ssim": quality["ssim"],
                "reference_ab_residual_mean": quality["residual_mean"],
                "reference_ab_residual_std": quality["residual_std"],
                "reference_a_gradient": gradient_mean(ref_a),
                "reference_b_gradient": gradient_mean(ref_b),
                "reference_gradient_ratio_b_to_a": gradient_mean(ref_b) / max(gradient_mean(ref_a), 1e-12),
            }
        )
    return references, rows


def load_conditions(path: Path) -> dict[int, float]:
    return {int(float(row["folder"])): float(row["condition_score"]) for row in read_csv(path)}


def load_selections(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    rows = read_csv(path)
    return {(int(float(row["heldout_folder"])), row["strategy"]): row for row in rows}


def strategy_alpha(selection: dict[str, str], score: float) -> float:
    if selection["kind"] == "hard":
        return 1.0 if score >= float(selection["threshold"]) else 0.0
    if selection["kind"] == "linear":
        low, high = float(selection["low"]), float(selection["high"])
        return float(np.clip((score - low) / max(high - low, 1e-12), 0.0, 1.0))
    raise ValueError(f"Unsupported LOFO selection kind: {selection['kind']}")


@torch.no_grad()
def evaluate(
    pairs: list[dict[str, str]],
    references: dict[int, dict[str, torch.Tensor]],
    conditions: dict[int, float],
    selections: dict[tuple[int, str], dict[str, str]],
    p99_model: torch.nn.Module,
    physical_model: torch.nn.Module,
    device: torch.device,
    range_max: float,
    primary_strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    metric_rows: list[dict[str, Any]] = []
    image_cache: dict[str, dict[str, torch.Tensor]] = {}
    for pair in pairs:
        folder = int(float(pair["folder"]))
        noisy = load_tiff_tensor(pair["noisy_path"], range_max).to(device)
        p99 = p99_model(noisy).clamp(0.0, 1.0)
        physical = physical_model(noisy).clamp(0.0, 1.0)
        images = {"always_noisy": noisy, "always_p99": p99, "always_physical": physical}
        for strategy in ("lofo_best_hard", "lofo_best_linear"):
            selection = selections[(folder, strategy)]
            alpha = strategy_alpha(selection, conditions[folder])
            images[strategy] = ((1.0 - alpha) * p99 + alpha * physical).clamp(0.0, 1.0)
        image_cache[pair["pair_key"]] = {key: value.cpu() for key, value in images.items()}
        image_cache[pair["pair_key"]].update(references[folder])

        noisy_gradient = gradient_mean(noisy)
        for reference_name, reference_cpu in references[folder].items():
            reference = reference_cpu.to(device)
            noisy_quality = image_quality(noisy, reference, data_range=1.0)
            for strategy, prediction in images.items():
                quality = image_quality(prediction, reference, data_range=1.0)
                metric_rows.append(
                    {
                        "reference": reference_name,
                        "strategy": strategy,
                        "pair_key": pair["pair_key"],
                        "folder": folder,
                        "condition_score": conditions[folder],
                        "psnr": quality["psnr"],
                        "psnr_gain": quality["psnr"] - noisy_quality["psnr"],
                        "ssim": quality["ssim"],
                        "ssim_gain": quality["ssim"] - noisy_quality["ssim"],
                        "residual_mean": quality["residual_mean"],
                        "residual_std": quality["residual_std"],
                        "gradient_ratio_to_noisy": gradient_mean(prediction) / max(noisy_gradient, 1e-12),
                        "removed_residual_std": float(torch.std(noisy - prediction, unbiased=False).cpu()),
                        "is_primary_strategy": int(strategy == primary_strategy),
                    }
                )
    return metric_rows, image_cache


def summarize_folders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["reference"], row["strategy"], int(row["folder"]))].append(row)
    out: list[dict[str, Any]] = []
    for (reference, strategy, folder), group in groups.items():
        gains = np.asarray([float(row["psnr_gain"]) for row in group])
        out.append(
            {
                "reference": reference,
                "strategy": strategy,
                "folder": folder,
                "pair_count": len(group),
                "mean_psnr_gain": float(np.mean(gains)),
                "std_psnr_gain": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
                "positive_pair_fraction": float(np.mean(gains > 0.0)),
                "mean_ssim": mean_field(group, "ssim"),
                "mean_ssim_gain": mean_field(group, "ssim_gain"),
                "mean_gradient_ratio_to_noisy": mean_field(group, "gradient_ratio_to_noisy"),
                "mean_residual_std": mean_field(group, "residual_std"),
                "mean_removed_residual_std": mean_field(group, "removed_residual_std"),
            }
        )
    return sorted(out, key=lambda row: (row["reference"], row["strategy"], row["folder"]))


def summarize_strategies(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["reference"], row["strategy"])].append(row)
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    iterations = int(config["bootstrap"]["iterations"])
    out: list[dict[str, Any]] = []
    for (reference, strategy), group in groups.items():
        folder_values: dict[int, list[float]] = defaultdict(list)
        for row in group:
            folder_values[int(row["folder"])].append(float(row["psnr_gain"]))
        folder_gains = np.asarray([np.mean(values) for values in folder_values.values()])
        bootstrap = np.asarray([
            np.mean(rng.choice(folder_gains, size=len(folder_gains), replace=True)) for _ in range(iterations)
        ])
        pair_gains = np.asarray([float(row["psnr_gain"]) for row in group])
        out.append(
            {
                "reference": reference,
                "strategy": strategy,
                "pair_count": len(group),
                "folder_count": len(folder_gains),
                "mean_folder_psnr_gain": float(np.mean(folder_gains)),
                "folder_gain_ci95_low": float(np.percentile(bootstrap, 2.5)),
                "folder_gain_ci95_high": float(np.percentile(bootstrap, 97.5)),
                "worst_folder_psnr_gain": float(np.min(folder_gains)),
                "positive_folder_count": int(np.sum(folder_gains > 0.0)),
                "mean_pair_psnr_gain": float(np.mean(pair_gains)),
                "positive_pair_fraction": float(np.mean(pair_gains > 0.0)),
                "mean_ssim": mean_field(group, "ssim"),
                "mean_ssim_gain": mean_field(group, "ssim_gain"),
                "mean_gradient_ratio_to_noisy": mean_field(group, "gradient_ratio_to_noisy"),
                "mean_residual_std": mean_field(group, "residual_std"),
            }
        )
    return sorted(out, key=lambda row: (row["reference"], -float(row["mean_folder_psnr_gain"])))


def summarize_reference_agreement(folder_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = str(config["fixed_models"]["primary_strategy"])
    by_key = {(row["reference"], row["strategy"], int(row["folder"])): row for row in folder_rows}
    folders = sorted({int(row["folder"]) for row in folder_rows if row["strategy"] == primary})
    out: list[dict[str, Any]] = []
    for strategy in sorted({str(row["strategy"]) for row in folder_rows}):
        gains_a = np.asarray([float(by_key[("reference_a_odd", strategy, f)]["mean_psnr_gain"]) for f in folders])
        gains_b = np.asarray([float(by_key[("reference_b_even", strategy, f)]["mean_psnr_gain"]) for f in folders])
        signs = np.sign(gains_a) == np.sign(gains_b)
        if len(gains_a) < 2 or np.std(gains_a) <= 1e-12 or np.std(gains_b) <= 1e-12:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(gains_a, gains_b)[0, 1])
        out.append(
            {
                "strategy": strategy,
                "folder_sign_agreement_fraction": float(np.mean(signs)),
                "mean_absolute_folder_gain_delta_db": float(np.mean(np.abs(gains_a - gains_b))),
                "max_absolute_folder_gain_delta_db": float(np.max(np.abs(gains_a - gains_b))),
                "folder_gain_correlation": correlation,
            }
        )
    return out


def summarize_strategy_contrasts(folder_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = str(config["fixed_models"]["primary_strategy"])
    by_key = {(row["reference"], row["strategy"], int(row["folder"])): row for row in folder_rows}
    references = sorted({str(row["reference"]) for row in folder_rows})
    folders = sorted({int(row["folder"]) for row in folder_rows if row["strategy"] == primary})
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    iterations = int(config["bootstrap"]["iterations"])
    out: list[dict[str, Any]] = []
    for reference in references:
        for comparator in ("always_p99", "always_physical"):
            differences = np.asarray([
                float(by_key[(reference, primary, folder)]["mean_psnr_gain"])
                - float(by_key[(reference, comparator, folder)]["mean_psnr_gain"])
                for folder in folders
            ])
            bootstrap = np.asarray([
                np.mean(rng.choice(differences, size=len(differences), replace=True))
                for _ in range(iterations)
            ])
            out.append(
                {
                    "reference": reference,
                    "primary_strategy": primary,
                    "comparator": comparator,
                    "mean_paired_folder_difference_db": float(np.mean(differences)),
                    "paired_difference_ci95_low": float(np.percentile(bootstrap, 2.5)),
                    "paired_difference_ci95_high": float(np.percentile(bootstrap, 97.5)),
                    "positive_difference_folder_count": int(np.sum(differences > 0.0)),
                    "equal_difference_folder_count": int(np.sum(np.abs(differences) <= 1e-12)),
                    "negative_difference_folder_count": int(np.sum(differences < 0.0)),
                    "minimum_folder_difference_db": float(np.min(differences)),
                    "maximum_folder_difference_db": float(np.max(differences)),
                }
            )
    return out


def decide(
    strategy_rows: list[dict[str, Any]],
    folder_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = str(config["fixed_models"]["primary_strategy"])
    criteria = config["go_no_go"]
    lookup = {(row["reference"], row["strategy"]): row for row in strategy_rows}
    agreement = {row["strategy"]: row for row in agreement_rows}[primary]
    checks: dict[str, bool] = {}
    advantages: dict[str, float] = {}
    for reference in ("reference_a_odd", "reference_b_even"):
        primary_row = lookup[(reference, primary)]
        best_fixed = max(
            float(lookup[(reference, "always_p99")]["mean_folder_psnr_gain"]),
            float(lookup[(reference, "always_physical")]["mean_folder_psnr_gain"]),
        )
        advantage = float(primary_row["mean_folder_psnr_gain"]) - best_fixed
        advantages[reference] = advantage
        checks[f"{reference}_advantage"] = advantage > float(criteria["minimum_advantage_over_best_fixed_db_each_reference"])
        checks[f"{reference}_positive_folders"] = int(primary_row["positive_folder_count"]) >= int(criteria["minimum_positive_folders_each_reference"])
        checks[f"{reference}_worst_folder"] = float(primary_row["worst_folder_psnr_gain"]) >= float(criteria["minimum_worst_folder_gain_db"])
        physical_grad = float(lookup[(reference, "always_physical")]["mean_gradient_ratio_to_noisy"])
        primary_grad = float(primary_row["mean_gradient_ratio_to_noisy"])
        checks[f"{reference}_gradient"] = primary_grad >= physical_grad - float(criteria["maximum_gradient_ratio_drop_vs_physical"])
    checks["folder_sign_agreement"] = float(agreement["folder_sign_agreement_fraction"]) >= float(criteria["minimum_folder_sign_agreement_fraction"])
    if all(checks.values()):
        status = "GO"
    elif all(checks[key] for key in checks if "advantage" in key) and checks["folder_sign_agreement"]:
        status = "PARTIAL"
    else:
        status = "NO_GO"
    return {
        "status": status,
        "checks": checks,
        "primary_advantage_over_best_fixed_db": advantages,
        "primary_folder_sign_agreement_fraction": agreement["folder_sign_agreement_fraction"],
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def mean_field(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def gradient_mean(image: Any) -> float:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.squeeze(np.asarray(image, dtype=np.float64))
    gy, gx = np.gradient(arr)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def save_panels(cache: dict[str, dict[str, torch.Tensor]], folder_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    primary_rows = [row for row in folder_rows if row["strategy"] == "lofo_best_linear"]
    mean_folder_gain: dict[int, list[float]] = defaultdict(list)
    for row in primary_rows:
        mean_folder_gain[int(row["folder"])].append(float(row["mean_psnr_gain"]))
    ranked = sorted((float(np.mean(values)), folder) for folder, values in mean_folder_gain.items())
    selected_folders = [ranked[0][1], ranked[len(ranked) // 2][1], ranked[-1][1]]
    for label, folder in zip(("worst", "median", "best"), selected_folders):
        pair_key = sorted(key for key in cache if key.startswith(f"folder_{folder}_"))[0]
        images = cache[pair_key]
        noisy = np.squeeze(images["always_noisy"].numpy())
        linear = np.squeeze(images["lofo_best_linear"].numpy())
        ref_a = np.squeeze(images["reference_a_odd"].numpy())
        ref_b = np.squeeze(images["reference_b_even"].numpy())
        panels = [
            ("model input", noisy, "gray"), ("p99", np.squeeze(images["always_p99"].numpy()), "gray"),
            ("physical", np.squeeze(images["always_physical"].numpy()), "gray"), ("LOFO linear", linear, "gray"),
            ("predicted residual", noisy - linear, "coolwarm"), ("surrogate A", ref_a, "gray"),
            ("error vs A", linear - ref_a, "coolwarm"), ("surrogate B", ref_b, "gray"),
            ("error vs B", linear - ref_b, "coolwarm"), ("A - B", ref_a - ref_b, "coolwarm"),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
        lo, hi = np.percentile(noisy, [1, 99])
        removed_limit = max(float(np.percentile(np.abs(noisy - linear), 99)), 1e-6)
        error_limit = max(
            float(np.percentile(np.abs(np.concatenate([(linear - ref_a).ravel(), (linear - ref_b).ravel()])), 99)),
            1e-6,
        )
        reference_delta_limit = max(float(np.percentile(np.abs(ref_a - ref_b), 99)), 1e-6)
        for axis, (title, image, cmap) in zip(axes.ravel(), panels):
            if cmap == "gray":
                axis.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
            else:
                if title == "predicted residual":
                    limit = removed_limit
                elif title == "A - B":
                    limit = reference_delta_limit
                else:
                    limit = error_limit
                axis.imshow(image, cmap=cmap, vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{label} folder {folder}: {pair_key}")
        fig.savefig(output_dir / f"{label}_{pair_key}.png", dpi=160)
        plt.close(fig)


def build_run_manifest(config: dict[str, Any], args: argparse.Namespace, device: torch.device, pair_count: int) -> dict[str, Any]:
    return {
        "config": config,
        "device": str(device),
        "git_commit": git_commit(),
        "pair_count": pair_count,
        "max_pairs": args.max_pairs,
        "torch_version": torch.__version__,
    }


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def write_report(
    path: Path,
    config: dict[str, Any],
    decision: dict[str, Any],
    strategies: list[dict[str, Any]],
    agreements: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> None:
    lines = [
        "# Surrogate Reference Reliability Audit",
        "",
        "## Preregistered Hypothesis",
        "",
        str(config["core_hypothesis"]),
        "",
        f"Decision: **{decision['status']}**",
        "",
        "## Strategy Results",
        "",
        "| Reference | Strategy | Folder gain | 95% CI | Positive folders | Worst folder | SSIM gain | Grad/noisy | Residual std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategies:
        lines.append(
            f"| {row['reference']} | {row['strategy']} | {float(row['mean_folder_psnr_gain']):.6f} | "
            f"[{float(row['folder_gain_ci95_low']):.6f}, {float(row['folder_gain_ci95_high']):.6f}] | "
            f"{int(row['positive_folder_count'])}/{int(row['folder_count'])} | {float(row['worst_folder_psnr_gain']):.6f} | "
            f"{float(row['mean_ssim_gain']):.6g} | {float(row['mean_gradient_ratio_to_noisy']):.4f} | "
            f"{float(row['mean_residual_std']):.6g} |"
        )
    lines.extend(["", "## Reference Agreement", "", "| Strategy | Folder sign agreement | Mean abs gain delta | Max abs gain delta | Correlation |", "|---|---:|---:|---:|---:|"])
    for row in agreements:
        lines.append(
            f"| {row['strategy']} | {float(row['folder_sign_agreement_fraction']):.3f} | "
            f"{float(row['mean_absolute_folder_gain_delta_db']):.6f} | {float(row['max_absolute_folder_gain_delta_db']):.6f} | "
            f"{float(row['folder_gain_correlation']):.4f} |"
        )
    lines.extend([
        "", "## Paired Strategy Contrasts", "",
        "| Reference | Comparator | Linear minus comparator | 95% paired CI | Positive/equal/negative folders |",
        "|---|---|---:|---:|---:|",
    ])
    for row in contrasts:
        lines.append(
            f"| {row['reference']} | {row['comparator']} | {float(row['mean_paired_folder_difference_db']):.6f} | "
            f"[{float(row['paired_difference_ci95_low']):.6f}, {float(row['paired_difference_ci95_high']):.6f}] | "
            f"{int(row['positive_difference_folder_count'])}/{int(row['equal_difference_folder_count'])}/"
            f"{int(row['negative_difference_folder_count'])} |"
        )
    lines.extend([
        "", "## Automatic Checks", "",
        *[f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in decision["checks"].items()],
        "", "## Claim Boundary", "",
        "- Both references are temporal means from repeated frames, not clean ground truth.",
        "- This audit tests evaluation stability; it does not establish recovery of unobserved scene detail.",
        "- Fixed-pattern content shared by both split references may remain and is not removed by split-half agreement alone.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
