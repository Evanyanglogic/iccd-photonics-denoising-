"""Inspect visual and residual behavior for condition-aware ICCD denoising."""

from __future__ import annotations

import argparse
import csv
import math
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
from train_manifest_denoiser_baseline import ResidualDenoiser, count_parameters


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    p99_model = load_model(Path(args.p99_checkpoint), device)
    physical_model = load_model(Path(args.physical_checkpoint), device)
    pair_rows = read_csv(Path(args.pairs_csv))
    physical_metrics = read_csv(Path(args.physical_metrics_csv))
    p99_metrics = read_csv(Path(args.p99_metrics_csv))
    selected = select_pairs(args.folders, pair_rows, physical_metrics, p99_metrics, args.selection_policy)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for selected_row in selected:
            rows.extend(inspect_pair(selected_row, p99_model, physical_model, device, panels_dir, args))

    metrics_csv = output_dir / "condition_visual_metrics.csv"
    report_path = output_dir / "condition_visual_report.md"
    write_csv(rows, metrics_csv)
    write_report(report_path, rows, metrics_csv, panels_dir, args, count_parameters(p99_model), count_parameters(physical_model))

    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote panels: {panels_dir}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--p99-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--p99-metrics-csv", required=True)
    parser.add_argument("--physical-metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folders", nargs="*", type=int, default=[2, 5, 1, 10])
    parser.add_argument(
        "--selection-policy",
        choices=["diagnostic", "median_physical_gain"],
        default="diagnostic",
        help="diagnostic preserves the original hand-picked E3.5-C low/high cases; median_physical_gain picks the median physical-gain pair for each folder.",
    )
    parser.add_argument(
        "--hybrid-physical-folders",
        nargs="*",
        type=int,
        default=[4, 5, 7, 8, 9, 10],
        help="Folders assigned to the physical checkpoint in the hybrid strategy.",
    )
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--panel-percentile-low", type=float, default=1.0)
    parser.add_argument("--panel-percentile-high", type=float, default=99.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, str(device))
    config = checkpoint.get("config", {})
    channels = int(config.get("channels", 16))
    depth = int(config.get("depth", 3))
    model = ResidualDenoiser(channels=channels, depth=depth).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_pairs(
    folders: list[int],
    pair_rows: list[dict[str, str]],
    physical_metrics: list[dict[str, str]],
    p99_metrics: list[dict[str, str]],
    selection_policy: str,
) -> list[dict[str, str]]:
    pairs_by_key = {row["pair_key"]: row for row in pair_rows}
    p99_by_key = {row["pair_key"]: row for row in p99_metrics}
    by_folder: dict[int, list[dict[str, str]]] = {}
    for row in physical_metrics:
        folder = int(float(row["meta_folder"]))
        by_folder.setdefault(folder, []).append(row)

    selected: list[dict[str, str]] = []
    for folder in folders:
        rows = by_folder[folder]
        if selection_policy == "median_physical_gain":
            ranked = sorted(rows, key=lambda item: float(item["psnr_gain"]))
            metric = ranked[len(ranked) // 2]
            reason = "median_physical_gain"
        elif folder == 2:
            metric = min(rows, key=lambda item: float(item["psnr_gain"]))
            reason = "worst_physical_low_condition"
        elif folder == 5:
            metric = max(rows, key=lambda item: float(item["psnr_gain"]))
            reason = "best_physical_high_condition"
        else:
            ranked = sorted(rows, key=lambda item: float(item["psnr_gain"]))
            metric = ranked[len(ranked) // 2]
            reason = "boundary_median_physical_gain"
        pair = dict(pairs_by_key[metric["pair_key"]])
        pair["selection_reason"] = reason
        pair["physical_pair_gain"] = metric["psnr_gain"]
        pair["p99_pair_gain"] = p99_by_key[metric["pair_key"]]["psnr_gain"]
        selected.append(pair)
    return selected


def inspect_pair(
    row: dict[str, str],
    p99_model: torch.nn.Module,
    physical_model: torch.nn.Module,
    device: torch.device,
    panels_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    clean_t = load_tiff_tensor(row["clean_path"], args.range_max).to(device)
    noisy_t = load_tiff_tensor(row["noisy_path"], args.range_max).to(device)
    p99_t = p99_model(noisy_t).clamp(0.0, 1.0)
    physical_t = physical_model(noisy_t).clamp(0.0, 1.0)
    folder = int(row["folder"])
    physical_folders = set(args.hybrid_physical_folders)
    hybrid_t = physical_t if folder in physical_folders else p99_t
    tensors = {
        "noisy": noisy_t,
        "p99": p99_t,
        "physical": physical_t,
        "hybrid": hybrid_t,
    }
    clean_np = to_image(clean_t)
    noisy_np = to_image(noisy_t)
    image_panel = panels_dir / f"{row['pair_key']}_image_panel.png"
    residual_panel = panels_dir / f"{row['pair_key']}_residual_panel.png"
    save_image_panel(image_panel, clean_np, {key: to_image(value) for key, value in tensors.items()}, args)
    save_residual_panel(residual_panel, clean_np, {key: to_image(value) for key, value in tensors.items()})

    rows: list[dict[str, Any]] = []
    for strategy, tensor in tensors.items():
        image = to_image(tensor)
        quality = image_quality(tensor, clean_t, data_range=1.0)
        correction = image - noisy_np
        rows.append(
            {
                "pair_key": row["pair_key"],
                "folder": folder,
                "heldout_frame_index": row["heldout_frame_index"],
                "selection_reason": row["selection_reason"],
                "strategy": strategy,
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "psnr_gain_vs_noisy": quality["psnr"] - image_quality(noisy_t, clean_t, data_range=1.0)["psnr"],
                "residual_mean": quality["residual_mean"],
                "residual_std": quality["residual_std"],
                "residual_mae": quality["residual_mae"],
                "correction_mean": float(np.mean(correction)),
                "correction_std": float(np.std(correction)),
                "correction_abs_mean": float(np.mean(np.abs(correction))),
                "gradient_mean": gradient_mean(image),
                "gradient_ratio_to_noisy": gradient_mean(image) / max(gradient_mean(noisy_np), 1e-12),
                "gradient_ratio_to_clean": gradient_mean(image) / max(gradient_mean(clean_np), 1e-12),
                "image_panel": image_panel.as_posix(),
                "residual_panel": residual_panel.as_posix(),
            }
        )
    return rows


def to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale tensor, got {arr.shape}")
    return arr


def gradient_mean(image: np.ndarray) -> float:
    dy = np.diff(image, axis=0)
    dx = np.diff(image, axis=1)
    return float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))


def save_image_panel(path: Path, clean: np.ndarray, images: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    panels = [images["noisy"], images["p99"], images["physical"], images["hybrid"], clean]
    labels = ["noisy", "p99", "physical", "hybrid", "clean"]
    scaled = [scale_display(panel, args.panel_percentile_low, args.panel_percentile_high) for panel in panels]
    save_labeled_panel(path, scaled, labels)


def save_residual_panel(path: Path, clean: np.ndarray, images: dict[str, np.ndarray]) -> None:
    residuals = [images["noisy"] - clean, images["p99"] - clean, images["physical"] - clean, images["hybrid"] - clean]
    max_abs = max(float(np.percentile(np.abs(residual), 99.5)) for residual in residuals)
    max_abs = max(max_abs, 1e-6)
    scaled = [np.clip((residual / max_abs + 1.0) * 0.5, 0.0, 1.0) for residual in residuals]
    save_labeled_panel(path, scaled, ["noisy-clean", "p99-clean", "physical-clean", "hybrid-clean"])


def scale_display(image: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    low = float(np.percentile(image, p_low))
    high = float(np.percentile(image, p_high))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def save_labeled_panel(path: Path, panels: list[np.ndarray], labels: list[str]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to write PNG panels") from exc

    height, width = panels[0].shape
    label_height = 24
    canvas = np.ones((height + label_height, width * len(panels)), dtype=np.uint8) * 255
    for index, panel in enumerate(panels):
        canvas[label_height:, index * width : (index + 1) * width] = np.rint(panel * 255).astype(np.uint8)
    image = Image.fromarray(canvas, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(labels):
        draw.text((index * width + 4, 4), label, fill=(255, 0, 0))
    image.save(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    metrics_csv: Path,
    panels_dir: Path,
    args: argparse.Namespace,
    p99_params: int,
    physical_params: int,
) -> None:
    lines = [
        "# E3.5 Visual and Residual Inspection",
        "",
        "This report inspects selected low/high/boundary ICCD folders to check whether condition-aware gains reflect real residual reduction rather than brightness drift or obvious oversmoothing.",
        "",
        f"- Pair manifest: `{args.pairs_csv}`",
        f"- p99 checkpoint: `{args.p99_checkpoint}` ({p99_params} parameters)",
        f"- physical checkpoint: `{args.physical_checkpoint}` ({physical_params} parameters)",
        f"- Selection policy: `{args.selection_policy}`",
        f"- Hybrid physical folders: `{', '.join(str(item) for item in args.hybrid_physical_folders)}`",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Panels: `{panels_dir}`",
        "",
        "## Selected Samples",
        "",
        "| folder | pair | reason | p99 gain | physical gain | hybrid strategy | image panel | residual panel |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_key"]), []).append(row)
    for pair_key, pair_rows in sorted(by_pair.items(), key=lambda item: int(item[1][0]["folder"])):
        folder = int(pair_rows[0]["folder"])
        p99 = find_strategy(pair_rows, "p99")
        physical = find_strategy(pair_rows, "physical")
        hybrid_strategy = "physical" if folder in set(args.hybrid_physical_folders) else "p99"
        lines.append(
            "| {folder} | {pair} | {reason} | {p99_gain:.4f} | {physical_gain:.4f} | {hybrid} | `{image_panel}` | `{residual_panel}` |".format(
                folder=folder,
                pair=pair_key,
                reason=pair_rows[0]["selection_reason"],
                p99_gain=float(p99["psnr_gain_vs_noisy"]),
                physical_gain=float(physical["psnr_gain_vs_noisy"]),
                hybrid=hybrid_strategy,
                image_panel=pair_rows[0]["image_panel"],
                residual_panel=pair_rows[0]["residual_panel"],
            )
        )

    lines.extend(["", "## Residual / Smoothing Checks", ""])
    lines.extend(
        [
            "| folder | strategy | residual mean | residual std | correction abs mean | grad/noisy | grad/clean |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (int(item["folder"]), str(item["strategy"]))):
        lines.append(
            "| {folder} | {strategy} | {rmean:.6g} | {rstd:.6g} | {corr:.6g} | {gnoisy:.4f} | {gclean:.4f} |".format(
                folder=int(row["folder"]),
                strategy=row["strategy"],
                rmean=float(row["residual_mean"]),
                rstd=float(row["residual_std"]),
                corr=float(row["correction_abs_mean"]),
                gnoisy=float(row["gradient_ratio_to_noisy"]),
                gclean=float(row["gradient_ratio_to_clean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Residual mean near zero is necessary but not sufficient; over-smoothed images can still show PSNR gains.",
            "- Gradient ratios far below the noisy input indicate smoothing and must be visually checked.",
            "- These panels are selected diagnostics, not a complete visual benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def find_strategy(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    for row in rows:
        if row["strategy"] == strategy:
            return row
    raise KeyError(strategy)


if __name__ == "__main__":
    raise SystemExit(main())
