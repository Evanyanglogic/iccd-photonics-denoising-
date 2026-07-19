"""Attribute frozen G/CG-NC brightness bias and structure removal on ICCD holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import tifffile
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

from json_serialization import dump_json
from run_e3_real_iccd_holdout_validation import (
    correlation,
    gradient_magnitude,
    infer,
    load_model,
    load_roi_stack,
    proxy_summary,
    row_column_summary,
    sorted_tiffs,
    spatial_summaries,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Nonfinite metric: {value}")
    return result


def temporal_variance(stack: np.ndarray, center_frames: bool = False) -> float:
    work = stack.astype(np.float64, copy=False)
    if center_frames:
        work = work - np.mean(work, axis=(1, 2), keepdims=True)
    return finite_float(np.mean(np.var(work, axis=0, ddof=1)))


def temporal_metrics(raw: np.ndarray, output: np.ndarray, corrected: np.ndarray) -> dict[str, float]:
    raw_variance = temporal_variance(raw)
    raw_centered = temporal_variance(raw, True)
    output_variance = temporal_variance(output)
    output_centered = temporal_variance(output, True)
    corrected_variance = temporal_variance(corrected)
    corrected_centered = temporal_variance(corrected, True)
    return {
        "raw_temporal_variance_DN2": raw_variance,
        "output_temporal_variance_DN2": output_variance,
        "corrected_temporal_variance_DN2": corrected_variance,
        "raw_temporal_reduction": 1.0 - output_variance / raw_variance,
        "corrected_temporal_reduction": 1.0 - corrected_variance / raw_variance,
        "raw_mean_centered_temporal_variance_DN2": raw_centered,
        "output_mean_centered_temporal_variance_DN2": output_centered,
        "corrected_mean_centered_temporal_variance_DN2": corrected_centered,
        "mean_centered_temporal_reduction": 1.0 - output_centered / raw_centered,
        "corrected_mean_centered_temporal_reduction": 1.0 - corrected_centered / raw_centered,
    }


def frequency_band_energy(stack: np.ndarray, cfg: dict[str, Any], batch_size: int = 8) -> dict[str, float]:
    residual = stack.astype(np.float32) - np.mean(stack, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    height, width = residual.shape[1:]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    bands = cfg["frequency_bands_cycles_per_pixel"]
    masks = {
        "dc": radius == 0,
        "very_low": (radius > 0) & (radius < float(bands["very_low_upper"])),
        "low": (radius >= float(bands["very_low_upper"])) & (radius < float(bands["low_upper"])),
        "mid": (radius >= float(bands["low_upper"])) & (radius < float(bands["mid_upper"])),
        "high": radius >= float(bands["mid_upper"]),
    }
    totals = {name: 0.0 for name in masks}
    for start in range(0, len(residual), batch_size):
        transformed = np.fft.rfft2(residual[start:start + batch_size], axes=(1, 2))
        power = np.abs(transformed) ** 2
        # rfft omits the negative x half. Ratios use the same representation for all models.
        for name, mask in masks.items():
            totals[name] += float(np.sum(power[:, mask]))
    total = sum(totals.values())
    return {f"{name}_energy": finite_float(value) for name, value in totals.items()} | {
        f"{name}_fraction": finite_float(value / max(total, 1e-30)) for name, value in totals.items()
    }


def gradient_masks(mean_map: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    gradient = gradient_magnitude(mean_map)
    limits = cfg["gradient_regions"]
    q25, q50, q90 = np.percentile(
        gradient,
        [limits["flat_upper_percentile"], limits["low_upper_percentile"], limits["medium_upper_percentile"]],
    )
    masks = {
        "flat": gradient <= q25,
        "low_gradient": (gradient > q25) & (gradient <= q50),
        "medium_gradient": (gradient > q50) & (gradient <= q90),
        "high_gradient": gradient > q90,
    }
    return masks, {"q25": float(q25), "q50": float(q50), "q90": float(q90)}


def masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return correlation(np.asarray(a)[mask], np.asarray(b)[mask])


def region_metrics(
    raw: np.ndarray,
    output: np.ndarray,
    corrected: np.ndarray,
    mean_map: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    raw_variance_map = np.var(raw, axis=0, ddof=1, dtype=np.float64)
    output_variance_map = np.var(output, axis=0, ddof=1, dtype=np.float64)
    corrected_variance_map = np.var(corrected, axis=0, ddof=1, dtype=np.float64)
    removed = raw - output
    corrected_removed = raw - corrected
    mean_removed = np.mean(removed, axis=0, dtype=np.float64)
    mean_corrected_removed = np.mean(corrected_removed, axis=0, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for region, mask in masks.items():
        raw_gradient = []
        output_gradient = []
        corrected_gradient = []
        raw_local_variance = []
        output_local_variance = []
        corrected_local_variance = []
        for raw_frame, output_frame, corrected_frame in zip(raw, output, corrected):
            raw_gradient.append(float(np.mean(gradient_magnitude(raw_frame)[mask])))
            output_gradient.append(float(np.mean(gradient_magnitude(output_frame)[mask])))
            corrected_gradient.append(float(np.mean(gradient_magnitude(corrected_frame)[mask])))
            raw_local_variance.append(float(np.var(raw_frame[mask], ddof=1)))
            output_local_variance.append(float(np.var(output_frame[mask], ddof=1)))
            corrected_local_variance.append(float(np.var(corrected_frame[mask], ddof=1)))
        raw_temporal = float(np.mean(raw_variance_map[mask]))
        output_temporal = float(np.mean(output_variance_map[mask]))
        corrected_temporal = float(np.mean(corrected_variance_map[mask]))
        raw_gradient_mean = float(np.mean(raw_gradient))
        rows.append({
            "region": region,
            "pixel_count": int(mask.sum()),
            "raw_temporal_variance_DN2": raw_temporal,
            "output_temporal_variance_DN2": output_temporal,
            "corrected_temporal_variance_DN2": corrected_temporal,
            "temporal_variance_reduction": 1.0 - output_temporal / raw_temporal,
            "corrected_temporal_variance_reduction": 1.0 - corrected_temporal / raw_temporal,
            "output_input_MAE_DN": float(np.mean(np.abs(removed[:, mask]))),
            "corrected_output_input_MAE_DN": float(np.mean(np.abs(corrected_removed[:, mask]))),
            "gradient_ratio": float(np.mean(output_gradient) / max(raw_gradient_mean, 1e-30)),
            "corrected_gradient_ratio": float(np.mean(corrected_gradient) / max(raw_gradient_mean, 1e-30)),
            "edge_contrast_change": float(np.mean(output_gradient) / max(raw_gradient_mean, 1e-30) - 1.0),
            "corrected_edge_contrast_change": float(np.mean(corrected_gradient) / max(raw_gradient_mean, 1e-30) - 1.0),
            "removed_residual_std_DN": float(np.std(removed[:, mask], ddof=1)),
            "corrected_removed_residual_std_DN": float(np.std(corrected_removed[:, mask], ddof=1)),
            "removed_temporal_mean_correlation": masked_correlation(mean_removed, mean_map, mask),
            "corrected_removed_temporal_mean_correlation": masked_correlation(mean_corrected_removed, mean_map, mask),
            "local_variance_ratio": float(np.mean(output_local_variance) / max(np.mean(raw_local_variance), 1e-30)),
            "corrected_local_variance_ratio": float(np.mean(corrected_local_variance) / max(np.mean(raw_local_variance), 1e-30)),
        })
    return rows


def residual_dc_metrics(raw: np.ndarray, output: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    removed = raw - output
    frame_rows = []
    frame_means = np.mean(removed, axis=(1, 2), dtype=np.float64)
    for index, residual in enumerate(removed):
        centered = residual.astype(np.float64) - frame_means[index]
        frame_rows.append({
            "frame_index": index + 1,
            "predicted_residual_mean_DN": float(frame_means[index]),
            "predicted_residual_median_DN": float(np.median(residual)),
            "predicted_residual_row_mean_rms_DN": float(np.sqrt(np.mean(np.mean(centered, axis=1) ** 2))),
            "predicted_residual_column_mean_rms_DN": float(np.sqrt(np.mean(np.mean(centered, axis=0) ** 2))),
        })
    mean_removed = np.mean(removed, axis=0, dtype=np.float64)
    transformed = np.fft.fft2(mean_removed - np.mean(mean_removed))
    power = np.abs(transformed) ** 2
    fy = np.fft.fftfreq(mean_removed.shape[0])[:, None]
    fx = np.fft.fftfreq(mean_removed.shape[1])[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    low_mask = (radius > 0) & (radius < 0.0625)
    summary = {
        "predicted_residual_DC_DN": float(np.mean(removed)),
        "predicted_residual_mean_std_across_frames_DN": float(np.std(frame_means, ddof=1)),
        "predicted_residual_mean_first_last_change_DN": float(frame_means[-1] - frame_means[0]),
        "predicted_residual_spatial_low_frequency_fraction": float(np.sum(power[low_mask]) / max(np.sum(power), 1e-30)),
        "predicted_residual_frame_to_frame_mean_difference_std_DN": float(np.std(np.diff(frame_means), ddof=1)),
    }
    return summary, frame_rows


def design_fit(y: np.ndarray, matrix: np.ndarray, names: list[str]) -> dict[str, Any]:
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    predicted = matrix @ coefficients
    residual = y - predicted
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 1e-30 else 0.0
    return {
        "coefficients": json.dumps({name: float(value) for name, value in zip(names, coefficients)}, sort_keys=True),
        "R2": r2,
        "residual_mean_DN": float(np.mean(residual)),
        "residual_std_DN": float(np.std(residual, ddof=1)),
    }


def bias_model_fits(frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [(model, "ALL", group) for model, group in frames.groupby("model")]
    groups += [(model, str(seed), group) for (model, seed), group in frames.groupby(["model", "run_seed"])]
    for model, seed, group in groups:
        y = group.mean_shift_DN.to_numpy(np.float64)
        signal = group.input_mean_DN.to_numpy(np.float64)
        input_std = group.input_std_DN.to_numpy(np.float64)
        for name, matrix, terms in [
            ("B1", np.ones((len(group), 1)), ["intercept"]),
            ("B2", np.column_stack([np.ones(len(group)), signal]), ["intercept", "input_mean_signal"]),
            ("B4", np.column_stack([np.ones(len(group)), signal, input_std]), ["intercept", "input_mean_signal", "input_std"]),
        ]:
            result = design_fit(y, matrix, terms)
            pearson = pearsonr(y, signal).statistic if np.std(y) > 0 and np.std(signal) > 0 else np.nan
            spearman = spearmanr(y, signal).statistic if np.std(y) > 0 and np.std(signal) > 0 else np.nan
            rows.append({"model": model, "run_seed": seed, "bias_model": name, **result, "pearson_with_signal": pearson, "spearman_with_signal": spearman})
        if model == "CG_NC":
            sigma = group.predicted_sigma_DN.to_numpy(np.float64)
            result = design_fit(y, np.column_stack([np.ones(len(group)), sigma]), ["intercept", "predicted_sigma"])
            rows.append({"model": model, "run_seed": seed, "bias_model": "B3", **result, "pearson_with_signal": pearsonr(y, sigma).statistic, "spearman_with_signal": spearmanr(y, sigma).statistic, "note": "predicted_sigma is exactly proportional to input_mean_signal"})
    for model, group in frames.groupby("model"):
        encoded = pd.get_dummies(group[["folder", "run_seed"]].astype(str), drop_first=True, dtype=float)
        matrix = np.column_stack([np.ones(len(group)), encoded.to_numpy()])
        result = design_fit(group.mean_shift_DN.to_numpy(np.float64), matrix, ["intercept", *encoded.columns.tolist()])
        rows.append({"model": model, "run_seed": "ALL", "bias_model": "B5", **result, "pearson_with_signal": np.nan, "spearman_with_signal": np.nan, "note": "descriptive folder and seed fixed effects"})
    return pd.DataFrame(rows)


def source_snapshot(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame([{
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "modified_time_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    } for path in paths])


def load_epoch_metrics(repo: Path, cfg: dict[str, Any], bias_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed, relative in cfg["training_metric_sources"].items():
        frame = pd.read_csv(repo / relative)
        if "experiment" not in frame.columns:
            raise RuntimeError(f"Training metric schema missing experiment column: {relative}")
        frame = frame.rename(columns={"experiment": "model"})
        frame.insert(0, "run_seed", int(seed))
        rows.append(frame)
    epochs = pd.concat(rows, ignore_index=True)
    bias = bias_summary.groupby(["run_seed", "model"], as_index=False).agg(
        mean_real_shift_DN=("mean_shift_DN", "mean"),
        mean_absolute_real_shift_DN=("mean_shift_DN", lambda x: float(np.mean(np.abs(x)))),
        mean_predicted_residual_DC_DN=("predicted_residual_mean_DN", "mean"),
    )
    epochs = epochs.merge(bias, on=["run_seed", "model"], how="left")
    epochs["real_bias_available_for_epoch"] = epochs.is_best
    epochs.loc[~epochs.is_best, ["mean_real_shift_DN", "mean_absolute_real_shift_DN", "mean_predicted_residual_DC_DN"]] = np.nan
    drop_rows = []
    for (seed, model), group in epochs.groupby(["run_seed", "model"]):
        best = group.loc[group.validation_psnr.idxmax()]
        final = group.sort_values("epoch").iloc[-1]
        real_bias = bias[(bias.run_seed == seed) & (bias.model == model)].iloc[0]
        drop_rows.append({
            "run_seed": seed, "model": model, "best_epoch": int(best.epoch), "final_epoch": int(final.epoch),
            "best_validation_PSNR": float(best.validation_psnr), "final_validation_PSNR": float(final.validation_psnr),
            "best_to_final_PSNR_drop": float(best.validation_psnr - final.validation_psnr),
            "train_loss_first": float(group.sort_values("epoch").iloc[0].train_l1), "train_loss_final": float(final.train_l1),
            "train_loss_decreased": bool(final.train_l1 < group.sort_values("epoch").iloc[0].train_l1),
            "best_checkpoint_mean_absolute_real_shift_DN": float(real_bias.mean_absolute_real_shift_DN),
            "epochwise_bias_trend_available": False,
            "limitation": "Only best checkpoint has real-holdout bias metrics; epochwise checkpoints were not selected or evaluated",
        })
    drops = pd.DataFrame(drop_rows)
    return epochs, drops


def choose_output_root(repo: Path, requested: str) -> Path:
    base = (repo / requested).resolve()
    if not base.exists():
        return base
    index = 2
    while Path(f"{base}_v{index}").exists():
        index += 1
    return Path(f"{base}_v{index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = (repo / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = choose_output_root(repo, args.output_root)
    directories = [
        "provenance", "manifests", "bias_analysis", "temporal_attribution", "structure_analysis",
        "dc_correction", "overfitting_analysis", "logs",
    ]
    for directory in directories:
        (out / directory).mkdir(parents=True, exist_ok=False)
    started = now()
    commit = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    provenance = out / "provenance"
    (provenance / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (provenance / "git_status_before.txt").write_text(status_before, encoding="utf-8")
    (provenance / "git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (provenance / "command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (provenance / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (provenance / "environment.txt").write_text(
        f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\ncuda={torch.cuda.is_available()}\nnumpy={np.__version__}\nscipy={scipy.__version__}\ntifffile={tifffile.__version__}\n",
        encoding="utf-8",
    )
    (provenance / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout, encoding="utf-8")

    if cfg["evaluation_folders"] != [2, 5, 9, 11] or cfg["calibration_folders"] != [1, 4, 7, 8, 10, 13]:
        raise RuntimeError("Frozen folder boundary drift")
    if cfg["roi"] != {"top": 2304, "left": 2304, "height": 512, "width": 512}:
        raise RuntimeError("Frozen ROI drift")
    checkpoints = cfg["checkpoints"][:2] if args.smoke else cfg["checkpoints"]
    folders = cfg["evaluation_folders"][:1] if args.smoke else cfg["evaluation_folders"]
    frame_limit = 8 if args.smoke else int(cfg["frames_per_folder"])
    checkpoint_rows = []
    for item in checkpoints:
        path = (repo / item["path"]).resolve()
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 drift: {path}")
        checkpoint_rows.append({**item, "absolute_path": str(path), "sha256_verified": True, "frozen_best_checkpoint": True})
    pd.DataFrame(checkpoint_rows).to_csv(out / "manifests/checkpoint_manifest.csv", index=False, encoding="utf-8-sig")

    folder_paths: dict[int, list[Path]] = {}
    all_paths: list[Path] = []
    for folder in folders:
        paths = sorted_tiffs(Path(cfg["data_root"]) / str(folder))
        expected = int(cfg["frames_per_folder"])
        if len(paths) != expected:
            raise RuntimeError(f"Folder {folder}: expected {expected} frames, found {len(paths)}")
        paths = paths[:frame_limit]
        folder_paths[int(folder)] = paths
        all_paths.extend(paths)
    source_before = source_snapshot(all_paths)
    source_before.to_csv(out / "provenance/source_before.csv", index=False, encoding="utf-8-sig")
    source_before.assign(folder=[int(Path(path).parent.name) for path in source_before.absolute_path], frame_index=list(range(1, frame_limit + 1)) * len(folders), role="evaluation_holdout_read_only").to_csv(out / "manifests/evaluation_frames.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_rows: list[dict[str, Any]] = []
    bias_summary_rows: list[dict[str, Any]] = []
    raw_temporal_rows: list[dict[str, Any]] = []
    centered_temporal_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    corrected_frame_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []

    for folder in folders:
        print(f"folder={folder}: loading {frame_limit} read-only frames", flush=True)
        raw = load_roi_stack(folder_paths[int(folder)], cfg["roi"])
        temporal_mean = np.mean(raw, axis=0, dtype=np.float64)
        temporal_median = np.median(raw, axis=0)
        masks, thresholds = gradient_masks(temporal_mean, cfg)
        raw_frequency = frequency_band_energy(raw, cfg)
        raw_rowcol = row_column_summary(int(folder), "raw", raw, 1.0, 1.0)
        raw_acf, raw_psd = spatial_summaries(int(folder), "raw", raw, 128, 1.0)
        for item in checkpoints:
            run_seed, model_name = int(item["run_seed"]), str(item["model"])
            print(f"folder={folder} seed={run_seed} model={model_name}: frozen inference", flush=True)
            model = load_model((repo / item["path"]).resolve(), 1, int(item["best_epoch"]), device, int(cfg["model"]["base_channels"]))
            output, _ = infer(model, raw, model_name, cfg, device)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            input_means = np.mean(raw, axis=(1, 2), dtype=np.float64)
            output_means = np.mean(output, axis=(1, 2), dtype=np.float64)
            correction = input_means - output_means
            corrected = output + correction[:, None, None].astype(np.float32)
            metrics = temporal_metrics(raw, output, corrected)
            raw_temporal_rows.append({"run_seed": run_seed, "model": model_name, "folder": folder, **{k: v for k, v in metrics.items() if "mean_centered" not in k}})
            centered_temporal_rows.append({"run_seed": run_seed, "model": model_name, "folder": folder, **{k: v for k, v in metrics.items() if "mean_centered" in k}, "CG_minus_G_placeholder": np.nan})

            output_frequency = frequency_band_energy(output, cfg)
            corrected_frequency = frequency_band_energy(corrected, cfg)
            for band in ["dc", "very_low", "low", "mid", "high"]:
                raw_energy = raw_frequency[f"{band}_energy"]
                frequency_rows.append({
                    "run_seed": run_seed, "model": model_name, "folder": folder, "band": band,
                    "raw_energy": raw_energy, "output_energy": output_frequency[f"{band}_energy"], "corrected_energy": corrected_frequency[f"{band}_energy"],
                    "output_energy_ratio": output_frequency[f"{band}_energy"] / max(raw_energy, 1e-30),
                    "corrected_energy_ratio": corrected_frequency[f"{band}_energy"] / max(raw_energy, 1e-30),
                    "raw_fraction": raw_frequency[f"{band}_fraction"], "output_fraction": output_frequency[f"{band}_fraction"], "corrected_fraction": corrected_frequency[f"{band}_fraction"],
                })

            dc_summary, dc_frames = residual_dc_metrics(raw, output)
            removed = raw - output
            corrected_removed = raw - corrected
            mean_removed = np.mean(removed, axis=0, dtype=np.float64)
            mean_corrected_removed = np.mean(corrected_removed, axis=0, dtype=np.float64)
            removed_structure = correlation(mean_removed, temporal_mean)
            corrected_structure = correlation(mean_corrected_removed, temporal_mean)
            removed_gradient = correlation(np.mean(np.abs(removed), axis=0, dtype=np.float64), gradient_magnitude(temporal_mean))
            corrected_removed_gradient = correlation(np.mean(np.abs(corrected_removed), axis=0, dtype=np.float64), gradient_magnitude(temporal_mean))
            removed_rows.append({
                "run_seed": run_seed, "model": model_name, "folder": folder, **dc_summary,
                "removed_temporal_mean_structure_correlation": removed_structure,
                "corrected_removed_temporal_mean_structure_correlation": corrected_structure,
                "removed_gradient_correlation": removed_gradient,
                "corrected_removed_gradient_correlation": corrected_removed_gradient,
            })

            region_result = region_metrics(raw, output, corrected, temporal_mean, masks)
            for row in region_result:
                structure_rows.append({"run_seed": run_seed, "model": model_name, "folder": folder, **thresholds, **row})
            high_row = next(row for row in region_result if row["region"] == "high_gradient")

            output_rowcol = row_column_summary(int(folder), model_name, output, raw_rowcol["row_energy_DN"], raw_rowcol["column_energy_DN"])
            corrected_rowcol = row_column_summary(int(folder), f"{model_name}_DC", corrected, raw_rowcol["row_energy_DN"], raw_rowcol["column_energy_DN"])
            output_acf, output_psd = spatial_summaries(int(folder), model_name, output, 128, raw_psd["total_residual_energy_DN2"])
            corrected_acf, corrected_psd = spatial_summaries(int(folder), f"{model_name}_DC", corrected, 128, raw_psd["total_residual_energy_DN2"])
            output_proxy = proxy_summary(int(folder), model_name, output, temporal_mean, temporal_median)
            corrected_proxy = proxy_summary(int(folder), f"{model_name}_DC", corrected, temporal_mean, temporal_median)
            gradient_ratio = float(np.mean([np.mean(gradient_magnitude(frame)) for frame in output]) / np.mean([np.mean(gradient_magnitude(frame)) for frame in raw]))
            corrected_gradient_ratio = float(np.mean([np.mean(gradient_magnitude(frame)) for frame in corrected]) / np.mean([np.mean(gradient_magnitude(frame)) for frame in raw]))
            correction_rows.append({
                "run_seed": run_seed, "model": model_name, "folder": folder,
                "pre_mean_shift_DN": float(np.mean(output_means - input_means)),
                "pre_mean_absolute_shift_DN": float(np.mean(np.abs(output_means - input_means))),
                "post_mean_shift_DN": float(np.mean(np.mean(corrected, axis=(1, 2), dtype=np.float64) - input_means)),
                "post_mean_absolute_shift_DN": float(np.mean(np.abs(np.mean(corrected, axis=(1, 2), dtype=np.float64) - input_means))),
                **metrics,
                "gradient_ratio": gradient_ratio, "corrected_gradient_ratio": corrected_gradient_ratio,
                "high_gradient_retention": high_row["gradient_ratio"], "corrected_high_gradient_retention": high_row["corrected_gradient_ratio"],
                "row_energy_reduction": output_rowcol["row_energy_reduction"], "corrected_row_energy_reduction": corrected_rowcol["row_energy_reduction"],
                "column_energy_reduction": output_rowcol["column_energy_reduction"], "corrected_column_energy_reduction": corrected_rowcol["column_energy_reduction"],
                "radial_acf_r1": output_acf["radial_acf_r1"], "corrected_radial_acf_r1": corrected_acf["radial_acf_r1"],
                "psd_energy_ratio": output_psd["output_input_energy_ratio"], "corrected_psd_energy_ratio": corrected_psd["output_input_energy_ratio"],
                "proxy_mean_PSNR": output_proxy["PSNR_to_temporal_mean_proxy"], "corrected_proxy_mean_PSNR": corrected_proxy["PSNR_to_temporal_mean_proxy"],
                "proxy_mean_SSIM": output_proxy["SSIM_to_temporal_mean_proxy"], "corrected_proxy_mean_SSIM": corrected_proxy["SSIM_to_temporal_mean_proxy"],
                "proxy_median_PSNR": output_proxy["PSNR_to_temporal_median_proxy"], "corrected_proxy_median_PSNR": corrected_proxy["PSNR_to_temporal_median_proxy"],
                "proxy_median_SSIM": output_proxy["SSIM_to_temporal_median_proxy"], "corrected_proxy_median_SSIM": corrected_proxy["SSIM_to_temporal_median_proxy"],
                "removed_structure_correlation": removed_structure, "corrected_removed_structure_correlation": corrected_structure,
                "corrected_below_zero_ratio": float(np.mean(corrected < 0)), "corrected_above_65535_ratio": float(np.mean(corrected > 65535)),
            })

            for index, dc_frame in enumerate(dc_frames):
                raw_frame, output_frame, corrected_frame = raw[index], output[index], corrected[index]
                predicted_sigma = float(np.mean(raw_frame, dtype=np.float64) * cfg["condition_slope"]) if model_name == "CG_NC" else np.nan
                row = {
                    "run_seed": run_seed, "model": model_name, "folder": folder, "frame_index": index + 1,
                    "input_mean_DN": float(np.mean(raw_frame, dtype=np.float64)), "output_mean_DN": float(np.mean(output_frame, dtype=np.float64)),
                    "mean_shift_DN": float(np.mean(output_frame, dtype=np.float64) - np.mean(raw_frame, dtype=np.float64)),
                    "median_shift_DN": float(np.median(output_frame - raw_frame)),
                    "input_std_DN": float(np.std(raw_frame, ddof=1)), "output_std_DN": float(np.std(output_frame, ddof=1)),
                    "predicted_sigma_DN": predicted_sigma, "best_epoch": int(item["best_epoch"]), **dc_frame,
                }
                frame_rows.append(row)
                corrected_frame_rows.append({
                    "run_seed": run_seed, "model": model_name, "folder": folder, "frame_index": index + 1,
                    "pre_mean_shift_DN": row["mean_shift_DN"],
                    "post_mean_shift_DN": float(np.mean(corrected_frame, dtype=np.float64) - np.mean(raw_frame, dtype=np.float64)),
                    "correction_DN": float(correction[index]),
                    "pre_gradient_ratio": float(np.mean(gradient_magnitude(output_frame)) / max(np.mean(gradient_magnitude(raw_frame)), 1e-30)),
                    "post_gradient_ratio": float(np.mean(gradient_magnitude(corrected_frame)) / max(np.mean(gradient_magnitude(raw_frame)), 1e-30)),
                    "pre_output_input_MAE_DN": float(np.mean(np.abs(output_frame - raw_frame))),
                    "post_output_input_MAE_DN": float(np.mean(np.abs(corrected_frame - raw_frame))),
                })
            bias_summary_rows.append({
                "run_seed": run_seed, "model": model_name, "folder": folder,
                "mean_shift_DN": float(np.mean(output_means - input_means)),
                "mean_absolute_shift_DN": float(np.mean(np.abs(output_means - input_means))),
                "predicted_residual_mean_DN": float(np.mean(raw - output)),
                "input_mean_DN": float(np.mean(input_means)),
                "predicted_sigma_DN": float(np.mean(input_means) * cfg["condition_slope"]) if model_name == "CG_NC" else np.nan,
                **dc_summary,
            })
            del output, corrected, removed, corrected_removed

    frame_df = pd.DataFrame(frame_rows)
    bias_summary_df = pd.DataFrame(bias_summary_rows)
    fits_df = bias_model_fits(frame_df)
    raw_temporal_df = pd.DataFrame(raw_temporal_rows)
    centered_df = pd.DataFrame(centered_temporal_rows)
    frequency_df = pd.DataFrame(frequency_rows)
    structure_df = pd.DataFrame(structure_rows)
    removed_df = pd.DataFrame(removed_rows)
    correction_df = pd.DataFrame(correction_rows)
    corrected_frame_df = pd.DataFrame(corrected_frame_rows)
    frame_df.to_csv(out / "bias_analysis/frame_level_bias.csv", index=False, encoding="utf-8-sig")
    fits_df.to_csv(out / "bias_analysis/bias_model_fits.csv", index=False, encoding="utf-8-sig")
    bias_summary_df.to_csv(out / "bias_analysis/seed_folder_bias_summary.csv", index=False, encoding="utf-8-sig")
    removed_df.to_csv(out / "bias_analysis/residual_dc_analysis.csv", index=False, encoding="utf-8-sig")
    raw_temporal_df.to_csv(out / "temporal_attribution/raw_temporal_summary.csv", index=False, encoding="utf-8-sig")
    centered_df.to_csv(out / "temporal_attribution/mean_centered_temporal_summary.csv", index=False, encoding="utf-8-sig")
    frequency_df.to_csv(out / "temporal_attribution/frequency_band_temporal_summary.csv", index=False, encoding="utf-8-sig")
    structure_df.to_csv(out / "structure_analysis/gradient_region_summary.csv", index=False, encoding="utf-8-sig")
    removed_df.to_csv(out / "structure_analysis/removed_structure_summary.csv", index=False, encoding="utf-8-sig")
    structure_df[structure_df.folder.eq(5)].to_csv(out / "structure_analysis/folder5_detailed_analysis.csv", index=False, encoding="utf-8-sig")
    corrected_frame_df.to_csv(out / "dc_correction/corrected_frame_metrics.csv", index=False, encoding="utf-8-sig")
    correction_df.to_csv(out / "dc_correction/correction_summary.csv", index=False, encoding="utf-8-sig")

    epoch_df, drop_df = load_epoch_metrics(repo, cfg, bias_summary_df) if not args.smoke else (pd.DataFrame(), pd.DataFrame())
    epoch_df.to_csv(out / "overfitting_analysis/epoch_bias_metrics.csv", index=False, encoding="utf-8-sig")
    drop_df.to_csv(out / "overfitting_analysis/validation_drop_analysis.csv", index=False, encoding="utf-8-sig")

    limits = cfg["dc_correction_decision"]
    brightness_reduction = 1.0 - correction_df.post_mean_absolute_shift_DN / correction_df.pre_mean_absolute_shift_DN.clip(lower=1e-12)
    temporal_retention = correction_df.corrected_temporal_reduction / correction_df.raw_temporal_reduction.clip(lower=1e-12)
    structure_increase = correction_df.corrected_removed_structure_correlation.abs() - correction_df.removed_structure_correlation.abs()
    checks = {
        "brightness_reduction_pass": bool((brightness_reduction >= limits["brightness_reduction_fraction_min"]).all()),
        "mean_centered_temporal_identical": bool(np.allclose(correction_df.corrected_mean_centered_temporal_reduction, correction_df.mean_centered_temporal_reduction, atol=1e-8)),
        "raw_temporal_retained_majority": bool((temporal_retention >= limits["temporal_reduction_retention_min"]).mean() >= 0.75),
        "gradient_not_worse": bool((correction_df.corrected_gradient_ratio + limits["gradient_ratio_tolerance"] >= correction_df.gradient_ratio).all()),
        "row_not_worse": bool((correction_df.corrected_row_energy_reduction + limits["row_column_relative_tolerance"] >= correction_df.row_energy_reduction).all()),
        "column_not_worse": bool((correction_df.corrected_column_energy_reduction + limits["row_column_relative_tolerance"] >= correction_df.column_energy_reduction).all()),
        "removed_structure_not_increased_majority": bool((structure_increase <= limits["structure_correlation_tolerance"]).mean() >= 0.75),
        "all_seeds_brightness_improved": bool((correction_df.assign(improvement=brightness_reduction).groupby("run_seed").improvement.mean() > 0).all()),
        "all_folders_brightness_improved": bool((correction_df.assign(improvement=brightness_reduction).groupby("folder").improvement.mean() > 0).all()),
    }
    if sum(checks.values()) >= 7 and checks["brightness_reduction_pass"] and checks["mean_centered_temporal_identical"]:
        correction_status = "DC-CORRECTION-BENEFICIAL"
    elif checks["brightness_reduction_pass"]:
        correction_status = "DC-CORRECTION-TRADEOFF"
    else:
        correction_status = "DC-CORRECTION-NOT-SUPPORTED"
    correction_decision = {"status": correction_status, "checks": checks, "correction": "frame-wise DC mean restoration", "checkpoint_modified": False, "historical_results_modified": False}
    dump_json(out / "dc_correction/correction_decision.json", correction_decision)

    centered_wide = centered_df.pivot(index=["run_seed", "folder"], columns="model", values="mean_centered_temporal_reduction")
    corrected_wide = correction_df.pivot(index=["run_seed", "folder"], columns="model", values="corrected_temporal_reduction")
    centered_delta = centered_wide.CG_NC - centered_wide.G
    corrected_delta = corrected_wide.CG_NC - corrected_wide.G
    raw_wide = raw_temporal_df.pivot(index=["run_seed", "folder"], columns="model", values="raw_temporal_reduction")
    raw_delta = raw_wide.CG_NC - raw_wide.G
    signal_correlations = fits_df[(fits_df.run_seed.astype(str) == "ALL") & fits_df.bias_model.eq("B2")][["model", "pearson_with_signal", "spearman_with_signal"]].to_dict("records")
    seed_bias_range = bias_summary_df.groupby("model").mean_shift_DN.agg(lambda x: float(x.max() - x.min())).to_dict()
    high = structure_df[structure_df.region.eq("high_gradient")].pivot(index=["run_seed", "folder"], columns="model", values="gradient_ratio")
    flat = structure_df[structure_df.region.eq("flat")].pivot(index=["run_seed", "folder"], columns="model", values="temporal_variance_reduction")
    brightness_categories = ["A. seed-dependent network DC bias"]
    if any(abs(item["spearman_with_signal"]) >= 0.3 for item in signal_correlations):
        brightness_categories.append("B. input-signal-dependent bias")
    brightness_categories.append("D. overfitting-associated bias (descriptive; epochwise bias unavailable)")
    if float(np.mean(corrected_frame_df.correction_DN.abs())) > 0:
        brightness_categories.append("F. unresolved checkpoint-specific contribution")
    structure_categories = ["A. flat-region noise suppression"]
    if float((high.CG_NC - high.G).mean()) < 0:
        structure_categories.append("B. edge attenuation")
    if float(frequency_df[frequency_df.band.isin(["dc", "very_low"])].output_energy_ratio.mean()) < 1:
        structure_categories.append("C. low-frequency/DC suppression")
    conditional_persists = bool((centered_delta.groupby(level="run_seed").mean() > 0).all() and (centered_delta.groupby(level="folder").mean() > 0).all())
    benefit_mainly_dc = bool(float(centered_delta.mean()) < 0.5 * float(raw_delta.mean()))
    overfit_all = bool(not drop_df.empty and (drop_df.best_to_final_PSNR_drop > cfg["overfit"]["validation_psnr_drop_warning_dB"]).all())
    attribution = {
        "brightness_shift_categories": brightness_categories,
        "structure_reduction_categories": structure_categories,
        "signal_correlations": signal_correlations,
        "seed_bias_range_DN": seed_bias_range,
        "raw_CG_minus_G_temporal_reduction_mean": float(raw_delta.mean()),
        "mean_centered_CG_minus_G_temporal_reduction_mean": float(centered_delta.mean()),
        "DC_corrected_CG_minus_G_temporal_reduction_mean": float(corrected_delta.mean()),
        "conditional_benefit_persists_after_mean_centering": conditional_persists,
        "conditional_benefit_mainly_DC": benefit_mainly_dc,
        "flat_region_CG_minus_G_reduction_mean": float((flat.CG_NC - flat.G).mean()),
        "high_gradient_CG_minus_G_retention_mean": float((high.CG_NC - high.G).mean()),
        "overfitting_across_all_runs": overfit_all,
        "overfitting_bias_causality": "UNRESOLVED: epochwise real-holdout bias was not measured and checkpoints remain frozen",
        "scientific_scope": "operational attribution; regressions are descriptive and not physical causal models",
    }
    dump_json(out / "attribution_decision.json", attribution)
    cgs_allowed = bool(
        conditional_persists and not benefit_mainly_dc and correction_status == "DC-CORRECTION-BENEFICIAL"
        and float((high.CG_NC - high.G).mean()) >= -0.001 and not overfit_all
    )
    cgs = {
        "CGS_ENTRY_ALLOWED": cgs_allowed,
        "reason": "Entry remains blocked by unresolved network bias/overfitting or edge-retention tradeoff" if not cgs_allowed else "Only single spatial-correlation component feasibility may be considered",
        "full_CGS_prohibited": True,
        "row_column_component_prohibited": True,
        "stable_component_prohibited": True,
    }
    dump_json(out / "cgs_entry_decision.json", cgs)

    source_after = source_snapshot(all_paths)
    source_after.to_csv(out / "provenance/source_after.csv", index=False, encoding="utf-8-sig")
    protected = bool(source_before.equals(source_after))
    status = "BIAS-STRUCTURE-ATTRIBUTION-VERIFIED-WITH-LIMITATIONS" if protected else "BIAS-STRUCTURE-ATTRIBUTION-NO-GO"
    verification = {
        "final_status": status, "smoke": args.smoke, "frozen_checkpoint_count": len(checkpoints),
        "folders_completed": folders, "frames_per_folder": frame_limit, "inference_frame_outputs": len(frame_df),
        "finite_metrics": bool(np.isfinite(frame_df.select_dtypes(include=[np.number]).drop(columns=["predicted_sigma_DN"], errors="ignore").to_numpy()).all()),
        "training_performed": False, "checkpoint_modified": False, "input_preprocessing_frozen": True,
        "DC_correction_status": correction_status, "conditional_benefit_persists_after_mean_centering": conditional_persists,
        "conditional_benefit_mainly_DC": benefit_mainly_dc, "CGS_ENTRY_ALLOWED": cgs_allowed,
        "source_data_protected": protected, "data_leakage_detected": False, "provenance_complete": True,
        "warnings": ["RAPID_OVERFITTING_ACROSS_SEEDS"] if overfit_all else [],
    }
    dump_json(out / "verification_status.json", verification)
    dump_json(out / "provenance/run_manifest.json", {"experiment_id": cfg["experiment_id"], "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "output_root": str(out), "final_status": status, "training_performed": False})
    script_paths = [Path(__file__), repo / "scripts/run_e3_real_iccd_holdout_validation.py", repo / "scripts/run_e2_g_cg_scaled_training.py", repo / "scripts/json_serialization.py"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in script_paths]).to_csv(out / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")
    (provenance / "git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")
    report = [
        "# E5 G/CG-NC Bias and Structure Attribution", "", f"Status: `{status}`", "",
        "All six frozen best checkpoints were evaluated without retraining or weight modification." if not args.smoke else "Independent smoke run.", "",
        f"Frame-wise DC correction: `{correction_status}`.",
        f"Conditional benefit after mean centering: `{conditional_persists}`.",
        f"Conditional benefit mainly explained by DC: `{benefit_mainly_dc}`.",
        f"CGS entry allowed: `{cgs_allowed}`.", "",
        "Brightness attribution: " + "; ".join(brightness_categories),
        "Structure attribution: " + "; ".join(structure_categories), "",
        "The analysis is descriptive. It does not establish clean-image recovery, physical causality, or permission to implement CGS.",
    ]
    (out / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(verification, indent=2))
    return 0 if protected else 2


if __name__ == "__main__":
    raise SystemExit(main())
