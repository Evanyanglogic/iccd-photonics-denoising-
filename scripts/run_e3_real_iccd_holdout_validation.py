"""Evaluate frozen G/CG checkpoints on real ICCD holdout repeated frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

from json_serialization import dump_json
from run_e2_g_cg_scaled_training import LightUNet


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


def sorted_tiffs(folder: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.match(r"^(\d+)", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)
    return sorted([path for path in folder.iterdir() if path.suffix.lower() in {".tif", ".tiff"}], key=key)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64).ravel().copy(); y = np.asarray(b, np.float64).ravel().copy()
    x -= x.mean(); y -= y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denominator) if denominator > 1e-12 else float("nan")


def load_roi_stack(paths: list[Path], roi: dict[str, int]) -> np.ndarray:
    top, left, height, width = (int(roi[key]) for key in ["top", "left", "height", "width"])
    stack = np.empty((len(paths), height, width), np.float32)
    for index, path in enumerate(paths):
        image = tifffile.memmap(path)
        if image.dtype != np.uint16 or image.shape != (5120, 5120):
            raise RuntimeError(f"Unexpected ICCD frame: {path} {image.dtype} {image.shape}")
        stack[index] = image[top:top + height, left:left + width]
    return stack


def temporal_residual(stack_dn: np.ndarray) -> np.ndarray:
    residual = stack_dn - np.mean(stack_dn, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    residual -= np.mean(residual, axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    return residual


def temporal_summary(folder: int, model: str, stack_dn: np.ndarray, raw_mean_variance: float) -> dict[str, Any]:
    variance = np.var(stack_dn, axis=0, ddof=1, dtype=np.float64)
    std = np.sqrt(np.maximum(variance, 0))
    mean_variance = float(np.mean(variance))
    return {
        "folder": folder, "model": model, "frame_count": len(stack_dn),
        "mean_temporal_variance_DN2": mean_variance,
        "median_temporal_variance_DN2": float(np.median(variance)),
        "mean_temporal_std_DN": float(np.mean(std)),
        "median_temporal_std_DN": float(np.median(std)),
        "p90_temporal_std_DN": float(np.percentile(std, 90)),
        "p95_temporal_std_DN": float(np.percentile(std, 95)),
        "p99_temporal_std_DN": float(np.percentile(std, 99)),
        "temporal_variance_reduction": 0.0 if model == "raw" else 1.0 - mean_variance / raw_mean_variance,
    }


def row_column_summary(folder: int, model: str, stack_dn: np.ndarray, raw_row: float, raw_column: float) -> dict[str, Any]:
    residual = temporal_residual(stack_dn)
    row_profiles = np.mean(residual, axis=2, dtype=np.float64)
    column_profiles = np.mean(residual, axis=1, dtype=np.float64)
    row_energy = float(np.sqrt(np.mean(row_profiles**2)))
    column_energy = float(np.sqrt(np.mean(column_profiles**2)))
    return {
        "folder": folder, "model": model,
        "row_energy_DN": row_energy, "column_energy_DN": column_energy,
        "row_energy_reduction": 0.0 if model == "raw" else 1.0 - row_energy / raw_row,
        "column_energy_reduction": 0.0 if model == "raw" else 1.0 - column_energy / raw_column,
    }


def average_power_spectrum(residual: np.ndarray, block_size: int = 8) -> np.ndarray:
    total = np.zeros(residual.shape[1:], np.float64)
    count = 0
    for start in range(0, len(residual), block_size):
        block = residual[start:start + block_size]
        fft = np.fft.fft2(block, axes=(1, 2))
        total += np.sum(np.abs(fft) ** 2, axis=0)
        count += len(block)
    return np.fft.fftshift(total / count)


def radial_average(image: np.ndarray, max_radius: int) -> np.ndarray:
    height, width = image.shape
    y, x = np.indices((height, width))
    radius = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2).astype(np.int32)
    mask = radius <= max_radius
    sums = np.bincount(radius[mask].ravel(), weights=image[mask].ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius[mask].ravel(), minlength=max_radius + 1)
    return sums[:max_radius + 1] / np.maximum(counts[:max_radius + 1], 1)


def spatial_summaries(folder: int, model: str, stack_dn: np.ndarray, max_radius: int, raw_energy: float) -> tuple[dict[str, Any], dict[str, Any]]:
    residual = temporal_residual(stack_dn)
    power = average_power_spectrum(residual)
    autocorrelation = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(power)).real)
    center = autocorrelation[autocorrelation.shape[0] // 2, autocorrelation.shape[1] // 2]
    autocorrelation = autocorrelation / center if abs(center) > 1e-12 else autocorrelation * np.nan
    radial_psd = radial_average(power, max_radius)
    radial_acf = radial_average(autocorrelation, max_radius)
    total = float(np.sum(radial_psd))
    low_end = max(2, max_radius // 16); mid_end = max(3, max_radius // 4)
    low = float(np.sum(radial_psd[1:low_end]) / total)
    mid = float(np.sum(radial_psd[low_end:mid_end]) / total)
    high = float(np.sum(radial_psd[mid_end:max_radius]) / total)
    energy = float(np.mean(residual.astype(np.float64) ** 2))
    acf = {
        "folder": folder, "model": model,
        "horizontal_lag1": correlation(residual[:, :, :-1], residual[:, :, 1:]),
        "vertical_lag1": correlation(residual[:, :-1, :], residual[:, 1:, :]),
        "radial_acf_r1": float(radial_acf[1]), "radial_acf_r2": float(radial_acf[2]),
        "radial_acf_r4": float(radial_acf[4]), "radial_acf_r8": float(radial_acf[8]),
    }
    psd = {
        "folder": folder, "model": model, "low_frequency_fraction": low,
        "mid_frequency_fraction": mid, "high_frequency_fraction": high,
        "total_residual_energy_DN2": energy,
        "output_input_energy_ratio": 1.0 if model == "raw" else energy / raw_energy,
    }
    return acf, psd


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = sobel(image, axis=1, mode="reflect") / 8.0
    gy = sobel(image, axis=0, mode="reflect") / 8.0
    return np.hypot(gx, gy)


def gradient_summary(folder: int, model: str, raw_stack: np.ndarray, stack_dn: np.ndarray, mean_map: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    proxy_gradient = gradient_magnitude(mean_map)
    high_threshold = np.percentile(proxy_gradient, cfg["gradient"]["high_percentile"])
    low_threshold = np.percentile(proxy_gradient, cfg["gradient"]["low_percentile"])
    high_mask = proxy_gradient >= high_threshold; low_mask = proxy_gradient <= low_threshold
    raw_gradients, output_gradients = [], []
    edge_densities = []
    for raw, output in zip(raw_stack, stack_dn):
        raw_gradient = gradient_magnitude(raw); output_gradient = gradient_magnitude(output)
        raw_gradients.append(float(np.mean(raw_gradient))); output_gradients.append(float(np.mean(output_gradient)))
        edge_densities.append(float(np.mean(output_gradient >= high_threshold)))
    raw_temporal_variance = np.var(raw_stack, axis=0, ddof=1, dtype=np.float64)
    output_temporal_variance = np.var(stack_dn, axis=0, ddof=1, dtype=np.float64)
    high_retention = float(np.mean([np.mean(gradient_magnitude(output)[high_mask]) / max(np.mean(gradient_magnitude(raw)[high_mask]), 1e-12) for raw, output in zip(raw_stack, stack_dn)]))
    return {
        "folder": folder, "model": model,
        "mean_gradient_energy_ratio": float(np.mean(output_gradients) / max(np.mean(raw_gradients), 1e-12)),
        "mean_sobel_magnitude_DN": float(np.mean(output_gradients)),
        "edge_density": float(np.mean(edge_densities)),
        "high_gradient_retention": high_retention,
        "high_region_temporal_variance_ratio": float(np.mean(output_temporal_variance[high_mask]) / np.mean(raw_temporal_variance[high_mask])),
        "low_region_temporal_variance_ratio": float(np.mean(output_temporal_variance[low_mask]) / np.mean(raw_temporal_variance[low_mask])),
    }


def brightness_summary(folder: int, model: str, raw_stack: np.ndarray, stack_dn: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    input_means = np.mean(raw_stack, axis=(1, 2), dtype=np.float64)
    output_means = np.mean(stack_dn, axis=(1, 2), dtype=np.float64)
    shift = output_means - input_means
    row = {
        "folder": folder, "model": model, "mean_shift_DN": float(np.mean(shift)),
        "median_absolute_shift_DN": float(np.median(np.abs(shift))),
        "p95_absolute_shift_DN": float(np.percentile(np.abs(shift), 95)),
        "max_absolute_shift_DN": float(np.max(np.abs(shift))),
        "input_mean_drift_DN": float(input_means[-1] - input_means[0]),
        "output_mean_drift_DN": float(output_means[-1] - output_means[0]),
        "warning_frame_count": int(np.sum(np.abs(shift) > 5)),
        "severe_warning_frame_count": int(np.sum(np.abs(shift) > 15)),
    }
    return row, shift


def split_stable_summary(folder: int, model: str, stack_dn: np.ndarray) -> dict[str, Any]:
    half = len(stack_dn) // 2
    first = np.mean(stack_dn[:half], axis=0, dtype=np.float64)
    last = np.mean(stack_dn[half:], axis=0, dtype=np.float64)
    difference = first - last
    return {"folder": folder, "model": model, "split_map_correlation": correlation(first, last), "split_map_difference_rms_DN": float(np.sqrt(np.mean(difference**2))), "split_map_difference_energy_DN2": float(np.mean(difference**2))}


def frame_consistency_summary(folder: int, model: str, stack_dn: np.ndarray) -> dict[str, Any]:
    differences = np.diff(stack_dn, axis=0)
    correlations = [correlation(stack_dn[index], stack_dn[index + 1]) for index in range(len(stack_dn) - 1)]
    return {"folder": folder, "model": model, "adjacent_difference_std_DN": float(np.std(differences, ddof=1)), "adjacent_difference_MAE_DN": float(np.mean(np.abs(differences))), "adjacent_temporal_correlation": float(np.mean(correlations))}


def proxy_summary(folder: int, model: str, stack_dn: np.ndarray, mean_proxy: np.ndarray, median_proxy: np.ndarray) -> dict[str, Any]:
    mean_psnr, mean_ssim, median_psnr, median_ssim = [], [], [], []
    data_range = 65535.0
    for frame in stack_dn:
        mse_mean = float(np.mean((frame.astype(np.float64) - mean_proxy) ** 2))
        mse_median = float(np.mean((frame.astype(np.float64) - median_proxy) ** 2))
        mean_psnr.append(10 * math.log10(data_range**2 / max(mse_mean, 1e-12)))
        median_psnr.append(10 * math.log10(data_range**2 / max(mse_median, 1e-12)))
        mean_ssim.append(structural_similarity(mean_proxy, frame, data_range=data_range))
        median_ssim.append(structural_similarity(median_proxy, frame, data_range=data_range))
    return {"folder": folder, "model": model, "PSNR_to_temporal_mean_proxy": float(np.mean(mean_psnr)), "SSIM_to_temporal_mean_proxy": float(np.mean(mean_ssim)), "PSNR_to_temporal_median_proxy": float(np.mean(median_psnr)), "SSIM_to_temporal_median_proxy": float(np.mean(median_ssim))}


def removed_summary(folder: int, model: str, raw_stack: np.ndarray, output_stack: np.ndarray, temporal_mean: np.ndarray, max_radius: int, threshold: float) -> dict[str, Any]:
    removed = raw_stack - output_stack
    mean_abs_removed = np.mean(np.abs(removed), axis=0, dtype=np.float64)
    mean_removed = np.mean(removed, axis=0, dtype=np.float64)
    gradient = gradient_magnitude(temporal_mean)
    residual = temporal_residual(removed)
    row_profiles = np.mean(residual, axis=2, dtype=np.float64); column_profiles = np.mean(residual, axis=1, dtype=np.float64)
    power = average_power_spectrum(residual); ac = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(power)).real); ac /= ac[256, 256]
    radial_ac = radial_average(ac, max_radius)
    gradient_corr = correlation(mean_abs_removed, gradient)
    stable_corr = correlation(mean_removed, temporal_mean)
    return {"folder": folder, "model": model, "removed_mean_DN": float(np.mean(removed)), "removed_std_DN": float(np.std(removed, ddof=1)), "removed_row_energy_DN": float(np.sqrt(np.mean(row_profiles**2))), "removed_column_energy_DN": float(np.sqrt(np.mean(column_profiles**2))), "removed_radial_acf_r1": float(radial_ac[1]), "removed_gradient_correlation": gradient_corr, "removed_temporal_mean_structure_correlation": stable_corr, "structure_warning": bool(abs(gradient_corr) > threshold or abs(stable_corr) > threshold)}


def load_model(checkpoint_path: Path, channels: int, epoch: int, device: torch.device, base_channels: int) -> LightUNet:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(payload["epoch"]) != epoch: raise RuntimeError(f"Checkpoint epoch mismatch: {checkpoint_path}")
    model = LightUNet(channels, base_channels).to(device, memory_format=torch.channels_last)
    model.load_state_dict(payload["model"]); model.eval()
    return model


@torch.no_grad()
def infer(model: LightUNet, stack_dn: np.ndarray, model_name: str, cfg: dict[str, Any], device: torch.device) -> tuple[np.ndarray, pd.DataFrame]:
    divisor = float(cfg["normalization_divisor"]); batch_size = int(cfg["inference"]["batch_size"])
    outputs = np.empty_like(stack_dn, dtype=np.float32); condition_rows = []
    use_amp = device.type == "cuda" and cfg["inference"]["precision"] == "cuda_amp_float16"
    for start in range(0, len(stack_dn), batch_size):
        block_dn = stack_dn[start:start + batch_size]
        block = block_dn / divisor
        channels = [block]
        if model_name == "CG_C":
            means = np.mean(block_dn, axis=(1, 2), dtype=np.float64)
            sigmas = means * float(cfg["condition_slope"])
            valid = np.isfinite(sigmas) & (sigmas > 0) & (sigmas <= float(cfg["sigma_safety_max_DN"]))
            if not valid.all(): raise RuntimeError("CG-C sigma safety gate failed")
            maps = np.broadcast_to((sigmas / divisor)[:, None, None], block.shape).astype(np.float32)
            channels.append(maps)
            for offset, (mean, sigma) in enumerate(zip(means, sigmas)):
                low, high = cfg["calibration_signal_range_DN"]
                condition_rows.append({"frame_index": start + offset + 1, "mean_signal_DN": mean, "predicted_sigma_DN": sigma, "signal_in_calibration_range": bool(low <= mean <= high), "extrapolation_flag": bool(not low <= mean <= high), "sigma_map_value": sigma / divisor})
        inputs = torch.from_numpy(np.stack(channels, axis=1)).to(device, memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=use_amp):
            predicted_residual = model(inputs)
            clean = torch.clamp(inputs[:, :1] - predicted_residual, 0.0, 1.0)
        result = clean[:, 0].float().cpu().numpy() * divisor
        if not np.isfinite(result).all(): raise RuntimeError(f"Nonfinite output from {model_name}")
        outputs[start:start + len(result)] = result
    return outputs, pd.DataFrame(condition_rows)


def save_preview(path: Path, image: np.ndarray, low: float, high: float) -> None:
    normalized = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
    Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); repo = Path(__file__).resolve().parents[1]
    cfg_path = (repo / args.config).resolve(); cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")); out = (repo / args.output_root).resolve()
    if out.exists(): raise FileExistsError(out)
    directories = ["provenance", "configs", "manifests", "metrics", "previews"] + [f"previews/folder_{folder}" for folder in cfg["evaluation_folders"]]
    for directory in directories: (out / directory).mkdir(parents=True, exist_ok=False)
    started = now(); commit = git(repo, "rev-parse", "HEAD").strip(); status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    provenance = out / "provenance"
    (provenance / "git_commit.txt").write_text(commit + "\n", encoding="utf-8"); (provenance / "git_status_before.txt").write_text(status_before, encoding="utf-8"); (provenance / "git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8"); (provenance / "command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8"); (provenance / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"); (out / "configs" / cfg_path.name).write_bytes(cfg_path.read_bytes())
    environment = f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\ncuda={torch.cuda.is_available()}\nnumpy={np.__version__}\ntifffile={tifffile.__version__}\n"
    (provenance / "environment.txt").write_text(environment, encoding="utf-8"); (provenance / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout, encoding="utf-8"); gpu = subprocess.run(["nvidia-smi"], text=True, capture_output=True); (provenance / "gpu_info.txt").write_text(gpu.stdout, encoding="utf-8")

    checkpoint_rows = []
    for model_name, item in cfg["checkpoints"].items():
        path = repo / item["path"]; digest = sha256_file(path)
        if digest.lower() != item["sha256"].lower(): raise RuntimeError(f"Checkpoint hash mismatch: {model_name}")
        checkpoint_rows.append({"model": model_name, "path": str(path), "best_epoch": item["epoch"], "input_channels": item["input_channels"], "sha256": digest, "training_report": cfg["training_report"], "frozen": True})
    pd.DataFrame(checkpoint_rows).to_csv(out / "manifests/checkpoint_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"folder": folder, "role": "evaluation_holdout", "formal_evaluation": True} for folder in cfg["evaluation_folders"]] + [{"folder": folder, "role": "calibration_not_read", "formal_evaluation": False} for folder in cfg["calibration_folders"]]).to_csv(out / "manifests/folder_role_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"dtype": "uint16", "shape": "5120x5120", "roi_top": cfg["roi"]["top"], "roi_left": cfg["roi"]["left"], "roi_height": cfg["roi"]["height"], "roi_width": cfg["roi"]["width"], "normalization": "float32 / 65535.0", "dark_subtraction": False, "p99_scaling": False, "exposure_scaling": False}]).to_csv(out / "manifests/preprocessing_manifest.csv", index=False, encoding="utf-8-sig")

    e1_manifest = pd.read_csv(repo / cfg["e1_input_manifest"]); frame_rows = []; source_before = {}
    folder_paths: dict[int, list[Path]] = {}
    selected_folders = cfg["evaluation_folders"][:1] if args.smoke else cfg["evaluation_folders"]
    for folder in selected_folders:
        paths = sorted_tiffs(Path(cfg["data_root"]) / str(folder)); folder_paths[folder] = paths
        if len(paths) != cfg["frames_per_folder"]: raise RuntimeError(f"Frame count mismatch folder {folder}")
        if args.smoke: paths = paths[:8]; folder_paths[folder] = paths
        expected = e1_manifest[e1_manifest.folder.eq(folder)].sort_values("frame_index")
        for index, path in enumerate(paths):
            stat = path.stat(); old = expected.iloc[index]
            if stat.st_size != int(old.file_size_bytes) or stat.st_mtime_ns != int(old.mtime_ns): raise RuntimeError(f"Input drift: {path}")
            source_before[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            frame_rows.append({"folder": folder, "frame_index": index + 1, "path": str(path), "file_size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "input_audit_match": True})
    pd.DataFrame(frame_rows).to_csv(out / "manifests/evaluation_frames.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); base_channels = int(cfg["model"]["base_channels"])
    models = {name: load_model(repo / item["path"], int(item["input_channels"]), int(item["epoch"]), device, base_channels) for name, item in cfg["checkpoints"].items()}
    frame_metrics = []; temporal_rows = []; rowcol_rows = []; acf_rows = []; psd_rows = []; gradient_rows = []; brightness_rows = []; split_rows = []; consistency_rows = []; proxy_rows = []; removed_rows = []; condition_rows = []
    for folder in selected_folders:
        print(f"folder={folder} load", flush=True); raw_stack = load_roi_stack(folder_paths[folder], cfg["roi"]); temporal_mean = np.mean(raw_stack, axis=0, dtype=np.float64); temporal_median = np.median(raw_stack, axis=0)
        raw_temp = temporal_summary(folder, "raw", raw_stack, 1.0); raw_rowcol = row_column_summary(folder, "raw", raw_stack, 1.0, 1.0); raw_acf, raw_psd = spatial_summaries(folder, "raw", raw_stack, int(cfg["spatial"]["max_radius"]), 1.0)
        temporal_rows.append(raw_temp); rowcol_rows.append(raw_rowcol); acf_rows.append(raw_acf); psd_rows.append(raw_psd); gradient_rows.append(gradient_summary(folder, "raw", raw_stack, raw_stack, temporal_mean, cfg)); brightness, shifts = brightness_summary(folder, "raw", raw_stack, raw_stack); brightness_rows.append(brightness); split_rows.append(split_stable_summary(folder, "raw", raw_stack)); consistency_rows.append(frame_consistency_summary(folder, "raw", raw_stack)); proxy_rows.append(proxy_summary(folder, "raw", raw_stack, temporal_mean, temporal_median))
        for index in range(len(raw_stack)): frame_metrics.append({"folder": folder, "frame_index": index + 1, "model": "raw", "input_mean_DN": float(np.mean(raw_stack[index])), "output_mean_DN": float(np.mean(raw_stack[index])), "mean_shift_DN": 0.0, "gradient_ratio": 1.0, "output_zero_ratio": float(np.mean(raw_stack[index] == 0)), "output_saturation_ratio": float(np.mean(raw_stack[index] == 65535))})
        preview_indices = [0, len(raw_stack) // 2 - 1, len(raw_stack) - 1]
        preview_cache = {index: {"raw": raw_stack[index].copy()} for index in preview_indices}
        for model_name, model in models.items():
            print(f"folder={folder} model={model_name} inference", flush=True); output_stack, conditions = infer(model, raw_stack, model_name, cfg, device)
            if not conditions.empty: conditions.insert(0, "folder", folder); condition_rows.extend(conditions.to_dict("records"))
            temporal_rows.append(temporal_summary(folder, model_name, output_stack, raw_temp["mean_temporal_variance_DN2"])); rowcol_rows.append(row_column_summary(folder, model_name, output_stack, raw_rowcol["row_energy_DN"], raw_rowcol["column_energy_DN"])); acf, psd = spatial_summaries(folder, model_name, output_stack, int(cfg["spatial"]["max_radius"]), raw_psd["total_residual_energy_DN2"]); acf_rows.append(acf); psd_rows.append(psd); gradient_rows.append(gradient_summary(folder, model_name, raw_stack, output_stack, temporal_mean, cfg)); brightness, shifts = brightness_summary(folder, model_name, raw_stack, output_stack); brightness_rows.append(brightness); split_rows.append(split_stable_summary(folder, model_name, output_stack)); consistency_rows.append(frame_consistency_summary(folder, model_name, output_stack)); proxy_rows.append(proxy_summary(folder, model_name, output_stack, temporal_mean, temporal_median)); removed_rows.append(removed_summary(folder, model_name, raw_stack, output_stack, temporal_mean, int(cfg["spatial"]["max_radius"]), float(cfg["removed_residual"]["structure_correlation_warning_abs"])))
            for index in range(len(output_stack)):
                raw_gradient = gradient_magnitude(raw_stack[index]); output_gradient = gradient_magnitude(output_stack[index])
                frame_metrics.append({"folder": folder, "frame_index": index + 1, "model": model_name, "input_mean_DN": float(np.mean(raw_stack[index])), "output_mean_DN": float(np.mean(output_stack[index])), "mean_shift_DN": float(shifts[index]), "gradient_ratio": float(np.mean(output_gradient) / max(np.mean(raw_gradient), 1e-12)), "output_zero_ratio": float(np.mean(output_stack[index] == 0)), "output_saturation_ratio": float(np.mean(output_stack[index] == 65535))})
            for index in preview_cache: preview_cache[index][model_name] = output_stack[index].copy(); preview_cache[index][f"removed_{model_name}"] = raw_stack[index] - output_stack[index]
            del output_stack
        for index, images in preview_cache.items():
            low, high = np.percentile(images["raw"], [1, 99]); residual_limit = max(np.percentile(np.abs(images[f"removed_{name}"]), 99) for name in models)
            for name in ["raw", *models]: save_preview(out / f"previews/folder_{folder}/frame_{index + 1:03d}_{name}.png", images[name], low, high)
            for name in models: save_preview(out / f"previews/folder_{folder}/frame_{index + 1:03d}_removed_{name}.png", images[f"removed_{name}"], -residual_limit, residual_limit)
        del raw_stack

    pd.DataFrame(frame_metrics).to_csv(out / "metrics/frame_level_metrics.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(temporal_rows).to_csv(out / "metrics/folder_temporal_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(rowcol_rows).to_csv(out / "metrics/row_column_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(acf_rows).to_csv(out / "metrics/autocorrelation_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(psd_rows).to_csv(out / "metrics/psd_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(gradient_rows).to_csv(out / "metrics/gradient_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(brightness_rows).to_csv(out / "metrics/brightness_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(split_rows).to_csv(out / "metrics/split_stable_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(consistency_rows).to_csv(out / "metrics/frame_consistency_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(proxy_rows).to_csv(out / "metrics/proxy_reference_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(removed_rows).to_csv(out / "metrics/removed_residual_summary.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(condition_rows).to_csv(out / "metrics/cgc_condition_inputs.csv", index=False, encoding="utf-8-sig")
    temporal_df = pd.DataFrame(temporal_rows); rowcol_df = pd.DataFrame(rowcol_rows); acf_df = pd.DataFrame(acf_rows); psd_df = pd.DataFrame(psd_rows); gradient_df = pd.DataFrame(gradient_rows); brightness_df = pd.DataFrame(brightness_rows); split_df = pd.DataFrame(split_rows); consistency_df = pd.DataFrame(consistency_rows); proxy_df = pd.DataFrame(proxy_rows); removed_df = pd.DataFrame(removed_rows)
    comparisons = []
    for folder in selected_folders:
        for model_name in models:
            get = lambda frame, model=model_name: frame[(frame.folder == folder) & (frame.model == model)].iloc[0]
            t, rc, ac, ps, gr, br, sp, fc, pr, rr = get(temporal_df), get(rowcol_df), get(acf_df), get(psd_df), get(gradient_df), get(brightness_df), get(split_df), get(consistency_df), get(proxy_df), get(removed_df)
            raw_ac = acf_df[(acf_df.folder == folder) & acf_df.model.eq("raw")].iloc[0]; raw_ps = psd_df[(psd_df.folder == folder) & psd_df.model.eq("raw")].iloc[0]; raw_sp = split_df[(split_df.folder == folder) & split_df.model.eq("raw")].iloc[0]; raw_fc = consistency_df[(consistency_df.folder == folder) & consistency_df.model.eq("raw")].iloc[0]
            comparisons.append({"folder": folder, "model": model_name, "temporal_variance_reduction": t.temporal_variance_reduction, "row_energy_reduction": rc.row_energy_reduction, "column_energy_reduction": rc.column_energy_reduction, "radial_acf_r1_change": ac.radial_acf_r1 - raw_ac.radial_acf_r1, "radial_acf_r1_relative_change": (ac.radial_acf_r1 - raw_ac.radial_acf_r1) / max(abs(raw_ac.radial_acf_r1), 1e-12), "psd_low_fraction_change": ps.low_frequency_fraction - raw_ps.low_frequency_fraction, "psd_mid_fraction_change": ps.mid_frequency_fraction - raw_ps.mid_frequency_fraction, "psd_high_fraction_change": ps.high_frequency_fraction - raw_ps.high_frequency_fraction, "psd_energy_ratio": ps.output_input_energy_ratio, "mean_shift_DN": br.mean_shift_DN, "p95_absolute_shift_DN": br.p95_absolute_shift_DN, "max_absolute_shift_DN": br.max_absolute_shift_DN, "high_gradient_retention": gr.high_gradient_retention, "low_region_temporal_variance_ratio": gr.low_region_temporal_variance_ratio, "split_map_correlation_change": sp.split_map_correlation - raw_sp.split_map_correlation, "adjacent_difference_std_ratio": fc.adjacent_difference_std_DN / raw_fc.adjacent_difference_std_DN, "adjacent_temporal_correlation_change": fc.adjacent_temporal_correlation - raw_fc.adjacent_temporal_correlation, "proxy_mean_PSNR": pr.PSNR_to_temporal_mean_proxy, "proxy_mean_SSIM": pr.SSIM_to_temporal_mean_proxy, "proxy_median_PSNR": pr.PSNR_to_temporal_median_proxy, "proxy_median_SSIM": pr.SSIM_to_temporal_median_proxy, "removed_gradient_correlation": rr.removed_gradient_correlation, "removed_structure_correlation": rr.removed_temporal_mean_structure_correlation, "removed_structure_warning": rr.structure_warning})
    comparison_df = pd.DataFrame(comparisons); comparison_df.to_csv(out / "metrics/model_comparison.csv", index=False, encoding="utf-8-sig")
    g = comparison_df[comparison_df.model.eq("G")].set_index("folder"); cg = comparison_df[comparison_df.model.eq("CG_NC")].set_index("folder")
    benefit_checks = {
        "better_temporal_reduction_folders": int((cg.temporal_variance_reduction > g.temporal_variance_reduction).sum()),
        "row_column_not_systematically_worse": bool(((cg.row_energy_reduction >= g.row_energy_reduction).sum() >= 2) and ((cg.column_energy_reduction >= g.column_energy_reduction).sum() >= 2)),
        "brightness_not_worse": bool((cg.p95_absolute_shift_DN <= g.p95_absolute_shift_DN).sum() >= 3),
        "gradient_not_worse": bool((cg.high_gradient_retention >= g.high_gradient_retention).sum() >= 3),
        "proxy_not_systematically_lower": bool((cg.proxy_mean_PSNR >= g.proxy_mean_PSNR).sum() >= 3),
        "residual_structure_not_higher": bool((cg.removed_structure_correlation.abs() <= g.removed_structure_correlation.abs()).sum() >= 3),
        "folder_5_temporal_better": bool(5 in cg.index and cg.loc[5].temporal_variance_reduction > g.loc[5].temporal_variance_reduction),
    }
    benefit = bool(sum([benefit_checks["better_temporal_reduction_folders"] >= 3] + [bool(value) for key, value in benefit_checks.items() if key != "better_temporal_reduction_folders"]) >= 5)
    metric_frames = [temporal_df, rowcol_df, acf_df, psd_df, gradient_df, brightness_df, split_df, consistency_df, proxy_df, removed_df, pd.DataFrame(frame_metrics)]
    finite = bool(all(np.isfinite(frame.select_dtypes("number").to_numpy()).all() for frame in metric_frames)); degenerate = bool((pd.DataFrame(frame_metrics).query("model != 'raw'").groupby("model").output_zero_ratio.max() >= 0.99).any())
    brightness_warning = bool((brightness_df.query("model != 'raw'").max_absolute_shift_DN > cfg["brightness"]["warning_DN"]).any()); severe = bool((brightness_df.query("model != 'raw'").max_absolute_shift_DN > cfg["brightness"]["severe_warning_DN"]).any()); structure_warning = bool(pd.DataFrame(removed_rows).structure_warning.any()); warnings = [name for flag, name in [(brightness_warning, "BRIGHTNESS_SHIFT_WARNING"), (severe, "SEVERE_BRIGHTNESS_SHIFT_WARNING"), (structure_warning, "REMOVED_RESIDUAL_STRUCTURE_WARNING"), (not benefit, "NO_CLEAR_REAL_DOMAIN_CONDITIONAL_BENEFIT")] if flag]
    source_rows = []
    for path_string, before in source_before.items():
        stat = Path(path_string).stat(); source_rows.append({"path": path_string, "size_before": before["size"], "size_after": stat.st_size, "mtime_ns_before": before["mtime_ns"], "mtime_ns_after": stat.st_mtime_ns, "unchanged": before["size"] == stat.st_size and before["mtime_ns"] == stat.st_mtime_ns})
    source = pd.DataFrame(source_rows); source.to_csv(provenance / "source_protection.csv", index=False, encoding="utf-8-sig"); protected = bool(source.unchanged.all())
    final_status = "REAL-HOLDOUT-NO-GO" if (not finite or degenerate or not protected) else ("REAL-HOLDOUT-VALID-WITH-WARNINGS" if warnings else "REAL-HOLDOUT-RUN-VALID")
    script_paths = [Path(__file__), cfg_path, repo / "scripts/run_e2_g_cg_scaled_training.py", repo / "scripts/json_serialization.py"]; pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in script_paths]).to_csv(provenance / "script_hashes.csv", index=False, encoding="utf-8-sig"); (provenance / "git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")
    completed_frames = len(frame_rows); run_manifest = {"experiment_id": cfg["experiment_id"], "smoke": args.smoke, "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "device": str(device), "frames": completed_frames, "model_outputs": completed_frames * len(models), "models": list(models), "training_performed": False, "calibration_folders_read": False, "evaluation_folders": selected_folders, "source_data_protected": protected, "final_status": final_status}; dump_json(provenance / "run_manifest.json", run_manifest)
    verification = {"final_status": final_status, "smoke": args.smoke, "all_folders_completed": True, "raw_frames_completed": completed_frames, "outputs_completed_per_model": completed_frames, "checkpoint_hashes_verified": True, "preprocessing_frozen": True, "finite_metrics": finite, "output_degenerate": degenerate, "calibration_folders_not_evaluated": True, "training_performed": False, "source_data_protected": protected, "conditional_benefit_observed": benefit, "conditional_benefit_checks": benefit_checks, "warnings": warnings, "provenance_complete": True}; dump_json(out / "verification_status.json", verification)
    report = ["# E3 Real ICCD Holdout Validation", "", f"Status: `{final_status}`", "", f"Conditional benefit observed: `{benefit}`", "", f"Warnings: {', '.join(warnings) if warnings else 'none'}", "", "This no-reference controlled evaluation reports changes in observed repeated-frame statistics. It does not establish clean-image recovery or real PSNR/SSIM."]
    (out / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv": hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig"); print(json.dumps(verification, indent=2)); return 0 if final_status != "REAL-HOLDOUT-NO-GO" else 5


if __name__ == "__main__": raise SystemExit(main())
