"""Run the pre-registered G/CG scaled-content training experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import kurtosis, skew
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset

from json_serialization import dump_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(content_sha256: str, condition_id: str, base_seed: int) -> int:
    payload = f"{content_sha256}|{condition_id}|{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=check)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def anchors_from_calibration(e1: pd.DataFrame, calibration_folders: list[int], slope: float) -> pd.DataFrame:
    calibration = e1[e1.folder.isin(calibration_folders)].copy()
    if sorted(calibration.folder.astype(int).tolist()) != sorted(calibration_folders):
        raise RuntimeError("Calibration folder statistics are incomplete")
    values = np.sort(calibration.mean_signal.to_numpy(np.float64))
    anchors = [
        ("low", float(values[0]), "minimum calibration-folder mean signal"),
        ("mid", float(np.median(values)), "arithmetic median of six calibration-folder means"),
        ("high", float(values[-1]), "maximum calibration-folder mean signal"),
    ]
    rows = []
    for condition_id, target, rule in anchors:
        rows.append(
            {
                "condition_id": condition_id,
                "target_signal_DN": target,
                "source_calibration_folders": ";".join(map(str, calibration_folders)),
                "anchor_rule": rule,
                "predicted_sigma_DN": slope * target,
            }
        )
    result = pd.DataFrame(rows)
    if not ((result.predicted_sigma_DN > 0) & (result.predicted_sigma_DN <= 300)).all():
        raise RuntimeError("Anchor sigma safety gate failed")
    return result


def pmrid_patch_coordinates(file_hash: str, height: int, width: int, patch_size: int = 512) -> tuple[int, int]:
    top = 2 * (int(file_hash[:8], 16) % ((height - patch_size) // 2 + 1))
    left = 2 * (int(file_hash[8:16], 16) % ((width - patch_size) // 2 + 1))
    return top, left


def scale_uint16(raw: np.ndarray, target_signal_dn: float) -> tuple[np.ndarray, dict[str, float | int]]:
    if raw.dtype != np.uint16:
        raise TypeError(f"Expected uint16, got {raw.dtype}")
    source_mean = float(np.mean(raw, dtype=np.float64))
    if not math.isfinite(source_mean) or source_mean <= 0:
        raise ValueError("Source mean must be finite and positive")
    gain = target_signal_dn / source_mean
    scaled_float = raw.astype(np.float32) * np.float32(gain)
    scaled = np.rint(np.clip(scaled_float, 0.0, 65535.0)).astype(np.uint16)
    scaled_mean = float(np.mean(scaled, dtype=np.float64))
    source_zero = float(np.mean(raw == 0))
    zero_ratio = float(np.mean(scaled == 0))
    metrics = {
        "source_mean_DN": source_mean,
        "target_iccd_signal_DN": target_signal_dn,
        "gain": gain,
        "scaled_mean_DN": scaled_mean,
        "relative_mean_error": abs(scaled_mean - target_signal_dn) / target_signal_dn,
        "source_zero_ratio": source_zero,
        "scaled_zero_ratio": zero_ratio,
        "added_zero_ratio": max(0.0, zero_ratio - source_zero),
        "clipping_high_ratio": float(np.mean(scaled == 65535)),
        "unique_values": int(np.unique(scaled).size),
        "scaled_std_DN": float(np.std(scaled, dtype=np.float64)),
        "effective_dynamic_range_DN": float(np.percentile(scaled, 99) - np.percentile(scaled, 1)),
        "round_trip_max_error_DN": float(np.max(np.abs(scaled_float - scaled.astype(np.float32)))),
    }
    return scaled, metrics


def scaling_pass(metrics: dict[str, Any], gates: dict[str, float]) -> tuple[bool, str]:
    numeric = [value for value in metrics.values() if isinstance(value, (float, int))]
    failures = []
    if not all(math.isfinite(float(value)) for value in numeric): failures.append("NONFINITE")
    if metrics["source_mean_DN"] <= 0 or metrics["gain"] <= 0: failures.append("NONPOSITIVE_SCALE")
    if metrics["relative_mean_error"] >= gates["relative_mean_error_max"]: failures.append("MEAN_ERROR")
    if metrics["clipping_high_ratio"] >= gates["high_clipping_ratio_max"]: failures.append("HIGH_CLIPPING")
    if metrics["added_zero_ratio"] >= gates["added_zero_ratio_max"]: failures.append("ZERO_INCREASE")
    if metrics["unique_values"] < gates["unique_values_min"]: failures.append("UNIQUE_VALUES")
    if metrics["scaled_std_DN"] < gates["scaled_std_DN_min"]: failures.append("STD_COLLAPSE")
    if metrics["round_trip_max_error_DN"] > gates["round_trip_max_error_DN"]: failures.append("ROUND_TRIP")
    return not failures, ";".join(failures)


def pair_metrics(reference_uint16: np.ndarray, sigma_dn: float, seed: int) -> tuple[np.ndarray, dict[str, float | int]]:
    reference = reference_uint16.astype(np.float32) / 65535.0
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=reference.shape).astype(np.float32)
    residual = z * np.float32(sigma_dn / 65535.0)
    noisy_unclipped = reference + residual
    noisy = np.clip(noisy_unclipped, 0.0, 1.0)
    noisy_uint16 = np.rint(noisy * 65535.0).astype(np.uint16)
    noisy_round_trip = noisy_uint16.astype(np.float32) / 65535.0
    reconstructed_residual_dn = (noisy_round_trip - reference) * 65535.0
    clipped_float_residual_dn = (noisy - reference) * 65535.0
    residual_dn = residual * 65535.0
    source_zero = float(np.mean(reference_uint16 == 0))
    source_one = float(np.mean(reference_uint16 == 65535))
    metrics = {
        "residual_mean_DN": float(np.mean(residual_dn, dtype=np.float64)),
        "residual_std_DN": float(np.std(residual_dn, dtype=np.float64)),
        "residual_std_relative_error": abs(float(np.std(residual_dn, dtype=np.float64)) - sigma_dn) / sigma_dn,
        "residual_skewness": float(skew(residual_dn.ravel())),
        "residual_excess_kurtosis": float(kurtosis(residual_dn.ravel())),
        "negative_before_clipping_ratio": float(np.mean(noisy_unclipped < 0)),
        "above_one_before_clipping_ratio": float(np.mean(noisy_unclipped > 1)),
        "added_zero_ratio": max(0.0, float(np.mean(noisy == 0)) - source_zero),
        "added_one_ratio": max(0.0, float(np.mean(noisy == 1)) - source_one),
        "brightness_shift_DN": float((np.mean(noisy, dtype=np.float64) - np.mean(reference, dtype=np.float64)) * 65535.0),
        "gradient_ratio": gradient_energy(noisy) / max(gradient_energy(reference), 1e-20),
        "noisy_round_trip_max_error_DN": float(np.max(np.abs(noisy_round_trip - noisy)) * 65535.0),
        "residual_reconstruction_max_error_DN": float(np.max(np.abs(reconstructed_residual_dn - clipped_float_residual_dn))),
    }
    return z, metrics


def pair_pass(metrics: dict[str, Any], gates: dict[str, float]) -> tuple[bool, str]:
    failures = []
    if not all(math.isfinite(float(value)) for value in metrics.values()): failures.append("NONFINITE")
    if metrics["residual_std_relative_error"] >= gates["residual_std_relative_error_max"]: failures.append("SIGMA_ERROR")
    if abs(metrics["brightness_shift_DN"]) >= gates["absolute_brightness_shift_DN_max"]: failures.append("BRIGHTNESS")
    if metrics["added_zero_ratio"] >= gates["added_zero_ratio_max"]: failures.append("ZERO_CLIPPING")
    if metrics["added_one_ratio"] >= gates["added_one_ratio_max"]: failures.append("ONE_CLIPPING")
    if metrics["noisy_round_trip_max_error_DN"] > gates["noisy_round_trip_max_error_DN"]: failures.append("NOISY_ROUND_TRIP")
    if metrics["residual_reconstruction_max_error_DN"] > gates["residual_reconstruction_max_error_DN"]: failures.append("RESIDUAL_ROUND_TRIP")
    return not failures, ";".join(failures)


def gradient_energy(image: np.ndarray) -> float:
    horizontal = np.diff(image, axis=-1)
    vertical = np.diff(image, axis=-2)
    return float((np.mean(horizontal * horizontal) + np.mean(vertical * vertical)) / 2.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightUNet(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


@dataclass
class PairRecord:
    pair_id: str
    content_id: str
    condition_id: str
    scene_id: str
    reference_uint16: np.ndarray
    z: np.ndarray
    sigma_dn: float
    seed: int


class PairDataset(Dataset):
    def __init__(self, records: list[PairRecord], conditional_channel: bool, augment: bool) -> None:
        self.records = records
        self.conditional_channel = conditional_channel
        self.augment = augment
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        reference = record.reference_uint16.astype(np.float32) / 65535.0
        noisy = np.clip(reference + record.z * np.float32(record.sigma_dn / 65535.0), 0.0, 1.0)
        if self.augment:
            aug_seed = stable_seed(str(record.seed), str(self.epoch), index)
            k = aug_seed % 4
            if (aug_seed >> 2) & 1:
                reference = np.flip(reference, axis=1); noisy = np.flip(noisy, axis=1)
            if (aug_seed >> 3) & 1:
                reference = np.flip(reference, axis=0); noisy = np.flip(noisy, axis=0)
            if k:
                reference = np.rot90(reference, k); noisy = np.rot90(noisy, k)
        noisy = np.ascontiguousarray(noisy)
        reference = np.ascontiguousarray(reference)
        channels = [noisy]
        if self.conditional_channel:
            channels.append(np.full_like(noisy, record.sigma_dn / 65535.0))
        return torch.from_numpy(np.stack(channels)), torch.from_numpy(reference[None]), index


@torch.no_grad()
def validation_loss_psnr(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    l1_values, psnr_values = [], []
    for inputs, target, _ in loader:
        inputs, target = inputs.to(device), target.to(device)
        clean_hat = inputs[:, :1] - model(inputs)
        l1_values.extend(torch.mean(torch.abs(clean_hat - target), dim=(1, 2, 3)).cpu().tolist())
        mse = torch.mean((torch.clamp(clean_hat, 0, 1) - target) ** 2, dim=(1, 2, 3))
        psnr_values.extend((-10 * torch.log10(torch.clamp(mse, min=1e-12))).cpu().tolist())
    return float(np.mean(l1_values)), float(np.mean(psnr_values))


def train_experiment(
    name: str, train_records: list[PairRecord], validation_records: list[PairRecord], conditional_channel: bool,
    cfg: dict[str, Any], output_root: Path, device: torch.device, smoke: bool,
) -> tuple[pd.DataFrame, Path, Path, dict[str, Any]]:
    train_cfg = cfg["training"]
    set_seed(int(train_cfg["model_seed"]))
    model = LightUNet(2 if conditional_channel else 1, int(cfg["network"]["base_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    train_dataset = PairDataset(train_records, conditional_channel, augment=True)
    validation_dataset = PairDataset(validation_records, conditional_channel, augment=False)
    generator = torch.Generator().manual_seed(int(train_cfg["dataloader_seed"]))
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    epochs = 1 if smoke else int(train_cfg["epochs"])
    metrics = []
    best_psnr, best_l1, best_epoch = -math.inf, math.inf, -1
    checkpoint_dir = output_root / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    best_path, final_path = checkpoint_dir / "best.pt", checkpoint_dir / "final.pt"
    start = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for inputs, target, _ in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_residual = model(inputs)
            clean_hat = inputs[:, :1] - predicted_residual
            loss = F.l1_loss(clean_hat, target)
            if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite training loss in {name} epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_l1, val_psnr = validation_loss_psnr(model, validation_loader, device)
        train_l1 = float(np.mean(losses))
        improved = val_psnr > best_psnr + 1e-12 or (abs(val_psnr - best_psnr) <= 1e-12 and val_l1 < best_l1)
        if improved:
            best_psnr, best_l1, best_epoch = val_psnr, val_l1, epoch
            torch.save({"experiment": name, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, best_path)
        metrics.append({"experiment": name, "epoch": epoch, "train_l1": train_l1, "validation_l1": val_l1, "validation_psnr": val_psnr, "is_best": improved})
        print(f"{name} epoch={epoch:02d}/{epochs} train_l1={train_l1:.8f} val_l1={val_l1:.8f} val_psnr={val_psnr:.4f}", flush=True)
    torch.save({"experiment": name, "epoch": epochs, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, final_path)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return pd.DataFrame(metrics), best_path, final_path, {"best_epoch": best_epoch, "best_validation_psnr": best_psnr, "best_validation_l1": best_l1, "elapsed_seconds": elapsed, "peak_gpu_bytes": peak, "parameter_count": sum(p.numel() for p in model.parameters())}


def image_metrics(reference: np.ndarray, noisy: np.ndarray, output: np.ndarray, true_residual: np.ndarray, predicted_residual: np.ndarray) -> dict[str, float]:
    output_clip = np.clip(output, 0, 1)
    mse = float(np.mean((output_clip - reference) ** 2, dtype=np.float64))
    mae = float(np.mean(np.abs(output_clip - reference), dtype=np.float64))
    noisy_mse = float(np.mean((noisy - reference) ** 2, dtype=np.float64))
    return {
        "noisy_psnr": -10 * math.log10(max(noisy_mse, 1e-20)),
        "noisy_ssim": float(structural_similarity(reference, noisy, data_range=1.0)),
        "output_psnr": -10 * math.log10(max(mse, 1e-20)),
        "output_ssim": float(structural_similarity(reference, output_clip, data_range=1.0)),
        "output_mae": mae,
        "output_rmse": math.sqrt(mse),
        "output_mean_shift_DN": float((np.mean(output_clip, dtype=np.float64) - np.mean(reference, dtype=np.float64)) * 65535),
        "gradient_energy_ratio": gradient_energy(output_clip) / max(gradient_energy(reference), 1e-20),
        "output_zero_ratio": float(np.mean(output_clip == 0)),
        "output_one_ratio": float(np.mean(output_clip == 1)),
        "predicted_residual_std_DN": float(np.std(predicted_residual, dtype=np.float64) * 65535),
        "residual_reconstruction_rmse_DN": float(np.sqrt(np.mean((predicted_residual - true_residual) ** 2, dtype=np.float64)) * 65535),
    }


@torch.no_grad()
def evaluate_best(name: str, checkpoint: Path, records: list[PairRecord], conditional_channel: bool, cfg: dict[str, Any], device: torch.device) -> pd.DataFrame:
    model = LightUNet(2 if conditional_channel else 1, int(cfg["network"]["base_channels"])).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"]); model.eval()
    rows = []
    for record in records:
        reference = record.reference_uint16.astype(np.float32) / 65535.0
        true_residual = record.z * np.float32(record.sigma_dn / 65535.0)
        noisy = np.clip(reference + true_residual, 0, 1)
        channels = [noisy]
        if conditional_channel: channels.append(np.full_like(noisy, record.sigma_dn / 65535.0))
        inputs = torch.from_numpy(np.stack(channels)[None]).to(device)
        predicted = model(inputs).cpu().numpy()[0, 0]
        output = noisy - predicted
        metrics = image_metrics(reference, noisy, output, true_residual, predicted)
        rows.append({"experiment": name, "pair_id": record.pair_id, "content_id": record.content_id, "scene_id": record.scene_id, "condition_id": record.condition_id, "sigma_DN": record.sigma_dn, "seed": record.seed, **metrics})
    return pd.DataFrame(rows)


def write_provenance_before(repo: Path, out: Path, cfg_path: Path, cfg: dict[str, Any], command: str) -> str:
    provenance = out / "provenance"
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout
    (provenance / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (provenance / "git_status_before.txt").write_text(status, encoding="utf-8")
    (provenance / "git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD").stdout, encoding="utf-8")
    (provenance / "command.txt").write_text(command + "\n", encoding="utf-8")
    (provenance / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    environment = f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\ncuda_available={torch.cuda.is_available()}\nnumpy={np.__version__}\npandas={pd.__version__}\ntifffile={tifffile.__version__}\n"
    (provenance / "environment.txt").write_text(environment, encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout
    (provenance / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    gpu = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    (provenance / "gpu_info.txt").write_text(gpu.stdout if gpu.returncode == 0 else gpu.stderr, encoding="utf-8")
    return commit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = (repo / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = (repo / args.output_root).resolve()
    if out.exists(): raise FileExistsError(f"Output exists: {out}")
    for directory in ["provenance", "configs", "manifests", "scale_preflight", "condition_model", "generated_data/limited_previews", "training/G", "training/CG_NC", "training/CG_C", "metrics", "checkpoints", "logs"]:
        (out / directory).mkdir(parents=True, exist_ok=False)
    started = now()
    commit = write_provenance_before(repo, out, cfg_path, cfg, subprocess.list2cmdline(sys.argv))
    (out / "configs" / cfg_path.name).write_bytes(cfg_path.read_bytes())
    (out / "configs" / Path(cfg["condition_model"]).name).write_bytes((repo / cfg["condition_model"]).read_bytes())

    split = pd.read_csv(repo / cfg["folder_split"])
    calibration_ids = split.loc[split.role.eq("calibration"), "folder"].astype(int).tolist()
    evaluation_ids = split.loc[split.role.eq("evaluation"), "folder"].astype(int).tolist()
    if calibration_ids != cfg["calibration_folders"] or evaluation_ids != cfg["evaluation_folders"]:
        raise RuntimeError("ICCD folder split drift")
    e1 = pd.read_csv(repo / cfg["e1_statistics"])
    anchors = anchors_from_calibration(e1, calibration_ids, float(cfg["condition_slope"]))
    anchors.to_csv(out / "manifests/iccd_signal_anchors.csv", index=False, encoding="utf-8-sig")
    condition_model = yaml.safe_load((repo / cfg["condition_model"]).read_text(encoding="utf-8"))
    if abs(float(condition_model["b"]) - float(cfg["condition_slope"])) > 1e-15: raise RuntimeError("Condition slope drift")
    dump_json(out / "condition_model/signal_condition_model.json", condition_model)

    scmos_manifest = pd.read_csv(repo / cfg["scmos_manifest"])
    pmrid_manifest = pd.read_csv(repo / cfg["pmrid_scene_manifest"])
    benchmark = json.loads(Path(cfg["pmrid_benchmark"]).read_text(encoding="utf-8"))
    if len(scmos_manifest) != 100 or len(pmrid_manifest) != 39 or len(benchmark) != 39: raise RuntimeError("Input count drift")
    if not set(scmos_manifest.allowed_role).issubset({"debug_only"}): raise RuntimeError("sCMOS source role drift")
    source_before: dict[str, dict[str, Any]] = {}
    for _, row in scmos_manifest.iterrows():
        path = Path(row.absolute_path); digest = sha256_file(path)
        if digest != row.sha256: raise RuntimeError(f"sCMOS hash drift: {path}")
        source_before[str(path)] = {"sha256": digest, "mtime_ns": path.stat().st_mtime_ns}
    for _, row in pmrid_manifest.iterrows():
        path = Path(row.paired_gt_path); digest = sha256_file(path)
        if digest != row.SHA256: raise RuntimeError(f"PMRID hash drift: {path}")
        source_before[str(path)] = {"sha256": digest, "mtime_ns": path.stat().st_mtime_ns}

    selected_scmos = scmos_manifest.sort_values("content_id").head(2 if args.smoke else 100)
    selected_pmrid = pmrid_manifest.sort_values("benchmark_entry").head(2 if args.smoke else 39)
    scaling_rows, scmos_scaled_rows, pmrid_scaled_rows = [], [], []
    train_scaled: list[tuple[str, str, str, np.ndarray, int, float]] = []
    validation_scaled: list[tuple[str, str, str, np.ndarray, int, float]] = []
    anchor_map = {row.condition_id: (float(row.target_signal_DN), float(row.predicted_sigma_DN)) for _, row in anchors.iterrows()}
    roi = cfg["scmos_roi"]
    for _, row in selected_scmos.iterrows():
        raw_full = tifffile.imread(row.absolute_path)
        raw = raw_full[roi["top"]:roi["top"] + roi["height"], roi["left"]:roi["left"] + roi["width"]]
        for condition_id, (target, sigma) in anchor_map.items():
            scaled, metrics = scale_uint16(raw, target); passed, reason = scaling_pass(metrics, cfg["scaling_gates"])
            seed = stable_seed(row.sha256, condition_id, int(cfg["base_seed"]))
            record = {"source": "sCMOS", "content_id": row.content_id, "content_sha256": row.sha256, "condition_id": condition_id, "seed": seed, **metrics, "scale_gate_pass": passed, "failure_reason": reason}
            scaling_rows.append(record); scmos_scaled_rows.append(record); train_scaled.append((row.content_id, condition_id, "single_unknown_source_group", scaled, seed, sigma))
    for _, row in selected_pmrid.iterrows():
        item = benchmark[int(row.benchmark_entry)]
        path = Path(row.paired_gt_path); height, width = item["meta"]["shape"]
        top, left = pmrid_patch_coordinates(row.SHA256, height, width)
        raw_map = np.memmap(path, dtype=np.uint16, mode="r", shape=(height, width))
        raw = np.asarray(raw_map[top:top + 512, left:left + 512])
        for condition_id, (target, sigma) in anchor_map.items():
            scaled, metrics = scale_uint16(raw, target); passed, reason = scaling_pass(metrics, cfg["scaling_gates"])
            seed = stable_seed(row.SHA256, condition_id, int(cfg["base_seed"]))
            record = {"source": "PMRID", "content_id": row.pmrid_content_id, "content_sha256": row.SHA256, "scene_id": row.scene_id, "condition_id": condition_id, "seed": seed, "patch_top": top, "patch_left": left, **metrics, "scale_gate_pass": passed, "failure_reason": reason}
            scaling_rows.append(record); pmrid_scaled_rows.append(record); validation_scaled.append((row.pmrid_content_id, condition_id, row.scene_id, scaled, seed, sigma))
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(out / "scale_preflight/scaling_pair_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scmos_scaled_rows).to_csv(out / "manifests/scmos_scaled_training_contents.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pmrid_scaled_rows).to_csv(out / "manifests/pmrid_scaled_validation_contents.csv", index=False, encoding="utf-8-sig")
    scaling.groupby(["source", "condition_id"]).agg(gain_min=("gain", "min"), gain_max=("gain", "max"), max_relative_mean_error=("relative_mean_error", "max"), min_unique_values=("unique_values", "min"), min_scaled_std_DN=("scaled_std_DN", "min"), max_high_clipping=("clipping_high_ratio", "max"), max_added_zero=("added_zero_ratio", "max"), all_pass=("scale_gate_pass", "all")).reset_index().to_csv(out / "scale_preflight/scaling_summary.csv", index=False, encoding="utf-8-sig")
    scaling[~scaling.scale_gate_pass].to_csv(out / "scale_preflight/warnings.csv", index=False, encoding="utf-8-sig")
    if not scaling.scale_gate_pass.all():
        dump_json(out / "verification_status.json", {"final_status": "SCALE-PREFLIGHT-NO-GO", "failed_count": int((~scaling.scale_gate_pass).sum())})
        return 3

    pair_rows, train_g, train_cg, validation_cg = [], [], [], []
    for content_id, condition_id, scene_id, reference, seed, cg_sigma in train_scaled:
        for pair_type, sigma in [("G", float(cfg["g_sigma_DN"])), ("CG_SHARED", cg_sigma)]:
            z, metrics = pair_metrics(reference, sigma, seed); passed, reason = pair_pass(metrics, cfg["pair_gates"])
            pair_id = f"train_{content_id}_{condition_id}_{pair_type}"
            pair_rows.append({"split": "training", "pair_type": pair_type, "pair_id": pair_id, "content_id": content_id, "condition_id": condition_id, "scene_id": scene_id, "sigma_DN": sigma, "seed": seed, **metrics, "pair_gate_pass": passed, "failure_reason": reason})
            record = PairRecord(pair_id, content_id, condition_id, scene_id, reference, z, sigma, seed)
            (train_g if pair_type == "G" else train_cg).append(record)
    for content_id, condition_id, scene_id, reference, seed, sigma in validation_scaled:
        z, metrics = pair_metrics(reference, sigma, seed); passed, reason = pair_pass(metrics, cfg["pair_gates"])
        pair_id = f"validation_{content_id}_{condition_id}_CG_SHARED"
        pair_rows.append({"split": "validation", "pair_type": "CG_SHARED", "pair_id": pair_id, "content_id": content_id, "condition_id": condition_id, "scene_id": scene_id, "sigma_DN": sigma, "seed": seed, **metrics, "pair_gate_pass": passed, "failure_reason": reason})
        validation_cg.append(PairRecord(pair_id, content_id, condition_id, scene_id, reference, z, sigma, seed))
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame[pair_frame.split.eq("training")].to_csv(out / "manifests/training_pairs.csv", index=False, encoding="utf-8-sig")
    pair_frame[pair_frame.split.eq("validation")].to_csv(out / "manifests/validation_pairs.csv", index=False, encoding="utf-8-sig")
    pair_frame[~pair_frame.pair_gate_pass].to_csv(out / "metrics/warnings.csv", index=False, encoding="utf-8-sig")
    if not pair_frame.pair_gate_pass.all():
        dump_json(out / "verification_status.json", {"final_status": "PAIR-PREFLIGHT-NO-GO", "failed_count": int((~pair_frame.pair_gate_pass).sum())})
        return 4

    pd.DataFrame([
        {"source": "sCMOS_500ms_100", "role": "exploratory_training_only", "source_count": len(selected_scmos), "scene_status": "single_unknown_source_group"},
        {"source": "PMRID_GT_RAW", "role": "validation_content_only", "source_count": len(selected_pmrid), "scene_status": "Scene1-Scene4"},
        {"source": "ICCD_calibration", "role": "condition_parameter_calibration_only", "source_count": 6, "scene_status": "folders 1,4,7,8,10,13"},
        {"source": "ICCD_evaluation", "role": "holdout_not_read", "source_count": 4, "scene_status": "folders 2,5,9,11"},
    ]).to_csv(out / "manifests/data_role_manifest.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_frames, training_info, checkpoints = [], {}, {}
    experiments = [("G", train_g, False), ("CG_NC", train_cg, False), ("CG_C", train_cg, True)]
    for name, records, conditional_channel in experiments:
        frame, best, final, info = train_experiment(name, records, validation_cg, conditional_channel, cfg, out, device, args.smoke)
        epoch_frames.append(frame); training_info[name] = info; checkpoints[name] = best
    epochs = pd.concat(epoch_frames, ignore_index=True)
    epochs.to_csv(out / "metrics/epoch_metrics.csv", index=False, encoding="utf-8-sig")
    validation_frames = []
    for name, _, conditional_channel in experiments:
        validation_frames.append(evaluate_best(name, checkpoints[name], validation_cg, conditional_channel, cfg, device))
    validation_metrics = pd.concat(validation_frames, ignore_index=True)
    validation_metrics.to_csv(out / "metrics/validation_pair_metrics.csv", index=False, encoding="utf-8-sig")
    scene_summary = validation_metrics.groupby(["experiment", "scene_id"]).agg(pair_count=("pair_id", "count"), noisy_psnr=("noisy_psnr", "mean"), noisy_ssim=("noisy_ssim", "mean"), output_psnr=("output_psnr", "mean"), output_ssim=("output_ssim", "mean"), output_mae=("output_mae", "mean"), mean_shift_DN=("output_mean_shift_DN", "mean")).reset_index()
    condition_summary = validation_metrics.groupby(["experiment", "condition_id"]).agg(pair_count=("pair_id", "count"), noisy_psnr=("noisy_psnr", "mean"), noisy_ssim=("noisy_ssim", "mean"), output_psnr=("output_psnr", "mean"), output_ssim=("output_ssim", "mean"), output_mae=("output_mae", "mean"), max_abs_mean_shift_DN=("output_mean_shift_DN", lambda x: float(np.max(np.abs(x))))).reset_index()
    scene_summary.to_csv(out / "metrics/scene_summary.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(out / "metrics/condition_summary.csv", index=False, encoding="utf-8-sig")
    comparison_rows = []
    for name, group in validation_metrics.groupby("experiment"):
        info = training_info[name]
        experiment_epochs = epochs[epochs.experiment.eq(name)].sort_values("epoch")
        final_train_l1 = float(experiment_epochs.iloc[-1].train_l1)
        comparison_rows.append({"experiment": name, "pair_count": len(group), "noisy_psnr": group.noisy_psnr.mean(), "noisy_ssim": group.noisy_ssim.mean(), "output_psnr": group.output_psnr.mean(), "output_ssim": group.output_ssim.mean(), "output_mae": group.output_mae.mean(), "output_rmse": group.output_rmse.mean(), "max_abs_output_mean_shift_DN": np.abs(group.output_mean_shift_DN).max(), "max_output_zero_ratio": group.output_zero_ratio.max(), "max_output_one_ratio": group.output_one_ratio.max(), "condition_psnr_variance": condition_summary[condition_summary.experiment.eq(name)].output_psnr.var(ddof=0), "first_train_l1": float(experiment_epochs.iloc[0].train_l1), "final_train_l1": final_train_l1, "train_validation_l1_gap": float(info["best_validation_l1"] - final_train_l1), **info})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "metrics/experiment_comparison.csv", index=False, encoding="utf-8-sig")

    finite = bool(np.isfinite(epochs.select_dtypes("number")).all().all() and np.isfinite(validation_metrics.select_dtypes("number")).all().all())
    completed_epochs = 1 if args.smoke else int(cfg["training"]["epochs"])
    training_completed = all(int((epochs.experiment == name).sum()) == completed_epochs for name in ["G", "CG_NC", "CG_C"])
    output_not_degenerate = bool((comparison.max_output_zero_ratio < 0.99).all() and (comparison.max_output_one_ratio < 0.99).all())
    psnr_improved = bool((comparison.output_psnr > comparison.noisy_psnr).all())
    ssim_not_lower = bool((comparison.output_ssim >= comparison.noisy_ssim).all())
    mean_shift_ok = bool((comparison.max_abs_output_mean_shift_DN <= cfg["evaluation"]["output_mean_shift_warning_DN"]).all())
    train_loss_decreased = bool((comparison.final_train_l1 < comparison.first_train_l1).all())
    warnings = []
    if not mean_shift_ok: warnings.append("OUTPUT_MEAN_SHIFT_WARNING")
    if not ssim_not_lower: warnings.append("SSIM_NOT_IMPROVED_FOR_ALL_ARMS")
    if not psnr_improved: warnings.append("PSNR_NOT_IMPROVED_FOR_ALL_ARMS")
    if not train_loss_decreased: warnings.append("TRAIN_LOSS_NOT_DECREASED_FOR_ALL_ARMS")
    if not all([training_completed, finite, output_not_degenerate]):
        final_status = "TRAINING-FAILED"
    elif all([psnr_improved, ssim_not_lower, mean_shift_ok, train_loss_decreased]):
        final_status = "TRAINING-RUN-VALID"
    else:
        final_status = "TRAINING-RUN-VALID-WITH-LIMITATIONS"
    conditional_benefit = bool(comparison.set_index("experiment").loc["CG_C", "output_psnr"] > comparison.set_index("experiment").loc["G", "output_psnr"] and comparison.set_index("experiment").loc["CG_C", "output_ssim"] > comparison.set_index("experiment").loc["G", "output_ssim"])

    source_rows = []
    for path_string, before in source_before.items():
        path = Path(path_string); after_hash = sha256_file(path); after_mtime = path.stat().st_mtime_ns
        source_rows.append({"path": path_string, "sha256_before": before["sha256"], "sha256_after": after_hash, "mtime_ns_before": before["mtime_ns"], "mtime_ns_after": after_mtime, "unchanged": before["sha256"] == after_hash and before["mtime_ns"] == after_mtime})
    source_protection = pd.DataFrame(source_rows)
    source_protection.to_csv(out / "provenance/source_protection.csv", index=False, encoding="utf-8-sig")
    protected = bool(source_protection.unchanged.all())
    script_paths = [Path(__file__), cfg_path, repo / cfg["condition_model"], repo / "scripts/json_serialization.py"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in script_paths]).to_csv(out / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")
    (out / "provenance/git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout, encoding="utf-8")
    run_manifest = {"experiment_id": cfg["experiment_id"], "smoke": args.smoke, "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "device": str(device), "final_status": final_status, "source_data_protected": protected, "evaluation_iccd_raw_read": False, "training_info": training_info}
    dump_json(out / "provenance/run_manifest.json", run_manifest)
    verification = {"experiment_id": cfg["experiment_id"], "final_status": final_status, "scale_preflight_passed": True, "scale_pairs": len(scaling), "pair_preflight_passed": True, "noise_pairs": len(pair_frame), "training_completed": training_completed, "completed_epochs": completed_epochs, "finite_metrics": finite, "output_not_degenerate": output_not_degenerate, "train_loss_decreased_all_arms": train_loss_decreased, "psnr_improved_all_arms": psnr_improved, "ssim_not_lower_all_arms": ssim_not_lower, "mean_shift_gate_passed": mean_shift_ok, "conditional_benefit_observed": conditional_benefit, "warnings": warnings, "source_data_protected": protected, "evaluation_iccd_holdout_preserved": True, "provenance_complete": True}
    dump_json(out / "verification_status.json", verification)
    report = ["# E2 G/CG Scaled Training", "", f"Status: `{final_status}`", "", f"Scale preflight: {len(scaling)}/{len(scaling)} passed. Pair preflight: {len(pair_frame)}/{len(pair_frame)} passed.", "", comparison.to_markdown(index=False), "", "This is independent-content synthetic validation only. It is not real ICCD denoising evidence or cross-camera radiometric calibration."]
    (out / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv": hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
