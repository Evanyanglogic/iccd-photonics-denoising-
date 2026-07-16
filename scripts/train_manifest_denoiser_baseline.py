"""Train a manifest-driven supervised denoising baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_data import ICCDPairDataset
from src.iccd_eval.metrics import image_quality


@dataclass(frozen=True)
class TrainConfig:
    experiment_id: str
    train_pairs: str
    train_splits: str
    train_split: str
    val_pairs: str
    val_splits: str
    val_split: str
    output_dir: str
    range_max: float
    patch_size: int
    val_patch_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    channels: int
    depth: int
    seed: int
    num_workers: int
    grad_clip: float
    amp: bool
    input_channels: int
    condition_column: str
    condition_value_scale: float
    max_train_batches: int
    max_val_batches: int
    device: str
    git_commit: str


def main() -> int:
    args = parse_args()
    config = build_config(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)

    set_seed(config.seed)
    write_json(output_dir / "config.json", asdict(config))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device(config.device)
    train_dataset = ICCDPairDataset(
        pairs_csv=config.train_pairs,
        splits_yaml=config.train_splits,
        split=config.train_split,
        range_max=config.range_max,
        patch_size=config.patch_size,
        crop_mode="random",
        augment=True,
        seed=config.seed,
    )
    val_dataset = ICCDPairDataset(
        pairs_csv=config.val_pairs,
        splits_yaml=config.val_splits,
        split=config.val_split,
        range_max=config.range_max,
        patch_size=config.val_patch_size,
        crop_mode="center",
        augment=False,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ResidualDenoiser(
        channels=config.channels,
        depth=config.depth,
        input_channels=config.input_channels,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(enabled=config.amp and device.type == "cuda")

    metrics_path = output_dir / "metrics.csv"
    val_rows_path = output_dir / "validation_predictions.csv"
    best_psnr = float("-inf")
    metric_rows: list[dict[str, Any]] = []
    last_val_rows: list[dict[str, Any]] = []
    start_time = time.time()

    print(f"Experiment: {config.experiment_id}")
    print(f"Device: {device}")
    print(f"Train/val samples: {len(train_dataset)} / {len(val_dataset)}")
    print(f"Model parameters: {count_parameters(model)}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_summary, val_rows = validate(model, val_loader, criterion, device, config)
        last_val_rows = val_rows
        is_best = val_summary["psnr_mean"] > best_psnr
        if is_best:
            best_psnr = val_summary["psnr_mean"]

        row = {
            "epoch": epoch,
            "train_l1": train_loss,
            "val_l1": val_summary["loss_mean"],
            "val_psnr_mean": val_summary["psnr_mean"],
            "val_psnr_std": val_summary["psnr_std"],
            "val_ssim_mean": val_summary["ssim_mean"],
            "val_ssim_std": val_summary["ssim_std"],
            "val_residual_mean": val_summary["residual_mean"],
            "val_residual_std": val_summary["residual_std"],
            "seconds_elapsed": time.time() - start_time,
            "is_best": is_best,
        }
        metric_rows.append(row)
        write_csv(metric_rows, metrics_path)
        write_csv(last_val_rows, val_rows_path)
        save_checkpoint(output_dir / "checkpoints" / "last.pth", model, optimizer, epoch, config, best_psnr)
        if is_best:
            save_checkpoint(output_dir / "checkpoints" / "best.pth", model, optimizer, epoch, config, best_psnr)
            save_ranked_samples(model, val_loader, device, output_dir / "samples", config, max_items=3)

        print(
            f"Epoch {epoch}/{config.epochs}: train_l1={train_loss:.6g}, "
            f"val_psnr={val_summary['psnr_mean']:.4f}, val_ssim={val_summary['ssim_mean']:.6f}"
        )

    write_report(output_dir / "training_report.md", config, metric_rows, last_val_rows, model)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="manifest_baseline_smoke")
    parser.add_argument(
        "--train-pairs",
        default="reports/target_scmos_iccd_like_synthetic_512_p99_0p25/pairs.csv",
    )
    parser.add_argument(
        "--train-splits",
        default="reports/target_scmos_iccd_like_synthetic_512_p99_0p25/splits.yaml",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-pairs", default="")
    parser.add_argument("--val-splits", default="")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", default="reports/e3_manifest_baseline_smoke")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--val-patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument(
        "--condition-column",
        default="",
        help="Optional metadata column used as a constant condition channel, for example assigned_condition_score.",
    )
    parser.add_argument(
        "--condition-value-scale",
        type=float,
        default=1.0,
        help="Divide condition values by this scale before adding the condition channel.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_pairs = args.val_pairs or args.train_pairs
    val_splits = args.val_splits or args.train_splits
    return TrainConfig(
        experiment_id=args.experiment_id,
        train_pairs=args.train_pairs,
        train_splits=args.train_splits,
        train_split=args.train_split,
        val_pairs=val_pairs,
        val_splits=val_splits,
        val_split=args.val_split,
        output_dir=args.output_dir,
        range_max=args.range_max,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        channels=args.channels,
        depth=args.depth,
        seed=args.seed,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        amp=args.amp,
        input_channels=args.input_channels,
        condition_column=args.condition_column,
        condition_value_scale=args.condition_value_scale,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        device=device,
        git_commit=current_git_commit(),
    )


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualDenoiser(torch.nn.Module):
    """Small residual CNN baseline for grayscale denoising.

    Input shape: (B, input_channels, H, W). Output shape: (B, 1, H, W).
    The first input channel must be the noisy image. Additional channels can
    carry constant condition values. The residual is bounded so the initial
    baseline stays near the noisy input instead of producing arbitrary images.
    """

    def __init__(self, channels: int = 32, depth: int = 4, input_channels: int = 1) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        self.input_channels = input_channels
        layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                ]
            )
        layers.append(torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1))
        self.net = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)
        last = self.net[-1]
        if isinstance(last, torch.nn.Conv2d):
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        noisy = model_input[:, :1]
        residual = 0.1 * torch.tanh(self.net(model_input))
        return noisy + residual

    @staticmethod
    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def make_grad_scaler(enabled: bool) -> Any:
    import torch

    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: Any, enabled: bool) -> Any:
    import contextlib
    import torch

    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    config: TrainConfig,
) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        model_input = make_model_input(noisy, batch, config).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler is not None):
            pred = model(model_input)
            loss = criterion(pred, clean)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if config.max_train_batches and batch_idx >= config.max_train_batches:
            break
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: Any,
    criterion: Any,
    device: Any,
    config: TrainConfig,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader, start=1):
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        model_input = make_model_input(noisy, batch, config).to(device, non_blocking=True)
        pred = model(model_input)
        loss = criterion(pred, clean)
        pred_metric = pred.clamp(0.0, 1.0)
        quality = image_quality(pred_metric, clean, data_range=1.0)
        noisy_quality = image_quality(noisy, clean, data_range=1.0)
        rows.append(
            {
                "pair_key": batch["pair_key"][0],
                "loss_l1": float(loss.detach().cpu().item()),
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "residual_mean": quality["residual_mean"],
                "residual_std": quality["residual_std"],
                "noisy_psnr": noisy_quality["psnr"],
                "noisy_ssim": noisy_quality["ssim"],
            }
        )
        if config.max_val_batches and batch_idx >= config.max_val_batches:
            break
    summary = summarize_validation(rows)
    return summary, rows


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "loss_mean": mean(rows, "loss_l1"),
        "psnr_mean": mean(rows, "psnr"),
        "psnr_std": std(rows, "psnr"),
        "ssim_mean": mean(rows, "ssim"),
        "ssim_std": std(rows, "ssim"),
        "residual_mean": mean(rows, "residual_mean"),
        "residual_std": mean(rows, "residual_std"),
    }


def make_model_input(noisy: torch.Tensor, batch: dict[str, Any], config: TrainConfig) -> torch.Tensor:
    if config.input_channels == 1:
        return noisy
    if config.input_channels != 2:
        raise ValueError(f"Only input_channels 1 or 2 are supported, got {config.input_channels}")
    values = condition_values(batch, config.condition_column, config.condition_value_scale, noisy.device)
    condition = values[:, None, None, None].expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
    return torch.cat([noisy, condition], dim=1)


def condition_values(batch: dict[str, Any], column: str, scale: float, device: Any) -> torch.Tensor:
    if not column:
        raise ValueError("--condition-column is required when --input-channels > 1")
    metadata = batch.get("metadata", {})
    if column not in metadata:
        raise KeyError(f"Condition column {column!r} not found in batch metadata")
    raw_values = metadata[column]
    if isinstance(raw_values, torch.Tensor):
        values = raw_values.detach().float().to(device)
    elif isinstance(raw_values, (list, tuple)):
        values = torch.tensor([float(item) for item in raw_values], dtype=torch.float32, device=device)
    else:
        values = torch.tensor([float(raw_values)], dtype=torch.float32, device=device)
    divisor = float(scale) if scale else 1.0
    return values / divisor


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def std(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanstd(values))


@torch.no_grad()
def save_ranked_samples(
    model: torch.nn.Module,
    loader: Any,
    device: Any,
    output_dir: Path,
    config: TrainConfig,
    max_items: int,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.tif"):
        old_file.unlink()
    samples: list[dict[str, Any]] = []
    tensors: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        model_input = make_model_input(noisy, batch, config).to(device)
        pred = model(model_input).clamp(0.0, 1.0)
        quality = image_quality(pred, clean, data_range=1.0)
        pair_key = batch["pair_key"][0]
        samples.append({"pair_key": pair_key, "psnr": quality["psnr"]})
        tensors.append((pair_key, clean.cpu(), noisy.cpu(), pred.cpu()))
    if not samples:
        return
    ranked = sorted(samples, key=lambda row: row["psnr"])
    wanted_keys = [ranked[0]["pair_key"], ranked[len(ranked) // 2]["pair_key"], ranked[-1]["pair_key"]]
    labels = ["worst", "median", "best"]
    for label, key in zip(labels[:max_items], wanted_keys[:max_items]):
        for pair_key, clean, noisy, pred in tensors:
            if pair_key == key:
                save_triplet_tiff(output_dir / f"{label}_{pair_key}.tif", clean, noisy, pred)
                break


def save_triplet_tiff(path: Path, clean: Any, noisy: Any, pred: Any) -> None:
    import tifffile

    clean_np = tensor_to_hwc(clean)
    noisy_np = tensor_to_hwc(noisy)
    pred_np = tensor_to_hwc(pred)
    triplet = np.concatenate([noisy_np, pred_np, clean_np], axis=1)
    tifffile.imwrite(path, np.rint(np.clip(triplet, 0.0, 1.0) * 65535.0).astype(np.uint16))


def tensor_to_hwc(tensor: Any) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale tensor, got {arr.shape}")
    return arr


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Any,
    epoch: int,
    config: TrainConfig,
    best_psnr: float,
) -> None:
    import torch

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "config": asdict(config),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        },
        path,
    )


def write_report(
    path: Path,
    config: TrainConfig,
    metric_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    model: torch.nn.Module,
) -> None:
    final = metric_rows[-1] if metric_rows else {}
    noisy_psnr = mean(val_rows, "noisy_psnr")
    noisy_ssim = mean(val_rows, "noisy_ssim")
    lines = [
        "# Manifest Denoiser Baseline",
        "",
        "## Configuration",
        "",
        f"- Experiment ID: `{config.experiment_id}`",
        f"- Git commit: `{config.git_commit}`",
        f"- Train manifest: `{config.train_pairs}` / split `{config.train_split}`",
        f"- Validation manifest: `{config.val_pairs}` / split `{config.val_split}`",
        f"- Seed: {config.seed}",
        f"- Device: {config.device}",
        f"- Model parameters: {count_parameters(model)}",
        f"- Input channels: {config.input_channels}",
        f"- Condition column: `{config.condition_column or 'none'}`",
        f"- Condition value scale: {config.condition_value_scale:g}",
        "",
        "## Final Result",
        "",
        f"- Train L1: {float(final.get('train_l1', float('nan'))):.6g}",
        f"- Validation L1: {float(final.get('val_l1', float('nan'))):.6g}",
        f"- Validation PSNR/SSIM: {float(final.get('val_psnr_mean', float('nan'))):.4f} / {float(final.get('val_ssim_mean', float('nan'))):.6f}",
        f"- Noisy-input PSNR/SSIM on same validation subset: {noisy_psnr:.4f} / {noisy_ssim:.6f}",
        "",
        "## Outputs",
        "",
        f"- Config: `{path.parent / 'config.json'}`",
        f"- Metrics: `{path.parent / 'metrics.csv'}`",
        f"- Validation rows: `{path.parent / 'validation_predictions.csv'}`",
        f"- Best checkpoint: `{path.parent / 'checkpoints' / 'best.pth'}`",
        f"- Last checkpoint: `{path.parent / 'checkpoints' / 'last.pth'}`",
        f"- Samples: `{path.parent / 'samples'}`",
        "",
        "## Claim Boundary",
        "",
        "- This is an engineering baseline for manifest/training correctness.",
        "- It does not establish real ICCD denoising performance.",
        "- Do not compare paper models against this run unless they use the same manifest, split, seed policy, and metric implementation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
