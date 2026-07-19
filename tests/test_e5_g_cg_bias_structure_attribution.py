from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_e5_g_cg_bias_structure_attribution import (
    bias_model_fits,
    frequency_band_energy,
    gradient_masks,
    load_epoch_metrics,
    region_metrics,
    residual_dc_metrics,
    temporal_metrics,
)


CFG = {
    "frequency_bands_cycles_per_pixel": {"very_low_upper": 1 / 64, "low_upper": 1 / 16, "mid_upper": 1 / 4},
    "gradient_regions": {"flat_upper_percentile": 25.0, "low_upper_percentile": 50.0, "medium_upper_percentile": 90.0},
}


def test_mean_centered_temporal_metrics_remove_frame_dc() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(1000, 10, size=(20, 32, 32)).astype(np.float32)
    output = raw + np.arange(20, dtype=np.float32)[:, None, None]
    correction = np.mean(raw, axis=(1, 2)) - np.mean(output, axis=(1, 2))
    corrected = output + correction[:, None, None]
    result = temporal_metrics(raw, output, corrected)
    assert result["output_temporal_variance_DN2"] > result["raw_temporal_variance_DN2"]
    assert abs(result["corrected_mean_centered_temporal_reduction"] - result["mean_centered_temporal_reduction"]) < 1e-8
    assert abs(result["corrected_temporal_reduction"]) < 1e-6


def test_dc_restoration_preserves_spatial_gradients_and_residual_dc() -> None:
    y, x = np.indices((32, 32), dtype=np.float32)
    base = x**2 + 0.25 * y**2 + 20 * np.sin(x / 3) * np.cos(y / 5)
    raw = np.stack([base + i for i in range(8)])
    output = raw - 5.0
    summary, frames = residual_dc_metrics(raw, output)
    assert abs(summary["predicted_residual_DC_DN"] - 5.0) < 1e-7
    assert all(abs(row["predicted_residual_mean_DN"] - 5.0) < 1e-7 for row in frames)
    masks, _ = gradient_masks(np.mean(raw, axis=0), CFG)
    corrected = output + 5.0
    rows = region_metrics(raw, output, corrected, np.mean(raw, axis=0), masks)
    assert all(abs(row["gradient_ratio"] - row["corrected_gradient_ratio"]) < 1e-7 for row in rows)


def test_frequency_energy_identifies_dc_and_high_frequency() -> None:
    stack = np.zeros((4, 32, 32), np.float32)
    stack[:, :, :] = np.arange(4)[:, None, None]
    dc = frequency_band_energy(stack, CFG, batch_size=2)
    assert dc["dc_fraction"] > 0.999
    checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 2 - 1).astype(np.float32)
    stack = np.stack([checker * value for value in [1, -1, 1, -1]])
    high = frequency_band_energy(stack, CFG, batch_size=2)
    assert high["high_fraction"] > 0.99


def test_bias_models_recover_signal_slope() -> None:
    signal = np.linspace(100, 1000, 40)
    frames = pd.DataFrame({
        "model": ["CG_NC"] * 40,
        "run_seed": [1] * 20 + [2] * 20,
        "folder": [2, 5, 9, 11] * 10,
        "mean_shift_DN": 2.0 + 0.01 * signal,
        "input_mean_DN": signal,
        "input_std_DN": np.linspace(10, 20, 40),
        "predicted_sigma_DN": signal * 0.059,
    })
    fits = bias_model_fits(frames)
    b2 = fits[(fits.run_seed.astype(str) == "ALL") & fits.bias_model.eq("B2")].iloc[0]
    coefficients = __import__("json").loads(b2.coefficients)
    assert abs(coefficients["intercept"] - 2.0) < 1e-10
    assert abs(coefficients["input_mean_signal"] - 0.01) < 1e-12
    assert b2.R2 > 0.999999


def test_epoch_metrics_schema_maps_experiment_to_model(tmp_path: Path) -> None:
    metrics = pd.DataFrame([
        {"experiment": model, "epoch": epoch, "train_l1": 0.1 / epoch, "validation_l1": 0.2, "validation_psnr": 50 + epoch, "is_best": epoch == 2}
        for model in ["G", "CG_NC"] for epoch in [1, 2]
    ])
    path = tmp_path / "epoch_metrics.csv"
    metrics.to_csv(path, index=False)
    bias = pd.DataFrame([
        {"run_seed": 1, "model": model, "folder": folder, "mean_shift_DN": shift, "predicted_residual_mean_DN": -shift}
        for model, shift in [("G", 1.0), ("CG_NC", 2.0)] for folder in [2, 5]
    ])
    cfg = {"training_metric_sources": {1: str(path)}}
    epochs, drops = load_epoch_metrics(Path("/"), cfg, bias)
    assert set(epochs.model) == {"G", "CG_NC"}
    assert len(drops) == 2
    assert set(drops.model) == {"G", "CG_NC"}
