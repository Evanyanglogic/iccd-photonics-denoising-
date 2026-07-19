from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_e2_g_cg_scaled_training import LightUNet, pair_metrics, scale_uint16, stable_seed


def test_stable_seed_is_deterministic_and_condition_specific() -> None:
    first = stable_seed("a" * 64, "low", 20260719)
    assert first == stable_seed("a" * 64, "low", 20260719)
    assert first != stable_seed("a" * 64, "high", 20260719)


def test_operational_scaling_hits_target_without_collapse() -> None:
    raw = np.arange(1, 513 * 513, dtype=np.uint32)[: 512 * 512].reshape(512, 512).astype(np.uint16)
    scaled, metrics = scale_uint16(raw, 1491.98291015625)
    assert scaled.dtype == np.uint16
    assert metrics["relative_mean_error"] < 0.005
    assert metrics["unique_values"] >= 256
    assert metrics["round_trip_max_error_DN"] <= 0.501


def test_pair_generation_and_network_shapes() -> None:
    raw = np.full((512, 512), 1492, dtype=np.uint16)
    z, metrics = pair_metrics(raw, 91.4432, 20260719)
    assert z.shape == raw.shape
    assert metrics["residual_std_relative_error"] < 0.01
    assert LightUNet(1)(torch.zeros(1, 1, 512, 512)).shape == (1, 1, 512, 512)
    assert LightUNet(2)(torch.zeros(1, 2, 512, 512)).shape == (1, 1, 512, 512)


def test_residual_round_trip_uses_post_clipping_residual() -> None:
    raw = np.full((512, 512), 4, dtype=np.uint16)
    for seed in [20260717, 20260718, 20260719, 2527312500]:
        _, metrics = pair_metrics(raw, 148.440509, seed)
        assert metrics["negative_before_clipping_ratio"] > 0
        assert metrics["noisy_round_trip_max_error_DN"] <= 0.5
        assert metrics["residual_reconstruction_max_error_DN"] <= 0.5
