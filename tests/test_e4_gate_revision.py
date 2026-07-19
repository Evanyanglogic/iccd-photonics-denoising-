from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_e2_g_cg_scaled_training import pair_pass
from run_e4_g_cg_gate_revision import grouped_summary


def test_grouped_summary_keeps_pre_and_post_clipping_means_separate() -> None:
    frame = pd.DataFrame({
        "experiment": ["CG", "CG"], "run_seed": [1, 1],
        "residual_mean_DN": [-0.2, 0.2], "brightness_shift_DN": [0.1, 0.3],
        "clipping_mean_contribution_DN": [0.3, 0.1], "z_mean": [-1.0, 1.0],
    })
    result = grouped_summary(frame, ["experiment", "run_seed"]).iloc[0]
    assert abs(result.mean_residual_DN) < 1e-12
    assert abs(result.mean_brightness_shift_DN - 0.2) < 1e-12
    assert abs(result.mean_clipping_contribution_DN - 0.2) < 1e-12


def test_revised_gate_uses_pre_clipping_z_not_post_clipping_brightness() -> None:
    metrics = {
        "residual_mean_DN": 0.4525933888, "residual_std_relative_error": 0.00092,
        "z_mean": 1.56108, "brightness_shift_DN": 1.06095,
        "added_zero_ratio": 0.0091, "added_one_ratio": 0.0,
        "noisy_round_trip_max_error_DN": 0.5, "residual_reconstruction_max_error_DN": 0.5,
    }
    common = {"residual_std_relative_error_max": 0.01, "added_zero_ratio_max": 0.05, "added_one_ratio_max": 0.01, "noisy_round_trip_max_error_DN": 0.501, "residual_reconstruction_max_error_DN": 1.0}
    old_pass, _ = pair_pass(metrics, {**common, "absolute_brightness_shift_DN_max": 1.0})
    revised_pass, _ = pair_pass(metrics, {**common, "residual_mean_z_max": 4.5, "absolute_residual_mean_DN_max": 2.0})
    assert old_pass is False
    assert revised_pass is True
