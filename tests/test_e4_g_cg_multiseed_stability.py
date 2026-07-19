from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_e4_g_cg_multiseed_stability.py"
SPEC = importlib.util.spec_from_file_location("e4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_decision_reports_repeatable_tradeoffs() -> None:
    synthetic = pd.DataFrame([
        {"run_seed": seed, "model": model, "output_psnr": 57.0 + (0.002 if model == "CG_NC" else 0), "output_ssim": 0.99}
        for seed in [1, 2, 3] for model in ["G", "CG_NC"]
    ])
    rows = []
    for seed in [1, 2, 3]:
        for folder in [2, 5, 9, 11]:
            for model in ["G", "CG_NC"]:
                rows.append({
                    "run_seed": seed, "folder": folder, "model": model,
                    "temporal_variance_reduction": 0.04 + (0.01 if model == "CG_NC" else 0),
                    "row_energy_reduction": 0.01, "column_energy_reduction": 0.01,
                    "max_absolute_shift_DN": 2.0 + (2.0 if model == "CG_NC" else 0),
                    "high_gradient_retention": 0.99 - (0.002 if model == "CG_NC" else 0),
                    "removed_structure_correlation": 0.05 + (0.01 if model == "CG_NC" else 0),
                    "removed_structure_warning": False,
                })
    cfg = {"decision": {"required_synthetic_noninferior_seeds": 2, "required_real_better_seeds": 2, "required_folders_mean_better": 3, "severe_brightness_shift_DN": 15.0, "obvious_gradient_retention_drop": 0.01, "structure_correlation_margin": 0.02}}
    decision, cgs = MODULE.decide(synthetic, pd.DataFrame(rows), cfg)
    assert decision["status"] == "CONDITIONAL-BENEFIT-WITH-TRADEOFFS"
    assert cgs["CGS_ENTRY_ALLOWED"] is False
