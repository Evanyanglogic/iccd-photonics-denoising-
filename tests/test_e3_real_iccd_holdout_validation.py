from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_e3_real_iccd_holdout_validation import correlation, row_column_summary, split_stable_summary, temporal_summary


def test_temporal_reduction_and_row_column_metrics() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(1000, 20, size=(20, 32, 32)).astype(np.float32)
    output = 1000 + (raw - 1000) * 0.5
    raw_summary = temporal_summary(2, "raw", raw, 1.0)
    out_summary = temporal_summary(2, "G", output, raw_summary["mean_temporal_variance_DN2"])
    assert 0.74 < out_summary["temporal_variance_reduction"] < 0.76
    raw_rc = row_column_summary(2, "raw", raw, 1.0, 1.0)
    out_rc = row_column_summary(2, "G", output, raw_rc["row_energy_DN"], raw_rc["column_energy_DN"])
    assert 0.49 < out_rc["row_energy_reduction"] < 0.51


def test_split_map_and_correlation() -> None:
    base = np.arange(64, dtype=np.float32).reshape(8, 8)
    stack = np.stack([base + index * 0.01 for index in range(200)])
    summary = split_stable_summary(2, "raw", stack)
    assert summary["split_map_correlation"] > 0.999
    assert correlation(base, base) == 1.0
