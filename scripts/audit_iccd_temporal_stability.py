"""Audit frame-level temporal stability and drift for the formal E1 rerun."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from e1_formal_common import (
    common_metadata,
    correlation,
    linear_slope,
    load_config,
    read_stack,
    safe_div,
    selected_paths,
    warnings_for_values,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run(config)
    return 0


def run(config: dict[str, Any]) -> None:
    cfg = config["temporal_stability"]
    output_dir = Path(config["output_root"]) / "temporal_stability"
    crop_size = int(config["primary_crop_size"])
    count = int(cfg["frame_count"])
    rolling_window = int(cfg["rolling_window"])
    grid = int(cfg["local_grid"])
    stride = int(cfg["correlation_stride"])
    trace_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        frame_means = np.mean(stack, axis=(1, 2), dtype=np.float64)
        frame_stds = np.std(stack, axis=(1, 2), dtype=np.float64)
        temporal_mean = np.mean(stack, axis=0, dtype=np.float64)
        residual = stack.astype(np.float64) - temporal_mean[None]
        half = count // 2
        mean_half_delta = float(np.mean(frame_means[half:]) - np.mean(frame_means[:half]))
        mean_half_relative = safe_div(mean_half_delta, float(np.mean(frame_means[:half])))
        std_half_delta = float(np.mean(frame_stds[half:]) - np.mean(frame_stds[:half]))

        local_relative_changes = []
        block_h = crop_size // grid
        block_w = crop_size // grid
        for row_index in range(grid):
            for col_index in range(grid):
                patch = stack[
                    :,
                    row_index * block_h : (row_index + 1) * block_h,
                    col_index * block_w : (col_index + 1) * block_w,
                ]
                local_means = np.mean(patch, axis=(1, 2), dtype=np.float64)
                local_relative_changes.append(
                    abs(safe_div(float(np.mean(local_means[half:]) - np.mean(local_means[:half])), float(np.mean(local_means[:half]))))
                )

        lag_correlations = {}
        for lag in [int(value) for value in cfg["correlation_lags"]]:
            values = [correlation(residual[index], residual[index + lag], stride) for index in range(count - lag)]
            finite = [value for value in values if math.isfinite(value)]
            lag_correlations[lag] = float(np.mean(finite)) if finite else float("nan")
            lag_rows.append(
                {
                    **common_metadata(
                        folder,
                        count,
                        crop_size,
                        coords,
                        "Mean correlation of temporal-mean-subtracted frame residuals at the stated lag.",
                        lag_correlations[lag],
                        len(finite),
                        warnings_for_values([lag_correlations[lag]]),
                    ),
                    "lag_frames": lag,
                    "mean_residual_correlation": lag_correlations[lag],
                }
            )

        rolling_means = rolling_average(frame_means, rolling_window)
        rolling_vars = rolling_average(frame_stds**2, rolling_window)
        for index in range(count):
            trace_rows.append(
                {
                    **common_metadata(
                        folder,
                        count,
                        crop_size,
                        coords,
                        "Frame mean in raw DN for temporal drift inspection.",
                        float(frame_means[index]),
                        crop_size * crop_size,
                    ),
                    "frame_index": index + 1,
                    "frame_mean_dn": float(frame_means[index]),
                    "frame_std_dn": float(frame_stds[index]),
                    "rolling_mean_dn": float(rolling_means[index]),
                    "rolling_variance_dn2": float(rolling_vars[index]),
                }
            )

        values = [
            linear_slope(frame_means),
            linear_slope(frame_stds),
            mean_half_relative,
            max(local_relative_changes),
            *lag_correlations.values(),
        ]
        warnings = warnings_for_values(values)
        if abs(mean_half_relative) > 0.01:
            warnings.append("global_brightness_drift_gt_1pct")
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "Relative change in frame mean between the second and first halves.",
            mean_half_relative,
            count,
            warnings,
        )
        row.update(
            {
                "frame_mean_slope_dn_per_frame": linear_slope(frame_means),
                "frame_std_slope_dn_per_frame": linear_slope(frame_stds),
                "frame_mean_first_half_dn": float(np.mean(frame_means[:half])),
                "frame_mean_second_half_dn": float(np.mean(frame_means[half:])),
                "frame_mean_delta_half_dn": mean_half_delta,
                "frame_mean_relative_change_half": mean_half_relative,
                "frame_std_delta_half_dn": std_half_delta,
                "max_local_brightness_relative_change_half": max(local_relative_changes),
                "lag1_residual_correlation": lag_correlations.get(1, float("nan")),
                "lag10_residual_correlation": lag_correlations.get(10, float("nan")),
                "lag50_residual_correlation": lag_correlations.get(50, float("nan")),
            }
        )
        summary_rows.append(row)
        print(f"temporal_stability folder={folder}", flush=True)

    write_csv(trace_rows, output_dir / "frame_traces.csv")
    write_csv(lag_rows, output_dir / "temporal_lag_correlations.csv")
    write_csv(summary_rows, output_dir / "temporal_drift_summary.csv")


def rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

