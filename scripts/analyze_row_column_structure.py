"""Analyze temporal row and column structure for the formal E1 rerun."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from e1_formal_common import (
    common_metadata,
    load_config,
    read_stack,
    safe_div,
    selected_paths,
    vector_correlation,
    warnings_for_values,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(Path(args.config)))
    return 0


def run(config: dict[str, Any]) -> None:
    cfg = config["row_column"]
    output_dir = Path(config["output_root"]) / "row_column"
    crop_size = int(config["primary_crop_size"])
    count = int(cfg["frame_count"])
    block_size = int(cfg["block_size"])
    summary_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        residual = stack - np.mean(stack, axis=0, keepdims=True)
        residual -= np.mean(residual, axis=(1, 2), keepdims=True)
        row_profiles = np.mean(residual, axis=2, dtype=np.float64)
        col_profiles = np.mean(residual, axis=1, dtype=np.float64)
        row_energy = float(np.sqrt(np.mean(row_profiles**2)))
        col_energy = float(np.sqrt(np.mean(col_profiles**2)))
        residual_std_before = float(np.std(residual, ddof=1))
        reconstructed = row_profiles[:, :, None] + col_profiles[:, None, :]
        corrected = residual - reconstructed
        residual_std_after = float(np.std(corrected, ddof=1))
        removed_fraction = 1.0 - safe_div(residual_std_after**2, residual_std_before**2)
        row_adjacent = adjacent_profile_correlations(row_profiles)
        col_adjacent = adjacent_profile_correlations(col_profiles)
        row_block_energies = []
        col_block_energies = []
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            if stop - start < 2:
                continue
            block_row_energy = float(np.sqrt(np.mean(row_profiles[start:stop] ** 2)))
            block_col_energy = float(np.sqrt(np.mean(col_profiles[start:stop] ** 2)))
            row_block_energies.append(block_row_energy)
            col_block_energies.append(block_col_energy)
            block_rows.append(
                {
                    **common_metadata(
                        folder,
                        stop - start,
                        crop_size,
                        coords,
                        "Root-mean-square row-profile energy in the stated non-overlapping frame block.",
                        block_row_energy,
                        (stop - start) * crop_size,
                    ),
                    "block_start_frame": start + 1,
                    "block_end_frame": stop,
                    "row_pattern_energy_dn": block_row_energy,
                    "column_pattern_energy_dn": block_col_energy,
                }
            )
        row_cv = safe_div(float(np.std(row_block_energies, ddof=1)), float(np.mean(row_block_energies)))
        col_cv = safe_div(float(np.std(col_block_energies, ddof=1)), float(np.mean(col_block_energies)))
        values = [row_energy, col_energy, residual_std_before, residual_std_after, removed_fraction, row_adjacent, col_adjacent, row_cv, col_cv]
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "RMS row-profile energy of temporal-mean-subtracted residual frames in raw DN.",
            row_energy,
            row_profiles.size,
            warnings_for_values(values),
        )
        row.update(
            {
                "row_pattern_energy_dn": row_energy,
                "column_pattern_energy_dn": col_energy,
                "row_adjacent_frame_profile_correlation": row_adjacent,
                "column_adjacent_frame_profile_correlation": col_adjacent,
                "residual_std_before_dn": residual_std_before,
                "residual_std_after_row_column_removal_dn": residual_std_after,
                "variance_fraction_removed": removed_fraction,
                "row_energy_block_cv": row_cv,
                "column_energy_block_cv": col_cv,
            }
        )
        summary_rows.append(row)
        arrays[f"folder_{folder}_row_profiles"] = row_profiles.astype(np.float32)
        arrays[f"folder_{folder}_column_profiles"] = col_profiles.astype(np.float32)
        print(f"row_column folder={folder}", flush=True)

    write_csv(summary_rows, output_dir / "row_column_summary.csv")
    write_csv(block_rows, output_dir / "row_column_by_block.csv")
    np.savez_compressed(output_dir / "row_column_profiles.npz", **arrays)


def adjacent_profile_correlations(profiles: np.ndarray) -> float:
    values = [vector_correlation(profiles[index], profiles[index + 1]) for index in range(len(profiles) - 1)]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
