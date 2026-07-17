"""Measure repeatable stable image components without assigning a physical cause."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from e1_formal_common import (
    common_metadata,
    correlation,
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
    run(load_config(Path(args.config)))
    return 0


def run(config: dict[str, Any]) -> None:
    from scipy.ndimage import gaussian_filter

    cfg = config["stable_component"]
    output_dir = Path(config["output_root"]) / "stable_component"
    crop_size = int(config["primary_crop_size"])
    count = int(cfg["frame_count"])
    sigma = float(cfg["highpass_sigma"])
    stride = int(cfg["correlation_stride"])
    block_size = int(cfg["block_size"])
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        half = count // 2
        splits: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("odd_even", stack[0::2], stack[1::2]),
            ("first_second_half", stack[:half], stack[half : half * 2]),
        ]
        blocks = [stack[start : start + block_size] for start in range(0, count, block_size) if len(stack[start : start + block_size]) == block_size]
        if len(blocks) >= 4:
            splits.append(("alternating_blocks", np.concatenate(blocks[0::2]), np.concatenate(blocks[1::2])))

        correlations = []
        stable_stds = []
        temporal_stds = []
        for name, first, second in splits:
            first_map = highpass(np.mean(first, axis=0, dtype=np.float64), sigma, gaussian_filter)
            second_map = highpass(np.mean(second, axis=0, dtype=np.float64), sigma, gaussian_filter)
            corr = correlation(first_map, second_map, stride)
            stable_map = (first_map + second_map) / 2.0
            difference_map = (first_map - second_map) / math.sqrt(2.0)
            stable_std = float(np.std(stable_map, ddof=1))
            temporal_std = float(np.std(difference_map, ddof=1))
            correlations.append(corr)
            stable_stds.append(stable_std)
            temporal_stds.append(temporal_std)
            arrays[f"folder_{folder}_{name}_first"] = first_map.astype(np.float32)
            arrays[f"folder_{folder}_{name}_second"] = second_map.astype(np.float32)
            split_rows.append(
                {
                    **common_metadata(
                        folder,
                        len(first) + len(second),
                        crop_size,
                        coords,
                        "Correlation between independently averaged high-pass observed maps for the stated split.",
                        corr,
                        first_map[::stride, ::stride].size,
                        warnings_for_values([corr, stable_std, temporal_std]),
                    ),
                    "split_method": name,
                    "first_group_frames": len(first),
                    "second_group_frames": len(second),
                    "split_map_correlation": corr,
                    "observed_stable_component_std_dn": stable_std,
                    "split_difference_component_std_dn": temporal_std,
                    "stable_to_split_difference_ratio": safe_div(stable_std, temporal_std),
                    "highpass_sigma_px": sigma,
                }
            )
        warnings = warnings_for_values([*correlations, *stable_stds, *temporal_stds])
        if min(correlations) < 0.5:
            warnings.append("split_repeatability_below_0p5")
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "Minimum correlation across odd/even, half, and non-overlapping block split maps.",
            min(correlations),
            len(correlations),
            warnings,
        )
        row.update(
            {
                "split_method_count": len(correlations),
                "minimum_split_map_correlation": min(correlations),
                "median_split_map_correlation": float(np.median(correlations)),
                "observed_stable_component_std_dn": float(np.median(stable_stds)),
                "split_difference_component_std_dn": float(np.median(temporal_stds)),
                "stable_to_temporal_ratio": safe_div(float(np.median(stable_stds)), float(np.median(temporal_stds))),
                "highpass_sigma_px": sigma,
            }
        )
        summary_rows.append(row)
        print(f"stable_component folder={folder}", flush=True)

    write_csv(split_rows, output_dir / "stable_component_by_split.csv")
    write_csv(summary_rows, output_dir / "stable_component_summary.csv")
    np.savez_compressed(output_dir / "stable_component_maps.npz", **arrays)


def highpass(image: np.ndarray, sigma: float, gaussian_filter: Any) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return image - gaussian_filter(image, sigma=sigma, mode="reflect")


if __name__ == "__main__":
    raise SystemExit(main())

