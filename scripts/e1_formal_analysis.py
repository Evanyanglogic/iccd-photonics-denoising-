"""Run formal E1 integrity, summary, robustness, mean-variance, and spatial modules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from e1_formal_common import (
    common_metadata,
    correlation,
    crop_coordinates,
    indexed_tiffs,
    load_config,
    read_csv,
    read_stack,
    safe_div,
    selected_paths,
    sha256_file,
    warnings_for_values,
    write_csv,
    write_json,
)


MODULES = {"input_audit", "noise_summary", "mean_variance", "robustness", "spatial", "combined"}


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_root = Path(config["output_root"])
    if args.module == "input_audit":
        run_input_audit(config, output_root)
    elif args.module == "noise_summary":
        run_noise_summary(config, output_root)
    elif args.module == "mean_variance":
        run_mean_variance(config, output_root)
    elif args.module == "robustness":
        run_robustness(config, output_root)
    elif args.module == "spatial":
        run_spatial(config, output_root)
    elif args.module == "combined":
        run_combined(config, output_root)
    else:
        raise ValueError(args.module)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", required=True, choices=sorted(MODULES))
    return parser.parse_args()


def run_input_audit(config: dict[str, Any], output_root: Path) -> None:
    import tifffile

    cfg = config["integrity"]
    expected_shape = tuple(int(value) for value in config["image_shape_expected"])
    expected_dtype = str(config["dtype_expected"])
    expected_count = int(cfg["expected_frame_count"])
    expected_first = int(cfg["expected_first_index"])
    stride = max(1, int(cfg["pixel_statistics_stride"]))
    manifest_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for folder in [int(value) for value in config["folders"]]:
        folder_path = Path(config["data_root"]) / str(folder)
        indexed = indexed_tiffs(folder_path)
        indices = [index for index, _ in indexed]
        expected_indices = set(range(expected_first, expected_first + expected_count))
        missing = sorted(expected_indices - set(indices))
        unexpected = sorted(set(indices) - expected_indices)
        unreadable = shape_bad = dtype_bad = zero_bad = saturation_bad = 0
        file_sizes = [path.stat().st_size for _, path in indexed]
        median_size = float(np.median(file_sizes)) if file_sizes else 0.0
        size_bad = 0
        seen_fingerprints: set[str] = set()
        duplicate_content = 0
        for frame_index, path in indexed:
            stat = path.stat()
            sample_count = 0
            manifest_rows.append(
                {
                    "folder": folder,
                    "frame_index": frame_index,
                    "relative_path": str(path.relative_to(Path(config["data_root"]))),
                    "file_size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path) if cfg.get("compute_file_sha256", False) else "not_computed_manifest_uses_size_and_mtime",
                }
            )
            warning_flags: list[str] = []
            try:
                image = tifffile.memmap(path)
                shape = tuple(int(value) for value in image.shape)
                dtype = str(image.dtype)
                if shape != expected_shape:
                    shape_bad += 1
                    warning_flags.append("shape_mismatch")
                if dtype != expected_dtype:
                    dtype_bad += 1
                    warning_flags.append("dtype_mismatch")
                # Copy once so the full formal scan performs one disk pass per TIFF;
                # subsequent statistics operate on resident memory.
                sample = np.array(image[::stride, ::stride], copy=True)
                sample_count = int(sample.size)
                minimum = float(np.min(sample))
                maximum = float(np.max(sample))
                mean = float(np.mean(sample, dtype=np.float64))
                std = float(np.std(sample, dtype=np.float64))
                zero_ratio = float(np.mean(sample == 0))
                saturation_ratio = float(np.mean(sample >= int(config["saturation_value"])))
                if zero_ratio > float(cfg["max_zero_ratio"]):
                    zero_bad += 1
                    warning_flags.append("excess_zero_pixels")
                if saturation_ratio > float(cfg["max_saturation_ratio"]):
                    saturation_bad += 1
                    warning_flags.append("excess_saturation")
                fingerprint = f"{shape}|{dtype}|{minimum}|{maximum}|{mean:.12g}|{std:.12g}|{stat.st_size}"
                duplicate_content += int(fingerprint in seen_fingerprints)
                seen_fingerprints.add(fingerprint)
            except Exception as exc:
                unreadable += 1
                shape = ()
                dtype = "unreadable"
                minimum = maximum = mean = std = zero_ratio = saturation_ratio = float("nan")
                warning_flags.append(f"unreadable:{type(exc).__name__}")
            if median_size and abs(stat.st_size - median_size) / median_size > float(cfg["file_size_relative_tolerance"]):
                size_bad += 1
                warning_flags.append("file_size_outlier")
            coords = crop_coordinates(expected_shape, int(config["primary_crop_size"]))
            row = common_metadata(
                folder,
                1,
                int(config["primary_crop_size"]),
                coords,
                "Per-frame sampled pixel mean in raw DN; stride recorded in pixel_statistics_stride.",
                mean,
                sample_count,
                warning_flags,
            )
            row.update(
                {
                    "frame_index": frame_index,
                    "file_name": path.name,
                    "file_size_bytes": stat.st_size,
                    "shape": "x".join(str(value) for value in shape),
                    "dtype": dtype,
                    "pixel_statistics_stride": stride,
                    "sample_min": minimum,
                    "sample_max": maximum,
                    "sample_mean": mean,
                    "sample_std": std,
                    "zero_pixel_ratio": zero_ratio,
                    "saturation_ratio": saturation_ratio,
                }
            )
            frame_rows.append(row)

        integrity_pass = not any(
            [missing, unexpected, unreadable, shape_bad, dtype_bad, size_bad, duplicate_content, zero_bad, saturation_bad]
        ) and len(indexed) == expected_count
        warnings = []
        if missing:
            warnings.append("missing_frames")
        if unexpected:
            warnings.append("unexpected_frames")
        if duplicate_content:
            warnings.append("possible_duplicate_content")
        if unreadable:
            warnings.append("unreadable_files")
        if shape_bad:
            warnings.append("shape_mismatch")
        if dtype_bad:
            warnings.append("dtype_mismatch")
        if size_bad:
            warnings.append("file_size_outlier")
        if zero_bad:
            warnings.append("excess_zero_pixels")
        if saturation_bad:
            warnings.append("excess_saturation")
        coords = crop_coordinates(expected_shape, int(config["primary_crop_size"]))
        row = common_metadata(
            folder,
            len(indexed),
            int(config["primary_crop_size"]),
            coords,
            "Folder integrity gate over indexed TIFF files.",
            1.0 if integrity_pass else 0.0,
            len(indexed),
            warnings,
        )
        row.update(
            {
                "status": "PASS" if integrity_pass else "FAIL",
                "tiff_count": len(indexed),
                "missing_frame_count": len(missing),
                "missing_frame_indices": ";".join(map(str, missing)),
                "unexpected_frame_count": len(unexpected),
                "unreadable_file_count": unreadable,
                "shape_mismatch_count": shape_bad,
                "dtype_mismatch_count": dtype_bad,
                "file_size_outlier_count": size_bad,
                "possible_duplicate_content_count": duplicate_content,
                "excess_zero_frame_count": zero_bad,
                "excess_saturation_frame_count": saturation_bad,
            }
        )
        integrity_rows.append(row)
        print(f"input_audit folder={folder} status={row['status']} files={len(indexed)}", flush=True)

    write_csv(manifest_rows, output_root / "input_audit" / "input_manifest.csv")
    write_csv(frame_rows, output_root / "input_audit" / "frame_level_statistics.csv")
    write_csv(integrity_rows, output_root / "input_audit" / "data_integrity_report.csv")
    write_json(
        {
            "folder_count": len(integrity_rows),
            "pass_count": sum(row["status"] == "PASS" for row in integrity_rows),
            "fail_count": sum(row["status"] == "FAIL" for row in integrity_rows),
            "pixel_statistics_stride": stride,
            "input_manifest_sha256": sha256_file(output_root / "input_audit" / "input_manifest.csv"),
        },
        output_root / "input_audit" / "data_integrity_summary.json",
    )


def run_noise_summary(config: dict[str, Any], output_root: Path) -> None:
    from evaluate_noise_robustness import summarize_stack

    count = int(config["noise_summary"]["frame_count"])
    crop_size = int(config["primary_crop_size"])
    rows = []
    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        legacy = summarize_stack(str(folder), stack, crop_size, count)
        warnings = warnings_for_values(float(value) for key, value in legacy.items() if key != "folder")
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "Mean per-pixel temporal standard deviation in raw DN.",
            float(legacy["temporal_std_mean"]),
            count * coords[2] * coords[3],
            warnings,
        )
        row.update(legacy)
        row["folder"] = folder
        row["observed_stable_component_std"] = row.pop("spatial_fixed_std")
        row["stable_to_temporal_std_ratio"] = row.pop("fixed_to_temporal_std_ratio")
        rows.append(row)
        print(f"noise_summary folder={folder}", flush=True)
    write_csv(rows, output_root / "noise_summary" / "folder_noise_summary.csv")


def run_mean_variance(config: dict[str, Any], output_root: Path) -> None:
    cfg = config["mean_variance"]
    crop_size = int(config["primary_crop_size"])
    count = int(cfg["frame_count"])
    bin_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        temporal_mean = np.mean(stack, axis=0, dtype=np.float64)
        temporal_var = np.var(stack, axis=0, ddof=1, dtype=np.float64)
        lo = float(np.percentile(temporal_mean, float(cfg["lower_percentile"])))
        hi = float(np.percentile(temporal_mean, float(cfg["upper_percentile"])))
        edges = np.linspace(lo, hi, int(cfg["bins"]) + 1)
        valid_bins = 0
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (temporal_mean >= lower) & (temporal_mean <= upper if index == len(edges) - 2 else temporal_mean < upper)
            pixels = int(np.count_nonzero(mask))
            if pixels < int(cfg["min_pixels_per_bin"]):
                continue
            valid_bins += 1
            mean_signal = float(np.mean(temporal_mean[mask]))
            variance = float(np.mean(temporal_var[mask]))
            row = common_metadata(
                folder,
                count,
                crop_size,
                coords,
                "Temporal variance conditional on the per-pixel temporal mean within one fixed scene; not a photon-transfer curve.",
                variance,
                pixels,
                ["single_scene_spatial_binning"],
            )
            row.update(
                {
                    "bin_index": index,
                    "bin_low_dn": lower,
                    "bin_high_dn": upper,
                    "mean_signal_dn": mean_signal,
                    "temporal_variance_dn2": variance,
                    "temporal_std_dn": float(np.mean(np.sqrt(np.maximum(temporal_var[mask], 0.0)))),
                    "fano_like_dn": safe_div(variance, mean_signal),
                }
            )
            bin_rows.append(row)
        mean_signal = float(np.mean(temporal_mean))
        variance = float(np.mean(temporal_var))
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "Folder-level temporal variance divided by mean signal in raw DN; operational Fano-like statistic.",
            safe_div(variance, mean_signal),
            stack.size,
            ["not_photon_transfer_curve"],
        )
        row.update(
            {
                "mean_signal_dn": mean_signal,
                "temporal_variance_dn2": variance,
                "fano_like_dn": safe_div(variance, mean_signal),
                "valid_bin_count": valid_bins,
            }
        )
        summary_rows.append(row)
        print(f"mean_variance folder={folder} bins={valid_bins}", flush=True)
    write_csv(bin_rows, output_root / "mean_variance" / "mean_variance_bins.csv")
    write_csv(summary_rows, output_root / "mean_variance" / "fano_like_summary.csv")


def run_robustness(config: dict[str, Any], output_root: Path) -> None:
    from evaluate_noise_robustness import center_crop_stack, summarize_stack

    crop_sizes = [int(value) for value in config["crop_sizes"]]
    frame_counts = [int(value) for value in config["frame_counts"]]
    max_crop = max(crop_sizes)
    max_count = max(frame_counts)
    rows = []
    for folder in [int(value) for value in config["folders"]]:
        stack, _ = read_stack(selected_paths(config, folder), max_crop, max_count)
        for crop_size in crop_sizes:
            cropped = center_crop_stack(stack, crop_size)
            coords = crop_coordinates(tuple(config["image_shape_expected"]), crop_size)
            for frame_count in frame_counts:
                legacy = summarize_stack(str(folder), cropped[:frame_count], crop_size, frame_count)
                values = [float(value) for key, value in legacy.items() if key != "folder"]
                row = common_metadata(
                    folder,
                    frame_count,
                    crop_size,
                    coords,
                    "Robustness endpoint: mean per-pixel temporal standard deviation in raw DN.",
                    float(legacy["temporal_std_mean"]),
                    frame_count * crop_size * crop_size,
                    warnings_for_values(values),
                )
                row.update(legacy)
                row["folder"] = folder
                row["observed_stable_component_std"] = row.pop("spatial_fixed_std")
                row["stable_to_temporal_std_ratio"] = row.pop("fixed_to_temporal_std_ratio")
                rows.append(row)
        print(f"robustness folder={folder}", flush=True)
    write_csv(rows, output_root / "robustness" / "robustness_by_crop_and_frames.csv")


def run_spatial(config: dict[str, Any], output_root: Path) -> None:
    from analyze_iccd_spatial_correlation import (
        analyze_folder,
        average_power_spectrum,
        normalized_autocorrelation_from_power,
        residual_after_fixed_pattern,
    )

    cfg = config["spatial"]
    count = int(cfg["frame_count"])
    crop_size = int(config["primary_crop_size"])
    max_radius = int(cfg["max_radius"])
    half_width = int(cfg["profile_half_width"])
    summaries: list[dict[str, Any]] = []
    radial_psd_rows: list[dict[str, Any]] = []
    radial_ac_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for folder in [int(value) for value in config["folders"]]:
        stack, coords = read_stack(selected_paths(config, folder), crop_size, count)
        residual = residual_after_fixed_pattern(stack)
        legacy, radial_psd, radial_ac = analyze_folder(str(folder), residual, max_radius)
        power = average_power_spectrum(residual)
        autocorr = normalized_autocorrelation_from_power(power)
        center_y, center_x = power.shape[0] // 2, power.shape[1] // 2
        horizontal = np.mean(power[center_y - half_width : center_y + half_width + 1, :], axis=0)
        vertical = np.mean(power[:, center_x - half_width : center_x + half_width + 1], axis=1)
        horizontal /= max(float(np.sum(horizontal)), 1e-12)
        vertical /= max(float(np.sum(vertical)), 1e-12)
        arrays[f"folder_{folder}_power2d"] = power.astype(np.float32)
        arrays[f"folder_{folder}_autocorr2d"] = autocorr.astype(np.float32)
        warnings = warnings_for_values(float(value) for key, value in legacy.items() if key != "folder")
        if not math.isfinite(float(legacy["corr_length_0p1_px"])):
            legacy["corr_length_0p1_px"] = float(max_radius + 1)
            warnings.append("corr_length_0p1_right_censored")
        if not math.isfinite(float(legacy["corr_length_1e_px"])):
            legacy["corr_length_1e_px"] = float(max_radius + 1)
            warnings.append("corr_length_1e_right_censored")
        row = common_metadata(
            folder,
            count,
            crop_size,
            coords,
            "Residual radial autocorrelation at one-pixel radius after temporal-mean and frame-mean subtraction.",
            float(legacy["radial_autocorr_r1"]),
            residual.size,
            warnings,
        )
        row.update(legacy)
        row["folder"] = folder
        summaries.append(row)
        for item in radial_psd:
            radial_psd_rows.append({"folder": folder, "frame_count": count, "crop_size": crop_size, **item})
        for item in radial_ac:
            radial_ac_rows.append({"folder": folder, "frame_count": count, "crop_size": crop_size, **item})
        frequencies = np.fft.fftshift(np.fft.fftfreq(crop_size))
        for index, frequency in enumerate(frequencies):
            directional_rows.append(
                {
                    "folder": folder,
                    "frame_count": count,
                    "crop_size": crop_size,
                    "frequency_cycles_per_pixel": float(frequency),
                    "horizontal_psd_norm": float(horizontal[index]),
                    "vertical_psd_norm": float(vertical[index]),
                }
            )
        print(f"spatial folder={folder}", flush=True)
    spatial_dir = output_root / "spatial"
    write_csv(summaries, spatial_dir / "spatial_correlation_summary.csv")
    write_csv(radial_psd_rows, spatial_dir / "radial_psd.csv")
    write_csv(radial_ac_rows, spatial_dir / "radial_autocorrelation.csv")
    write_csv(directional_rows, spatial_dir / "directional_psd.csv")
    np.savez_compressed(spatial_dir / "spatial_arrays.npz", **arrays)


def run_combined(config: dict[str, Any], output_root: Path) -> None:
    source_files = {
        "integrity": output_root / "input_audit" / "data_integrity_report.csv",
        "noise": output_root / "noise_summary" / "folder_noise_summary.csv",
        "temporal": output_root / "temporal_stability" / "temporal_drift_summary.csv",
        "stable": output_root / "stable_component" / "stable_component_summary.csv",
        "row_column": output_root / "row_column" / "row_column_summary.csv",
        "spatial": output_root / "spatial" / "spatial_correlation_summary.csv",
    }
    by_source = {name: {int(float(row["folder"])): row for row in read_csv(path)} for name, path in source_files.items()}
    rows = []
    for folder in [int(value) for value in config["folders"]]:
        integrity = by_source["integrity"][folder]
        temporal = by_source["temporal"][folder]
        stable = by_source["stable"][folder]
        row_column = by_source["row_column"][folder]
        spatial = by_source["spatial"][folder]
        warnings = []
        if integrity["status"] != "PASS":
            warnings.append("integrity_failed")
        if abs(float(temporal["frame_mean_relative_change_half"])) > 0.01:
            warnings.append("global_brightness_drift_gt_1pct")
        if float(stable["minimum_split_map_correlation"]) < 0.5:
            warnings.append("stable_component_not_repeatable")
        coords = crop_coordinates(tuple(config["image_shape_expected"]), int(config["primary_crop_size"]))
        status = "PASS" if not warnings else "WARN"
        row = common_metadata(
            folder,
            int(config["max_frames"]),
            int(config["primary_crop_size"]),
            coords,
            "E1 folder eligibility for operational noise characterization; not a physical component-identification gate.",
            1.0 if status == "PASS" else 0.5,
            int(config["max_frames"]),
            warnings,
        )
        row.update(
            {
                "status": status,
                "integrity_status": integrity["status"],
                "noise_characterization": "PASS" if integrity["status"] == "PASS" else "FAIL",
                "temporal_variability_metric": "eligible",
                "conditional_temporal_variance_metric": "eligible_with_single_scene_warning",
                "repeatable_stable_component_metric": "eligible_operational_only",
                "row_column_structure_metric": "eligible_operational_only",
                "spatial_correlation_metric": "eligible",
                "physical_fixed_pattern_attribution": "not_eligible_without_dark_flat",
                "strict_photon_transfer_curve": "not_eligible_without_controlled_illumination",
                "physical_noise_component_decomposition": "not_eligible",
                "frame_mean_relative_change_half": temporal["frame_mean_relative_change_half"],
                "minimum_split_map_correlation": stable["minimum_split_map_correlation"],
                "row_pattern_energy_dn": row_column["row_pattern_energy_dn"],
                "column_pattern_energy_dn": row_column["column_pattern_energy_dn"],
                "radial_autocorr_r1": spatial["radial_autocorr_r1"],
            }
        )
        rows.append(row)
    write_csv(rows, output_root / "combined" / "folder_eligibility_summary.csv")


if __name__ == "__main__":
    raise SystemExit(main())
