"""Audit E2 clean-content inputs, historical outputs, and E1 parameter provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(config)
    return 0


def run(config: dict[str, Any]) -> dict[str, Any]:
    import tifffile

    output_root = Path(config["output_root"])
    input_dir = output_root / "input_audit"
    generation_dir = output_root / "generation_audit"
    input_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(Path(config["source_pairs_csv"]))
    splits = yaml.safe_load(Path(config["source_splits_yaml"]).read_text(encoding="utf-8"))
    key_to_split = {key: split for split, keys in splits.items() for key in keys}
    dark = np.load(config["dark_offset_path"])
    bad = np.asarray(np.load(config["bad_pixel_mask_path"]), dtype=bool)
    expected_shape = tuple(config["expected_shape"])
    range_max = float(config["range_max"])
    crop_size = int(config["crop_size"])
    manifest_rows = []
    audit_rows = []
    thumbnails = []

    for row in source_rows:
        pair_key = row["pair_key"]
        path = Path(row[config["clean_column"]])
        stat = path.stat()
        sha = sha256_file(path)
        image = tifffile.memmap(path)
        crop, coords = center_crop(np.asarray(image), crop_size)
        dark_crop, _ = center_crop(dark, crop_size)
        bad_crop, _ = center_crop(bad, crop_size)
        valid = (~bad_crop) & (crop > 0) & (crop < range_max)
        corrected = np.clip((crop.astype(np.float32) - dark_crop.astype(np.float32)) / range_max, 0.0, 1.0)
        corrected_valid = corrected[valid]
        raw = np.asarray(image)
        raw_float = raw.astype(np.float64, copy=False)
        highpass_std = estimate_high_frequency_std(corrected)
        warning_flags = []
        if np.mean(raw == 0) > 0.01:
            warning_flags.append("excess_zero_pixels")
        if np.mean(raw >= range_max) > 0.01:
            warning_flags.append("excess_saturation")
        if float(np.median(dark_crop)) > 0:
            warning_flags.append("dark_offset_correction_required")
        if float(np.mean(corrected == 0)) > 0.01:
            warning_flags.append("corrected_zero_clipping_gt_1pct")
        warning_flags.append("single_scmos_frame_contains_unresolved_sensor_noise")
        manifest_rows.append(
            {
                "source_pair_key": pair_key,
                "clean_path": str(path),
                "relative_to_parent": str(path.relative_to(path.parents[1])),
                "file_size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha,
                "hash_strategy": config["hash_strategy"],
                "split": key_to_split.get(pair_key, "UNASSIGNED"),
            }
        )
        audit_rows.append(
            {
                "source_pair_key": pair_key,
                "clean_path": str(path),
                "split": key_to_split.get(pair_key, "UNASSIGNED"),
                "source_device": row.get("source_device", config["clean_device"]),
                "source_exposure": row.get("clean_exposure", config["clean_exposure"]),
                "dtype": str(image.dtype),
                "shape": "x".join(map(str, image.shape)),
                "raw_min_dn": float(np.min(raw)),
                "raw_max_dn": float(np.max(raw)),
                "raw_mean_dn": float(np.mean(raw_float)),
                "raw_std_dn": float(np.std(raw_float, ddof=1)),
                "raw_p1_dn": float(np.percentile(raw, 1)),
                "raw_p50_dn": float(np.percentile(raw, 50)),
                "raw_p99_dn": float(np.percentile(raw, 99)),
                "raw_zero_ratio": float(np.mean(raw == 0)),
                "raw_saturation_ratio": float(np.mean(raw >= range_max)),
                "crop_size": crop_size,
                "crop_top": coords[0],
                "crop_left": coords[1],
                "dark_offset_median_dn": float(np.median(dark_crop)),
                "valid_fraction": float(np.mean(valid)),
                "corrected_mean_norm": float(np.mean(corrected_valid)),
                "corrected_std_norm": float(np.std(corrected_valid, ddof=1)),
                "corrected_p1_norm": float(np.percentile(corrected_valid, 1)),
                "corrected_p50_norm": float(np.percentile(corrected_valid, 50)),
                "corrected_p99_norm": float(np.percentile(corrected_valid, 99)),
                "corrected_zero_ratio": float(np.mean(corrected == 0)),
                "corrected_saturation_ratio": float(np.mean(corrected >= 1.0)),
                "corrected_high_frequency_std_norm": highpass_std,
                "warning_flags": ";".join(warning_flags),
                "sha256": sha,
            }
        )
        thumbnails.append((pair_key, normalized_thumbnail(raw, config["near_duplicate"])))
        print(f"clean_input {pair_key}", flush=True)

    duplicate_rows = duplicate_report(manifest_rows, thumbnails, config["near_duplicate"])
    thumbnail_stats = thumbnail_correlation_summary(thumbnails)
    exact_duplicates = sum(row["match_type"] == "exact_sha256" for row in duplicate_rows)
    near_duplicates = sum(row["match_type"] == "near_content" for row in duplicate_rows)
    dtype_ok = all(row["dtype"] == config["expected_dtype"] for row in audit_rows)
    shape_ok = all(row["shape"] == "x".join(map(str, expected_shape)) for row in audit_rows)
    summary = {
        "status": "PASS" if dtype_ok and shape_ok and not exact_duplicates else "FAIL",
        "file_count": len(audit_rows),
        "source_device": config["clean_device"],
        "source_exposure": config["clean_exposure"],
        "dtype_counts": counts(row["dtype"] for row in audit_rows),
        "shape_counts": counts(row["shape"] for row in audit_rows),
        "raw_min_dn": min(row["raw_min_dn"] for row in audit_rows),
        "raw_max_dn": max(row["raw_max_dn"] for row in audit_rows),
        "raw_mean_dn_range": range_of(audit_rows, "raw_mean_dn"),
        "raw_p99_dn_range": range_of(audit_rows, "raw_p99_dn"),
        "raw_zero_ratio_max": max(row["raw_zero_ratio"] for row in audit_rows),
        "raw_saturation_ratio_max": max(row["raw_saturation_ratio"] for row in audit_rows),
        "corrected_mean_norm_range": range_of(audit_rows, "corrected_mean_norm"),
        "corrected_p99_norm_range": range_of(audit_rows, "corrected_p99_norm"),
        "corrected_zero_ratio_range": range_of(audit_rows, "corrected_zero_ratio"),
        "corrected_saturation_ratio_max": max(row["corrected_saturation_ratio"] for row in audit_rows),
        "corrected_high_frequency_std_norm_range": range_of(audit_rows, "corrected_high_frequency_std_norm"),
        "exact_duplicate_pair_count": exact_duplicates,
        "near_duplicate_pair_count": near_duplicates,
        "near_duplicate_threshold": config["near_duplicate"]["correlation_threshold"],
        "thumbnail_pair_correlation_p50": thumbnail_stats["p50"],
        "thumbnail_pair_correlation_p90": thumbnail_stats["p90"],
        "thumbnail_pair_correlation_p99": thumbnail_stats["p99"],
        "thumbnail_pair_correlation_max": thumbnail_stats["max"],
        "scene_metadata_present": all("source_scene" in row and row.get("source_scene", "") for row in source_rows),
        "clean_content_boundary": "single-frame sCMOS long-exposure content source with unresolved offset and sensor noise; not ICCD clean ground truth",
    }
    write_csv(manifest_rows, input_dir / "input_clean_manifest.csv")
    write_csv(audit_rows, input_dir / "clean_content_audit.csv")
    write_csv_or_header(duplicate_rows, input_dir / "duplicate_or_near_duplicate_report.csv", duplicate_fields())
    write_json(summary, input_dir / "clean_content_summary.json")

    mapping = build_parameter_mapping(config)
    write_csv(mapping, generation_dir / "e2_parameter_to_e1_mapping.csv")
    historical = audit_historical_outputs(config, source_rows)
    write_csv(historical["rows"], generation_dir / "historical_e2_output_audit.csv")
    write_json(historical["summary"], generation_dir / "historical_e2_summary.json")
    return {"clean": summary, "historical": historical["summary"]}


def build_parameter_mapping(config: dict[str, Any]) -> list[dict[str, Any]]:
    old = yaml.safe_load(Path("configs/iccd_prior_20260319.yaml").read_text(encoding="utf-8"))
    e1_noise = read_csv(Path(config["e1_formal_root"]) / "noise_summary" / "folder_noise_summary.csv")
    e1_fano = read_csv(Path(config["e1_formal_root"]) / "mean_variance" / "fano_like_summary.csv")
    e1_row = read_csv(Path(config["e1_formal_root"]) / "row_column" / "row_column_summary.csv")
    e1_spatial = read_csv(Path(config["e1_formal_root"]) / "spatial" / "spatial_correlation_summary.csv")
    temporal = [float(row["temporal_std_mean"]) for row in e1_noise]
    fano = [float(row["fano_like_dn"]) for row in e1_fano]
    rows = [
        mapping("residual_std", old["iccd"]["read_noise_sigma"], "old E1 minimum temporal std / 65535", False, f"formal temporal std {min(temporal):.6g}..{max(temporal):.6g} DN", "normalized vs raw DN", "partly interpretable only as an old lower-bound scale"),
        mapping("gain/photon_scale", old["iccd"]["photon_scale"], "old single-scene mean-variance linear slope", False, "no strict photon-transfer gain in formal E1", "dimensionless effective count", "not physically identifiable"),
        mapping("Fano-like", "not used directly", "none", False, f"formal operational range {min(fano):.6g}..{max(fano):.6g} DN", "not applicable", "recorded but not mapped"),
        mapping("signal_dependence", "Poisson(clean*photon_scale)", "old slope proxy", False, "formal conditional temporal variance only", "normalized signal to effective counts", "mechanism not validated"),
        mapping("row_energy", 0.0, "generator has no row term", False, range_text(e1_row, "row_pattern_energy_dn", "DN"), "missing mapping", "not represented"),
        mapping("column_energy", 0.0, "generator has no column term", False, range_text(e1_row, "column_pattern_energy_dn", "DN"), "missing mapping", "not represented"),
        mapping("PSD", "not used", "none", False, "formal radial and directional PSD available", "missing mapping", "not represented"),
        mapping("autocorrelation", old["iccd"]["phosphor_sigma"], "thresholded old lag correlation; currently zero", False, range_text(e1_spatial, "radial_autocorr_r1", "correlation"), "sigma px vs correlation", "not represented in active config"),
        mapping("stable_component_amplitude", "not used", "old value recorded but not injected", False, "formal repeatable observed stable component", "missing mapping", "must not be called fixed-pattern"),
        mapping("clipping_threshold", "[0,1] then uint16 [0,65535]", "generic numerical bound", False, "formal saturation value 65535 DN", "consistent bound, different domains", "numerically clear"),
        mapping("pedestal", 0.0, "model offset; clean dark offset is subtracted", False, "formal E1 did not identify a transferable pedestal", "normalized", "no transferable mapping"),
        mapping("content_p99", "0 or per-image 0.25", "manual E2 option", False, "not an E1 noise statistic", "normalized intensity", "changes absolute content scale per image"),
        mapping("physical_scaling", "content_p99_target=0", "preserve corrected sCMOS content scale", False, "not a physical E1 parameter", "normalized intensity", "rename legacy_unscaled_content"),
    ]
    return rows


def mapping(parameter: str, value: Any, source: str, formal: bool, e1_metric: str, units: str, interpretation: str) -> dict[str, Any]:
    return {"synthetic_parameter": parameter, "current_code_value_or_operation": value, "current_code_source": source, "from_formal_e1": formal, "formal_e1_corresponding_metric": e1_metric, "unit_consistency": units, "interpretability": interpretation}


def audit_historical_outputs(config: dict[str, Any], source_rows: list[dict[str, str]]) -> dict[str, Any]:
    result_rows = []
    summaries = {}
    for variant, variant_cfg in config["historical_variants"].items():
        root = Path(variant_cfg["output_root"])
        pairs_path = root / "pairs.csv"
        metrics_path = root / "synthetic_pair_metrics.csv"
        split_path = root / "splits.yaml"
        rows = read_csv(pairs_path) if pairs_path.exists() else []
        missing_outputs = 0
        for row in rows:
            missing_outputs += int(not Path(row["clean_path"]).exists()) + int(not Path(row["noisy_path"]).exists())
        provenance = {
            "raw_log": (root / "run.log").exists() or (root / "generation.log").exists(),
            "commit": (root / "git_commit.txt").exists(),
            "environment": (root / "environment.txt").exists(),
            "resolved_config": (root / "resolved_config.yaml").exists(),
            "input_hashes": (root / "input_file_hashes.csv").exists(),
        }
        status = "PARTIAL-RUN" if rows and metrics_path.exists() and split_path.exists() and missing_outputs == 0 else "INVALID"
        if all(provenance.values()):
            status = "VERIFIED-GENERATION"
        result_rows.append(
            {
                "variant": variant,
                "historical_name": "physical-scale" if variant == "legacy_unscaled_content" else "p99",
                "recommended_name": variant,
                "pair_count": len(rows),
                "missing_output_file_count": missing_outputs,
                "pairs_csv": str(pairs_path),
                "metrics_csv_exists": metrics_path.exists(),
                "splits_yaml_exists": split_path.exists(),
                "raw_log_exists": provenance["raw_log"],
                "commit_record_exists": provenance["commit"],
                "environment_record_exists": provenance["environment"],
                "resolved_config_exists": provenance["resolved_config"],
                "input_hashes_exist": provenance["input_hashes"],
                "content_p99_target": variant_cfg["content_p99_target"],
                "per_pair_noise_seed_recorded": bool(rows and "noise_seed" in rows[0]),
                "crop_coordinates_recorded": bool(rows and all(field in rows[0] for field in ["crop_top", "crop_left"])),
                "split_recorded_in_pair_manifest": bool(rows and "split" in rows[0]),
                "status": status,
                "formal_use_gate": "NO-GO until scene-isolated split and formal parameter definition exist",
            }
        )
        summaries[variant] = status
    return {"rows": result_rows, "summary": {"variants": summaries, "historical_chain_status": "INVALID_FOR_FORMAL_USE", "reason": "generation artifacts exist, but run provenance is incomplete, source_scene metadata is absent, and the physical label is unsupported"}}


def normalized_thumbnail(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    stride = int(cfg["thumbnail_stride"])
    thumbnail = np.asarray(image[::stride, ::stride], dtype=np.float32)
    low, high = np.percentile(thumbnail, cfg["robust_percentiles"])
    thumbnail = np.clip((thumbnail - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    thumbnail -= float(np.mean(thumbnail))
    norm = float(np.linalg.norm(thumbnail))
    return (thumbnail / max(norm, 1e-12)).ravel()


def thumbnail_correlation_summary(thumbnails: list[tuple[str, np.ndarray]]) -> dict[str, float]:
    matrix = np.stack([value for _, value in thumbnails])
    correlations = matrix @ matrix.T
    values = correlations[np.triu_indices(len(thumbnails), 1)]
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def duplicate_report(manifest: list[dict[str, Any]], thumbnails: list[tuple[str, np.ndarray]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_sha: dict[str, list[str]] = {}
    for row in manifest:
        by_sha.setdefault(row["sha256"], []).append(row["source_pair_key"])
    for sha, keys in by_sha.items():
        for index, first in enumerate(keys):
            for second in keys[index + 1 :]:
                rows.append({"source_pair_key_a": first, "source_pair_key_b": second, "match_type": "exact_sha256", "correlation": 1.0, "threshold": 1.0, "sha256": sha})
    threshold = float(cfg["correlation_threshold"])
    matrix = np.stack([value for _, value in thumbnails])
    correlations = matrix @ matrix.T
    for first in range(len(thumbnails)):
        for second in range(first + 1, len(thumbnails)):
            corr = float(correlations[first, second])
            if corr >= threshold:
                rows.append({"source_pair_key_a": thumbnails[first][0], "source_pair_key_b": thumbnails[second][0], "match_type": "near_content", "correlation": corr, "threshold": threshold, "sha256": ""})
    return rows


def estimate_high_frequency_std(image: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter

    high = image - gaussian_filter(image, sigma=1.0, mode="reflect")
    return float(np.std(high, ddof=1))


def center_crop(array: np.ndarray, crop_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = array.shape
    size = min(crop_size, height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return np.asarray(array[top : top + size, left : left + size]), (top, left, size, size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def range_text(rows: list[dict[str, str]], key: str, unit: str) -> str:
    values = [float(row[key]) for row in rows]
    return f"{min(values):.6g}..{max(values):.6g} {unit}"


def range_of(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [min(values), max(values)]


def counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def duplicate_fields() -> list[str]:
    return ["source_pair_key_a", "source_pair_key_b", "match_type", "correlation", "threshold", "sha256"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_or_header(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
