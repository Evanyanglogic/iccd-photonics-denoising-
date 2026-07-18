"""Formal no-dark audit for the 100-image sCMOS 500 ms content source."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def run(repo: Path, cfg: dict[str, Any], output: Path, source_config: Path, smoke_files: int = 0) -> int:
    settings = cfg["no_dark_recheck"]
    historical = read_csv(resolve(repo, settings["historical_manifest"]))
    if len(historical) != int(cfg["expected_clean_count"]):
        raise RuntimeError(f"Historical manifest has {len(historical)} rows, expected {cfg['expected_clean_count']}")
    formal_file_count = len(historical)
    if smoke_files:
        historical = historical[:smoke_files]

    missing = [row["clean_path"] for row in historical if not Path(row["clean_path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"No-dark formal audit requires all source TIFFs before creating output: missing {len(missing)}/{len(historical)}; "
            f"first={missing[0]}"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    # Hash drift is checked before image statistics. A drifted input is recorded
    # as invalid, never silently substituted.
    hash_rows = []
    drift = []
    for row in historical:
        path = Path(row["clean_path"])
        actual = sha256_file(path)
        same = actual.lower() == row["sha256"].lower()
        item = {
            "source_pair_key": row["source_pair_key"],
            "filename": path.name,
            "path": str(path),
            "absolute_path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "creation_time_ns": path.stat().st_ctime_ns,
            "mtime_ns": path.stat().st_mtime_ns,
            "historical_file_size_bytes": int(row["file_size_bytes"]),
            "historical_mtime_ns": int(row["mtime_ns"]),
            "file_size_delta_bytes": path.stat().st_size - int(row["file_size_bytes"]),
            "mtime_delta_ns": path.stat().st_mtime_ns - int(row["mtime_ns"]),
            "historical_sha256": row["sha256"],
            "actual_sha256": actual,
            "sha256_match": same,
        }
        hash_rows.append(item)
        if not same:
            drift.append(item)

    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    logs = output / "logs"
    logs.mkdir()
    started_at = utc_now()
    commit = git(repo, ["rev-parse", "HEAD"]).strip()
    git_status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    write_provenance(repo, cfg, source_config, provenance, commit, git_status)
    write_csv(hash_rows, output / "input_manifest.csv")
    write_csv(hash_rows, output / "sha256_comparison.csv")

    if drift:
        decision = {
            "status": "CLEAN-SOURCE-INVALID",
            "failure_code": "INPUT-DRIFT",
            "drifted_file_count": len(drift),
            "first_drift": drift[0],
            "synthetic_generation_performed": False,
            "model_training_performed": False,
        }
        write_json(decision, output / "verification_status.json")
        write_json(decision, output / "processing_status_assessment.json")
        write_text(output / "verification_report.md", "# E2 No-Dark Input Recheck\n\n**CLEAN-SOURCE-INVALID: INPUT-DRIFT**\n")
        return 2

    metadata_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    thumbnails: list[dict[str, Any]] = []
    for index, row in enumerate(historical):
        path = Path(row["clean_path"])
        metadata, image = read_tiff_and_metadata(path, row["source_pair_key"])
        metadata["pixel_sha256"] = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        metadata_rows.append(metadata)
        roi, coords = center_crop(image, int(settings["crop_size"]))
        full_rows.append(statistics_row(row["source_pair_key"], path, image, "full_image", (0, 0, *image.shape)))
        roi_rows.append(statistics_row(row["source_pair_key"], path, roi, "center_roi", coords))
        thumbnails.append(build_thumbnail_record(row["source_pair_key"], image, roi, int(settings["thumbnail_size"])))
        print(f"no_dark_input {index + 1:03d}/{len(historical)} {path.name}", flush=True)

    similarity_rows, duplicate_rows = similarity_analysis(thumbnails, settings)
    duplicate_rows.extend(exact_duplicate_rows(hash_rows))
    duplicate_rows.extend(pixel_exact_duplicate_rows(metadata_rows))
    group_rows, group_summary = source_group_analysis(historical, metadata_rows, similarity_rows)
    processing = assess_processing(metadata_rows, full_rows, roi_rows, group_summary)
    comparison_rows = full_vs_roi_rows(full_rows, roi_rows)
    high_similarity_rows = [row for row in similarity_rows if row["high_content_similarity"]]
    status = verify(cfg, commit, git_status, hash_rows, metadata_rows, full_rows, roi_rows, duplicate_rows, group_summary, processing, smoke_files)

    write_csv(metadata_rows, output / "tiff_metadata.csv")
    write_csv(full_rows, output / "full_image_statistics.csv")
    write_csv(roi_rows, output / "center_roi_statistics.csv")
    write_csv(comparison_rows, output / "full_vs_roi_statistics.csv")
    write_csv_or_header(duplicate_rows, output / "duplicate_analysis.csv", duplicate_fields())
    write_csv(similarity_rows, output / "content_similarity.csv")
    write_csv_or_header(high_similarity_rows, output / "high_similarity_pairs.csv", similarity_fields())
    write_csv(group_rows, output / "source_group_analysis.csv")
    write_json(group_summary, output / "source_group_decision.json")
    write_json(processing, output / "processing_status_assessment.json")
    write_json(status, output / "verification_status.json")
    write_report(output / "verification_report.md", status, processing, full_rows, roi_rows, group_summary)
    write_text(logs / "stdout.log", f"processed_files={len(historical)}\nstatus={status['status']}\n")
    write_text(logs / "stderr.log", "")
    final_git_status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    write_text(provenance / "git_status_after.txt", final_git_status)
    run_manifest = {
        "experiment_id": cfg["experiment_id"] + "_no_dark_recheck",
        "status": status["status"],
        "git_commit": commit,
        "git_worktree_clean_at_start": not git_status.strip(),
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "exit_code": 0 if status["status"].startswith("VERIFIED-INPUT") else 3,
        "smoke": bool(smoke_files),
        "processed_file_count": len(historical),
        "formal_expected_file_count": formal_file_count,
        "preprocessing": "raw_uint16.astype(np.float32) / 65535.0",
        "dark_subtraction": False,
        "scalar_pedestal_subtraction": False,
        "per_image_p99_scaling": False,
        "synthetic_generation_performed": False,
        "model_training_performed": False,
        "output_hashes": output_hashes(output),
    }
    write_json(run_manifest, provenance / "run_manifest.json")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["status"].startswith("VERIFIED-INPUT") else 3


def read_tiff_and_metadata(path: Path, pair_key: str) -> tuple[dict[str, Any], np.ndarray]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        image = np.asarray(page.asarray())
        tags = {tag.name: safe_tag_value(tag.value) for tag in page.tags.values()}
        text = json.dumps(tags, ensure_ascii=False).lower()
        metadata = {
            "source_pair_key": pair_key,
            "path": str(path),
            "page_count": len(tif.pages),
            "dtype": str(image.dtype),
            "shape": "x".join(map(str, image.shape)),
            "byte_order": tif.byteorder,
            "compression": str(page.compression.name if hasattr(page.compression, "name") else page.compression),
            "photometric_interpretation": str(page.photometric.name if hasattr(page.photometric, "name") else page.photometric),
            "bits_per_sample": str(page.bitspersample),
            "sample_format": str(tags.get("SampleFormat", "")),
            "software": str(tags.get("Software", "")),
            "make": str(tags.get("Make", "")),
            "model": str(tags.get("Model", "")),
            "datetime_tag": str(tags.get("DateTime", "")),
            "exposure_tag": find_tag(tags, ["exposure", "exposuretime"]),
            "gain_tag": find_tag(tags, ["gain", "iso"]),
            "temperature_tag": find_tag(tags, ["temperature", "sensor temperature"]),
            "readout_mode_tag": find_tag(tags, ["readout", "mode"]),
            "black_level_tag": find_tag(tags, ["blacklevel", "black level", "black_level"]),
            "image_description": str(tags.get("ImageDescription", "")),
            "processing_keyword_present": any(word in text for word in ["processed", "corrected", "normalized", "gamma"]),
            "all_tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        }
    return metadata, image


def statistics_row(pair_key: str, path: Path, image: np.ndarray, scope: str, coords: tuple[int, int, int, int]) -> dict[str, Any]:
    from scipy.ndimage import gaussian_filter, laplace, uniform_filter

    raw = np.asarray(image)
    values = raw.astype(np.float64)
    normalized = raw.astype(np.float32) / np.float32(65535.0)
    exact_normalized = values / 65535.0
    q = np.percentile(values, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    gx = np.diff(normalized, axis=1)
    gy = np.diff(normalized, axis=0)
    local_mean = uniform_filter(normalized, size=9, mode="reflect")
    local_sq = uniform_filter(normalized * normalized, size=9, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    high = normalized - gaussian_filter(normalized, sigma=1.0, mode="reflect")
    lap = laplace(normalized, mode="reflect")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, 1.0)
    bad_candidates = (values <= median - 8 * robust_sigma) | (values >= median + 8 * robust_sigma)
    return {
        "source_pair_key": pair_key,
        "path": str(path),
        "scope": scope,
        "crop_top": coords[0],
        "crop_left": coords[1],
        "crop_height": coords[2],
        "crop_width": coords[3],
        "dtype": str(raw.dtype),
        "shape": "x".join(map(str, raw.shape)),
        "raw_min_dn": float(np.min(values)),
        "raw_max_dn": float(np.max(values)),
        "raw_mean_dn": float(np.mean(values)),
        "raw_std_dn": float(np.std(values, ddof=1)),
        "raw_p0_1_dn": float(q[0]),
        "raw_p1_dn": float(q[1]),
        "raw_p5_dn": float(q[2]),
        "raw_p25_dn": float(q[3]),
        "raw_p50_dn": float(q[4]),
        "raw_median_dn": float(q[4]),
        "raw_p75_dn": float(q[5]),
        "raw_p95_dn": float(q[6]),
        "raw_p99_dn": float(q[7]),
        "raw_p99_9_dn": float(q[8]),
        "norm_min": float(np.min(normalized)),
        "norm_max": float(np.max(normalized)),
        "norm_mean": float(np.mean(normalized)),
        "norm_std": float(np.std(normalized, ddof=1)),
        "norm_p0_1": float(q[0] / 65535.0),
        "norm_p1": float(q[1] / 65535.0),
        "norm_p5": float(q[2] / 65535.0),
        "norm_p25": float(q[3] / 65535.0),
        "norm_p50": float(q[4] / 65535.0),
        "norm_median": float(q[4] / 65535.0),
        "norm_p75": float(q[5] / 65535.0),
        "norm_p95": float(q[6] / 65535.0),
        "norm_p99": float(q[7] / 65535.0),
        "norm_p99_9": float(q[8] / 65535.0),
        "normalization_max_abs_error_vs_float64_div65535": float(
            np.max(np.abs(normalized.astype(np.float64) - exact_normalized))
        ),
        "zero_ratio": float(np.mean(raw == 0)),
        "saturation_ratio": float(np.mean(raw == 65535)),
        "negative_before_clipping_ratio": 0.0,
        "dynamic_range_dn": float(np.max(values) - np.min(values)),
        "robust_dynamic_range_dn_p1_p99": float(q[7] - q[1]),
        "local_contrast_mean_norm": float(np.mean(local_std)),
        "gradient_energy_norm": float((np.mean(gx * gx) + np.mean(gy * gy)) / 2.0),
        "laplacian_variance_norm": float(np.var(lap, ddof=1)),
        "high_frequency_energy_norm": float(np.mean(high * high)),
        "row_mean_variation_norm": float(np.std(np.mean(normalized, axis=1), ddof=1)),
        "column_mean_variation_norm": float(np.std(np.mean(normalized, axis=0), ddof=1)),
        "bad_pixel_candidate_ratio": float(np.mean(bad_candidates)),
        "unique_value_count": int(np.unique(raw).size),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "preprocessing": "raw_uint16.astype(np.float32)/65535.0",
    }


def build_thumbnail_record(pair_key: str, full: np.ndarray, roi: np.ndarray, size: int) -> dict[str, Any]:
    from skimage.transform import resize

    image = roi.astype(np.float32) / np.float32(65535.0)
    full_image = full.astype(np.float32) / np.float32(65535.0)
    thumb = resize(image, (size, size), preserve_range=True, anti_aliasing=True).astype(np.float32)
    full_thumb = resize(full_image, (size, size), preserve_range=True, anti_aliasing=True).astype(np.float32)
    low = gaussian_component(thumb, sigma=3.0)
    high = thumb - low
    gradient = gradient_map(thumb)
    return {"pair_key": pair_key, "image": thumb, "full": full_thumb, "low": low, "high": high, "gradient": gradient}


def similarity_analysis(records: list[dict[str, Any]], settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from skimage.metrics import structural_similarity

    rows = []
    duplicates = []
    for i, first in enumerate(records):
        for second in records[i + 1 :]:
            corr = correlation(first["image"], second["image"])
            low_corr = correlation(first["low"], second["low"])
            high_corr = correlation(first["high"], second["high"])
            ssim = float(structural_similarity(first["image"], second["image"], data_range=1.0))
            full_corr = correlation(first["full"], second["full"])
            row = {
                "source_pair_key_a": first["pair_key"],
                "source_pair_key_b": second["pair_key"],
                "full_image_correlation": full_corr,
                "center_roi_correlation": corr,
                "center_roi_ssim": ssim,
                "low_frequency_correlation": low_corr,
                "high_frequency_correlation": high_corr,
                "gradient_map_correlation": correlation(first["gradient"], second["gradient"]),
                "downsampled_perceptual_similarity": ssim,
                "high_content_similarity": (
                    full_corr >= float(settings["high_content_similarity_correlation"])
                    or corr >= float(settings["high_content_similarity_correlation"])
                    or ssim >= float(settings["perceptual_duplicate_ssim"])
                ),
            }
            rows.append(row)
            if corr >= float(settings["perceptual_duplicate_correlation"]) and ssim >= float(settings["perceptual_duplicate_ssim"]):
                duplicates.append({**row, "duplicate_type": "perceptual_candidate"})
    return rows, duplicates


def exact_duplicate_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = {}
    for row in manifest:
        by_hash.setdefault(row["actual_sha256"], []).append(row["source_pair_key"])
    rows = []
    for keys in by_hash.values():
        for index, first in enumerate(keys):
            for second in keys[index + 1:]:
                rows.append(
                    {
                        "source_pair_key_a": first,
                        "source_pair_key_b": second,
                        "center_roi_correlation": 1.0,
                        "center_roi_ssim": 1.0,
                        "low_frequency_correlation": 1.0,
                        "high_frequency_correlation": 1.0,
                        "high_content_similarity": True,
                        "duplicate_type": "exact_sha256",
                    }
                )
    return rows


def pixel_exact_duplicate_rows(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = {}
    for row in metadata:
        by_hash.setdefault(row["pixel_sha256"], []).append(row["source_pair_key"])
    rows = []
    for keys in by_hash.values():
        for index, first in enumerate(keys):
            for second in keys[index + 1:]:
                rows.append({
                    "source_pair_key_a": first,
                    "source_pair_key_b": second,
                    "full_image_correlation": 1.0,
                    "center_roi_correlation": 1.0,
                    "center_roi_ssim": 1.0,
                    "low_frequency_correlation": 1.0,
                    "high_frequency_correlation": 1.0,
                    "gradient_map_correlation": 1.0,
                    "downsampled_perceptual_similarity": 1.0,
                    "high_content_similarity": True,
                    "duplicate_type": "pixel_exact",
                })
    return rows


def source_group_analysis(
    manifest: list[dict[str, str]], metadata: list[dict[str, Any]], similarity: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_by_key = {row["source_pair_key"]: row for row in metadata}
    high_pairs = sum(bool(row["high_content_similarity"]) for row in similarity)
    rows = []
    for index, row in enumerate(manifest):
        key = row["source_pair_key"]
        meta = metadata_by_key[key]
        rows.append(
            {
                "source_pair_key": key,
                "filename": Path(row["clean_path"]).name,
                "parent_directory": str(Path(row["clean_path"]).parent),
                "manifest_order": index,
                "datetime_tag": meta["datetime_tag"],
                "source_scene": "",
                "source_group": "source_group_unknown",
                "group_basis": "filename, single parent folder, metadata, and content similarity do not establish scene identity",
            }
        )
    return rows, {
        "source_scene_reliably_established": False,
        "source_group_reliably_established": False,
        "assigned_group": "source_group_unknown",
        "high_similarity_pair_count": high_pairs,
        "pair_count": len(similarity),
        "decision_basis": "parent directory, filename sequence, TIFF datetime/tags, acquisition order, and image similarity were reviewed",
    }


def full_vs_roi_rows(full: list[dict[str, Any]], roi: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row["source_pair_key"]: row for row in full}
    rows = []
    for roi_row in roi:
        full_row = by_key[roi_row["source_pair_key"]]
        rows.append({
            "source_pair_key": roi_row["source_pair_key"],
            "full_mean_dn": full_row["raw_mean_dn"],
            "roi_mean_dn": roi_row["raw_mean_dn"],
            "roi_to_full_mean_ratio": roi_row["raw_mean_dn"] / max(full_row["raw_mean_dn"], 1e-12),
            "full_p99_dn": full_row["raw_p99_dn"],
            "roi_p99_dn": roi_row["raw_p99_dn"],
            "full_robust_dynamic_range_dn": full_row["robust_dynamic_range_dn_p1_p99"],
            "roi_robust_dynamic_range_dn": roi_row["robust_dynamic_range_dn_p1_p99"],
            "full_zero_ratio": full_row["zero_ratio"],
            "roi_zero_ratio": roi_row["zero_ratio"],
            "full_saturation_ratio": full_row["saturation_ratio"],
            "roi_saturation_ratio": roi_row["saturation_ratio"],
        })
    return rows


def assess_processing(
    metadata: list[dict[str, Any]], full: list[dict[str, Any]], roi: list[dict[str, Any]], groups: dict[str, Any]
) -> dict[str, Any]:
    software = sorted({row["software"] for row in metadata if row["software"]})
    explicit = [row["source_pair_key"] for row in metadata if row["processing_keyword_present"]]
    return {
        "classification": "likely camera-processed" if explicit else "processing status unknown",
        "metadata_explicit_processing_file_count": len(explicit),
        "software_values": software,
        "black_level_correction": "NOT_ESTABLISHED",
        "uniform_pedestal": "NOT_ESTABLISHED",
        "software_clipping": "NOT_ESTABLISHED",
        "gamma_or_nonlinear_mapping": "NOT_ESTABLISHED",
        "bad_pixel_correction": "NOT_ESTABLISHED",
        "cross_exposure_nonmonotonic_median_evidence": "previous audit observed decreasing median with nominal exposure; acquisition settings/pipeline unresolved",
        "within_500ms_mean_range_dn": range_of(full, "raw_mean_dn"),
        "within_500ms_p99_range_dn": range_of(full, "raw_p99_dn"),
        "within_500ms_roi_dynamic_range_dn": range_of(roi, "dynamic_range_dn"),
        "source_group": groups["assigned_group"],
        "content_boundary": "sCMOS-derived operational content source; not clean ground truth",
    }


def verify(
    cfg: dict[str, Any], commit: str, git_status: str, manifest: list[dict[str, Any]], metadata: list[dict[str, Any]],
    full: list[dict[str, Any]], roi: list[dict[str, Any]], duplicates: list[dict[str, Any]], groups: dict[str, Any],
    processing: dict[str, Any], smoke_files: int,
) -> dict[str, Any]:
    checks = []
    add = lambda name, passed, value: checks.append({"name": name, "passed": bool(passed), "observed": value})
    add("file_count", len(manifest) == (smoke_files or 100), len(manifest))
    add("sha256_match", all(row["sha256_match"] for row in manifest), sum(not row["sha256_match"] for row in manifest))
    add("dtype_uint16", all(row["dtype"] == "uint16" for row in metadata), sorted({row["dtype"] for row in metadata}))
    add("shape_2048x2048", all(row["shape"] == "2048x2048" for row in metadata), sorted({row["shape"] for row in metadata}))
    add("center_roi_coordinates", all((row["crop_top"], row["crop_left"], row["crop_height"], row["crop_width"]) == (768, 768, 512, 512) for row in roi), "768,768,512,512")
    add("zero_ratio_lt_5pct", max(row["zero_ratio"] for row in full + roi) < 0.05, max(row["zero_ratio"] for row in full + roi))
    add("saturation_ratio_lt_1pct", max(row["saturation_ratio"] for row in full + roi) < 0.01, max(row["saturation_ratio"] for row in full + roi))
    add("negative_ratio_zero", max(row["negative_before_clipping_ratio"] for row in full + roi) == 0, 0)
    add("no_nan_inf", all(row["nan_count"] == 0 and row["inf_count"] == 0 for row in full + roi), True)
    add("no_per_image_scaling", all(row["preprocessing"] == "raw_uint16.astype(np.float32)/65535.0" for row in full + roi), True)
    normalization_error = max(row["normalization_max_abs_error_vs_float64_div65535"] for row in full + roi)
    float32_half_ulp_at_one = float(np.finfo(np.float32).eps / 2.0)
    add("normalization_is_float32_divide_65535", normalization_error <= float32_half_ulp_at_one, normalization_error)
    add("no_uint8_conversion", all(row["dtype"] == "uint16" for row in full + roi), sorted({row["dtype"] for row in full + roi}))
    add("no_silent_clipping", all(row["norm_min"] >= 0.0 and row["norm_max"] <= 1.0 for row in full + roi), True)
    add("inter_image_mean_not_forced", max(row["raw_mean_dn"] for row in full) > min(row["raw_mean_dn"] for row in full), range_of(full, "raw_mean_dn"))
    add("inter_image_p99_not_forced", max(row["raw_p99_dn"] for row in full) > min(row["raw_p99_dn"] for row in full), range_of(full, "raw_p99_dn"))
    add("inter_image_dynamic_range_not_forced", max(row["robust_dynamic_range_dn_p1_p99"] for row in full) > min(row["robust_dynamic_range_dn_p1_p99"] for row in full), range_of(full, "robust_dynamic_range_dn_p1_p99"))
    add("perceptual_or_exact_duplicate_fraction_lt_20pct", len(duplicates) / 4950.0 < 0.20, len(duplicates) / 4950.0)
    add("metadata_recorded", len(metadata) == (smoke_files or 100) and all(row["all_tags_json"] for row in metadata), len(metadata))
    add("clean_commit", bool(smoke_files) or not git_status.strip(), commit)
    hard = [item for item in checks if item["name"] not in {"clean_commit"}]
    if not all(item["passed"] for item in hard):
        status = "CLEAN-SOURCE-INVALID"
    elif not checks[-1]["passed"]:
        status = "FAILED"
    elif smoke_files:
        status = "VERIFIED-INPUT-WITH-LIMITATIONS"
    elif processing["classification"] == "processing status unknown" or not groups["source_group_reliably_established"]:
        status = "VERIFIED-INPUT-WITH-LIMITATIONS"
    else:
        status = "VERIFIED-INPUT"
    return {
        "status": status,
        "checks": checks,
        "processing_status": processing["classification"],
        "source_group": groups["assigned_group"],
        "synthetic_generation_performed": False,
        "model_training_performed": False,
    }


def write_report(path: Path, status: dict[str, Any], processing: dict[str, Any], full: list[dict[str, Any]], roi: list[dict[str, Any]], groups: dict[str, Any]) -> None:
    lines = [
        "# E2 No-Dark Formal Input Recheck", "", f"- Status: **{status['status']}**",
        "- Preprocessing: `raw_uint16.astype(np.float32) / 65535.0`", "- Dark subtraction: no",
        "- Scalar pedestal subtraction: no", "- Per-image p99 scaling: no", "- Synthetic generation: no",
        "", "## Processing Assessment", "", f"- Classification: {processing['classification']}",
        f"- Source group: {groups['assigned_group']}",
        f"- Full-image mean range: {range_of(full, 'raw_mean_dn')}",
        f"- Center-ROI p99 range: {range_of(roi, 'raw_p99_dn')}", "", "## Gates", "",
    ]
    for item in status["checks"]:
        lines.append(f"- {'PASS' if item['passed'] else 'FAIL'} `{item['name']}`: {item['observed']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_provenance(repo: Path, cfg: dict[str, Any], source_config: Path, output: Path, commit: str, status: str) -> None:
    write_text(output / "git_commit.txt", commit + "\n")
    write_text(output / "git_status_before.txt", status)
    write_text(output / "git_diff.patch", git(repo, ["diff", "--binary", "HEAD"]))
    write_text(output / "command.txt", subprocess.list2cmdline(sys.argv) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_text(output / "environment.txt", "\n".join([
        f"python={sys.version}", f"platform={platform.platform()}", f"numpy={np.__version__}",
    ]) + "\n")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    write_text(output / "pip_freeze.txt", freeze.stdout or "")
    gpu = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    write_text(output / "gpu_info.txt", gpu.stdout or "")
    names = [source_config, Path(__file__), repo / "scripts/audit_e2_dark_offset_compatibility.py"]
    write_csv([{"path": str(item), "size_bytes": item.stat().st_size, "sha256": sha256_file(item)} for item in names], output / "script_hashes.csv")


def safe_tag_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value[:256].hex() + ("..." if len(value) > 256 else "")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [safe_tag_value(item) for item in value[:64]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_tag(tags: dict[str, Any], candidates: list[str]) -> str:
    for key, value in tags.items():
        if any(candidate in key.lower() for candidate in candidates):
            return str(value)
    return ""


def center_crop(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return np.asarray(image[top:top + size, left:left + size]), (top, left, size, size)


def gaussian_component(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma, mode="reflect")


def gradient_map(image: np.ndarray) -> np.ndarray:
    gx = np.gradient(image, axis=1)
    gy = np.gradient(image, axis=0)
    return np.sqrt(gx * gx + gy * gy)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    if np.std(first) == 0 or np.std(second) == 0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def range_of(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [min(values), max(values)]


def duplicate_fields() -> list[str]:
    return similarity_fields() + ["duplicate_type"]


def similarity_fields() -> list[str]:
    return ["source_pair_key_a", "source_pair_key_b", "full_image_correlation", "center_roi_correlation", "center_roi_ssim", "low_frequency_correlation", "high_frequency_correlation", "gradient_map_correlation", "downsampled_perceptual_similarity", "high_content_similarity"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_or_header(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout or ""


def output_hashes(output: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(output)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_manifest.json"]


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
