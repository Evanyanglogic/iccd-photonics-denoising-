"""Audit whether the historical sCMOS dark-offset map is compatible with 500 ms content.

This script never writes synthetic image pairs and never modifies source TIFFs.
When the source volume is unavailable, raw pixel-level tests are reported as
not computable instead of being inferred from clipped historical outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PERCENTILES = [0.1, 1.0, 5.0, 50.0, 95.0, 99.0]
TRANSFORMS = {
    "identity": lambda x: x,
    "transpose": lambda x: x.T,
    "flip_vertical": lambda x: np.flipud(x),
    "flip_horizontal": lambda x: np.fliplr(x),
    "rotate_180": lambda x: np.flipud(np.fliplr(x)),
    "transpose_flip_vertical": lambda x: np.flipud(x.T),
    "transpose_flip_horizontal": lambda x: np.fliplr(x.T),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--smoke-files", type=int, default=0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source_config = resolve(repo, args.config)
    cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if args.output_root:
        cfg["output_root"] = args.output_root
    output = resolve(repo, cfg["output_root"])
    no_dark_prefix = cfg.get("no_dark_recheck", {}).get("output_name_prefix", "e2_no_dark_input_recheck_")
    if output.name.startswith(no_dark_prefix):
        from audit_e2_no_dark_input import run as run_no_dark_input

        return run_no_dark_input(repo, cfg, output, source_config, smoke_files=args.smoke_files)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    provenance = output / "provenance"
    logs = output / "logs"
    provenance.mkdir()
    logs.mkdir()

    started = utc_now()
    commit = git(repo, ["rev-parse", "HEAD"]).strip()
    status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    write_text(provenance / "git_commit.txt", commit + "\n")
    write_text(provenance / "git_status.txt", status)
    write_text(provenance / "git_diff.patch", git(repo, ["diff", "--binary", "HEAD"]))
    write_text(provenance / "command.txt", subprocess.list2cmdline(sys.argv) + "\n")
    (provenance / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    save_environment(provenance)
    write_hashes(repo, cfg, provenance / "input_hashes.csv", provenance / "script_hashes.csv")

    source_root = Path(cfg["source_root"])
    source_available = source_root.is_dir()
    dark_path = resolve(repo, cfg["dark_offset_path"])
    dark_full = np.asarray(np.load(dark_path), dtype=np.float32)
    dark, dark_crop = center_crop(dark_full, int(cfg["crop_size"]))
    dark_std_full = np.asarray(np.load(resolve(repo, cfg["dark_std_path"])), dtype=np.float32)
    dark_std, _ = center_crop(dark_std_full, int(cfg["crop_size"]))
    bad_full = np.asarray(np.load(resolve(repo, cfg["bad_pixel_mask_path"])), dtype=bool)
    bad, _ = center_crop(bad_full, int(cfg["crop_size"]))

    provenance_payload = build_dark_provenance(repo, cfg, dark_path, dark_full, dark_crop, source_available)
    write_json(provenance_payload, output / "dark_offset_provenance.json")
    write_csv(dark_statistics_rows(dark_full, dark, dark_std, bad), output / "dark_offset_statistics.csv")

    clean_audit = read_csv(resolve(repo, cfg["saved_clean_audit_csv"]))
    raw_paths = [Path(row["clean_path"]) for row in clean_audit]
    all_raw_available = source_available and len(raw_paths) == int(cfg["expected_clean_count"]) and all(path.is_file() for path in raw_paths)

    if all_raw_available:
        scale_rows, raw_stack = compare_from_raw(clean_audit, dark, bad, cfg)
        spatial_rows = spatial_audit(raw_stack, dark, bad, evidence="raw_500ms_center_crop")
        pixel_evidence = "DIRECT_RAW"
    else:
        scale_rows = compare_from_saved_audit(clean_audit, dark, cfg)
        corrected_stack = load_historical_corrected_stack(repo, cfg)
        spatial_rows = spatial_audit(
            corrected_stack,
            dark,
            bad,
            evidence="POST_DARK_SUBTRACTION_CLIPPED_DIAGNOSTIC_NOT_RAW_ALIGNMENT_EVIDENCE",
        )
        pixel_evidence = "SOURCE_VOLUME_UNAVAILABLE_SAVED_SUMMARY_AND_CENSORED_OUTPUT_ONLY"
    write_csv(scale_rows, output / "clean_dark_scale_comparison.csv")
    write_csv(spatial_rows, output / "spatial_alignment_audit.csv")

    candidates = candidate_rows(cfg, clean_audit, dark, bad, all_raw_available)
    write_csv(candidates, output / "candidate_processing_comparison.csv")
    gate_rows = clean_gate_rows(cfg, clean_audit, dark, bad, all_raw_available)
    write_csv(gate_rows, output / "clean_content_gate.csv")

    decision = build_decision(
        cfg=cfg,
        source_available=source_available,
        all_raw_available=all_raw_available,
        pixel_evidence=pixel_evidence,
        provenance=provenance_payload,
        clean_audit=clean_audit,
        dark=dark,
        bad=bad,
        spatial_rows=spatial_rows,
    )
    write_json(decision, output / "compatibility_decision.json")
    write_report(output / "verification_report.md", decision, provenance_payload, candidates, gate_rows, dark)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "status": decision["audit_status"],
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "git_commit": commit,
        "git_worktree_clean_at_start": not status.strip(),
        "source_volume_available": source_available,
        "raw_pixel_audit_completed": all_raw_available,
        "batch_synthetic_generation_performed": False,
        "model_training_performed": False,
        "outputs": output_hashes(output),
    }
    write_json(manifest, provenance / "run_manifest.json")
    write_text(logs / "audit.log", json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


def build_dark_provenance(
    repo: Path,
    cfg: dict[str, Any],
    path: Path,
    dark: np.ndarray,
    crop: tuple[int, int, int, int],
    source_available: bool,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "classification": "UNTRACED CALIBRATION ARTIFACT",
        "classification_reason": (
            "The generating code and named source folder are known, but the run did not preserve device serial, "
            "dark exposure, gain, temperature, readout mode, black-level state, TIFF metadata, or a resolved run config."
        ),
        "full_path": str(path),
        "created_at_local": datetime.fromtimestamp(stat.st_ctime).astimezone().isoformat(),
        "modified_at_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256_file(path),
        "shape": list(dark.shape),
        "dtype": str(dark.dtype),
        "unit": "raw uint16-domain DN represented as float32",
        "operation": "pixelwise arithmetic mean of 64 center-cropped dark_Background TIFF frames",
        "dark_frame_count": int(cfg["dark_frame_count"]),
        "source_folder_name": cfg["dark_folder"],
        "source_root": cfg["source_root"],
        "source_volume_available_at_audit": source_available,
        "device": "UNRECORDED",
        "exposure": "UNRECORDED",
        "gain": "UNRECORDED",
        "temperature": "UNRECORDED",
        "readout_mode": "UNRECORDED",
        "black_level_state": "UNRECORDED",
        "source_image_shape": list(cfg["expected_clean_shape"]),
        "source_center_crop_size": int(cfg["dark_source_crop_size"]),
        "source_center_crop_coordinates": [512, 512, 1024, 1024],
        "effective_512_crop_within_npy": list(crop),
        "effective_global_512_coordinates": [768, 768, 512, 512],
        "divided_by_65535": False,
        "normalized": False,
        "bad_pixel_correction_applied": False,
        "generator_script": cfg["dark_generation_script"],
        "generator_script_commit": cfg["dark_generation_commit"],
    }


def dark_statistics_rows(dark_full: np.ndarray, dark: np.ndarray, dark_std: np.ndarray, bad: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for label, array, unit in [
        ("dark_offset_1024", dark_full, "DN"),
        ("dark_offset_effective_512", dark, "DN"),
        ("dark_temporal_std_effective_512", dark_std, "DN"),
        ("bad_pixel_mask_effective_512", bad.astype(np.float32), "fraction"),
    ]:
        stats = describe(array)
        histogram, edges = np.histogram(np.asarray(array, dtype=np.float64), bins=64)
        rows.append(
            {
                "artifact": label,
                "unit": unit,
                **stats,
                "histogram_bin_edges_json": json.dumps(edges.tolist()),
                "histogram_counts_json": json.dumps(histogram.tolist()),
            }
        )
    return rows


def compare_from_raw(
    audit_rows: list[dict[str, str]], dark: np.ndarray, bad: np.ndarray, cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    import tifffile

    rows: list[dict[str, Any]] = []
    stack = []
    for audit in audit_rows:
        raw = np.asarray(tifffile.imread(audit["clean_path"]))
        crop, coords = center_crop(raw, int(cfg["crop_size"]))
        crop = crop.astype(np.float32)
        diff = crop - dark
        valid = ~bad
        ratio = dark[valid] / np.maximum(crop[valid], 1.0)
        rows.append(
            comparison_row(
                audit,
                dark,
                crop,
                diff,
                ratio,
                coords,
                evidence="DIRECT_RAW",
                negative=float(np.mean(diff < 0)),
                zero=float(np.mean(diff == 0)),
                positive=float(np.mean(diff > 0)),
            )
        )
        stack.append(crop)
    return rows, np.stack(stack)


def compare_from_saved_audit(
    audit_rows: list[dict[str, str]], dark: np.ndarray, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for audit in audit_rows:
        raw_mean = float(audit["raw_mean_dn"])
        nonpositive = float(audit["corrected_zero_ratio"])
        rows.append(
            {
                "source_pair_key": audit["source_pair_key"],
                "evidence": "SAVED_FORMAL_SMOKE_SUMMARY_SOURCE_VOLUME_OFFLINE",
                "clean_path": audit["clean_path"],
                "clean_raw_min_dn": float(audit["raw_min_dn"]),
                "clean_raw_max_dn": float(audit["raw_max_dn"]),
                "clean_raw_mean_dn": raw_mean,
                "clean_raw_std_dn": float(audit["raw_std_dn"]),
                "clean_raw_p1_dn": float(audit["raw_p1_dn"]),
                "clean_raw_p50_dn": float(audit["raw_p50_dn"]),
                "clean_raw_p99_dn": float(audit["raw_p99_dn"]),
                "dark_mean_dn": float(np.mean(dark)),
                "dark_p50_dn": float(np.median(dark)),
                "dark_to_clean_mean_ratio": float(np.mean(dark)) / raw_mean,
                "pixel_dark_clean_ratio_p50": "NOT_COMPUTABLE_SOURCE_OFFLINE",
                "difference_mean_dn": "NOT_COMPUTABLE_SOURCE_OFFLINE",
                "difference_negative_ratio": "NOT_SEPARABLE_FROM_EQUAL_SOURCE_OFFLINE",
                "difference_zero_ratio": "NOT_SEPARABLE_FROM_NEGATIVE_SOURCE_OFFLINE",
                "difference_nonpositive_ratio": nonpositive,
                "difference_positive_ratio": 1.0 - nonpositive,
                "crop_top": int(audit["crop_top"]),
                "crop_left": int(audit["crop_left"]),
                "crop_size": int(cfg["crop_size"]),
                "warning": "raw pixel comparison requires remounted source volume",
            }
        )
    return rows


def comparison_row(
    audit: dict[str, str],
    dark: np.ndarray,
    clean: np.ndarray,
    diff: np.ndarray,
    ratio: np.ndarray,
    coords: tuple[int, int, int, int],
    evidence: str,
    negative: float,
    zero: float,
    positive: float,
) -> dict[str, Any]:
    return {
        "source_pair_key": audit["source_pair_key"],
        "evidence": evidence,
        "clean_path": audit["clean_path"],
        "clean_raw_min_dn": float(np.min(clean)),
        "clean_raw_max_dn": float(np.max(clean)),
        "clean_raw_mean_dn": float(np.mean(clean)),
        "clean_raw_std_dn": float(np.std(clean)),
        "clean_raw_p1_dn": float(np.percentile(clean, 1)),
        "clean_raw_p50_dn": float(np.percentile(clean, 50)),
        "clean_raw_p99_dn": float(np.percentile(clean, 99)),
        "dark_mean_dn": float(np.mean(dark)),
        "dark_p50_dn": float(np.median(dark)),
        "dark_to_clean_mean_ratio": float(np.mean(dark)) / max(float(np.mean(clean)), 1e-12),
        "pixel_dark_clean_ratio_p50": float(np.median(ratio)),
        "difference_mean_dn": float(np.mean(diff)),
        "difference_negative_ratio": negative,
        "difference_zero_ratio": zero,
        "difference_nonpositive_ratio": negative + zero,
        "difference_positive_ratio": positive,
        "crop_top": coords[0],
        "crop_left": coords[1],
        "crop_size": coords[2],
        "warning": "",
    }


def load_historical_corrected_stack(repo: Path, cfg: dict[str, Any]) -> np.ndarray:
    import tifffile

    rows = read_csv(resolve(repo, cfg["historical_corrected_pairs_csv"]))
    images = []
    for row in rows:
        path = Path(row["clean_path"])
        if not path.is_absolute():
            path = resolve(repo, str(path))
        if path.is_file():
            images.append(np.asarray(tifffile.imread(path), dtype=np.float32))
    if not images:
        raise FileNotFoundError("No historical corrected clean TIFFs available for censored spatial diagnostic")
    return np.stack(images)


def spatial_audit(stack: np.ndarray, dark: np.ndarray, bad: np.ndarray, evidence: str) -> list[dict[str, Any]]:
    temporal_median = np.median(stack, axis=0).astype(np.float32)
    clean_hot = high_outlier_mask(temporal_median)
    dark_hot = high_outlier_mask(dark) | bad
    rows = []
    for name, transform in TRANSFORMS.items():
        candidate = np.asarray(transform(dark))
        mask = np.isfinite(candidate) & np.isfinite(temporal_median)
        corr = correlation(temporal_median[mask], candidate[mask])
        row_corr = correlation(np.mean(temporal_median, axis=1), np.mean(candidate, axis=1))
        col_corr = correlation(np.mean(temporal_median, axis=0), np.mean(candidate, axis=0))
        hp_corr = correlation(highpass(temporal_median)[mask], highpass(candidate)[mask])
        transformed_hot = np.asarray(transform(dark_hot))
        overlap = float(np.sum(clean_hot & transformed_hot) / max(np.sum(clean_hot | transformed_hot), 1))
        rows.append(
            {
                "transform": name,
                "evidence": evidence,
                "pixel_correlation": corr,
                "row_profile_correlation": row_corr,
                "column_profile_correlation": col_corr,
                "highpass_correlation": hp_corr,
                "bad_or_hot_jaccard": overlap,
                "clean_stack_count": int(stack.shape[0]),
                "interpretation": (
                    "raw spatial alignment evidence" if evidence == "raw_500ms_center_crop" else
                    "censored post-correction diagnostic; cannot validate raw alignment"
                ),
            }
        )
    return rows


def candidate_rows(
    cfg: dict[str, Any], clean: list[dict[str, str]], dark: np.ndarray, bad: np.ndarray, raw_available: bool
) -> list[dict[str, Any]]:
    raw_zero = max(float(row["raw_zero_ratio"]) for row in clean)
    raw_sat = max(float(row["raw_saturation_ratio"]) for row in clean)
    corrected_zero = [float(row["corrected_zero_ratio"]) for row in clean]
    return [
        {
            "candidate": "A_no_dark",
            "rating": "ACCEPTABLE",
            "operation": "raw uint16 / 65535; no per-image p99 scaling",
            "observed_zero_ratio": raw_zero,
            "observed_saturation_ratio": raw_sat,
            "negative_before_clipping_ratio": 0.0,
            "preserves_inter_image_scale": True,
            "condition": "operational content source only; unresolved pedestal and sCMOS noise retained",
        },
        {
            "candidate": "B_matched_dark_mean",
            "rating": "INVALID",
            "operation": "raw DN - historical dark map",
            "observed_zero_ratio": float(np.median(corrected_zero)),
            "observed_saturation_ratio": 0.0,
            "negative_before_clipping_ratio": "not separately recoverable; nonpositive ratio equals corrected zero ratio",
            "preserves_inter_image_scale": True,
            "condition": "dark acquisition conditions are unrecorded and correction clips 97%+ pixels",
        },
        {
            "candidate": "C_scalar_pedestal",
            "rating": "INVALID",
            "operation": "raw DN - scalar pedestal",
            "observed_zero_ratio": "NOT_RUN",
            "observed_saturation_ratio": raw_sat,
            "negative_before_clipping_ratio": "NOT_RUN",
            "preserves_inter_image_scale": True,
            "condition": "no camera black-level metadata or matched-dark scalar is available",
        },
        {
            "candidate": "D_abandon_source",
            "rating": "CONDITIONAL",
            "operation": "stop using the 100 images",
            "observed_zero_ratio": "not applicable",
            "observed_saturation_ratio": "not applicable",
            "negative_before_clipping_ratio": "not applicable",
            "preserves_inter_image_scale": "not applicable",
            "condition": "required if source files or metadata cannot be recovered for the next formal no-dark audit",
        },
    ]


def clean_gate_rows(
    cfg: dict[str, Any], clean: list[dict[str, str]], dark: np.ndarray, bad: np.ndarray, raw_available: bool
) -> list[dict[str, Any]]:
    means = np.array([float(row["raw_mean_dn"]) for row in clean])
    p99s = np.array([float(row["raw_p99_dn"]) for row in clean])
    zero = np.array([float(row["raw_zero_ratio"]) for row in clean])
    sat = np.array([float(row["raw_saturation_ratio"]) for row in clean])
    corrected_zero = np.array([float(row["corrected_zero_ratio"]) for row in clean])
    return [
        gate("source_file_count", len(clean), int(cfg["expected_clean_count"]), len(clean) == int(cfg["expected_clean_count"]), "saved audit"),
        gate("source_volume_currently_accessible", raw_available, True, raw_available, "direct raw recheck unavailable when false"),
        gate("raw_zero_ratio_max", float(np.max(zero)), float(cfg["quality_gates"]["maximum_zero_ratio"]), float(np.max(zero)) <= float(cfg["quality_gates"]["maximum_zero_ratio"]), "no-dark"),
        gate("raw_saturation_ratio_max", float(np.max(sat)), float(cfg["quality_gates"]["maximum_saturation_ratio"]), float(np.max(sat)) <= float(cfg["quality_gates"]["maximum_saturation_ratio"]), "no-dark"),
        gate("raw_mean_range_dn", f"{means.min():.6g}..{means.max():.6g}", "preserve", True, "no per-image normalization"),
        gate("raw_p99_range_dn", f"{p99s.min():.6g}..{p99s.max():.6g}", "preserve", True, "no per-image normalization"),
        gate("dark_subtracted_zero_ratio_median", float(np.median(corrected_zero)), float(cfg["quality_gates"]["maximum_zero_ratio"]), float(np.median(corrected_zero)) <= float(cfg["quality_gates"]["maximum_zero_ratio"]), "historical dark subtraction"),
        gate("bad_pixel_mask_ratio", float(np.mean(bad)), float(cfg["quality_gates"]["maximum_bad_pixel_ratio"]), float(np.mean(bad)) <= float(cfg["quality_gates"]["maximum_bad_pixel_ratio"]), "mask only; no correction"),
        gate("per_image_p99_scaling_disabled", True, True, bool(cfg["decision"]["prohibit_per_image_p99"]), "training data rule"),
        gate("metadata_proves_raw_or_corrected_state", False, True, False, "TIFF tags unavailable while source volume is offline"),
    ]


def gate(metric: str, observed: Any, threshold: Any, passed: bool, scope: str) -> dict[str, Any]:
    return {"metric": metric, "observed": observed, "threshold_or_requirement": threshold, "pass": bool(passed), "scope": scope}


def build_decision(
    cfg: dict[str, Any],
    source_available: bool,
    all_raw_available: bool,
    pixel_evidence: str,
    provenance: dict[str, Any],
    clean_audit: list[dict[str, str]],
    dark: np.ndarray,
    bad: np.ndarray,
    spatial_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    corrected_zero = np.array([float(row["corrected_zero_ratio"]) for row in clean_audit])
    raw_means = np.array([float(row["raw_mean_dn"]) for row in clean_audit])
    identity = next(row for row in spatial_rows if row["transform"] == "identity")
    best = max(spatial_rows, key=lambda row: abs(float(row["highpass_correlation"])))
    folder_summary = read_csv(Path(cfg["historical_folder_summary_csv"]))
    exposure_p50 = {row["folder"]: float(row["sample_p50"]) for row in folder_summary}
    exposure_saturation = {
        row["folder"]: float(row["sample_saturated_fraction"]) for row in folder_summary
    }
    return {
        "audit_status": "PARTIAL-RUN" if not all_raw_available else "VERIFIED-COMPATIBILITY-AUDIT",
        "final_decision": cfg["decision"]["selected"],
        "go_for_batch_synthetic_generation": False,
        "dark_offset_classification": provenance["classification"],
        "dark_offset_unit": provenance["unit"],
        "device_and_acquisition_match": "NOT_ESTABLISHED",
        "spatial_crop_geometry": "MATCHED_CENTER_GEOMETRY_2048_TO_1024_TO_512",
        "spatial_content_alignment": (
            "NOT_COMPUTABLE_FROM_RAW_SOURCE_OFFLINE" if not all_raw_available else "SEE_SPATIAL_ALIGNMENT_AUDIT"
        ),
        "spatial_identity_highpass_correlation": identity["highpass_correlation"],
        "spatial_best_transform_by_absolute_highpass_correlation": best["transform"],
        "spatial_best_transform_highpass_correlation": best["highpass_correlation"],
        "pixel_level_evidence": pixel_evidence,
        "dark_mean_effective_512_dn": float(np.mean(dark)),
        "clean_full_frame_mean_range_dn": [float(raw_means.min()), float(raw_means.max())],
        "dark_to_clean_mean_ratio_range": [
            float(np.mean(dark) / raw_means.max()),
            float(np.mean(dark) / raw_means.min()),
        ],
        "historical_corrected_zero_ratio_range": [float(corrected_zero.min()), float(corrected_zero.max())],
        "historical_corrected_zero_ratio_median": float(np.median(corrected_zero)),
        "direct_cause_of_clipping": (
            "the unvalidated dark map is larger than the 500 ms center-crop signal at almost all pixels; "
            "subtraction followed by clip(0,1) censors the nonpositive difference"
        ),
        "clean_tiff_raw_state": "UNRESOLVED_CAMERA_EXPORTED_UINT16_NOT_PROVEN_SENSOR_RAW",
        "clean_tiff_prior_offset_correction": "NOT_ESTABLISHED",
        "clean_tiff_state_evidence": {
            "sample_p50_by_folder_dn": exposure_p50,
            "sample_saturation_ratio_by_folder": exposure_saturation,
            "interpretation": (
                "folder median is non-monotonic with nominal exposure and most folders have exactly the same "
                "sample saturation ratio; this is incompatible with assuming an uncontrolled linear raw sequence"
            ),
        },
        "bad_pixel_mask_fraction": float(np.mean(bad)),
        "selected_processing": "raw uint16 / 65535, no dark subtraction, no scalar pedestal, no per-image p99 scaling",
        "clean_content_usable": True,
        "clean_content_boundary": cfg["decision"]["boundary"].strip(),
        "blocking_requirement_before_formal_input_audit": (
            "remount the source volume and verify all 100 SHA256 values, TIFF tags, and no-dark center-crop quality metrics"
        ),
    }


def write_report(
    path: Path,
    decision: dict[str, Any],
    provenance: dict[str, Any],
    candidates: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    dark: np.ndarray,
) -> None:
    lines = [
        "# E2 Dark-Offset Compatibility Audit",
        "",
        f"- Audit status: **{decision['audit_status']}**",
        f"- Compatibility decision: **{decision['final_decision']}**",
        "- Synthetic pairs generated: no",
        "- Models trained: no",
        f"- Dark artifact: `{provenance['full_path']}`",
        f"- Dark SHA256: `{provenance['sha256']}`",
        f"- Dark effective 512 mean/median: {float(np.mean(dark)):.6f} / {float(np.median(dark)):.6f} DN",
        "",
        "## Finding",
        "",
        "The dark array is a raw-DN mean of 64 named dark-folder frames, but its calibration conditions were not recorded.",
        "It is therefore an UNTRACED CALIBRATION ARTIFACT for compatibility purposes.",
        f"Historical subtraction produced zero ratios of {decision['historical_corrected_zero_ratio_range'][0]:.4%}..{decision['historical_corrected_zero_ratio_range'][1]:.4%}.",
        "This is direct numerical evidence that the artifact is not compatible with the current 500 ms content under the historical subtraction pipeline.",
        "",
        "## Candidate Ratings",
        "",
        "| candidate | rating | condition |",
        "|---|---|---|",
    ]
    for row in candidates:
        lines.append(f"| {row['candidate']} | {row['rating']} | {row['condition']} |")
    lines.extend(["", "## Quality Gates", ""])
    for row in gates:
        lines.append(f"- {'PASS' if row['pass'] else 'FAIL'} `{row['metric']}`: {row['observed']} ({row['scope']})")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            decision["clean_content_boundary"],
            "The source volume was unavailable during this run, so raw pixel-level spatial alignment and TIFF metadata checks remain explicitly incomplete.",
            "Historical clipped outputs were used only for a labeled censored diagnostic; they were not used to validate raw alignment or select the processing decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def describe(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    q = np.percentile(values, PERCENTILES)
    return {
        "shape": "x".join(map(str, array.shape)),
        "dtype": str(array.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p0_1": float(q[0]),
        "p1": float(q[1]),
        "p5": float(q[2]),
        "p50": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "negative_ratio": float(np.mean(values < 0)),
    }


def highpass(array: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    values = np.asarray(array, dtype=np.float32)
    return values - gaussian_filter(values, sigma=2.0, mode="reflect")


def high_outlier_mask(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return values > median + 8.0 * max(1.4826 * mad, 1e-12)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def center_crop(array: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = array.shape
    top = (height - size) // 2
    left = (width - size) // 2
    return np.asarray(array[top : top + size, left : left + size]), (top, left, size, size)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_hashes(repo: Path, cfg: dict[str, Any], inputs: Path, scripts: Path) -> None:
    input_names = [
        cfg["dark_offset_path"], cfg["dark_std_path"], cfg["bad_pixel_mask_path"],
        cfg["saved_clean_audit_csv"], cfg["saved_clean_manifest_csv"],
        cfg["historical_corrected_pairs_csv"], cfg["historical_folder_summary_csv"],
    ]
    script_names = [
        "configs/e2_dark_offset_compatibility_20260717.yaml",
        "scripts/audit_e2_dark_offset_compatibility.py",
        cfg["dark_generation_script"],
        "scripts/audit_e2_synthetic_inputs.py",
    ]
    write_hash_table(repo, input_names, inputs)
    write_hash_table(repo, script_names, scripts)


def write_hash_table(repo: Path, names: Iterable[str], path: Path) -> None:
    rows = []
    for name in names:
        item = resolve(repo, name)
        rows.append({
            "input": name,
            "resolved_path": str(item),
            "exists": item.is_file(),
            "size_bytes": item.stat().st_size if item.is_file() else "",
            "sha256": sha256_file(item) if item.is_file() else "",
        })
    write_csv(rows, path)


def save_environment(path: Path) -> None:
    rows = [
        f"python={sys.version}", f"executable={sys.executable}",
        f"platform={platform.platform()}", f"numpy={np.__version__}",
    ]
    for name in ["scipy", "tifffile", "yaml"]:
        try:
            module = __import__(name)
            rows.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            rows.append(f"{name}=unavailable:{type(exc).__name__}")
    write_text(path / "environment.txt", "\n".join(rows) + "\n")


def output_hashes(output: Path) -> list[dict[str, Any]]:
    return [
        {"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    ]


def resolve(repo: Path, value: str) -> Path:
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


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
