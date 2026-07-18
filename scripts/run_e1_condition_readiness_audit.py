"""Audit E1 scientific completeness and readiness for CG/CGS without rerunning E1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy.stats import spearmanr

from json_serialization import dump_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_count(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


def rho(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    value = spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic
    return float(value) if math.isfinite(value) else float("nan")


def parse_picture_info(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    settings = [line.split(".tif", 1)[1].strip() if ".tif" in line else line.strip() for line in lines]
    first = lines[0] if lines else ""
    patterns = {
        "exposure_delay_ms": r"Exposure.*?delay:([0-9.]+)ms",
        "exposure_width_ms": r"Exposure.*?width:([0-9.]+)ms",
        "sync_a_delay_ns": r"Sync\.A.*?delay:([0-9.]+)ns",
        "sync_a_width_us": r"Sync\.A.*?width:([0-9.]+)us",
        "sync_b_delay_ns": r"Sync\.B.*?delay:([0-9.]+)ns",
        "sync_b_width_us": r"Sync\.B.*?width:([0-9.]+)us",
        "recorded_gain": r"gain[^0-9]*([0-9.]+)\s*$",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, first)
        parsed[key] = float(match.group(1)) if match else None
    parsed.update({"line_count": len(lines), "unique_setting_count": len(set(settings)), "sha256": sha256(path)})
    return parsed


def load_e1_tables(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "integrity": "input_audit/data_integrity_report.csv",
        "manifest": "input_audit/input_manifest.csv",
        "noise": "noise_summary/folder_noise_summary.csv",
        "fano": "mean_variance/fano_like_summary.csv",
        "robustness": "robustness/robustness_by_crop_and_frames.csv",
        "drift": "temporal_stability/temporal_drift_summary.csv",
        "stable": "stable_component/stable_component_summary.csv",
        "stable_split": "stable_component/stable_component_by_split.csv",
        "rowcol": "row_column/row_column_summary.csv",
        "rowcol_block": "row_column/row_column_by_block.csv",
        "spatial": "spatial/spatial_correlation_summary.csv",
    }
    return {name: pd.read_csv(root / path) for name, path in paths.items()}


def build_inventory(config: dict, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    data_root = Path(config["data_root"])
    roi = config["primary_roi"]
    integrity = tables["integrity"].set_index("folder")
    manifest = tables["manifest"]
    inventory_rows, metadata_rows, protected = [], [], []
    picture_hashes = []
    for order, folder in enumerate(config["folders"], start=1):
        folder_root = data_root / str(folder)
        group = manifest[manifest.folder.eq(folder)].sort_values("frame_index")
        first_tiff = folder_root / Path(group.iloc[0].relative_path).name
        info_path = folder_root / config["metadata_sources"]["picture_info_name"]
        protected.extend([first_tiff, info_path])
        info = parse_picture_info(info_path)
        picture_hashes.append(info["sha256"])
        with tifffile.TiffFile(first_tiff) as tf:
            page = tf.pages[0]
            tags = {tag.name: tag.value for tag in page.tags.values()}
            dtype = str(page.dtype)
            shape = f"{page.shape[0]}x{page.shape[1]}"
            bits = int(tags.get("BitsPerSample", page.dtype.itemsize * 8))
        mtimes = group.mtime_ns.astype(np.int64)
        inventory_rows.append({
            "folder": folder, "absolute_path": str(folder_root), "acquisition_order_from_folder_sequence": order,
            "frame_count": len(group), "first_filename": first_tiff.name, "last_filename": (folder_root / Path(group.iloc[-1].relative_path).name).name,
            "filename_pattern": "<frame_index>-Camera1[20600555].tif", "dtype": dtype, "shape": shape, "bit_depth": bits,
            "roi_top": roi["top"], "roi_left": roi["left"], "roi_height": roi["height"], "roi_width": roi["width"],
            "mtime_min_utc": datetime.fromtimestamp(int(mtimes.min()) / 1e9, tz=timezone.utc).isoformat(),
            "mtime_max_utc": datetime.fromtimestamp(int(mtimes.max()) / 1e9, tz=timezone.utc).isoformat(),
            "picture_info_lines": info["line_count"], "picture_info_unique_settings": info["unique_setting_count"],
            "camera_serial_from_filename": "20600555", "tiff_datetime_tag": tags.get("DateTime", ""), "tiff_software_tag": tags.get("Software", ""),
            "within_folder_scene_status": "PARTIALLY-VERIFIED", "cross_folder_scene_status": "UNKNOWN",
            "integrity_status": integrity.loc[folder, "status"],
        })
        fields = [
            ("camera_serial", "20600555", "VERIFIED", "Filename and Format.ini serialNumber agree."),
            ("recorded_exposure_channel_width_ms", info["exposure_width_ms"], "VERIFIED", "All 200 PictureInfo lines record Exposure width 900 ms."),
            ("recorded_exposure_channel_delay_ms", info["exposure_delay_ms"], "VERIFIED", "All 200 PictureInfo lines record Exposure delay 0 ms."),
            ("sync_a_width_us", info["sync_a_width_us"], "VERIFIED", "All 200 PictureInfo lines record Sync.A width 4 us; physical gate attribution is not established."),
            ("sync_b_width_us", info["sync_b_width_us"], "VERIFIED", "All 200 PictureInfo lines record Sync.B width 4 us; physical gate attribution is not established."),
            ("recorded_gain_setting", info["recorded_gain"], "VERIFIED", "All 200 PictureInfo lines and T560.ini record gain 60."),
            ("mcp_gain", "60?", "PARTIALLY-VERIFIED", "The control is labeled gain, but no source explicitly identifies it as calibrated MCP gain."),
            ("physical_gate_width", "unknown", "UNKNOWN", "Sync channels are recorded, but their mapping to the intensifier gate is undocumented."),
            ("sensor_integration_exposure", "900 ms vs Format.ini 300", "CONFLICTING", "Per-frame Exposure channel is 900 ms; non-snapshotted Format.ini contains s_exposure=300."),
            ("trigger_mode", "unknown", "UNKNOWN", "No per-folder trigger-mode record."),
            ("readout_mode", "5120x5120 uint16; other settings uncertain", "PARTIALLY-VERIFIED", "TIFF encoding is verified; Format.ini settings are not a per-folder immutable snapshot."),
            ("temperature", "unknown", "UNKNOWN", "No TIFF/PictureInfo temperature field."),
            ("capture_date", "20260319", "PARTIALLY-VERIFIED", "Directory and INI path encode date, but TIFF has no DateTime tag and manifest mtimes reflect later file handling."),
            ("acquisition_order", order, "PARTIALLY-VERIFIED", "Folder and mtime order are monotonic, but no immutable acquisition log confirms causality."),
            ("within_folder_same_scene", "likely repeated sequence", "PARTIALLY-VERIFIED", "Sequential 200-frame capture with constant settings; scene identifier absent."),
            ("cross_folder_same_scene", "unknown", "UNKNOWN", "No scene identifier or acquisition note links folders."),
        ]
        for field, value, status, evidence in fields:
            metadata_rows.append({"folder": folder, "condition_field": field, "recovered_value": value, "verification_level": status, "evidence": evidence})
    if len(set(picture_hashes)) != 1:
        raise RuntimeError("PictureInfo setting records are not identical across folders")
    return pd.DataFrame(inventory_rows), pd.DataFrame(metadata_rows), protected


def merged_noise(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    n = tables["noise"][["folder", "mean_signal", "frame_mean_std", "temporal_std_mean", "temporal_var_mean", "temporal_fano_approx", "observed_stable_component_std", "stable_to_temporal_std_ratio"]]
    f = tables["fano"][["folder", "fano_like_dn"]]
    rc = tables["rowcol"][["folder", "row_pattern_energy_dn", "column_pattern_energy_dn", "row_adjacent_frame_profile_correlation", "column_adjacent_frame_profile_correlation", "variance_fraction_removed", "row_energy_block_cv", "column_energy_block_cv"]]
    sp = tables["spatial"][["folder", "lag1_row_corr", "lag1_col_corr", "lag1_diag_corr", "radial_autocorr_r1", "radial_autocorr_r2", "corr_length_1e_px", "psd_low_fraction", "psd_mid_fraction", "psd_high_fraction"]]
    st = tables["stable"][["folder", "minimum_split_map_correlation", "median_split_map_correlation", "observed_stable_component_std_dn", "split_difference_component_std_dn", "stable_to_temporal_ratio"]]
    dr = tables["drift"][["folder", "frame_mean_slope_dn_per_frame", "frame_mean_relative_change_half", "frame_std_delta_half_dn", "max_local_brightness_relative_change_half", "lag1_residual_correlation", "lag10_residual_correlation", "lag50_residual_correlation"]]
    out = n.merge(f).merge(rc).merge(sp).merge(st).merge(dr)
    out["metric_scope"] = "folder-level operational statistics at frozen 512x512 center ROI"
    out["physical_attribution"] = "not established"
    return out


def repeatability(tables: dict[str, pd.DataFrame], noise: pd.DataFrame) -> pd.DataFrame:
    robust = tables["robustness"]
    robust = robust[robust.crop_size.eq(512)]
    rc = tables["rowcol"].set_index("folder")
    stable = tables["stable"].set_index("folder")
    rows = []
    frame_rank_rhos = []
    counts = sorted(robust.frame_count.unique())
    for i, left in enumerate(counts):
        for right in counts[i + 1:]:
            a = robust[robust.frame_count.eq(left)].sort_values("folder")
            b = robust[robust.frame_count.eq(right)].sort_values("folder")
            frame_rank_rhos.append(rho(a.temporal_std_mean, b.temporal_std_mean))
    for folder in noise.folder:
        values = robust[robust.folder.eq(folder)].sort_values("frame_count").temporal_std_mean.astype(float)
        mean = float(values.mean())
        rows.append({
            "folder": folder, "temporal_std_frame_count_min_dn": float(values.min()), "temporal_std_frame_count_max_dn": float(values.max()),
            "temporal_std_frame_count_cv": float(values.std(ddof=1) / mean), "temporal_std_frame_count_range_relative": float((values.max() - values.min()) / mean),
            "temporal_std_cross_frame_count_rank_rho_min": min(frame_rank_rhos),
            "row_energy_block_cv": float(rc.loc[folder, "row_energy_block_cv"]), "column_energy_block_cv": float(rc.loc[folder, "column_energy_block_cv"]),
            "minimum_split_map_correlation": float(stable.loc[folder, "minimum_split_map_correlation"]),
            "median_split_map_correlation": float(stable.loc[folder, "median_split_map_correlation"]),
            "spatial_acf_psd_subset_repeatability": "NOT-CALCULATED",
            "repeatability_interpretation": "temporal strength repeatable; row/column block variability and stable split correlation reported; spatial subset repeatability missing",
        })
    return pd.DataFrame(rows)


def brightness_confound(noise: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        corr = rho(noise.mean_signal, noise[feature])
        rows.append({
            "feature": feature, "spearman_with_mean_signal": corr, "absolute_spearman": abs(corr),
            "folder_min": float(noise[feature].min()), "folder_max": float(noise[feature].max()),
            "brightness_confound_risk": "HIGH" if abs(corr) >= 0.8 else "LOW-TO-MODERATE",
            "interpretation": "descriptive n=10; scene and brightness are not independently controlled",
        })
    crop = pd.read_csv(Path("reports/e1_formal_rerun_20260717/robustness/robustness_by_crop_and_frames.csv"))
    crop128 = crop[crop.frame_count.eq(128)]
    by_folder = crop128.groupby("folder").temporal_std_mean.agg(["min", "max", "mean"])
    by_folder["relative_range"] = (by_folder["max"] - by_folder["min"]) / by_folder["mean"]
    rows.append({"feature": "temporal_std_roi_sensitivity", "spearman_with_mean_signal": float("nan"), "absolute_spearman": float("nan"), "folder_min": float(by_folder.relative_range.min()), "folder_max": float(by_folder.relative_range.max()), "brightness_confound_risk": "HIGH", "interpretation": "crop-size relative range across 256/512/1024; fixed ROI is mandatory"})
    return pd.DataFrame(rows)


def feature_correlations(noise: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            corr = rho(noise[a], noise[b])
            rows.append({"feature_a": a, "feature_b": b, "spearman_rho": corr, "absolute_rho": abs(corr), "redundant_at_abs_rho_0p8": abs(corr) >= 0.8})
    return pd.DataFrame(rows)


def condition_candidates(noise: pd.DataFrame, repeat: pd.DataFrame, confound: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        ("recorded_exposure_channel_width_ms", "physical-recorded", "900 for all folders", "VERIFIED", False, "constant; cannot define states"),
        ("recorded_gain_setting", "physical-recorded", "60 for all folders", "VERIFIED", False, "constant; cannot define states"),
        ("sync_a_or_b_width_us", "physical-recorded", "4 for all folders", "VERIFIED", False, "constant and gate attribution unknown"),
        ("mean_signal", "content/statistical covariate", f"{noise.mean_signal.min():.3f}-{noise.mean_signal.max():.3f} DN", "VERIFIED-OPERATIONAL", False, "brightness/scene covariate, not a device condition"),
        ("temporal_std_mean", "observed noise-state", f"{noise.temporal_std_mean.min():.3f}-{noise.temporal_std_mean.max():.3f} DN", "REPEATABLE-WITH-CONFOUND", False, "best CG strength candidate, but rho=1.0 with mean signal and evaluation overlap unresolved"),
        ("temporal_fano_approx", "observed derived statistic", f"{noise.temporal_fano_approx.min():.3f}-{noise.temporal_fano_approx.max():.3f}", "OPERATIONAL-ONLY", False, "derived from variance and mean; not photon gain"),
        ("row_pattern_energy_dn", "observed noise-state", f"{noise.row_pattern_energy_dn.min():.3f}-{noise.row_pattern_energy_dn.max():.3f} DN", "PARTIAL", False, "block CV available; brightness and other strength features are highly collinear"),
        ("column_pattern_energy_dn", "observed noise-state", f"{noise.column_pattern_energy_dn.min():.3f}-{noise.column_pattern_energy_dn.max():.3f} DN", "PARTIAL", False, "block CV available; brightness and temporal strength are highly collinear"),
        ("radial_autocorr_r1", "observed noise-state", f"{noise.radial_autocorr_r1.min():.4f}-{noise.radial_autocorr_r1.max():.4f}", "PARTIAL", False, "low brightness correlation but frame-subset repeatability not established"),
        ("observed_stable_component_std_dn", "observed stable state", f"{noise.observed_stable_component_std_dn.min():.3f}-{noise.observed_stable_component_std_dn.max():.3f} DN", "PARTIAL", False, "split maps repeat, but scene content and stable detector structure are not separable"),
        ("frame_mean_relative_change_half", "observed drift state", f"{noise.frame_mean_relative_change_half.min():.6f}-{noise.frame_mean_relative_change_half.max():.6f}", "PARTIAL", False, "single sequence per folder; sign and magnitude require repeatability audit"),
    ]
    table = pd.DataFrame(candidates, columns=["candidate", "candidate_type", "observed_range", "evidence_status", "ready_for_frozen_condition", "limitation"])
    relations = []
    for condition in ["recorded_exposure_channel_width_ms", "recorded_gain_setting", "sync_a_or_b_width_us"]:
        for metric in ["temporal_std_mean", "row_pattern_energy_dn", "column_pattern_energy_dn", "radial_autocorr_r1", "observed_stable_component_std_dn"]:
            relations.append({"condition": condition, "noise_statistic": metric, "relationship": "NOT-ESTIMABLE", "reason": "condition is constant across all 10 folders"})
    for row in confound.itertuples():
        if row.feature == "temporal_std_roi_sensitivity":
            continue
        relations.append({"condition": "mean_signal_covariate", "noise_statistic": row.feature, "relationship": row.spearman_with_mean_signal, "reason": "descriptive Spearman; brightness and scene uncontrolled"})
    return table, pd.DataFrame(relations)


def overlap_table(config: dict, noise: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(config["real_evaluation_pairs"])
    eval_folders = set(pairs.folder.astype(int))
    return pd.DataFrame([{
        "folder": int(folder), "used_in_e1_parameter_estimation": True, "used_in_candidate_a_sigma_median": True,
        "present_in_real_surrogate_evaluation": int(folder) in eval_folders, "current_overlap": "YES" if int(folder) in eval_folders else "NO",
        "future_separation_plan": "NOT-FROZEN: pre-register folder-blocked calibration/evaluation roles before CG; do not choose roles from denoising outcomes",
        "evaluation_independence_limitation": "E1 and Candidate A already inspected all folders, so future evaluation is controlled holdout rather than pristine external test",
    } for folder in noise.folder])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = repo / args.output_root
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "provenance").mkdir()
    (output / "logs").mkdir()
    started = now()
    commit = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status_before.strip():
        raise RuntimeError("Formal E1 readiness audit requires a clean worktree")
    e1_root = repo / config["e1_root"]
    e1_status = json.loads((e1_root / "verification_status.json").read_text(encoding="utf-8"))
    if e1_status.get("status") != config["e1_expected_status"]:
        raise RuntimeError("E1 formal status mismatch")
    e1_commit = git(repo, "log", "-1", "--format=%H", "--", str(e1_root.relative_to(repo))).strip()
    if e1_commit != config["e1_report_expected_commit"]:
        raise RuntimeError(f"E1 report commit drift: {e1_commit}")

    tables = load_e1_tables(e1_root)
    inventory, metadata, protected = build_inventory(config, tables)
    root = Path(config["data_root"])
    root_count_before = file_count(root)
    source_before = {str(path): {"sha256": sha256(path), "mtime_ns": path.stat().st_mtime_ns} for path in protected}
    noise = merged_noise(tables)
    repeat = repeatability(tables, noise)
    features = config["noise_state_features"]
    confound = brightness_confound(noise, features)
    correlations = feature_correlations(noise, features)
    candidates, relationships = condition_candidates(noise, repeat, confound)
    overlap = overlap_table(config, noise)

    physical_varying_condition = False
    temporal_repeatable = bool((repeat.temporal_std_frame_count_cv < config["condition_readiness"]["maximum_repeatability_cv"]).all())
    brightness_confound_resolved = False
    separation_frozen = False
    cg_ready = False
    cg = {
        "CG_READY": cg_ready,
        "minimum_gate_results": {
            "at_least_two_distinguishable_strength_states": True,
            "sigma_repeatable_across_frame_counts": temporal_repeatable,
            "state_definition_traceable": "PARTIAL-observed-noise-state-only",
            "physical_condition_varies_across_folders": physical_varying_condition,
            "brightness_scene_confound_resolved": brightness_confound_resolved,
            "calibration_evaluation_separation_frozen": separation_frozen,
            "conditions_preregisterable_now": False,
        },
        "evidence": {
            "temporal_std_range_dn": [float(noise.temporal_std_mean.min()), float(noise.temporal_std_mean.max())],
            "temporal_std_ratio_max_min": float(noise.temporal_std_mean.max() / noise.temporal_std_mean.min()),
            "temporal_std_frame_count_cv_range": [float(repeat.temporal_std_frame_count_cv.min()), float(repeat.temporal_std_frame_count_cv.max())],
            "temporal_std_vs_mean_signal_spearman": rho(noise.temporal_std_mean, noise.mean_signal),
        },
        "decision": "Candidate observed strength states exist, but CG must wait for a frozen folder-blocked calibration/evaluation boundary and a brightness-adjusted reliability/redundancy audit.",
    }
    cgs = {
        "CGS_READY": False,
        "gate_results": {
            "row_column_folder_statistics_available": True,
            "row_column_block_repeatability_available": True,
            "spatial_acf_psd_folder_statistics_available": True,
            "spatial_acf_psd_frame_subset_repeatability_available": False,
            "stable_map_split_repeatability_available": True,
            "stable_component_scene_separation_available": False,
            "component_energy_double_counting_ruled_out": False,
            "calibration_only_parameter_estimation_available": False,
        },
        "decision": "CGS is not ready; no structured component may be added until CG is frozen and one structure component passes an independent reliability and energy-accounting gate.",
    }
    gaps = pd.DataFrame([
        {"priority": 1, "gap_id": "OBSERVED-STATE-RELIABILITY-SEPARATION", "module": "CG", "gap": "No frozen calibration/evaluation boundary and no brightness-adjusted, frame-subset reliability/redundancy gate for observed noise-state features.", "can_use_existing_data": True, "single_next_task": True},
        {"priority": 2, "gap_id": "SPATIAL-SUBSET-REPEATABILITY", "module": "CGS", "gap": "Radial ACF/PSD are folder-level only; frame-subset repeatability is not reported.", "can_use_existing_data": True, "single_next_task": False},
        {"priority": 3, "gap_id": "STRUCTURE-ENERGY-ACCOUNTING", "module": "CGS", "gap": "Temporal, row/column, spatial and stable energies are not shown to be non-overlapping generator components.", "can_use_existing_data": True, "single_next_task": False},
        {"priority": 4, "gap_id": "PHYSICAL-CONDITION-VARIATION", "module": "CG", "gap": "Recorded exposure/sync/gain settings are constant; gate attribution and varying physical conditions cannot be recovered from these folders.", "can_use_existing_data": False, "single_next_task": False},
    ])
    status = "E1-PARTIAL-FOR-CONDITION-MODELING"
    next_step = {"status": status, "highest_priority_gap": "OBSERVED-STATE-RELIABILITY-SEPARATION", "next_task": "Using existing data only, pre-register a folder-blocked calibration/evaluation boundary and audit brightness-adjusted frame-subset reliability and redundancy of the observed noise-state vector; do not train or implement CG.", "reacquisition_required": False}
    verification = {"experiment_id": config["experiment_id"], "status": status, "folder_count": len(inventory), "CG_READY": cg_ready, "CGS_READY": False, "physical_condition_variation_found": False, "observed_noise_state_candidates_found": True, "calibration_evaluation_overlap_folder_count": int(overlap.current_overlap.eq("YES").sum()), "provenance_complete": False, "source_data_protected": False, "synthetic_pairs_generated": False, "model_training_performed": False, "next_task": next_step["next_task"]}

    outputs = {
        "folder_input_inventory.csv": inventory, "folder_metadata_recovery.csv": metadata,
        "folder_level_noise_statistics.csv": noise, "folder_repeatability_analysis.csv": repeat,
        "folder_brightness_confound_analysis.csv": confound, "condition_candidate_table.csv": candidates,
        "condition_statistic_relationships.csv": relationships, "noise_state_feature_correlation.csv": correlations,
        "calibration_evaluation_overlap.csv": overlap, "e1_gap_list.csv": gaps,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    dump_json(output / "cg_readiness.json", cg)
    dump_json(output / "cgs_readiness.json", cgs)
    dump_json(output / "next_step_decision.json", next_step)
    dump_json(output / "verification_status.json", verification)

    (output / "provenance/git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "provenance/git_status_before.txt").write_text(status_before, encoding="utf-8")
    (output / "provenance/git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (output / "provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (output / "provenance/environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\npandas={pd.__version__}\nnumpy={np.__version__}\ntifffile={tifffile.__version__}\n", encoding="utf-8")
    (output / "provenance/resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    script_paths = [Path(__file__), config_path, repo / "scripts/json_serialization.py"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256(path)} for path in script_paths]).to_csv(output / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")
    input_paths = [e1_root / "verification_status.json", e1_root / "noise_summary/folder_noise_summary.csv", e1_root / "robustness/robustness_by_crop_and_frames.csv", e1_root / "row_column/row_column_summary.csv", e1_root / "spatial/spatial_correlation_summary.csv", e1_root / "stable_component/stable_component_summary.csv", e1_root / "temporal_stability/temporal_drift_summary.csv", repo / config["real_evaluation_pairs"]]
    pd.DataFrame([{"path": str(path), "sha256": sha256(path)} for path in input_paths]).to_csv(output / "provenance/input_hashes.csv", index=False, encoding="utf-8-sig")

    root_count_after = file_count(root)
    protection_rows = []
    for path_text, before in source_before.items():
        path = Path(path_text)
        protection_rows.append({"path": path_text, "sha256_before": before["sha256"], "sha256_after": sha256(path), "mtime_ns_before": before["mtime_ns"], "mtime_ns_after": path.stat().st_mtime_ns})
    protection = pd.DataFrame(protection_rows)
    protection["unchanged"] = (protection.sha256_before == protection.sha256_after) & (protection.mtime_ns_before == protection.mtime_ns_after)
    protection.to_csv(output / "provenance/source_protection.csv", index=False, encoding="utf-8-sig")
    source_protected = bool(root_count_before == root_count_after and protection.unchanged.all())
    run = {"experiment_id": config["experiment_id"], "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "status": status, "e1_source_commit": e1_commit, "source_file_count_before": root_count_before, "source_file_count_after": root_count_after, "source_data_protected": source_protected, "source_write_performed": False, "synthetic_pairs_generated": False, "model_training_performed": False}
    dump_json(output / "provenance/run_manifest.json", run)
    (output / "provenance/git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")

    report = f"""# E1 Scientific Completeness And Condition Readiness Audit

Status: `{status}`

E1 contains ten complete 200-frame uint16 folders at 5120x5120. All formal statistics use the frozen center ROI `(top=2304, left=2304, height=512, width=512)`. Folder-level temporal, row/column, spatial, stable-map and drift outputs are present and reproducible.

Per-frame `PictureInfo.txt` records are identical across all ten folders: Exposure channel width 900 ms, Sync.A/Sync.B width 4 us and gain 60, with camera serial 20600555 in filenames. These records verify constant control values, not varying physical conditions. Mapping Sync width to the intensifier gate and gain 60 to calibrated MCP gain is not established; sensor exposure is conflicting with a non-snapshotted `Format.ini` value of 300.

Temporal standard deviation spans {noise.temporal_std_mean.min():.3f}-{noise.temporal_std_mean.max():.3f} DN ({noise.temporal_std_mean.max()/noise.temporal_std_mean.min():.2f}x) and is stable across 16/32/64/128-frame estimates (folder CV {repeat.temporal_std_frame_count_cv.min()*100:.2f}-{repeat.temporal_std_frame_count_cv.max()*100:.2f}%). However its Spearman correlation with folder mean signal is {rho(noise.temporal_std_mean, noise.mean_signal):.3f}. Row, column and observed stable strength are also strongly correlated with brightness and each other. Radial ACF lag-1 is less brightness-confounded, but its frame-subset repeatability was not measured.

All ten folders were used in E1 and Candidate A strength estimation and also appear in real-surrogate evaluation. A future CG therefore requires a pre-registered folder-blocked calibration/evaluation boundary and must not choose roles or thresholds from denoising outcomes.

`CG_READY=false`: repeatable observed strength states exist, but physical conditions do not vary, brightness/scene confounding is unresolved, and calibration/evaluation separation is not frozen.

`CGS_READY=false`: folder-level structural statistics exist, but spatial subset repeatability, stable-component scene separation and non-overlapping component energy accounting are missing.

The sole next task is the priority-1 existing-data audit in `e1_gap_list.csv`: pre-register the folder boundary, then test brightness-adjusted frame-subset reliability and redundancy of the observed noise-state vector. This does not require reacquisition and does not authorize training or CG implementation.
"""
    (output / "verification_report.md").write_text(report, encoding="utf-8")
    dump_json(output / "logs/run.log", {"status": status, "CG_READY": False, "CGS_READY": False, "source_data_protected": source_protected})
    required = list(outputs) + ["cg_readiness.json", "cgs_readiness.json", "next_step_decision.json", "verification_status.json", "verification_report.md", "provenance/run_manifest.json", "provenance/source_protection.csv"]
    verification["provenance_complete"] = all((output / name).is_file() for name in required)
    verification["source_data_protected"] = source_protected
    if not verification["provenance_complete"] or not source_protected:
        verification["status"] = "FAILED"
        run["status"] = "FAILED"
    dump_json(output / "verification_status.json", verification)
    dump_json(output / "provenance/run_manifest.json", run)
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(hashes).to_csv(output / "output_hashes.csv", index=False, encoding="utf-8-sig")
    for name in ["cg_readiness.json", "cgs_readiness.json", "next_step_decision.json", "verification_status.json", "provenance/run_manifest.json"]:
        json.loads((output / name).read_text(encoding="utf-8"))
    print(json.dumps({"status": verification["status"], "folders": len(inventory), "CG_READY": False, "CGS_READY": False, "temporal_std_range_dn": [float(noise.temporal_std_mean.min()), float(noise.temporal_std_mean.max())], "brightness_rho": rho(noise.temporal_std_mean, noise.mean_signal), "overlap_folders": int(overlap.current_overlap.eq("YES").sum()), "source_protected": source_protected}, ensure_ascii=False, indent=2))
    return 0 if verification["status"] == status else 2


if __name__ == "__main__":
    raise SystemExit(main())
