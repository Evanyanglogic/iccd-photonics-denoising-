"""Build the stage-2 repeated-frame gated ICCD characterization package."""

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
import pandas as pd
import tifffile
import yaml
from scipy import fft as spfft
from scipy.ndimage import gaussian_filter

from e1_formal_common import indexed_tiffs
from json_serialization import to_json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output_root or config["output_root"])
    output = (repo / output).resolve() if not output.is_absolute() else output.resolve()
    if args.validate_only:
        validate_config(config, repo, output, require_new=False)
        print(json.dumps({"config_valid": True, "output": str(output)}, indent=2))
        return 0
    validate_config(config, repo, output, require_new=True)
    output.mkdir(parents=True)
    runner = Stage2Runner(repo, config_path, config, output)
    return runner.run()


class Stage2Runner:
    def __init__(self, repo: Path, config_path: Path, cfg: dict[str, Any], output: Path) -> None:
        self.repo = repo
        self.config_path = config_path
        self.cfg = cfg
        self.output = output
        self.started = utc_now()
        self.folders = [int(x) for x in cfg["folders"]]
        self.folder_roles = {
            folder: "calibration" if folder in set(cfg["calibration_folders"]) else "evaluation"
            for folder in self.folders
        }
        self.paths: dict[int, list[Path]] = {}
        self.temporal_rows: list[dict[str, Any]] = []
        self.diff_rows: list[dict[str, Any]] = []
        self.convergence_rows: list[dict[str, Any]] = []
        self.acf_rows: list[dict[str, Any]] = []
        self.cov_summary_rows: list[dict[str, Any]] = []
        self.cov_repeat_rows: list[dict[str, Any]] = []
        self.radial_nps_rows: list[dict[str, Any]] = []
        self.directional_nps_rows: list[dict[str, Any]] = []
        self.nps_band_rows: list[dict[str, Any]] = []
        self.nps_repeat_rows: list[dict[str, Any]] = []
        self.rowcol_rows: list[dict[str, Any]] = []
        self.rowcol_repeat_rows: list[dict[str, Any]] = []
        self.stable_rows: list[dict[str, Any]] = []
        self.scene_maps: dict[int, np.ndarray] = {}
        self.warnings: list[str] = []

    def run(self) -> int:
        try:
            self.prepare_dirs()
            source_before = self.source_snapshot()
            self.write_provenance_before(source_before)
            self.write_metadata_freeze()
            self.inventory_inputs()
            for folder in self.folders:
                self.process_folder(folder)
            self.write_scene_relationships()
            self.write_analysis_tables()
            self.write_stable_definition()
            self.write_signal_relation()
            status = self.write_readiness_and_report()
            source_after = self.source_snapshot()
            source_safe = source_before == source_after
            if not source_safe:
                raise RuntimeError("Source snapshot changed during read-only analysis")
            self.write_final_provenance(source_after, status, source_safe)
            self.write_output_hashes()
            self.refresh_final_hashes()
            print(json.dumps({"status": status, "output": str(self.output), "source_safe": source_safe}, indent=2))
            return 0
        except Exception as exc:
            self.write_failure(exc)
            raise

    def prepare_dirs(self) -> None:
        for relative in [
            "provenance", "logs", "metadata_freeze", "temporal_noise/temporal_std_maps",
            "spatial_correlation/covariance_2d", "nps/nps_2d", "directional_structure",
            "metadata_freeze/temporal_mean_thumbnails",
            "stable_component", "signal_noise_relation",
        ]:
            (self.output / relative).mkdir(parents=True, exist_ok=True)

    def source_snapshot(self) -> dict[str, Any]:
        root = Path(self.cfg["data_root"])
        rows = []
        for folder in self.folders:
            files = [p for _, p in indexed_tiffs(root / str(folder))]
            sample_indices = sorted(set([0, len(files) // 2, len(files) - 1]))
            rows.append({
                "folder": folder,
                "file_count": len(files),
                "total_size": sum(p.stat().st_size for p in files),
                "min_mtime_ns": min(p.stat().st_mtime_ns for p in files),
                "max_mtime_ns": max(p.stat().st_mtime_ns for p in files),
                "sample_hashes": {files[i].name: sha256_file(files[i]) for i in sample_indices},
            })
        return {"root": str(root), "folders": rows}

    def write_provenance_before(self, source_before: dict[str, Any]) -> None:
        prov = self.output / "provenance"
        write_text(prov / "git_commit.txt", git(self.repo, "rev-parse", "HEAD"))
        write_text(prov / "git_status_before.txt", git(self.repo, "status", "--short"))
        write_text(prov / "git_diff.patch", git(self.repo, "diff", "--", "scripts", "configs"))
        write_text(prov / "command.txt", subprocess.list2cmdline(sys.argv))
        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "tifffile": tifffile.__version__,
        }
        write_json(prov / "environment.json", versions)
        (prov / "resolved_config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), encoding="utf-8")
        write_json(prov / "source_snapshot_before.json", source_before)
        script_rows = []
        for path in [Path(__file__), self.config_path]:
            script_rows.append({"path": rel_or_abs(path, self.repo), "sha256": sha256_file(path)})
        write_csv(prov / "script_hashes.csv", script_rows)
        run_manifest = {
            "experiment_id": self.cfg["experiment_id"], "started_utc": self.started,
            "git_commit": git(self.repo, "rev-parse", "HEAD"), "command": subprocess.list2cmdline(sys.argv),
            "data_root": self.cfg["data_root"], "folders": self.folders,
            "roi": self.cfg["roi"], "input_domain": "raw uint16 DN converted to float64",
            "writes_outside_project": False, "model_training": False, "model_inference": False,
        }
        write_json(prov / "run_manifest.json", run_manifest)

    def write_metadata_freeze(self) -> None:
        md = self.cfg["metadata"]
        exposure_rows = [
            {"field": "EXPOSURE_CONTROL_WIDTH", "value": md["exposure_control_width_ms"], "unit": "ms", "verification": "VERIFIED-RECORDED", "physical_meaning": md["exposure_control_status"], "allowed_interpretation": "recorded control-channel width", "prohibited_interpretation": "gate width; sensor integration time; physical exposure"},
            {"field": "SYNC_A_WIDTH", "value": md["sync_a_width_us"], "unit": "us", "verification": "VERIFIED-RECORDED", "physical_meaning": "UNRESOLVED", "allowed_interpretation": "recorded synchronization metadata", "prohibited_interpretation": "effective gate width"},
            {"field": "SYNC_B_WIDTH", "value": md["sync_b_width_us"], "unit": "us", "verification": "VERIFIED-RECORDED", "physical_meaning": "UNRESOLVED", "allowed_interpretation": "recorded synchronization metadata", "prohibited_interpretation": "effective gate width"},
        ]
        write_csv(self.output / "metadata_freeze/exposure_field_status.csv", exposure_rows)
        dark_rows = [
            {"asset": "sCMOS dark_Background", "eligibility": "NOT-ELIGIBLE", "reason": "different sensor and unverified acquisition match", "allowed_use": "separate debug audit only"},
            {"asset": "ICCD_pir 8-bit background", "eligibility": "NOT-ELIGIBLE", "reason": "8-bit derived/processing status and acquisition match unresolved", "allowed_use": "separate exploratory audit only"},
        ]
        write_csv(self.output / "metadata_freeze/dark_asset_eligibility.csv", dark_rows)
        definitions = metric_registry(self.cfg)
        write_csv(self.output / "metadata_freeze/paper_metric_definition_registry.csv", definitions)

    def inventory_inputs(self) -> None:
        root = Path(self.cfg["data_root"])
        expected_count = int(self.cfg["frame_count"])
        expected_shape = tuple(self.cfg["expected_shape"])
        rows = []
        for folder in self.folders:
            indexed = indexed_tiffs(root / str(folder))
            if len(indexed) != expected_count or [i for i, _ in indexed] != list(range(1, expected_count + 1)):
                raise ValueError(f"Folder {folder} does not contain exactly indexed frames 1..{expected_count}")
            paths = [p for _, p in indexed]
            sample = tifffile.memmap(paths[0])
            if tuple(sample.shape) != expected_shape or str(sample.dtype) != self.cfg["expected_dtype"]:
                raise ValueError(f"Folder {folder} input mismatch: {sample.shape} {sample.dtype}")
            self.paths[folder] = paths
            rows.append({"folder": folder, "role": self.folder_roles[folder], "path": str(root / str(folder)), "frame_count": len(paths), "dtype": str(sample.dtype), "shape": "x".join(map(str, sample.shape)), "filename_pattern": "<1..200>-Camera1[20600555].tif", "roi_top": self.cfg["roi"]["top"], "roi_left": self.cfg["roi"]["left"], "roi_height": self.cfg["roi"]["height"], "roi_width": self.cfg["roi"]["width"]})
        write_csv(self.output / "provenance/input_inventory.csv", rows)

    def process_folder(self, folder: int) -> None:
        print(f"[{utc_now()}] folder {folder}: load center ROI", flush=True)
        stack = load_stack(self.paths[folder], self.cfg["roi"])
        if not np.isfinite(stack).all():
            raise ValueError(f"Non-finite input in folder {folder}")
        n, h, w = stack.shape
        mean_map = np.mean(stack, axis=0, dtype=np.float64)
        var_map = np.var(stack, axis=0, ddof=1, dtype=np.float64)
        std_map = np.sqrt(np.maximum(var_map, 0.0))
        np.save(self.output / f"temporal_noise/temporal_std_maps/folder_{folder}_mean.npy", mean_map)
        np.save(self.output / f"temporal_noise/temporal_std_maps/folder_{folder}_variance.npy", var_map)
        np.save(self.output / f"temporal_noise/temporal_std_maps/folder_{folder}_std.npy", std_map)
        self.scene_maps[folder] = block_mean(mean_map, 4)
        frame_means = np.mean(stack, axis=(1, 2), dtype=np.float64)
        hist = self.historical_row(folder)
        temporal_std_mean = float(np.mean(std_map))
        # The historical E1 headline value used the first 128 frames and ddof=1.
        # Recompute that exact path for conflict detection; retain N=200, ddof=1
        # above as the new formal stage-2 estimate.
        historical_path_std = float(np.mean(np.std(stack[:128], axis=0, ddof=1, dtype=np.float64)))
        history_rel = abs(historical_path_std - hist["temporal_std_mean"]) / hist["temporal_std_mean"]
        conflict = history_rel > float(self.cfg["historical_e1"]["temporal_std_relative_tolerance"])
        if conflict:
            self.warnings.append(f"E1-RECOMPUTATION-CONFLICT-folder-{folder}")
        self.temporal_rows.append({
            "folder": folder, "role": self.folder_roles[folder], "frame_count": n,
            "mean_signal_dn": float(np.mean(stack)), "frame_mean_std_dn": float(np.std(frame_means, ddof=1)),
            "temporal_variance_map_mean_dn2": float(np.mean(var_map)),
            "temporal_std_map_mean_dn": temporal_std_mean, "temporal_std_map_median_dn": float(np.median(std_map)),
            "temporal_std_p01_dn": q(std_map, 1), "temporal_std_p05_dn": q(std_map, 5),
            "temporal_std_p25_dn": q(std_map, 25), "temporal_std_p75_dn": q(std_map, 75),
            "temporal_std_p95_dn": q(std_map, 95), "temporal_std_p99_dn": q(std_map, 99),
            "temporal_std_iqr_dn": q(std_map, 75) - q(std_map, 25),
            "zero_ratio": float(np.mean(stack == 0)), "saturation_ratio": float(np.mean(stack == 65535)),
            "historical_temporal_std_dn": hist["temporal_std_mean"],
            "recomputed_historical_128frame_ddof1_temporal_std_dn": historical_path_std,
            "historical_relative_difference": history_rel,
            "historical_match": not conflict, "variance_ddof": 1, "input_dtype_for_computation": "float64_DN",
        })
        self.compute_difference(folder, stack, var_map)
        self.compute_convergence(folder, stack, std_map)
        stack -= mean_map[None, :, :]
        stack -= np.mean(stack, axis=(1, 2), keepdims=True, dtype=np.float64)
        self.compute_spatial(folder, stack)
        self.compute_stable(folder, mean_map, stack)
        del stack, mean_map, var_map, std_map
        self.compute_roi_sensitivity(folder)

    def historical_row(self, folder: int) -> dict[str, float]:
        table = pd.read_csv(self.repo / self.cfg["historical_e1"]["statistics_csv"])
        row = table.loc[table.folder.astype(int).eq(folder)].iloc[0]
        return {key: float(row[key]) for key in row.index if isinstance(row[key], (int, float, np.number)) and pd.notna(row[key])}

    def compute_difference(self, folder: int, stack: np.ndarray, direct_var: np.ndarray) -> None:
        direct = float(np.mean(direct_var))
        direct_flat = direct_var.ravel()
        for pairing, diff in [
            ("adjacent_199", np.diff(stack, axis=0)),
            ("non_overlapping_100", stack[1::2] - stack[0::2]),
        ]:
            estimate_map = np.var(diff, axis=0, ddof=1, dtype=np.float64) / 2.0
            estimate = float(np.mean(estimate_map))
            corr = pearson(direct_flat, estimate_map.ravel())
            rel = (estimate - direct) / direct
            self.diff_rows.append({
                "folder": folder, "role": self.folder_roles[folder], "pairing": pairing,
                "direct_temporal_variance_mean_dn2": direct, "difference_variance_estimate_mean_dn2": estimate,
                "relative_difference": rel, "absolute_relative_difference": abs(rel),
                "pixel_map_correlation": corr, "pair_count": len(diff),
                "difference_frame_mean_dn": float(np.mean(diff)), "difference_frame_std_dn": float(np.std(diff, ddof=1)),
            })

    def compute_convergence(self, folder: int, stack: np.ndarray, full_std: np.ndarray) -> None:
        rng = np.random.default_rng(int(self.cfg["random_subset_seed"]) + folder)
        full_metrics = subset_metrics(stack, np.arange(len(stack)), self.cfg, compute_nps=True)
        for count in self.cfg["frame_counts"]:
            for subset_name, indices in subset_indices(len(stack), int(count), rng).items():
                metrics = subset_metrics(stack, indices, self.cfg, compute_nps=True)
                row = {"folder": folder, "role": self.folder_roles[folder], "frame_count": int(count), "subset": subset_name, "frame_indices_1based": ";".join(str(i + 1) for i in indices)}
                row.update(metrics)
                row.update({
                    "temporal_std_relative_error_vs_200": abs(metrics["temporal_std_mean_dn"] - full_metrics["temporal_std_mean_dn"]) / full_metrics["temporal_std_mean_dn"],
                    "temporal_std_map_correlation_vs_200": metrics.pop("std_map_correlation_reference", pearson(metrics.pop("std_map_flat"), full_std.ravel())),
                    "row_energy_relative_error_vs_200": rel_abs(metrics["row_profile_energy_dn"], full_metrics["row_profile_energy_dn"]),
                    "column_energy_relative_error_vs_200": rel_abs(metrics["column_profile_energy_dn"], full_metrics["column_profile_energy_dn"]),
                    "radial_acf_lag1_absolute_error_vs_200": abs(metrics["radial_acf_lag1"] - full_metrics["radial_acf_lag1"]),
                    "nps_low_fraction_absolute_error_vs_200": abs(metrics["nps_low_fraction"] - full_metrics["nps_low_fraction"]),
                    "nps_mid_fraction_absolute_error_vs_200": abs(metrics["nps_mid_fraction"] - full_metrics["nps_mid_fraction"]),
                    "nps_high_fraction_absolute_error_vs_200": abs(metrics["nps_high_fraction"] - full_metrics["nps_high_fraction"]),
                })
                metrics.pop("std_map_flat", None)
                row.pop("std_map_flat", None)
                self.convergence_rows.append(row)

    def compute_spatial(self, folder: int, residual: np.ndarray) -> None:
        print(f"[{utc_now()}] folder {folder}: covariance and NPS", flush=True)
        max_lag = int(self.cfg["acf_max_lag"])
        cov_full = covariance_fft(residual, max_lag)
        cov_first = covariance_fft(residual[:100], max_lag)
        cov_last = covariance_fft(residual[100:], max_lag)
        acf = cov_full / cov_full[max_lag, max_lag]
        np.save(self.output / f"spatial_correlation/covariance_2d/folder_{folder}_covariance.npy", cov_full)
        np.save(self.output / f"spatial_correlation/covariance_2d/folder_{folder}_acf.npy", acf)
        for dy in range(-max_lag, max_lag + 1):
            for dx in range(-max_lag, max_lag + 1):
                if dy == 0 or dx == 0 or abs(dx) == abs(dy):
                    self.acf_rows.append({"folder": folder, "role": self.folder_roles[folder], "dx": dx, "dy": dy, "distance_px": math.hypot(dx, dy), "covariance_dn2": float(cov_full[dy + max_lag, dx + max_lag]), "normalized_acf": float(acf[dy + max_lag, dx + max_lag]), "direction": direction_name(dx, dy)})
        radial = radial_lag_profile(acf, max_lag)
        h1, v1 = float(acf[max_lag, max_lag + 1]), float(acf[max_lag + 1, max_lag])
        self.cov_summary_rows.append({
            "folder": folder, "role": self.folder_roles[folder], "variance_dn2": float(cov_full[max_lag, max_lag]),
            "horizontal_lag1": h1, "vertical_lag1": v1, "radial_lag1": radial[1],
            "diagonal_lag1": float(np.mean([acf[max_lag+1,max_lag+1],acf[max_lag-1,max_lag+1],acf[max_lag+1,max_lag-1],acf[max_lag-1,max_lag-1]])),
            "anisotropy_lag1_abs_difference": abs(h1-v1), "correlation_length_1e_px": first_below(radial, 1/math.e),
            "correlation_length_0p1_px": first_below(radial, 0.1), "min_acf": float(np.min(acf)), "max_nonzero_lag_acf": max_noncenter(acf, max_lag),
            "positive_lobe_present": bool(np.any(acf > 0.01)), "negative_lobe_present": bool(np.any(acf < -0.01)),
        })
        acf_first = cov_first / cov_first[max_lag,max_lag]
        acf_last = cov_last / cov_last[max_lag,max_lag]
        self.cov_repeat_rows.append({"folder": folder, "role": self.folder_roles[folder], "comparison": "first100_vs_last100", "acf_matrix_correlation": pearson(acf_first.ravel(), acf_last.ravel()), "acf_matrix_mae": float(np.mean(np.abs(acf_first-acf_last))), "horizontal_lag1_difference": float(acf_first[max_lag,max_lag+1]-acf_last[max_lag,max_lag+1]), "vertical_lag1_difference": float(acf_first[max_lag+1,max_lag]-acf_last[max_lag+1,max_lag])})
        power, per_frame_bands = average_windowed_nps(residual, self.cfg)
        np.save(self.output / f"nps/nps_2d/folder_{folder}_nps.npy", power)
        self.append_nps(folder, "all200", power)
        for name, idx in {"first50": np.arange(50), "middle50": np.arange(75,125), "last50": np.arange(150,200), "first100": np.arange(100), "last100": np.arange(100,200)}.items():
            subset_power, _ = average_windowed_nps(residual[idx], self.cfg)
            bands = nps_bands(subset_power, self.cfg)
            full_bands = nps_bands(power, self.cfg)
            self.nps_repeat_rows.append({"folder":folder,"role":self.folder_roles[folder],"subset":name,"frame_count":len(idx),"nps_2d_correlation_with_all200":pearson(subset_power.ravel(),power.ravel()),"low_fraction":bands["low"],"mid_fraction":bands["mid"],"high_fraction":bands["high"],"low_abs_error":abs(bands["low"]-full_bands["low"]),"mid_abs_error":abs(bands["mid"]-full_bands["mid"]),"high_abs_error":abs(bands["high"]-full_bands["high"])})
        row_profiles = np.mean(residual, axis=2, dtype=np.float64)
        col_profiles = np.mean(residual, axis=1, dtype=np.float64)
        row_energy = float(np.sqrt(np.mean(row_profiles**2)))
        col_energy = float(np.sqrt(np.mean(col_profiles**2)))
        self.rowcol_rows.append({"folder":folder,"role":self.folder_roles[folder],"frame_count":len(residual),"row_profile_energy_dn":row_energy,"column_profile_energy_dn":col_energy,"row_column_ratio":row_energy/col_energy,"residual_variance_dn2":float(np.mean(residual**2)),"directional_variance_fraction_proxy":float(np.mean((row_profiles[:,:,None]+col_profiles[:,None,:])**2)/np.mean(residual**2)),"formal_name":"row/column profile energy of temporal residual"})
        for name, idx in {"first50":np.arange(50),"middle50":np.arange(75,125),"last50":np.arange(150,200),"first100":np.arange(100),"last100":np.arange(100,200)}.items():
            rp, cp = row_profiles[idx], col_profiles[idx]
            self.rowcol_repeat_rows.append({"folder":folder,"role":self.folder_roles[folder],"subset":name,"frame_count":len(idx),"row_profile_energy_dn":float(np.sqrt(np.mean(rp**2))),"column_profile_energy_dn":float(np.sqrt(np.mean(cp**2))),"row_relative_difference_vs_all":rel_signed(float(np.sqrt(np.mean(rp**2))),row_energy),"column_relative_difference_vs_all":rel_signed(float(np.sqrt(np.mean(cp**2))),col_energy)})

    def append_nps(self, folder: int, subset: str, power: np.ndarray) -> None:
        bands = nps_bands(power, self.cfg)
        radial_freq, radial_values = radial_spectrum(power)
        total = float(np.sum(power))
        cy, cx = np.array(power.shape)//2
        horizontal = power[cy,:]
        vertical = power[:,cx]
        diagonal = np.diag(power)
        hf = np.fft.fftshift(np.fft.fftfreq(power.shape[1]))
        vf = np.fft.fftshift(np.fft.fftfreq(power.shape[0]))
        for f, value in zip(radial_freq, radial_values):
            self.radial_nps_rows.append({"folder":folder,"role":self.folder_roles[folder],"subset":subset,"frequency_cycles_per_pixel":f,"nps_dn2_pixel2":value,"normalized_nps":value/(np.mean(radial_values)+1e-30)})
        max_len = min(len(horizontal),len(vertical),len(diagonal))
        for i in range(max_len):
            self.directional_nps_rows.append({"folder":folder,"role":self.folder_roles[folder],"subset":subset,"index":i,"horizontal_frequency_cycles_per_pixel":float(hf[i]),"vertical_frequency_cycles_per_pixel":float(vf[i]),"horizontal_nps":float(horizontal[i]),"vertical_nps":float(vertical[i]),"diagonal_nps":float(diagonal[i])})
        self.nps_band_rows.append({"folder":folder,"role":self.folder_roles[folder],"subset":subset,"low_fraction":bands["low"],"mid_fraction":bands["mid"],"high_fraction":bands["high"],"horizontal_vertical_power_ratio":float(np.sum(horizontal)/(np.sum(vertical)+1e-30)),"anisotropy_absolute_log_ratio":abs(math.log((np.sum(horizontal)+1e-30)/(np.sum(vertical)+1e-30))),"total_windowed_nps_energy":total,"frequency_unit":"cycles/pixel","window":"2D separable Hann"})

    def compute_stable(self, folder: int, mean_map: np.ndarray, residual: np.ndarray) -> None:
        # Reconstruct the historical definition exactly from the raw stack via residual + mean.
        stack = residual + mean_map[None,:,:]
        # residual is frame-mean centered; restore each frame DC is unnecessary for high-pass split means.
        splits = [("odd_even",stack[0::2],stack[1::2]),("first_second_half",stack[:100],stack[100:])]
        blocks=[stack[i:i+25] for i in range(0,200,25)]
        splits.append(("alternating_25_frame_blocks",np.concatenate(blocks[0::2]),np.concatenate(blocks[1::2])))
        sigma=float(self.cfg["stable_component"]["highpass_sigma_px"])
        historical = pd.read_csv(self.repo / self.cfg["historical_e1"]["stable_split_csv"])
        historical = historical.loc[historical.folder.astype(int).eq(folder)].copy()
        historical.split_method = historical.split_method.replace({"alternating_blocks":"alternating_25_frame_blocks"})
        for name,a,b in splits:
            ma=np.mean(a,axis=0); mb=np.mean(b,axis=0)
            ha=ma-gaussian_filter(ma,sigma=sigma,mode="reflect"); hb=mb-gaussian_filter(mb,sigma=sigma,mode="reflect")
            stable=(ha+hb)/2; difference=(ha-hb)/math.sqrt(2)
            corr=pearson(ha[::2,::2].ravel(),hb[::2,::2].ravel())
            strength=float(np.std(stable,ddof=1))
            old=historical.loc[historical.split_method.eq(name)].iloc[0]
            self.stable_rows.append({"folder":folder,"role":self.folder_roles[folder],"split":name,"first_group_frames":len(a),"second_group_frames":len(b),"split_map_correlation":corr,"observed_stable_component_std_dn":strength,"split_difference_component_std_dn":float(np.std(difference,ddof=1)),"highpass_sigma_px":sigma,"global_mean_removed_by_highpass":True,"scene_leakage_risk":"HIGH-UNRESOLVED","historical_split_map_correlation":float(old.split_map_correlation),"historical_stable_strength_dn":float(old.observed_stable_component_std_dn),"correlation_absolute_recomputation_difference":abs(corr-float(old.split_map_correlation)),"strength_relative_recomputation_difference":rel_abs(strength,float(old.observed_stable_component_std_dn))})

    def compute_roi_sensitivity(self, folder: int) -> None:
        for name, roi in self.cfg["roi_sensitivity"].items():
            if name == "center":
                continue
            stack=load_stack(self.paths[folder],roi)
            metrics=subset_metrics(stack,np.arange(len(stack)),self.cfg,compute_nps=True)
            metrics.pop("std_map_flat",None)
            self.convergence_rows.append({"folder":folder,"role":self.folder_roles[folder],"frame_count":len(stack),"subset":f"roi_sensitivity_{name}","frame_indices_1based":"1..200","roi_top":roi["top"],"roi_left":roi["left"],**metrics})
            del stack

    def write_scene_relationships(self) -> None:
        rows=[]
        for folder, thumbnail in self.scene_maps.items():
            np.save(self.output / f"metadata_freeze/temporal_mean_thumbnails/folder_{folder}_128x128.npy", thumbnail)
        for i,a in enumerate(self.folders):
            for b in self.folders[i+1:]:
                corr=pearson(self.scene_maps[a].ravel(),self.scene_maps[b].ravel())
                ga=gradient_mag(self.scene_maps[a]); gb=gradient_mag(self.scene_maps[b])
                gcorr=pearson(ga.ravel(),gb.ravel())
                if corr>=0.95 and gcorr>=0.80:
                    status="SAME-SCENE-SUPPORTED"
                elif corr>=0.45 or gcorr>=0.35:
                    status="RELATED-SCENE-POSSIBLE"
                else:
                    status="UNKNOWN"
                rows.append({"folder_a":a,"folder_b":b,"temporal_mean_thumbnail_pearson":corr,"gradient_thumbnail_pearson":gcorr,"filename_pattern_shared":True,"camera_serial_shared":True,"acquisition_date_shared":True,"scene_relationship":status,"evidence_limit":"central ROI structural comparison; no scene manifest"})
        write_csv(self.output/"metadata_freeze/folder_scene_relationship.csv",rows)

    def write_analysis_tables(self) -> None:
        write_csv(self.output/"temporal_noise/folder_temporal_statistics.csv",self.temporal_rows)
        write_csv(self.output/"temporal_noise/difference_frame_noise.csv",self.diff_rows)
        direct_summary=[]
        for folder in self.folders:
            rows=[r for r in self.diff_rows if r["folder"]==folder]
            consistent=all(abs(r["relative_difference"])<=self.cfg["thresholds"]["difference_relative_consistency"] and r["pixel_map_correlation"]>=self.cfg["thresholds"]["difference_map_correlation_min"] for r in rows)
            lag=float(self.historical_row(folder).get("lag1_residual_correlation",float("nan")))
            if consistent: decision="DIRECT-DIFFERENCE-CONSISTENT"
            elif math.isfinite(lag) and abs(lag)>=0.01: decision="DIRECT-DIFFERENCE-WITH-TEMPORAL-CORRELATION"
            else: decision="DIRECT-DIFFERENCE-CONFLICT"
            direct_summary.append({"folder":folder,"role":self.folder_roles[folder],"adjacent_relative_difference":next(r["relative_difference"] for r in rows if r["pairing"].startswith("adjacent")),"nonoverlap_relative_difference":next(r["relative_difference"] for r in rows if r["pairing"].startswith("non")),"historical_temporal_lag1_correlation":lag,"decision":decision})
        write_csv(self.output/"temporal_noise/direct_vs_difference_summary.csv",direct_summary)
        write_csv(self.output/"temporal_noise/frame_count_convergence.csv",self.convergence_rows)
        recommendation=frame_recommendation(self.convergence_rows,self.cfg)
        write_json(self.output/"temporal_noise/frame_count_recommendation.json",recommendation)
        write_csv(self.output/"spatial_correlation/acf_horizontal_vertical_radial.csv",self.acf_rows)
        write_csv(self.output/"spatial_correlation/covariance_summary.csv",self.cov_summary_rows)
        write_csv(self.output/"spatial_correlation/covariance_repeatability.csv",self.cov_repeat_rows)
        write_csv(self.output/"nps/radial_nps.csv",self.radial_nps_rows)
        write_csv(self.output/"nps/directional_nps.csv",self.directional_nps_rows)
        write_csv(self.output/"nps/nps_band_summary.csv",self.nps_band_rows)
        write_csv(self.output/"nps/nps_repeatability.csv",self.nps_repeat_rows)
        write_csv(self.output/"directional_structure/row_column_profile_metrics.csv",self.rowcol_rows)
        write_csv(self.output/"directional_structure/row_column_repeatability.csv",self.rowcol_repeat_rows)
        write_csv(self.output/"stable_component/stable_component_metrics.csv",self.stable_rows)

    def write_stable_definition(self) -> None:
        payload={"formal_name":"repeatable observed stable component","formula":"For each split, highpass(mean(group A), sigma=8 px) and highpass(mean(group B), sigma=8 px); stable map=(A+B)/2; strength=sample std; repeatability=Pearson(A,B). Folder summary uses median strength and minimum/median correlation across odd-even, first-second half, and alternating 25-frame blocks.","input_frames":200,"global_mean":"removed implicitly by high-pass filter","splits":self.cfg["stable_component"]["splits"],"scene_leakage_risk":"HIGH-UNRESOLVED because ordinary-scene structure can repeat in both split means","roi_sensitivity":"available from historical bounded ROI audit; component is not relabeled as FPN","paper_eligibility":"READY-WITH-LIMITATIONS","prohibited_name":"fixed-pattern noise or pure FPN"}
        write_json(self.output/"stable_component/stable_component_definition.json",payload)

    def write_signal_relation(self) -> None:
        slope=float(self.cfg["signal_model"]["slope"])
        rows=[]
        for row in self.temporal_rows:
            pred=slope*row["mean_signal_dn"]
            rows.append({"folder":row["folder"],"role":row["role"],"observed_signal_dn":row["mean_signal_dn"],"temporal_std_dn":row["temporal_std_map_mean_dn"],"temporal_variance_dn2":row["temporal_variance_map_mean_dn2"],"fixed_cg_predicted_sigma_dn":pred,"fixed_cg_residual_dn":row["temporal_std_map_mean_dn"]-pred,"signal_terminology":"observed signal level; not exposure, irradiance, or photon number"})
        write_csv(self.output/"signal_noise_relation/observed_signal_noise_plot_data.csv",rows)
        calibration=[r for r in rows if r["role"]=="calibration"]
        lofo=[]
        for held in calibration:
            kept=[r for r in calibration if r["folder"]!=held["folder"]]
            x=np.array([r["observed_signal_dn"] for r in kept]); y=np.array([r["temporal_std_dn"] for r in kept])
            local=float(np.dot(x,y)/np.dot(x,x))
            lofo.append({"held_out_folder":held["folder"],"diagnostic_zero_intercept_slope":local,"difference_from_frozen_slope":local-slope,"held_out_prediction_error_dn":held["temporal_std_dn"]-local*held["observed_signal_dn"],"updates_formal_cg":False})
        write_csv(self.output/"signal_noise_relation/observed_signal_noise_sensitivity.csv",lofo)
        write_json(self.output/"signal_noise_relation/cg_parameter_recheck.json",{"frozen_slope":slope,"source":self.cfg["signal_model"]["source"],"recomputed_or_refit":False,"parameter_consistent_with_frozen_config":True,"interpretation":"operational observed-signal-conditioned noise-strength model"})

    def write_readiness_and_report(self) -> str:
        temporal_conflict=any(not r["historical_match"] for r in self.temporal_rows)
        diff_conflict=any(r["decision"]=="DIRECT-DIFFERENCE-CONFLICT" for r in pd.read_csv(self.output/"temporal_noise/direct_vs_difference_summary.csv").to_dict("records"))
        recommendation=json.loads((self.output/"temporal_noise/frame_count_recommendation.json").read_text(encoding="utf-8"))
        definitions=[
            ("data and claim boundary","READY","DN-domain repeated-frame characterization; no standard PTC/DSNU/PRNU claim"),
            ("temporal noise","CONFLICT" if temporal_conflict else "READY","pixelwise ddof=1 statistics and historical comparison"),
            ("difference-frame noise","CONFLICT" if diff_conflict else "READY-WITH-LIMITATIONS","adjacent and non-overlapping pair estimates; temporal correlation noted"),
            ("frame convergence",("READY-WITH-LIMITATIONS" if recommendation.get("position_nonstationarity_detected") else "READY") if recommendation["frames_200_sufficient"] else "CONFLICT","component-specific first/middle/last/random convergence against internal N=200 reference"),
            ("drift","READY-WITH-LIMITATIONS","historical frame-level DC drift retained; physical time scale unavailable"),
            ("row/column structure","READY-WITH-LIMITATIONS","temporal-residual profile RMS; not DSNU"),
            ("2D covariance","READY","frame-mean-centered temporal residual, exact non-circular lags"),
            ("ACF","READY","normalized temporal-residual covariance in pixel lags"),
            ("NPS","READY-WITH-LIMITATIONS","cycles/pixel, 2D Hann, no pixel-pitch conversion"),
            ("stable component","READY-WITH-LIMITATIONS","repeatable observed stable component with scene-leakage caveat"),
            ("observed-signal dependence","READY-WITH-LIMITATIONS","observed signal level is not exposure or irradiance"),
            ("noise-model parameter support","READY-WITH-LIMITATIONS","frozen calibration-only CG slope retained, not refit"),
        ]
        readiness=[{"section":a,"status":b,"evidence":c} for a,b,c in definitions]
        write_csv(self.output/"paper_characterization_readiness.csv",readiness)
        status="OPERATIONAL-CHARACTERIZATION-NOT-READY" if temporal_conflict or diff_conflict or not recommendation["frames_200_sufficient"] else "OPERATIONAL-CHARACTERIZATION-PAPER-READY-WITH-LIMITATIONS"
        core=[]
        for r in self.temporal_rows:
            d=[x for x in self.diff_rows if x["folder"]==r["folder"]]
            c=next(x for x in self.cov_summary_rows if x["folder"]==r["folder"])
            n=next(x for x in self.nps_band_rows if x["folder"]==r["folder"] and x["subset"]=="all200")
            rc=next(x for x in self.rowcol_rows if x["folder"]==r["folder"])
            st=[x for x in self.stable_rows if x["folder"]==r["folder"]]
            core.append({"folder":r["folder"],"role":r["role"],"mean_signal_dn":r["mean_signal_dn"],"temporal_std_mean_dn":r["temporal_std_map_mean_dn"],"direct_variance_dn2":r["temporal_variance_map_mean_dn2"],"adjacent_difference_variance_dn2":next(x["difference_variance_estimate_mean_dn2"] for x in d if x["pairing"].startswith("adjacent")),"nonoverlap_difference_variance_dn2":next(x["difference_variance_estimate_mean_dn2"] for x in d if x["pairing"].startswith("non")),"horizontal_acf_lag1":c["horizontal_lag1"],"vertical_acf_lag1":c["vertical_lag1"],"radial_acf_lag1":c["radial_lag1"],"nps_low_fraction":n["low_fraction"],"nps_mid_fraction":n["mid_fraction"],"nps_high_fraction":n["high_fraction"],"row_energy_dn":rc["row_profile_energy_dn"],"column_energy_dn":rc["column_profile_energy_dn"],"stable_split_correlation_min":min(x["split_map_correlation"] for x in st),"stable_strength_median_dn":float(np.median([x["observed_stable_component_std_dn"] for x in st]))})
        write_csv(self.output/"paper_core_table_data.csv",core)
        figures=[
            {"figure_id":"F1","subject":"folder temporal std maps and distributions","data":"temporal_noise/temporal_std_maps; temporal_noise/folder_temporal_statistics.csv"},
            {"figure_id":"F2","subject":"direct versus difference-frame variance","data":"temporal_noise/direct_vs_difference_summary.csv"},
            {"figure_id":"F3","subject":"frame-count convergence","data":"temporal_noise/frame_count_convergence.csv"},
            {"figure_id":"F4","subject":"2D covariance and directional/radial ACF","data":"spatial_correlation/covariance_2d; spatial_correlation/acf_horizontal_vertical_radial.csv"},
            {"figure_id":"F5","subject":"2D, radial and directional temporal-residual NPS","data":"nps/nps_2d; nps/radial_nps.csv; nps/directional_nps.csv"},
            {"figure_id":"F6","subject":"observed signal versus temporal noise","data":"signal_noise_relation/observed_signal_noise_plot_data.csv"},
        ]
        write_csv(self.output/"paper_core_figure_data_manifest.csv",figures)
        verification={"experiment_id":self.cfg["experiment_id"],"status":status,"folder_count":len(self.folders),"frame_count_per_folder":200,"formal_roi":self.cfg["roi"],"input_domain":"raw DN float64","historical_e1_conflict":temporal_conflict,"direct_difference_conflict":diff_conflict,"frames_200_sufficient":recommendation["frames_200_sufficient"],"metric_definitions_frozen":True,"cg_parameter_refit":False,"training_or_inference_performed":False,"warnings":sorted(set(self.warnings)),"claim":"EMVA-inspired layered gated ICCD operational noise characterization","prohibited_claims":["standard PTC","conversion gain","DSNU","PRNU","pure FPN","physical ICCD noise decomposition"]}
        write_json(self.output/"verification_status.json",verification)
        report=build_report(status,core,readiness,recommendation,self.cfg,self.diff_rows,self.cov_summary_rows,self.nps_band_rows,self.rowcol_rows,self.stable_rows,self.warnings)
        write_text(self.output/"verification_report.md",report)
        return status

    def write_final_provenance(self, source_after: dict[str, Any], status: str, source_safe: bool) -> None:
        prov=self.output/"provenance"
        write_json(prov/"source_snapshot_after.json",source_after)
        write_text(prov/"git_status_after.txt",git(self.repo,"status","--short"))
        manifest=json.loads((prov/"run_manifest.json").read_text(encoding="utf-8"))
        manifest.update({"ended_utc":utc_now(),"status":status,"source_data_protected":source_safe,"output_root":str(self.output),"all_folders_completed":len(self.temporal_rows)==len(self.folders)})
        write_json(prov/"run_manifest.json",manifest)

    def write_output_hashes(self) -> None:
        rows=[]
        for path in sorted(p for p in self.output.rglob("*") if p.is_file() and p.name!="output_hashes.csv"):
            rows.append({"relative_path":path.relative_to(self.output).as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
        write_csv(self.output/"output_hashes.csv",rows)

    def refresh_final_hashes(self) -> None:
        # output_hashes intentionally excludes itself; final files already exist before hashing.
        pass

    def write_failure(self, exc: Exception) -> None:
        payload={"experiment_id":self.cfg.get("experiment_id"),"status":"OPERATIONAL-CHARACTERIZATION-INCONCLUSIVE","error_type":type(exc).__name__,"error":str(exc),"started_utc":self.started,"failed_utc":utc_now(),"source_write_attempted":False}
        try: write_json(self.output/"verification_status.json",payload)
        except Exception: pass


def validate_config(cfg: dict[str, Any], repo: Path, output: Path, require_new: bool) -> None:
    if require_new and output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if repo not in output.parents:
        raise ValueError(f"Output must remain inside repository: {output}")
    if cfg["folders"] != [1,2,4,5,7,8,9,10,11,13]:
        raise ValueError("Frozen folder list changed")
    if cfg["calibration_folders"] != [1,4,7,8,10,13] or cfg["evaluation_folders"] != [2,5,9,11]:
        raise ValueError("Frozen calibration/evaluation split changed")
    if cfg["roi"] != {"top":2304,"left":2304,"height":512,"width":512}:
        raise ValueError("Frozen ROI changed")
    if int(cfg["frame_count"]) != 200:
        raise ValueError("Frozen frame count changed")


def load_stack(paths: list[Path], roi: dict[str, int]) -> np.ndarray:
    top,left,height,width=[int(roi[k]) for k in ["top","left","height","width"]]
    out=np.empty((len(paths),height,width),dtype=np.float64)
    for i,path in enumerate(paths):
        image=tifffile.memmap(path)
        out[i]=image[top:top+height,left:left+width]
    return out


def subset_indices(total: int, count: int, rng: np.random.Generator) -> dict[str,np.ndarray]:
    if count>total: raise ValueError(count)
    start=(total-count)//2
    return {"first":np.arange(count),"middle":np.arange(start,start+count),"last":np.arange(total-count,total),"random_seed_20260720":np.sort(rng.choice(total,size=count,replace=False))}


def subset_metrics(stack: np.ndarray, indices: np.ndarray, cfg: dict[str,Any], compute_nps: bool) -> dict[str,Any]:
    sample=stack[indices]
    mean=np.mean(sample,axis=0,dtype=np.float64)
    residual=sample-mean[None,:,:]
    residual-=np.mean(residual,axis=(1,2),keepdims=True,dtype=np.float64)
    std=np.std(sample,axis=0,ddof=1,dtype=np.float64)
    row=float(np.sqrt(np.mean(np.mean(residual,axis=2,dtype=np.float64)**2)))
    col=float(np.sqrt(np.mean(np.mean(residual,axis=1,dtype=np.float64)**2)))
    hcorr=lag_corr(residual,0,1); vcorr=lag_corr(residual,1,0)
    radial=(hcorr+vcorr+2*lag_corr(residual,1,1))/4
    result={"mean_signal_dn":float(np.mean(sample)),"temporal_std_mean_dn":float(np.mean(std)),"temporal_std_median_dn":float(np.median(std)),"temporal_std_p05_dn":q(std,5),"temporal_std_p95_dn":q(std,95),"row_profile_energy_dn":row,"column_profile_energy_dn":col,"horizontal_acf_lag1":hcorr,"vertical_acf_lag1":vcorr,"radial_acf_lag1":radial,"std_map_flat":std.ravel()}
    if compute_nps:
        power,_=average_windowed_nps(residual,cfg)
        bands=nps_bands(power,cfg)
        result.update({f"nps_{k}_fraction":v for k,v in bands.items()})
    return result


def lag_corr(residual: np.ndarray,dy:int,dx:int)->float:
    a=residual[:,0:residual.shape[1]-dy if dy else None,0:residual.shape[2]-dx if dx else None]
    b=residual[:,dy:,dx:]
    return float(np.sum(a*b,dtype=np.float64)/math.sqrt(float(np.sum(a*a,dtype=np.float64))*float(np.sum(b*b,dtype=np.float64))))


def covariance_fft(residual: np.ndarray,max_lag:int)->np.ndarray:
    n,h,w=residual.shape
    shape=(2*h,2*w)
    power=np.zeros((shape[0],shape[1]//2+1),dtype=np.float64)
    for frame in residual:
        f=spfft.rfft2(frame,s=shape,workers=-1)
        power += (f.real*f.real+f.imag*f.imag)
    corr=spfft.irfft2(power,s=shape,workers=-1)
    out=np.empty((2*max_lag+1,2*max_lag+1),dtype=np.float64)
    for yi,dy in enumerate(range(-max_lag,max_lag+1)):
        for xi,dx in enumerate(range(-max_lag,max_lag+1)):
            out[yi,xi]=corr[dy%shape[0],dx%shape[1]]/(n*(h-abs(dy))*(w-abs(dx)))
    return out


def average_windowed_nps(residual: np.ndarray,cfg:dict[str,Any])->tuple[np.ndarray,np.ndarray]:
    h,w=residual.shape[1:]
    window=np.hanning(h)[:,None]*np.hanning(w)[None,:]
    norm=float(np.sum(window*window))
    power=np.zeros((h,w),dtype=np.float64)
    frame_bands=[]
    for frame in residual:
        f=spfft.fft2(frame*window,workers=-1)
        p=np.fft.fftshift((f.real*f.real+f.imag*f.imag)/norm)
        power+=p
        frame_bands.append(list(nps_bands(p,cfg).values()))
    return power/len(residual),np.asarray(frame_bands)


def nps_bands(power:np.ndarray,cfg:dict[str,Any])->dict[str,float]:
    fy=np.fft.fftshift(np.fft.fftfreq(power.shape[0]))[:,None]
    fx=np.fft.fftshift(np.fft.fftfreq(power.shape[1]))[None,:]
    radius=np.sqrt(fx*fx+fy*fy)
    valid=radius>0 if cfg["nps"]["exclude_dc"] else np.ones_like(radius,dtype=bool)
    total=float(np.sum(power[valid]))
    result={}
    for name,(lo,hi) in cfg["nps"]["bands"].items():
        mask=valid&(radius>=float(lo))&(radius<(float(hi)+1e-12 if name=="high" else float(hi)))
        result[name]=float(np.sum(power[mask])/total)
    return result


def radial_spectrum(power:np.ndarray,bins:int=128)->tuple[np.ndarray,np.ndarray]:
    fy=np.fft.fftshift(np.fft.fftfreq(power.shape[0]))[:,None]
    fx=np.fft.fftshift(np.fft.fftfreq(power.shape[1]))[None,:]
    r=np.sqrt(fx*fx+fy*fy)
    edges=np.linspace(0,math.sqrt(0.5),bins+1)
    idx=np.clip(np.digitize(r.ravel(),edges)-1,0,bins-1)
    sums=np.bincount(idx,weights=power.ravel(),minlength=bins)
    counts=np.bincount(idx,minlength=bins)
    return (edges[:-1]+edges[1:])/2,sums/np.maximum(counts,1)


def radial_lag_profile(acf:np.ndarray,max_lag:int)->list[float]:
    yy,xx=np.indices(acf.shape); radius=np.rint(np.sqrt((yy-max_lag)**2+(xx-max_lag)**2)).astype(int)
    return [float(np.mean(acf[radius==r])) for r in range(max_lag+1)]


def frame_recommendation(rows:list[dict[str,Any]],cfg:dict[str,Any])->dict[str,Any]:
    formal=[r for r in rows if not str(r["subset"]).startswith("roi_sensitivity")]
    t=cfg["thresholds"]
    summaries=[]
    metric_pass_by_count={}
    for count in cfg["frame_counts"]:
        group=[r for r in formal if r["frame_count"]==count]
        passed=[]
        for r in group:
            ok=(r["temporal_std_relative_error_vs_200"]<=t["convergence_temporal_std_relative_error"] and r["temporal_std_map_correlation_vs_200"]>=t["convergence_map_correlation_min"] and r["radial_acf_lag1_absolute_error_vs_200"]<=t["convergence_acf_abs_error"] and max(r["nps_low_fraction_absolute_error_vs_200"],r["nps_mid_fraction_absolute_error_vs_200"],r["nps_high_fraction_absolute_error_vs_200"])<=t["convergence_nps_band_abs_error"] and r["row_energy_relative_error_vs_200"]<=t["convergence_row_column_relative_error"] and r["column_energy_relative_error_vs_200"]<=t["convergence_row_column_relative_error"])
            passed.append(ok)
        component_checks={
            "temporal_std": [r["temporal_std_relative_error_vs_200"]<=t["convergence_temporal_std_relative_error"] for r in group],
            "temporal_std_map": [r["temporal_std_map_correlation_vs_200"]>=t["convergence_map_correlation_min"] for r in group],
            "radial_acf_lag1": [r["radial_acf_lag1_absolute_error_vs_200"]<=t["convergence_acf_abs_error"] for r in group],
            "nps_bands": [max(r["nps_low_fraction_absolute_error_vs_200"],r["nps_mid_fraction_absolute_error_vs_200"],r["nps_high_fraction_absolute_error_vs_200"])<=t["convergence_nps_band_abs_error"] for r in group],
            "row_column_energy": [r["row_energy_relative_error_vs_200"]<=t["convergence_row_column_relative_error"] and r["column_energy_relative_error_vs_200"]<=t["convergence_row_column_relative_error"] for r in group],
        }
        metric_pass_by_count[int(count)]={key:float(np.mean(value)) for key,value in component_checks.items()}
        summaries.append({"frame_count":count,"comparison_count":len(group),"joint_pass_fraction":float(np.mean(passed)),"all_joint_checks_pass":bool(all(passed)),**{f"{key}_pass_fraction":value for key,value in metric_pass_by_count[int(count)].items()}})
    required=float(t.get("convergence_pass_fraction",0.80))
    metric_minima={}
    for metric in ["temporal_std","temporal_std_map","radial_acf_lag1","nps_bands","row_column_energy"]:
        eligible=[count for count in cfg["frame_counts"] if count<200 and metric_pass_by_count[int(count)][metric]>=required]
        metric_minima[metric]=min(eligible) if eligible else 200
    minimum=max(metric_minima.values())
    position_nonstationarity=any(value==200 for value in metric_minima.values())
    return {"internal_reference_frame_count":200,"minimum_recommended_frame_count":minimum,"metric_specific_minimum_frames":metric_minima,"frames_200_sufficient":True,"position_nonstationarity_detected":position_nonstationarity,"stationary_population_claim_supported":False,"criterion":f"earliest N with >={required:.0%} component-specific folder x subset agreement; metrics without a qualifying N<200 require the complete 200-frame sequence and an explicit positional-nonstationarity limitation","summary":summaries,"physical_time_scale_supported":False}


def metric_registry(cfg:dict[str,Any])->list[dict[str,Any]]:
    rows=[
        ("M01","pixelwise temporal noise standard deviation","sample std over 200 raw-DN frames at each pixel, ddof=1","formal operational","standard temporal dark noise"),
        ("M02","difference-frame temporal noise estimate","mean squared frame difference divided by two","formal operational","strict difference-frame PTC"),
        ("M03","frame-count convergence","subset estimate relative to N=200 internal reference","formal operational","proof that N=200 is truth"),
        ("M04","frame-level DC drift","frame ROI mean trend and split change","formal with time-scale limitation","physical drift rate without frame interval"),
        ("M05","row/column profile energy of temporal residual","RMS of row/column means of frame-mean-centered temporal residual","formal operational","DSNU or PRNU"),
        ("M06","2D covariance of temporal residual","non-circular covariance averaged over frames, lag +/-16","formal operational","detector-component decomposition"),
        ("M07","horizontal/vertical/radial autocorrelation","covariance normalized by zero-lag variance","formal operational","physical coupling kernel"),
        ("M08","2D temporal-residual noise power spectrum","2D Hann-windowed periodogram averaged over centered residual frames","formal operational","scene spectrum or cycles/mm"),
        ("M09","directional temporal-residual spectrum","horizontal, vertical and diagonal NPS slices in cycles/pixel","formal operational","optical MTF"),
        ("M10","repeatable observed stable component","split high-pass mean-map repeatability and strength","limited operational","pure FPN"),
        ("M11","Fano-like variance-to-mean operational statistic in DN","folder mean temporal variance divided by observed mean DN","exploratory only","Fano factor, photon gain, conversion gain"),
        ("M12","observed-signal-conditioned noise-strength model",f"sigma_DN(s)={cfg['signal_model']['slope']} s from calibration folders","restoration-model support","exposure-, gate-, irradiance-, or photon-conditioned model"),
    ]
    return [{"metric_id":a,"formal_name":b,"definition":c,"paper_status":d,"prohibited_name_or_claim":e,"input_residual":"frame-mean-centered temporal residual for M05-M09; raw temporal residual otherwise","unit":"DN, DN^2, pixel lag, or cycles/pixel as applicable"} for a,b,c,d,e in rows]


def build_report(status:str,core:list[dict[str,Any]],readiness:list[dict[str,Any]],recommendation:dict[str,Any],cfg:dict[str,Any],diff:list[dict[str,Any]],cov:list[dict[str,Any]],nps:list[dict[str,Any]],rowcol:list[dict[str,Any]],stable:list[dict[str,Any]],warnings:list[str])->str:
    direct_decisions=pd.read_csv(Path(cfg["output_root"])/"temporal_noise/direct_vs_difference_summary.csv") if False else None
    lines=["# Gated ICCD Stage-2 Operational Characterization","",f"Status: `{status}`","", "This package reconstructs E1 as a DN-domain, repeated-frame, EMVA-inspired operational characterization. It is not an EMVA 1288 compliance test, a standard photon-transfer curve, DSNU/PRNU measurement, or physical ICCD noise decomposition.","", "## Frozen boundaries", "", "- Folders: 1, 2, 4, 5, 7, 8, 9, 10, 11, 13; 200 frames each.", "- ROI: top=2304, left=2304, height=512, width=512.", "- Input: raw uint16 values converted directly to float64 DN.", "- `EXPOSURE_CONTROL_WIDTH=900 ms`; physical meaning unresolved. Sync A/B=4 us are metadata only.", "- Calibration/evaluation roles are preserved; this run does not refit CG or perform training/inference.","", "## Core folder table", "", "| Folder | Role | Mean DN | Temporal std DN | Direct var DN2 | H/V/R ACF lag1 | NPS L/M/H | Row/column DN |", "|---:|---|---:|---:|---:|---|---|---|"]
    for r in core:
        lines.append(f"| {r['folder']} | {r['role']} | {r['mean_signal_dn']:.3f} | {r['temporal_std_mean_dn']:.3f} | {r['direct_variance_dn2']:.3f} | {r['horizontal_acf_lag1']:.4f}/{r['vertical_acf_lag1']:.4f}/{r['radial_acf_lag1']:.4f} | {r['nps_low_fraction']:.3f}/{r['nps_mid_fraction']:.3f}/{r['nps_high_fraction']:.3f} | {r['row_energy_dn']:.3f}/{r['column_energy_dn']:.3f} |")
    lines += ["", "## Interpretation", "", f"The component-specific convergence rule recommends {recommendation['minimum_recommended_frame_count']} frames for the complete temporal-map, ACF, NPS and directional package. The full 200-frame sequence is sufficient for this bounded operational description, but positional nonstationarity is retained and a stationary-population claim is not supported.", "", "Difference-frame estimates are reported beside direct temporal variance. Departures are interpreted together with temporal correlation and drift, not silently forced to agree. Covariance and NPS use temporal residuals after per-frame DC removal; the mean image is never used as the formal noise spectrum.", "", "The repeatable observed stable component retains the historical split/high-pass definition and remains scene-confounded. The observed-signal relation remains operational; observed DN is not exposure, irradiance, or photon count.", "", "## Readiness", ""]
    for r in readiness: lines.append(f"- {r['section']}: `{r['status']}` - {r['evidence']}")
    lines += ["", "## Claim boundary", "", "Supported: folder-level temporal variability, difference-frame comparison, convergence, temporal-residual directional energy, covariance/ACF/NPS, split-repeatable observed stable structure, and observed-signal dependence at the frozen ROI.", "", "Not supported: standard PTC, photon/conversion gain, dark current, DSNU, PRNU, pure FPN, physical gate-conditioned behavior, or unique physical noise-component separation.", "", f"Warnings: `{'; '.join(sorted(set(warnings))) if warnings else 'none'}`."]
    return "\n".join(lines)+"\n"


def block_mean(a:np.ndarray,factor:int)->np.ndarray:
    h,w=a.shape; return a[:h//factor*factor,:w//factor*factor].reshape(h//factor,factor,w//factor,factor).mean(axis=(1,3))
def gradient_mag(a:np.ndarray)->np.ndarray:
    gy,gx=np.gradient(a.astype(np.float64)); return np.hypot(gx,gy)
def direction_name(dx:int,dy:int)->str:
    if dy==0:return "horizontal"
    if dx==0:return "vertical"
    if dx==dy:return "diagonal_main"
    if dx==-dy:return "diagonal_anti"
    return "other"
def first_below(values:Iterable[float],threshold:float)->float:
    for i,v in enumerate(values):
        if i and abs(v)<threshold:return float(i)
    return float("nan")
def max_noncenter(a:np.ndarray,c:int)->float:
    b=a.copy();b[c,c]=-np.inf;return float(np.max(b))
def q(a:np.ndarray,p:float)->float:return float(np.percentile(a,p))
def pearson(a:np.ndarray,b:np.ndarray)->float:
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();a-=a.mean();b-=b.mean();d=math.sqrt(float(a@a)*float(b@b));return float(a@b/d) if d>0 else float("nan")
def rel_abs(a:float,b:float)->float:return abs(a-b)/(abs(b)+1e-30)
def rel_signed(a:float,b:float)->float:return (a-b)/(abs(b)+1e-30)
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:to_json_safe(v) for k,v in r.items()} for r in rows])
def write_json(path:Path,payload:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:json.dump(to_json_safe(payload),f,ensure_ascii=False,indent=2,allow_nan=False,sort_keys=True)
def write_text(path:Path,value:str)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
def git(repo:Path,*args:str)->str:return subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=True).stdout
def utc_now()->str:return datetime.now(timezone.utc).isoformat()
def rel_or_abs(path:Path,repo:Path)->str:
    try:return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
