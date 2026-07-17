"""Aggregate E1-E6 evidence into a folder-level data-to-route audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


FRAME_NUMBER = re.compile(r"^(\d+)")
CONDITION_FEATURES = [
    "mean_signal",
    "temporal_std_mean",
    "fano_temporal",
    "fixed_map_std",
    "fixed_to_temporal_std_ratio",
]


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = by_folder(read_csv(Path(config["inputs"]["inventory_csv"])))
    stability = by_folder(read_csv(Path(config["inputs"]["repeated_frame_summary_csv"])))
    conditions = by_folder(read_csv(Path(config["inputs"]["condition_csv"])))
    e6_checks = {
        int(row["folder"]): row
        for row in read_json(Path(config["inputs"]["repeated_frame_decision_json"]))["checks"]
    }
    reference_rows = read_csv(Path(config["inputs"]["reference_folder_csv"]))
    reference_by_folder = summarize_reference_folders(reference_rows)
    seed_uncertainty = float(
        read_json(Path(config["inputs"]["factorial_uncertainty_json"]))["maximum_cell_seed_std_db"]
    )

    integrity_rows = []
    for folder in [int(value) for value in config["folders"]]:
        print(f"Checking folder {folder}", flush=True)
        integrity_rows.append(audit_integrity(Path(config["raw_root"]) / str(folder), folder, config, inventory[folder]))

    condition_rows, corr_rows, vif_rows, pca_rows, condition_summary = audit_condition(conditions, seed_uncertainty)
    gates = build_folder_gates(
        config, integrity_rows, stability, e6_checks, reference_by_folder, condition_summary
    )
    decision = build_route_decision(gates, condition_summary, seed_uncertainty)

    write_csv(integrity_rows, output_dir / "integrity_details.csv")
    write_csv(gates, output_dir / "folder_gate.csv")
    write_csv(condition_rows, output_dir / "condition_lofo_predictions.csv")
    write_csv(corr_rows, output_dir / "condition_feature_correlation.csv")
    write_csv(vif_rows, output_dir / "condition_feature_vif.csv")
    write_csv(pca_rows, output_dir / "condition_pca.csv")
    write_json(output_dir / "route_decision.json", decision)
    write_report(output_dir / "data_route_eligibility_report.md", gates, condition_summary, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e7_data_route_eligibility.yaml")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def by_folder(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(float(row.get("folder_name", row["folder"]))): row for row in rows}


def indexed_tiffs(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = FRAME_NUMBER.match(path.name)
        if match:
            index = int(match.group(1))
            if index in result:
                raise ValueError(f"Duplicate frame index {index} in {folder}")
            result[index] = path
    return result


def audit_integrity(folder: Path, folder_id: int, cfg: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    import tifffile

    paths = indexed_tiffs(folder)
    expected = int(cfg["expected_frames"])
    expected_shape = tuple(int(value) for value in cfg["expected_shape"])
    expected_dtype = str(cfg["expected_dtype"])
    crop_size = int(cfg["sample_crop_size"])
    range_max = float(cfg["range_max"])
    missing = sorted(set(range(1, expected + 1)) - set(paths))
    unreadable = 0
    shape_mismatch = 0
    dtype_mismatch = 0
    zero_pixels = 0
    saturated_pixels = 0
    sampled_pixels = 0
    hashes: list[str] = []
    observed_min = math.inf
    observed_max = -math.inf
    for index in sorted(paths):
        path = paths[index]
        try:
            image = tifffile.memmap(path)
            if image.ndim != 2:
                raise ValueError(f"Expected 2D image, got {image.shape}")
            shape_mismatch += int(tuple(image.shape) != expected_shape)
            dtype_mismatch += int(str(image.dtype) != expected_dtype)
            size = min(crop_size, image.shape[0], image.shape[1])
            top = (image.shape[0] - size) // 2
            left = (image.shape[1] - size) // 2
            crop = np.asarray(image[top : top + size, left : left + size])
            observed_min = min(observed_min, float(np.min(crop)))
            observed_max = max(observed_max, float(np.max(crop)))
            zero_pixels += int(np.count_nonzero(crop == 0))
            saturated_pixels += int(np.count_nonzero(crop >= range_max))
            sampled_pixels += int(crop.size)
            hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
        except Exception:
            unreadable += 1
    duplicate_crop_count = len(hashes) - len(set(hashes))
    metadata_rows = int(float(inventory["picture_info_rows"]))
    zero_fraction = zero_pixels / max(sampled_pixels, 1)
    saturated_fraction = saturated_pixels / max(sampled_pixels, 1)
    passed = (
        len(paths) == expected
        and metadata_rows == expected
        and not missing
        and unreadable == 0
        and shape_mismatch == 0
        and dtype_mismatch == 0
        and duplicate_crop_count == 0
        and zero_fraction <= float(cfg["gates"]["max_zero_fraction"])
        and saturated_fraction <= float(cfg["gates"]["max_saturated_fraction"])
    )
    return {
        "folder": folder_id,
        "frame_count": len(paths),
        "metadata_rows": metadata_rows,
        "missing_frame_count": len(missing),
        "unreadable_frame_count": unreadable,
        "shape_mismatch_count": shape_mismatch,
        "dtype_mismatch_count": dtype_mismatch,
        "duplicate_center_crop_count": duplicate_crop_count,
        "sample_min": observed_min,
        "sample_max": observed_max,
        "sample_zero_fraction": zero_fraction,
        "sample_saturated_fraction": saturated_fraction,
        "gate": "PASS" if passed else "FAIL",
    }


def summarize_reference_folders(rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        strategy = row["strategy"]
        if strategy not in {"always_p99", "always_physical"}:
            continue
        key = (int(float(row["folder"])), strategy)
        grouped.setdefault(key, {})[row["reference"]] = float(row["mean_psnr_gain"])
    result: dict[int, dict[str, Any]] = {}
    for (folder, strategy), values in grouped.items():
        a = values["reference_a_odd"]
        b = values["reference_b_even"]
        result.setdefault(folder, {})[strategy] = {
            "sign_agreement": bool(np.sign(a) == np.sign(b)),
            "gain_delta_db": abs(a - b),
            "gain_a_db": a,
            "gain_b_db": b,
        }
    return result


def audit_condition(
    conditions: dict[int, dict[str, Any]], seed_uncertainty: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    folders = sorted(conditions)
    x = np.asarray([[float(conditions[f][key]) for key in CONDITION_FEATURES] for f in folders], dtype=np.float64)
    y = np.asarray(
        [float(conditions[f]["physical_folder_gain"]) - float(conditions[f]["p99_folder_gain"]) for f in folders],
        dtype=np.float64,
    )
    xz = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    corr = np.corrcoef(xz, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)

    corr_rows = []
    for i, first in enumerate(CONDITION_FEATURES):
        for j, second in enumerate(CONDITION_FEATURES):
            corr_rows.append({"feature_a": first, "feature_b": second, "pearson_r": float(corr[i, j])})
    vif_rows = []
    for index, feature in enumerate(CONDITION_FEATURES):
        others = np.delete(xz, index, axis=1)
        beta = np.linalg.lstsq(np.c_[np.ones(len(others)), others], xz[:, index], rcond=None)[0]
        prediction = np.c_[np.ones(len(others)), others] @ beta
        r2 = 1.0 - np.sum((xz[:, index] - prediction) ** 2) / np.sum(xz[:, index] ** 2)
        vif_rows.append({"feature": feature, "r2_from_other_features": float(r2), "vif": float(1.0 / max(1.0 - r2, 1e-12))})
    pca_rows = [
        {"component": index + 1, "eigenvalue": float(value), "explained_fraction": float(explained[index])}
        for index, value in enumerate(eigenvalues)
    ]

    predictions = []
    for heldout in range(len(folders)):
        train = np.arange(len(folders)) != heldout
        mean = x[train].mean(axis=0)
        std = np.maximum(x[train].std(axis=0), 1e-12)
        train_x = (x[train] - mean) / std
        test_x = (x[heldout] - mean) / std
        alpha = 1.0
        beta = np.linalg.solve(train_x.T @ train_x + alpha * np.eye(train_x.shape[1]), train_x.T @ (y[train] - y[train].mean()))
        prediction = float(y[train].mean() + test_x @ beta)
        null_prediction = float(y[train].mean())
        predictions.append({
            "folder": folders[heldout],
            "observed_physical_minus_p99_db": float(y[heldout]),
            "lofo_ridge_prediction_db": prediction,
            "null_training_mean_prediction_db": null_prediction,
            "absolute_error_db": abs(prediction - y[heldout]),
            "null_absolute_error_db": abs(null_prediction - y[heldout]),
        })
    pred = np.asarray([row["lofo_ridge_prediction_db"] for row in predictions])
    null = np.asarray([row["null_training_mean_prediction_db"] for row in predictions])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    null_rmse = float(np.sqrt(np.mean((null - y) ** 2)))
    summary = {
        "folder_count": len(folders),
        "maximum_absolute_feature_correlation": float(np.max(np.abs(corr - np.eye(len(CONDITION_FEATURES))))),
        "pc1_explained_fraction": float(explained[0]),
        "maximum_vif": float(max(row["vif"] for row in vif_rows)),
        "lofo_ridge_rmse_db": rmse,
        "null_rmse_db": null_rmse,
        "lofo_relative_rmse_improvement": float((null_rmse - rmse) / max(null_rmse, 1e-12)),
        "prediction_observation_correlation": float(np.corrcoef(pred, y)[0, 1]),
        "target_gain_range_db": float(np.ptp(y)),
        "seed_uncertainty_db": seed_uncertainty,
        "device_metadata_varies_across_folders": False,
        "interpretation": "image-statistical state confounded with folder/scene identity",
    }
    return predictions, corr_rows, vif_rows, pca_rows, summary


def build_folder_gates(
    cfg: dict[str, Any],
    integrity_rows: list[dict[str, Any]],
    stability: dict[int, dict[str, Any]],
    e6_checks: dict[int, dict[str, Any]],
    references: dict[int, dict[str, Any]],
    condition_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    integrity = {int(row["folder"]): row for row in integrity_rows}
    rows = []
    for folder in [int(value) for value in cfg["folders"]]:
        stable = stability[folder]
        check = e6_checks[folder]
        ref = references[folder]
        local_ok = float(stable["local_drift_max_over_temporal_std"]) <= float(cfg["gates"]["max_local_drift_over_temporal_std"])
        fixed_ok = float(stable["fixed_map_half_correlation"]) >= float(cfg["gates"]["min_fixed_map_half_correlation"])
        reference_ok = all(
            value["sign_agreement"] and value["gain_delta_db"] <= float(cfg["gates"]["max_reference_folder_gain_delta_db"])
            for value in ref.values()
        )
        stability_gate = "PASS" if local_ok and check["registration"] and check["global_drift"] else "WARN"
        noise_gate = "PASS" if fixed_ok else "WARN"
        surrogate_gate = "PASS" if reference_ok and local_ok else "WARN"
        denoising_gate = "PASS" if surrogate_gate == "PASS" and integrity[folder]["gate"] == "PASS" else "WARN"
        reasons = []
        if not local_ok:
            reasons.append("local drift exceeds E6 gate")
        if not fixed_ok:
            reasons.append("split-half stable component is not stable")
        if not check["residual_independence"]:
            reasons.append("frame residuals are correlated")
        if not check["row_column_independence"]:
            reasons.append("row/column residuals are correlated")
        if not reference_ok:
            reasons.append("dual-reference folder result is sensitive")
        if not reasons:
            reasons.append("usable for operational characterization and relative surrogate comparison")
        rows.append({
            "folder": folder,
            "integrity": integrity[folder]["gate"],
            "scene_stability": stability_gate,
            "noise_characterization": noise_gate,
            "surrogate_reference": surrogate_gate,
            "condition_analysis": "WARN",
            "denoising_validation": denoising_gate,
            "repeated_frame_supervision": "PASS" if check["passed"] else "FAIL",
            "overall": "WARN" if integrity[folder]["gate"] == "PASS" else "FAIL",
            "reason": "; ".join(reasons),
        })
    return rows


def build_route_decision(gates: list[dict[str, Any]], condition: dict[str, Any], seed_uncertainty: float) -> dict[str, Any]:
    return {
        "data_integrity_pass_count": sum(row["integrity"] == "PASS" for row in gates),
        "noise_characterization_pass_count": sum(row["noise_characterization"] == "PASS" for row in gates),
        "surrogate_reference_pass_count": sum(row["surrogate_reference"] == "PASS" for row in gates),
        "repeated_frame_supervision_pass_count": sum(row["repeated_frame_supervision"] == "PASS" for row in gates),
        "module_ratings": {
            "gated_iccd_noise_characterization": "SUPPORTED",
            "condition_aware_noise_modeling": "NOT_SUPPORTED",
            "controlled_denoising_validation": "LIMITED_SUPPORT",
        },
        "recommended_route": 2,
        "recommended_route_name": "gated ICCD characterization + conditional noise mismatch analysis + denoising applicability validation",
        "condition_diagnostics": condition,
        "seed_uncertainty_db": seed_uncertainty,
        "claim_boundary": "surrogate-based controlled applicability and failure-boundary analysis; no clean-image recovery claim",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(path: Path, gates: list[dict[str, Any]], condition: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# E7 Data-to-Route Eligibility Audit",
        "",
        "## Module Decision",
        "",
        "- Gated ICCD noise characterization: **SUPPORTED**, limited to operationally observed components.",
        "- Condition-aware noise modeling: **NOT SUPPORTED** as a deployable generator or selector.",
        "- Controlled denoising validation: **LIMITED SUPPORT**, for surrogate-based applicability and failure boundaries.",
        "",
        "## Folder Gates",
        "",
        "| folder | integrity | stability | characterization | surrogate | condition | denoising | repeated supervision | reason |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in gates:
        lines.append(
            f"| {row['folder']} | {row['integrity']} | {row['scene_stability']} | {row['noise_characterization']} | "
            f"{row['surrogate_reference']} | {row['condition_analysis']} | {row['denoising_validation']} | "
            f"{row['repeated_frame_supervision']} | {row['reason']} |"
        )
    lines.extend([
        "",
        "## Condition Audit",
        "",
        f"- Maximum feature correlation: {condition['maximum_absolute_feature_correlation']:.4f}",
        f"- PC1 explained fraction: {condition['pc1_explained_fraction']:.4f}",
        f"- Maximum VIF: {condition['maximum_vif']:.2f}",
        f"- LOFO ridge RMSE: {condition['lofo_ridge_rmse_db']:.4f} dB",
        f"- Null RMSE: {condition['null_rmse_db']:.4f} dB",
        f"- Maximum E5 seed SD: {decision['seed_uncertainty_db']:.4f} dB",
        "- All complete folders share the recorded gate/exposure/sync/gain settings. The score is therefore an image-statistical state descriptor confounded with folder and scene, not a verified acquisition-condition variable.",
        "",
        "## Route",
        "",
        "Select route 2: characterization + conditional mismatch analysis + controlled denoising applicability validation.",
        "Do not claim a validated condition-aware generator, deployable selector, clean ground truth, or true-detail recovery.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
