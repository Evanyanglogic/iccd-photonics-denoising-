"""Audit repeated ICCD frames before any real-domain denoiser training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


FRAME_NUMBER = re.compile(r"^(\d+)")


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    data_cfg = config["data"]
    audit_cfg = config["audit"]
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    brightness_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for folder in [int(value) for value in data_cfg["folders"]]:
        print(f"Auditing folder {folder}", flush=True)
        paths = indexed_tiffs(Path(data_cfg["raw_root"]) / str(folder))
        expected = int(data_cfg["expected_frames"])
        if len(paths) != expected:
            raise ValueError(f"Folder {folder}: expected {expected} indexed TIFFs, found {len(paths)}")
        frames = np.stack(
            [read_center_crop(paths[index], int(data_cfg["crop_size"])) for index in range(1, expected + 1)]
        ).astype(np.float32)
        results = audit_folder(folder, frames, audit_cfg)
        folder_rows.append(results["folder"])
        lag_rows.extend(results["lags"])
        brightness_rows.extend(results["brightness"])
        target_rows.extend(results["targets"])
        trace_rows.extend(results["traces"])
        difference_rows.extend(results["differences"])

    checks, decision = decide(folder_rows, target_rows, config)
    write_csv(folder_rows, output_dir / "folder_stability_summary.csv")
    write_csv(lag_rows, output_dir / "correlation_decay.csv")
    write_csv(brightness_rows, output_dir / "brightness_residual_correlation.csv")
    write_csv(target_rows, output_dir / "target_candidate_summary.csv")
    write_csv(trace_rows, output_dir / "frame_traces.csv")
    write_csv(difference_rows, output_dir / "frame_difference_statistics.csv")
    write_json(output_dir / "audit_decision.json", {"checks": checks, **decision})
    save_plots(folder_rows, lag_rows, trace_rows, target_rows, output_dir)
    write_report(output_dir / "repeated_frame_audit_report.md", folder_rows, target_rows, checks, decision, config)
    print(json.dumps(decision, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e6_repeated_frame_audit.yaml")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def indexed_tiffs(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = FRAME_NUMBER.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def read_center_crop(path: Path, crop_size: int) -> np.ndarray:
    import tifffile

    try:
        image = tifffile.memmap(path)
    except Exception:
        image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale TIFF at {path}, got {image.shape}")
    size = min(crop_size, image.shape[0], image.shape[1])
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return np.asarray(image[top : top + size, left : left + size], dtype=np.float32)


def audit_folder(folder: int, frames: np.ndarray, cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    from scipy.ndimage import gaussian_filter
    from scipy.stats import kurtosis, skew
    from skimage.registration import phase_cross_correlation

    count, height, width = frames.shape
    indices = np.arange(1, count + 1, dtype=np.float64)
    frame_means = frames.mean(axis=(1, 2), dtype=np.float64)
    frame_stds = frames.std(axis=(1, 2), dtype=np.float64)
    temporal_mean = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    temporal_std_map = frames.std(axis=0, dtype=np.float64).astype(np.float32)
    temporal_std = float(np.mean(temporal_std_map))
    residuals = frames - temporal_mean[None]
    residuals_demean = residuals - residuals.mean(axis=(1, 2), keepdims=True)

    mean_slope = linear_slope(indices, frame_means)
    std_slope = linear_slope(indices, frame_stds)
    first_last_mean_shift = float(np.mean(frame_means[-50:]) - np.mean(frame_means[:50]))
    first_last_std_shift = float(np.mean(frame_stds[-50:]) - np.mean(frame_stds[:50]))
    local_drift = local_drift_summary(frames, int(cfg["local_grid"]), temporal_std)

    group_size = int(cfg["registration_group_size"])
    group_means = [frames[start : start + group_size].mean(axis=0) for start in range(0, count, group_size)]
    registration_reference = np.mean(group_means, axis=0)
    shifts = []
    for group in group_means:
        shift, _, _ = phase_cross_correlation(
            gaussian_filter(registration_reference, 1.0),
            gaussian_filter(group, 1.0),
            upsample_factor=int(cfg["registration_upsample_factor"]),
        )
        shifts.append(float(np.linalg.norm(shift)))

    fixed_first = frames[: count // 2].mean(axis=0)
    fixed_second = frames[count // 2 :].mean(axis=0)
    fixed_first -= fixed_first.mean()
    fixed_second -= fixed_second.mean()
    fixed_corr = correlation(fixed_first, fixed_second, stride=int(cfg["correlation_stride"]))
    fixed_rmse = float(np.sqrt(np.mean((fixed_first - fixed_second) ** 2)))

    lag_rows: list[dict[str, Any]] = []
    brightness_rows: list[dict[str, Any]] = []
    stride = int(cfg["correlation_stride"])
    pair_limit = int(cfg["correlation_pairs_per_lag"])
    brightness_edges = np.quantile(temporal_mean, np.linspace(0.0, 1.0, int(cfg["brightness_bins"]) + 1))
    highpass = np.empty_like(residuals_demean)
    for index in range(count):
        highpass[index] = residuals_demean[index] - gaussian_filter(residuals_demean[index], float(cfg["highpass_sigma"]))
    row_profiles = residuals_demean.mean(axis=2)
    column_profiles = residuals_demean.mean(axis=1)
    for lag in [int(value) for value in cfg["lags"]]:
        starts = np.linspace(0, count - lag - 1, min(pair_limit, count - lag), dtype=int)
        starts = np.unique(starts)
        raw_corrs, hp_corrs, row_corrs, col_corrs = [], [], [], []
        for start in starts:
            other = start + lag
            raw_corrs.append(correlation(residuals_demean[start], residuals_demean[other], stride))
            hp_corrs.append(correlation(highpass[start], highpass[other], stride))
            row_corrs.append(correlation(row_profiles[start], row_profiles[other], 1))
            col_corrs.append(correlation(column_profiles[start], column_profiles[other], 1))
        lag_rows.append(
            {
                "folder": folder,
                "lag": lag,
                "pair_count": len(starts),
                "residual_correlation_mean": finite_mean(raw_corrs),
                "residual_correlation_max_abs": finite_max_abs(raw_corrs),
                "highpass_correlation_mean": finite_mean(hp_corrs),
                "highpass_correlation_max_abs": finite_max_abs(hp_corrs),
                "row_correlation_mean": finite_mean(row_corrs),
                "column_correlation_mean": finite_mean(col_corrs),
            }
        )
        if lag in {1, 8, 32, 100}:
            for bin_index in range(len(brightness_edges) - 1):
                lo, hi = brightness_edges[bin_index], brightness_edges[bin_index + 1]
                mask = (temporal_mean >= lo) & (temporal_mean <= hi if bin_index == len(brightness_edges) - 2 else temporal_mean < hi)
                mask_sample = mask[::stride, ::stride]
                values = []
                for start in starts:
                    a = residuals_demean[start, ::stride, ::stride][mask_sample]
                    b = residuals_demean[start + lag, ::stride, ::stride][mask_sample]
                    values.append(vector_correlation(a, b))
                brightness_rows.append(
                    {
                        "folder": folder,
                        "lag": lag,
                        "brightness_bin": bin_index,
                        "brightness_low": float(lo),
                        "brightness_high": float(hi),
                        "pixel_count": int(np.count_nonzero(mask)),
                        "residual_correlation_mean": finite_mean(values),
                        "residual_correlation_max_abs": finite_max_abs(values),
                    }
                )

    difference_rows = []
    for lag in (1, 8, 32, 100):
        differences = frames[lag:] - frames[:-lag]
        sample = differences[:, ::stride, ::stride].reshape(-1)
        difference_rows.append(
            {
                "folder": folder,
                "lag": lag,
                "mean": float(np.mean(sample)),
                "std": float(np.std(sample)),
                "skewness": float(skew(sample, bias=False)),
                "excess_kurtosis": float(kurtosis(sample, fisher=True, bias=False)),
            }
        )

    targets = target_candidates(folder, frames, fixed_corr)
    lag1 = next(row for row in lag_rows if row["lag"] == 1)
    lag32 = next(row for row in lag_rows if row["lag"] == 32)
    folder_row = {
        "folder": folder,
        "frame_count": count,
        "frame_mean_mean": float(np.mean(frame_means)),
        "frame_mean_std": float(np.std(frame_means, ddof=1)),
        "frame_std_mean": float(np.mean(frame_stds)),
        "frame_std_std": float(np.std(frame_stds, ddof=1)),
        "mean_drift_dn_per_frame": mean_slope,
        "std_drift_dn_per_frame": std_slope,
        "first_last_mean_shift_dn": first_last_mean_shift,
        "first_last_mean_shift_over_temporal_std": abs(first_last_mean_shift) / max(temporal_std, 1e-12),
        "first_last_std_shift_dn": first_last_std_shift,
        "local_drift_max_over_temporal_std": local_drift["max_abs_shift_over_temporal_std"],
        "registration_shift_mean_px": float(np.mean(shifts)),
        "registration_shift_p95_px": float(np.percentile(shifts, 95)),
        "registration_shift_max_px": float(np.max(shifts)),
        "adjacent_frame_correlation": mean_frame_correlation(frames, lag=1, stride=stride),
        "far_frame_correlation": mean_frame_correlation(frames, lag=100, stride=stride),
        "fixed_map_half_correlation": fixed_corr,
        "fixed_map_half_rmse_dn": fixed_rmse,
        "temporal_std_mean_dn": temporal_std,
        "lag1_residual_correlation": lag1["residual_correlation_mean"],
        "lag32_residual_correlation": lag32["residual_correlation_mean"],
        "lag1_highpass_correlation": lag1["highpass_correlation_mean"],
        "lag1_row_correlation": lag1["row_correlation_mean"],
        "lag1_column_correlation": lag1["column_correlation_mean"],
    }
    traces = [
        {"folder": folder, "frame_index": int(index), "frame_mean": float(mean), "frame_std": float(std)}
        for index, mean, std in zip(indices, frame_means, frame_stds)
    ]
    return {
        "folder": folder_row,
        "lags": lag_rows,
        "brightness": brightness_rows,
        "targets": targets,
        "traces": traces,
        "differences": difference_rows,
    }


def target_candidates(folder: int, frames: np.ndarray, fixed_corr: float) -> list[dict[str, Any]]:
    definitions = [
        ("single_to_single", 1, 1, True, "single-frame"),
        ("single_to_8mean", 1, 8, True, "single-frame"),
        ("single_to_16mean", 1, 16, True, "single-frame"),
        ("8mean_to_8mean", 8, 8, True, "eight-frame mean"),
        ("odd_mean_to_even_mean", 100, 100, False, "temporal mean"),
    ]
    rows = []
    for name, input_count, target_count, blockwise, inference_input in definitions:
        if blockwise:
            block = input_count + target_count
            pair_count = len(frames) // block
            input_means, target_means, independent_references = [], [], []
            for pair_index in range(pair_count):
                start = pair_index * block
                stop = start + block
                input_means.append(frames[start : start + input_count].mean(axis=0))
                target_means.append(frames[start + input_count : stop].mean(axis=0))
                complement = np.concatenate((frames[:start], frames[stop:]), axis=0)
                independent_references.append(complement.mean(axis=0))
            target_noise = [float(np.std(target - reference)) for target, reference in zip(target_means, independent_references)]
            input_noise = [float(np.std(source - reference)) for source, reference in zip(input_means, independent_references)]
            bias = [float(np.mean(target - reference)) for target, reference in zip(target_means, independent_references)]
        else:
            pair_count = 1
            input_means = [frames[0::2].mean(axis=0)]
            target_means = [frames[1::2].mean(axis=0)]
            split_difference = target_means[0] - input_means[0]
            symmetric_std = float(np.std(split_difference) / math.sqrt(2.0))
            target_noise = [symmetric_std]
            input_noise = [symmetric_std]
            bias = [float(np.mean(split_difference) / 2.0)]
        pair_difference = [float(np.std(source - target)) for source, target in zip(input_means, target_means)]
        rows.append(
            {
                "folder": folder,
                "candidate": name,
                "pair_count_nonreused": pair_count,
                "input_frame_count": input_count,
                "target_frame_count": target_count,
                "input_residual_std_dn": float(np.mean(input_noise)),
                "target_residual_std_dn": float(np.mean(target_noise)),
                "input_target_difference_std_dn": float(np.mean(pair_difference)),
                "target_bias_dn": float(np.mean(bias)),
                "fixed_pattern_retained": True,
                "fixed_map_half_correlation": fixed_corr,
                "inference_input": inference_input,
                "oversmoothing_risk": "high" if target_count >= 16 else ("moderate" if target_count >= 8 else "low"),
            }
        )
    rows.append(
        {
            "folder": folder,
            "candidate": "synthetic_noisy_to_temporal_mean",
            "pair_count_nonreused": 0,
            "input_frame_count": 1,
            "target_frame_count": 100,
            "input_residual_std_dn": float("nan"),
            "target_residual_std_dn": float("nan"),
            "input_target_difference_std_dn": float("nan"),
            "target_bias_dn": float("nan"),
            "fixed_pattern_retained": "domain_mismatch",
            "fixed_map_half_correlation": fixed_corr,
            "inference_input": "synthetic content",
            "oversmoothing_risk": "not_pairable",
        }
    )
    return rows


def local_drift_summary(frames: np.ndarray, grid: int, temporal_std: float) -> dict[str, float]:
    count, height, width = frames.shape
    shifts = []
    for row in range(grid):
        for col in range(grid):
            tile = frames[:, row * height // grid : (row + 1) * height // grid, col * width // grid : (col + 1) * width // grid]
            means = tile.mean(axis=(1, 2))
            shifts.append(float(np.mean(means[-50:]) - np.mean(means[:50])))
    return {"max_abs_shift_over_temporal_std": max(abs(value) for value in shifts) / max(temporal_std, 1e-12)}


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def mean_frame_correlation(frames: np.ndarray, lag: int, stride: int) -> float:
    starts = np.linspace(0, len(frames) - lag - 1, min(24, len(frames) - lag), dtype=int)
    return finite_mean([correlation(frames[start], frames[start + lag], stride) for start in np.unique(starts)])


def correlation(a: np.ndarray, b: np.ndarray, stride: int) -> float:
    if a.ndim == 1 and b.ndim == 1:
        return vector_correlation(a[::stride], b[::stride])
    if a.ndim == 2 and b.ndim == 2:
        return vector_correlation(a[::stride, ::stride].reshape(-1), b[::stride, ::stride].reshape(-1))
    raise ValueError(f"Correlation dimensionality mismatch: {a.shape} vs {b.shape}")


def vector_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array))


def finite_max_abs(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmax(np.abs(array)))


def decide(
    folders: list[dict[str, Any]], targets: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limits = config["decision"]
    checks = []
    passing_folders = 0
    for row in folders:
        items = {
            "registration": float(row["registration_shift_p95_px"]) <= float(limits["max_registration_shift_p95_px"]),
            "global_drift": float(row["first_last_mean_shift_over_temporal_std"]) <= float(limits["max_first_last_mean_shift_over_temporal_std"]),
            "local_drift": float(row["local_drift_max_over_temporal_std"]) <= float(limits["max_local_drift_over_temporal_std"]),
            "residual_independence": abs(float(row["lag1_residual_correlation"])) <= float(limits["max_abs_residual_correlation"]),
            "highpass_independence": abs(float(row["lag1_highpass_correlation"])) <= float(limits["max_abs_highpass_correlation"]),
            "row_column_independence": max(abs(float(row["lag1_row_correlation"])), abs(float(row["lag1_column_correlation"]))) <= float(limits["max_abs_row_or_column_correlation"]),
            "fixed_map_stable": float(row["fixed_map_half_correlation"]) >= float(limits["min_fixed_map_half_correlation"]),
        }
        passed = all(items.values())
        passing_folders += int(passed)
        checks.append({"folder": int(row["folder"]), **items, "passed": passed})
    reductions = []
    for folder in folders:
        folder_id = int(folder["folder"])
        single = next(row for row in targets if int(row["folder"]) == folder_id and row["candidate"] == "single_to_single")
        mean8 = next(row for row in targets if int(row["folder"]) == folder_id and row["candidate"] == "single_to_8mean")
        reductions.append(float(single["target_residual_std_dn"]) / max(float(mean8["target_residual_std_dn"]), 1e-12))
    target_reduction = float(np.median(reductions))
    eligible = passing_folders >= int(limits["min_passing_folders"]) and target_reduction >= float(limits["min_target_noise_reduction_8frame"])
    ranked = sorted(folders, key=lambda row: (float(row["temporal_std_mean_dn"]), int(row["folder"])))
    representative = [int(ranked[0]["folder"]), int(ranked[(len(ranked) - 1) // 2]["folder"]), int(ranked[-1]["folder"])]
    return checks, {
        "status": "ELIGIBLE_REAL_REPEATED_FRAME" if eligible else "STOP_REACQUIRE",
        "passing_folder_count": passing_folders,
        "folder_count": len(folders),
        "median_8frame_target_noise_reduction": target_reduction,
        "representative_folders_low_mid_high": representative,
        "selected_protocol_if_eligible": "B_SINGLE_TO_DISJOINT_8FRAME_MEAN" if eligible else "E_REACQUIRE",
    }


def save_plots(
    folders: list[dict[str, Any]], lags: list[dict[str, Any]], traces: list[dict[str, Any]],
    targets: list[dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for axis, folder in zip(axes.flat, sorted({int(row["folder"]) for row in traces})):
        rows = [row for row in traces if int(row["folder"]) == folder]
        axis.plot([row["frame_index"] for row in rows], [row["frame_mean"] for row in rows], linewidth=1)
        axis.set_title(f"Folder {folder}")
        axis.set_xlabel("frame")
        axis.set_ylabel("mean DN")
    fig.savefig(plot_dir / "frame_mean_traces.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for folder in sorted({int(row["folder"]) for row in lags}):
        rows = [row for row in lags if int(row["folder"]) == folder]
        axis.plot([row["lag"] for row in rows], [row["residual_correlation_mean"] for row in rows], marker="o", label=str(folder))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("frame lag")
    axis.set_ylabel("residual correlation")
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(plot_dir / "residual_correlation_decay.png", dpi=140)
    plt.close(fig)

    names = ["single_to_single", "single_to_8mean", "single_to_16mean", "8mean_to_8mean", "odd_mean_to_even_mean"]
    medians = [np.median([float(row["target_residual_std_dn"]) for row in targets if row["candidate"] == name]) for name in names]
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.bar(names, medians)
    axis.set_ylabel("median target residual std (DN)")
    axis.tick_params(axis="x", rotation=25)
    fig.savefig(plot_dir / "target_noise_by_protocol.png", dpi=140)
    plt.close(fig)


def write_report(
    path: Path, folders: list[dict[str, Any]], targets: list[dict[str, Any]], checks: list[dict[str, Any]],
    decision: dict[str, Any], config: dict[str, Any]
) -> None:
    lines = [
        "# Repeated-Frame Supervision Audit", "",
        f"Decision: **{decision['status']}**", "",
        "## Folder Evidence", "",
        "| folder | temporal std | drift/std | local drift/std | shift p95 | residual corr | highpass corr | row corr | col corr | fixed-map corr | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_check = {int(row["folder"]): row for row in checks}
    for row in folders:
        check = by_check[int(row["folder"])]
        lines.append(
            f"| {int(row['folder'])} | {row['temporal_std_mean_dn']:.3f} | {row['first_last_mean_shift_over_temporal_std']:.3f} | "
            f"{row['local_drift_max_over_temporal_std']:.3f} | {row['registration_shift_p95_px']:.3f} | "
            f"{row['lag1_residual_correlation']:.4f} | {row['lag1_highpass_correlation']:.4f} | "
            f"{row['lag1_row_correlation']:.4f} | {row['lag1_column_correlation']:.4f} | "
            f"{row['fixed_map_half_correlation']:.5f} | {'yes' if check['passed'] else 'no'} |"
        )
    lines.extend([
        "", "## Target Evidence", "",
        f"- Median 8-frame target noise reduction: {decision['median_8frame_target_noise_reduction']:.3f}x",
        f"- Representative low/mid/high folders: {decision['representative_folders_low_mid_high']}",
        f"- Selected protocol: `{decision['selected_protocol_if_eligible']}`", "",
        "## Interpretation Boundary", "",
        "- Temporal means are surrogate expectations, not clean ground truth.",
        "- Repeated-frame supervision can suppress conditionally independent temporal noise but cannot identify static fixed-pattern bias shared by input and target.",
        "- A stable split-half fixed map can be learned or preserved as scene content; it is not evidence that fixed-pattern noise has been removed.",
        "- Registration is estimated from 8-frame means and may be anchored partly by fixed-pattern structure.",
        "- Synthetic images are not content-paired with real temporal means and therefore cannot form a valid direct supervision pair.", "",
        "## Preregistered Thresholds", "",
        "```json", json.dumps(config["decision"], indent=2), "```", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
