"""Validate strength/structure decoupling in the 2x2 synthetic noise dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    root = Path(args.input_root or config["output_root"])
    validation = config["validation"]
    variants = list(config["construction"]["variants"])
    construction_rows = read_csv(root / "construction_metrics.csv")
    construction_by_key = {(row["variant"], row["pair_key"]): row for row in construction_rows}

    pair_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    curve_accumulators: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    autocorr_accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
    radius = int(validation["autocorrelation_radius"])
    bins = int(validation["intensity_bins"])
    max_pairs = args.max_pairs

    for variant in variants:
        pairs = read_csv(root / variant / "pairs.csv")
        if max_pairs > 0:
            pairs = pairs[:max_pairs]
        for pair in pairs:
            clean = load_tiff(pair["clean_path"], float(config["source"]["range_max"]))
            noisy = load_tiff(pair["noisy_path"], float(config["source"]["range_max"]))
            residual = noisy - clean
            std = safe_std(residual)
            normalized = (residual - float(np.mean(residual))) / std
            radial, horizontal, vertical = psd_profiles(normalized)
            autocorr = autocorrelation_crop(normalized, radius)
            curve_accumulators[(variant, "radial_psd")].append(normalize_curve(radial))
            curve_accumulators[(variant, "horizontal_psd")].append(normalize_curve(horizontal))
            curve_accumulators[(variant, "vertical_psd")].append(normalize_curve(vertical))
            autocorr_accumulators[variant].append(autocorr)
            construction = construction_by_key[(variant, pair["pair_key"])]
            pair_rows.append(summarize_pair(variant, pair["pair_key"], clean, residual, construction, validation))
            conditional_rows.extend(conditional_variance(variant, pair["pair_key"], clean, residual, bins))

    summary_rows = summarize_variants(pair_rows)
    curve_rows, mean_curves = summarize_curves(curve_accumulators)
    autocorr_rows, mean_autocorr = summarize_autocorr(autocorr_accumulators)
    checks = decoupling_checks(summary_rows, mean_curves, mean_autocorr, config)
    decision = {
        "status": "GO_TO_TRAIN" if all(checks.values()) else "STOP_AND_REPAIR",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "pair_count_per_variant": len(pair_rows) // len(variants),
    }

    write_csv(pair_rows, root / "validation_pair_statistics.csv")
    write_csv(summary_rows, root / "validation_variant_summary.csv")
    write_csv(conditional_rows, root / "validation_conditional_variance.csv")
    write_csv(curve_rows, root / "validation_psd_profiles.csv")
    write_csv(autocorr_rows, root / "validation_autocorrelation.csv")
    write_json(root / "decoupling_decision.json", decision)
    save_plots(root, mean_curves, mean_autocorr, summary_rows)
    write_report(root / "decoupling_validation_report.md", decision, summary_rows, mean_curves, mean_autocorr, config)
    print(json.dumps(decision, indent=2))
    return 0 if decision["status"] == "GO_TO_TRAIN" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5_noise_factorial.yaml")
    parser.add_argument("--input-root", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_tiff(path_value: str, range_max: float) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(Path(path_value)), dtype=np.float32) / range_max


def summarize_pair(
    variant: str,
    pair_key: str,
    clean: np.ndarray,
    residual: np.ndarray,
    construction: dict[str, str],
    validation: dict[str, Any],
) -> dict[str, Any]:
    mean = float(np.mean(residual))
    std = safe_std(residual)
    z = (residual - mean) / std
    autocorr = autocorrelation_crop(z, int(validation["autocorrelation_radius"]))
    return {
        "variant": variant,
        "pair_key": pair_key,
        "structure_source": construction["structure_source"],
        "strength_level": construction["strength_level"],
        "residual_mean": mean,
        "residual_std": std,
        "target_residual_std": float(construction["target_residual_std"]),
        "target_std_relative_error": abs(std - float(construction["target_residual_std"])) / float(construction["target_residual_std"]),
        "skewness": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "q001": float(np.quantile(residual, 0.001)),
        "q01": float(np.quantile(residual, 0.01)),
        "q50": float(np.quantile(residual, 0.5)),
        "q99": float(np.quantile(residual, 0.99)),
        "q999": float(np.quantile(residual, 0.999)),
        "tail_probability_abs_gt_3sigma": float(np.mean(np.abs(z) > float(validation["tail_sigma"]))),
        "correlation_length_px": correlation_length(autocorr),
        "row_fixed_pattern_energy": fixed_pattern_energy(residual, axis=1),
        "column_fixed_pattern_energy": fixed_pattern_energy(residual, axis=0),
        "signal_residual_correlation": safe_corr(clean, residual),
        "fano_like_residual_var_over_signal_mean": float(np.var(residual) / max(float(np.mean(clean)), 1e-12)),
        "preclip_pixel_ratio": float(construction["preclip_pixel_ratio"]),
        "saved_zero_pixel_ratio": float(construction["saved_zero_pixel_ratio"]),
        "saved_one_pixel_ratio": float(construction["saved_one_pixel_ratio"]),
    }


def conditional_variance(
    variant: str, pair_key: str, clean: np.ndarray, residual: np.ndarray, bins: int
) -> list[dict[str, Any]]:
    edges = np.quantile(clean, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (clean >= low) & (clean <= high) if index == bins - 1 else (clean >= low) & (clean < high)
        values = residual[mask]
        rows.append(
            {
                "variant": variant,
                "pair_key": pair_key,
                "bin_index": index,
                "clean_low": low,
                "clean_high": high,
                "pixel_count": int(values.size),
                "residual_mean": float(np.mean(values)) if values.size else float("nan"),
                "residual_variance": float(np.var(values)) if values.size else float("nan"),
            }
        )
    return rows


def psd_profiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
    h, w = spectrum.shape
    y, x = np.indices((h, w))
    radius = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2).astype(np.int32)
    radial_sum = np.bincount(radius.ravel(), weights=spectrum.ravel())
    radial_count = np.bincount(radius.ravel())
    radial = radial_sum / np.maximum(radial_count, 1)
    horizontal = np.mean(spectrum, axis=0)
    vertical = np.mean(spectrum, axis=1)
    return radial, horizontal, vertical


def autocorrelation_crop(image: np.ndarray, radius: int) -> np.ndarray:
    spectrum = np.fft.fft2(image)
    corr = np.real(np.fft.fftshift(np.fft.ifft2(np.abs(spectrum) ** 2)))
    center = (corr.shape[0] // 2, corr.shape[1] // 2)
    corr /= max(float(corr[center]), 1e-12)
    return corr[center[0] - radius : center[0] + radius + 1, center[1] - radius : center[1] + radius + 1]


def correlation_length(autocorr: np.ndarray) -> float:
    center = autocorr.shape[0] // 2
    profile = 0.5 * (autocorr[center, center:] + autocorr[center:, center])
    indices = np.flatnonzero(profile <= np.exp(-1.0))
    return float(indices[0]) if indices.size else float(len(profile) - 1)


def fixed_pattern_energy(residual: np.ndarray, axis: int) -> float:
    total = float(np.var(residual))
    means = np.mean(residual, axis=axis)
    return float(np.var(means) / max(total, 1e-12))


def summarize_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["variant"])].append(row)
    fields = [
        "residual_mean", "residual_std", "target_std_relative_error", "skewness", "excess_kurtosis",
        "q001", "q01", "q50", "q99", "q999", "tail_probability_abs_gt_3sigma",
        "correlation_length_px", "row_fixed_pattern_energy", "column_fixed_pattern_energy",
        "signal_residual_correlation", "fano_like_residual_var_over_signal_mean", "preclip_pixel_ratio",
        "saved_zero_pixel_ratio", "saved_one_pixel_ratio",
    ]
    out: list[dict[str, Any]] = []
    for variant, group in groups.items():
        summary: dict[str, Any] = {"variant": variant, "pair_count": len(group)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            summary[f"{field}_mean"] = float(np.nanmean(values))
            summary[f"{field}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{field}_min"] = float(np.nanmin(values))
            summary[f"{field}_max"] = float(np.nanmax(values))
        out.append(summary)
    return sorted(out, key=lambda row: row["variant"])


def summarize_curves(
    accumulators: dict[tuple[str, str], list[np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    means: dict[tuple[str, str], np.ndarray] = {}
    for key, curves in accumulators.items():
        length = min(len(curve) for curve in curves)
        mean_curve = np.mean(np.stack([curve[:length] for curve in curves]), axis=0)
        means[key] = mean_curve
        for index, value in enumerate(mean_curve):
            rows.append({"variant": key[0], "profile": key[1], "frequency_index": index, "normalized_power": float(value)})
    return rows, means


def summarize_autocorr(
    accumulators: dict[str, list[np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    means: dict[str, np.ndarray] = {}
    for variant, arrays in accumulators.items():
        mean_array = np.mean(np.stack(arrays), axis=0)
        means[variant] = mean_array
        center = mean_array.shape[0] // 2
        for y, x in np.ndindex(mean_array.shape):
            rows.append({"variant": variant, "dy": y - center, "dx": x - center, "autocorrelation": float(mean_array[y, x])})
    return rows, means


def decoupling_checks(
    summaries: list[dict[str, Any]],
    curves: dict[tuple[str, str], np.ndarray],
    autocorr: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, bool]:
    cfg = config["validation"]
    by_variant = {row["variant"]: row for row in summaries}
    checks = {
        "low_strength_std_matched": relative_difference(by_variant["P-L"]["residual_std_mean"], by_variant["H-L"]["residual_std_mean"]) <= float(cfg["std_relative_difference_max"]),
        "high_strength_std_matched": relative_difference(by_variant["P-H"]["residual_std_mean"], by_variant["H-H"]["residual_std_mean"]) <= float(cfg["std_relative_difference_max"]),
        "low_is_lower_than_high": by_variant["P-L"]["residual_std_mean"] < by_variant["P-H"]["residual_std_mean"],
        "target_std_realized": max(row["target_std_relative_error_max"] for row in summaries) <= float(cfg["std_relative_difference_max"]),
        "residual_mean_controlled": max(
            max(abs(row["residual_mean_min"]), abs(row["residual_mean_max"])) for row in summaries
        ) <= float(config["construction"]["residual_mean_tolerance"]),
        "clipping_controlled": max(row["preclip_pixel_ratio_max"] for row in summaries) <= float(config["construction"]["clipping_ratio_max"]),
    }
    for structure, low, high in (("p99", "P-L", "P-H"), ("physical", "H-L", "H-H")):
        checks[f"{structure}_scale_only_psd"] = cosine(curves[(low, "radial_psd")], curves[(high, "radial_psd")]) >= float(cfg["normalized_same_structure_psd_cosine_min"])
        checks[f"{structure}_scale_only_autocorr"] = float(np.sqrt(np.mean((autocorr[low] - autocorr[high]) ** 2))) <= float(cfg["normalized_same_structure_autocorr_rmse_max"])
    for level, p_variant, h_variant in (("low", "P-L", "H-L"), ("high", "P-H", "H-H")):
        psd_l1 = float(np.mean(np.abs(curves[(p_variant, "radial_psd")] - curves[(h_variant, "radial_psd")])))
        signal_delta = abs(float(by_variant[p_variant]["signal_residual_correlation_mean"]) - float(by_variant[h_variant]["signal_residual_correlation_mean"]))
        skew_delta = abs(float(by_variant[p_variant]["skewness_mean"]) - float(by_variant[h_variant]["skewness_mean"]))
        kurtosis_delta = abs(float(by_variant[p_variant]["excess_kurtosis_mean"]) - float(by_variant[h_variant]["excess_kurtosis_mean"]))
        tail_delta = abs(
            float(by_variant[p_variant]["tail_probability_abs_gt_3sigma_mean"])
            - float(by_variant[h_variant]["tail_probability_abs_gt_3sigma_mean"])
        )
        checks[f"{level}_structure_remains_distinct"] = (
            psd_l1 >= float(cfg["structure_difference_psd_l1_min"])
            or signal_delta >= float(cfg["structure_difference_signal_corr_min"])
            or skew_delta >= float(cfg["structure_difference_skew_min"])
            or kurtosis_delta >= float(cfg["structure_difference_kurtosis_min"])
            or tail_delta >= float(cfg["structure_difference_tail_probability_min"])
        )
    return checks


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    return curve / max(float(np.sum(curve)), 1e-12)


def relative_difference(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(0.5 * (abs(float(a)) + abs(float(b))), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    x, y = a[:length], b[:length]
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def safe_std(array: np.ndarray) -> float:
    return max(float(np.std(np.asarray(array, dtype=np.float64), ddof=1)), 1e-12)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_plots(
    root: Path,
    curves: dict[tuple[str, str], np.ndarray],
    autocorr: dict[str, np.ndarray],
    summaries: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = root / "validation_plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for variant in sorted(autocorr):
        axes[0].plot(curves[(variant, "radial_psd")], label=variant)
        axes[1].plot(curves[(variant, "horizontal_psd")], label=variant)
        axes[2].plot(curves[(variant, "vertical_psd")], label=variant)
    for axis, title in zip(axes, ("Radial PSD", "Horizontal PSD", "Vertical PSD")):
        axis.set_title(title)
        axis.set_yscale("log")
        axis.legend()
    fig.savefig(plot_dir / "normalized_psd_profiles.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, variant in zip(axes, sorted(autocorr)):
        image = axis.imshow(autocorr[variant], cmap="coolwarm", vmin=-0.1, vmax=1.0)
        axis.set_title(variant)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(plot_dir / "autocorrelation_maps.png", dpi=160)
    plt.close(fig)

    by_variant = {row["variant"]: row for row in summaries}
    labels = ["P-L", "P-H", "H-L", "H-H"]
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(labels, [by_variant[label]["residual_std_mean"] for label in labels])
    axis.set_ylabel("Residual std")
    axis.set_title("Realized strength levels")
    fig.savefig(plot_dir / "residual_std_by_variant.png", dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    decision: dict[str, Any],
    summaries: list[dict[str, Any]],
    curves: dict[tuple[str, str], np.ndarray],
    autocorr: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    lines = [
        "# E5 Factorial Noise Decoupling Validation",
        "",
        f"Decision: **{decision['status']}**",
        "",
        "| Variant | Residual mean | Residual std | Skew | Kurtosis | Tail >3sigma | Signal-residual corr | Row energy | Column energy | Clip ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {float(row['residual_mean_mean']):.6g} | {float(row['residual_std_mean']):.6g} | "
            f"{float(row['skewness_mean']):.4f} | {float(row['excess_kurtosis_mean']):.4f} | "
            f"{float(row['tail_probability_abs_gt_3sigma_mean']):.6g} | {float(row['signal_residual_correlation_mean']):.4f} | "
            f"{float(row['row_fixed_pattern_energy_mean']):.6g} | {float(row['column_fixed_pattern_energy_mean']):.6g} | "
            f"{float(row['preclip_pixel_ratio_mean']):.6g} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend([f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in decision["checks"].items()])
    lines.extend([
        "", "## Construction Boundary", "",
        f"- Structure: {config['construction']['structure_definition']}",
        f"- Strength: {config['construction']['strength_definition']}",
        "- A common 1024-DN pedestal is applied to all four shared-clean variants; it is not factor-dependent.",
        "- Training is prohibited when the decision is `STOP_AND_REPAIR`.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
