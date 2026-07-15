"""Build an evidence-backed ICCD prior config from E1 reports."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    args = parse_args()
    output_config = Path(args.output_config)
    output_report = Path(args.output_report)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    mv_rows = read_csv(Path(args.mean_variance_csv))
    fp_rows = read_csv(Path(args.fixed_pattern_csv))
    corr_rows = read_csv(Path(args.spatial_correlation_csv))
    aux_rows = read_csv(Path(args.aux_background_csv)) if args.aux_background_csv else []

    params = build_params(
        mv_rows=mv_rows,
        fp_rows=fp_rows,
        corr_rows=corr_rows,
        aux_rows=aux_rows,
        range_max=float(args.range_max),
        seed=int(args.seed),
    )
    output_config.write_text(render_yaml(params), encoding="utf-8")
    output_report.write_text(render_report(params, output_config), encoding="utf-8")
    print(f"Wrote ICCD prior config: {output_config}")
    print(f"Wrote ICCD prior report: {output_report}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-variance-csv", default="reports/gated_iccd_20260319_mean_variance/mean_variance_summary.csv")
    parser.add_argument("--fixed-pattern-csv", default="reports/gated_iccd_20260319_fixed_pattern/fixed_pattern_correction_summary.csv")
    parser.add_argument("--spatial-correlation-csv", default="reports/gated_iccd_20260319_spatial_correlation/spatial_correlation_summary.csv")
    parser.add_argument("--aux-background-csv", default="reports/iccd_pir_20250709_background_57_187/background_summary.csv")
    parser.add_argument("--output-config", default="configs/iccd_prior_20260319.yaml")
    parser.add_argument("--output-report", default="reports/e2_1_iccd_prior/prior_parameter_report.md")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_params(
    mv_rows: list[dict[str, str]],
    fp_rows: list[dict[str, str]],
    corr_rows: list[dict[str, str]],
    aux_rows: list[dict[str, str]],
    range_max: float,
    seed: int,
) -> dict[str, Any]:
    slopes = finite_column(mv_rows, "linear_slope_var_per_dn", where=lambda row: to_float(row.get("linear_r2")) >= 0.95)
    fano = finite_column(mv_rows, "fano_temporal")
    temporal_std = finite_column(mv_rows, "temporal_std_mean")
    mean_signal = finite_column(mv_rows, "mean_signal")
    fixed_std = finite_column(fp_rows, "fixed_map_std")
    fixed_reduction = finite_column(fp_rows, "spatial_reduction_fraction")
    row_corr = finite_column(corr_rows, "lag1_row_corr")
    col_corr = finite_column(corr_rows, "lag1_col_corr")
    corr_len = finite_column(corr_rows, "corr_length_0p1_px")

    median_slope = median(slopes)
    effective_peak_photons = range_max / median_slope if median_slope > 0 else 120.0
    read_noise_sigma = min(temporal_std) / range_max if temporal_std else 0.001
    fixed_pattern_sigma = median(fixed_std) / range_max if fixed_std else 0.0
    weak_corr = max(abs(median(row_corr)), abs(median(col_corr))) if row_corr and col_corr else 0.0
    phosphor_sigma = 0.0 if weak_corr < 0.05 else 0.5

    aux = aux_rows[0] if aux_rows else {}
    aux_background = {
        "available": bool(aux_rows),
        "source": aux.get("root", ""),
        "dtype": aux.get("dtype", ""),
        "frame_count": coerce_number(aux.get("frame_count", "")),
        "mean_signal_dn": coerce_number(aux.get("mean_signal", "")),
        "temporal_std_mean_dn": coerce_number(aux.get("temporal_std_mean", "")),
        "spatial_mean_std_dn": coerce_number(aux.get("spatial_mean_std", "")),
        "saturated_fraction_mean": coerce_number(aux.get("saturated_fraction_mean", "")),
        "hot_pixel_fraction_p999": coerce_number(aux.get("hot_pixel_fraction_p999", "")),
        "matching_main_batch": False,
    }

    return {
        "range_max": int(range_max),
        "bins": 8,
        "seed": seed,
        "poisson_gaussian": {
            "peak_photons": round_float(effective_peak_photons),
            "read_noise_sigma": round_float(read_noise_sigma),
            "clip": True,
        },
        "scmos_like": {
            "signal_gain": round_float(median_slope / range_max if median_slope > 0 else 0.00025),
            "read_noise_sigma": round_float(read_noise_sigma),
            "row_noise_sigma": 0.0,
            "column_noise_sigma": 0.0,
            "offset": 0.0,
            "clip": True,
        },
        "iccd": {
            "photon_scale": round_float(effective_peak_photons),
            "photocathode_qe": 1.0,
            "mcp_gain_mean": 1.0,
            "mcp_gain_var": 0.0001,
            "dark_count_rate": 0.0,
            "phosphor_sigma": round_float(phosphor_sigma),
            "read_noise_sigma": round_float(read_noise_sigma),
            "offset": 0.0,
            "clip": True,
        },
        "iccd_empirical_components": {
            "source_main_batch": "D:/iccd/data/20260319",
            "calibration_status": "repeated-frame empirical prior; no matching dark/flat",
            "mean_signal_dn_min": round_float(min(mean_signal)),
            "mean_signal_dn_max": round_float(max(mean_signal)),
            "temporal_fano_min": round_float(min(fano)),
            "temporal_fano_median": round_float(median(fano)),
            "temporal_fano_max": round_float(max(fano)),
            "valid_linear_slope_var_per_dn_median_r2_ge_0p95": round_float(median_slope),
            "valid_linear_slope_count": len(slopes),
            "effective_peak_photons_from_raw_slope": round_float(effective_peak_photons),
            "fixed_pattern_sigma_norm_median": round_float(fixed_pattern_sigma),
            "fixed_pattern_reduction_median": round_float(median(fixed_reduction)),
            "lag1_row_corr_median": round_float(median(row_corr)),
            "lag1_col_corr_median": round_float(median(col_corr)),
            "corr_length_0p1_px_median": round_float(median(corr_len)),
            "auxiliary_8bit_background": aux_background,
            "claim_boundary": [
                "Parameters are derived from repeated-frame empirical reports.",
                "Do not present as strict dark/flat physical calibration.",
                "The current runnable ICCD model uses an effective photon scale to match raw-domain variance slope.",
                "Fixed-pattern statistics are recorded but not yet injected by src.iccd_noise.ICCDNoiseModel.",
            ],
        },
    }


def finite_column(rows: list[dict[str, str]], key: str, where: Any | None = None) -> list[float]:
    values = []
    for row in rows:
        if where is not None and not where(row):
            continue
        value = to_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return values


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def coerce_number(value: Any) -> Any:
    number = to_float(value)
    if math.isfinite(number):
        if abs(number - round(number)) < 1e-9:
            return int(round(number))
        return round_float(number)
    return value


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def round_float(value: float) -> float:
    if not math.isfinite(float(value)):
        return float("nan")
    return float(f"{float(value):.8g}")


def render_yaml(data: dict[str, Any]) -> str:
    lines = [
        "# Evidence-backed ICCD prior generated from local E1 reports.",
        "# Runnable sections are compatible with scripts/compare_noise_priors.py.",
        "# Empirical components record evidence not yet represented by the simple runnable model.",
    ]
    lines.extend(yaml_lines(data))
    return "\n".join(lines) + "\n"


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {format_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {format_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{format_yaml_scalar(value)}"]


def format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "/")
    if text == "" or any(char in text for char in [":", "#", "[", "]", "{", "}", ","]):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def render_report(params: dict[str, Any], config_path: Path) -> str:
    empirical = params["iccd_empirical_components"]
    lines = [
        "# E2.1 ICCD Prior Parameter Report",
        "",
        f"- Config: `{config_path}`",
        f"- Source main batch: `{empirical['source_main_batch']}`",
        f"- Calibration status: {empirical['calibration_status']}",
        "",
        "## Runnable Prior Parameters",
        "",
        "| prior | key parameters |",
        "|---|---|",
        f"| poisson_gaussian | peak_photons={params['poisson_gaussian']['peak_photons']}, read_noise_sigma={params['poisson_gaussian']['read_noise_sigma']} |",
        f"| scmos_like | signal_gain={params['scmos_like']['signal_gain']}, read_noise_sigma={params['scmos_like']['read_noise_sigma']} |",
        f"| iccd | photon_scale={params['iccd']['photon_scale']}, phosphor_sigma={params['iccd']['phosphor_sigma']}, read_noise_sigma={params['iccd']['read_noise_sigma']} |",
        "",
        "## Evidence Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean_signal_dn_min | {empirical['mean_signal_dn_min']} |",
        f"| mean_signal_dn_max | {empirical['mean_signal_dn_max']} |",
        f"| temporal_fano_min | {empirical['temporal_fano_min']} |",
        f"| temporal_fano_median | {empirical['temporal_fano_median']} |",
        f"| temporal_fano_max | {empirical['temporal_fano_max']} |",
        f"| linear_slope_median_r2_ge_0p95 | {empirical['valid_linear_slope_var_per_dn_median_r2_ge_0p95']} |",
        f"| fixed_pattern_sigma_norm_median | {empirical['fixed_pattern_sigma_norm_median']} |",
        f"| fixed_pattern_reduction_median | {empirical['fixed_pattern_reduction_median']} |",
        f"| lag1_row_corr_median | {empirical['lag1_row_corr_median']} |",
        f"| lag1_col_corr_median | {empirical['lag1_col_corr_median']} |",
        "",
        "## Claim Boundary",
        "",
    ]
    for item in empirical["claim_boundary"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
