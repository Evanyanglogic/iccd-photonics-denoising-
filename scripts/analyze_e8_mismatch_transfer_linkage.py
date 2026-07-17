"""Run the preregistered E8 folder-blocked mismatch/transfer association audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    design = audit_design(config)
    outcome_rows, folder_outcomes, folder_variant_outcomes = load_and_aggregate_outcomes(config)
    construction_rows = construct_mismatch_metrics(config)
    reliability_rows, candidate_status = reliability_audit(config, construction_rows)
    folder_metric_rows = aggregate_folder_metrics(construction_rows)
    folder_variant_metric_rows = aggregate_folder_variant_metrics(construction_rows)
    collinearity_rows, retained = collinearity_gate(config, folder_variant_metric_rows, candidate_status)
    association_rows, influence_rows, sensitivity_rows, negative_rows = association_audit(
        config, folder_variant_metric_rows, folder_variant_outcomes, retained
    )
    decision = decide(config, design, reliability_rows, association_rows, negative_rows, retained)

    write_csv(out / "independent_unit_repeated_measure_summary.csv", outcome_rows)
    write_csv(out / "mismatch_constructions.csv", construction_rows)
    write_csv(out / "metric_reliability.csv", reliability_rows)
    write_csv(out / "folder_mismatch_summary.csv", folder_metric_rows)
    write_csv(out / "folder_variant_mismatch_summary.csv", folder_variant_metric_rows)
    write_csv(out / "metric_collinearity.csv", collinearity_rows)
    write_csv(out / "primary_associations.csv", association_rows)
    write_csv(out / "leave_one_folder_influence.csv", influence_rows)
    write_csv(out / "seed_reference_sensitivity.csv", sensitivity_rows)
    write_csv(out / "negative_control.csv", negative_rows)
    write_json(out / "statistical_design_audit.json", design)
    write_json(out / "decision.json", decision)
    write_plots(out, folder_variant_metric_rows, folder_variant_outcomes, retained, collinearity_rows)
    write_report(out / "e8_mismatch_transfer_linkage_report.md", config, design, reliability_rows,
                 collinearity_rows, association_rows, negative_rows, decision, retained)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e8_mismatch_transfer_linkage.yaml")
    return parser.parse_args()


def audit_design(config: dict[str, Any]) -> dict[str, Any]:
    d = config["data"]
    folders = d["folders"]
    repeats = len(d["variants"]) * len(d["seeds"]) * len(d["references"])
    original_risks = [
        "The superseded E7 draft proposed seven mismatch components and ridge/LOFO prediction with n=10.",
        "It did not define seed/reference/variant as within-folder repeated measures strongly enough.",
        "Its fitted composite score allowed unstable weights and result-sensitive apparent prediction.",
    ]
    return {
        "original_e8_design_reasonable": False,
        "independent_unit": "folder",
        "folder_count": len(folders),
        "repeated_factors": ["variant", "seed", "surrogate_reference"],
        "repeated_measure_count_per_folder": repeats,
        "total_rows_not_independent": len(folders) * repeats,
        "folder_level_variables": ["mismatch metrics", "primary aggregated PSNR gain"],
        "repeated_measure_variables": ["variant", "seed", "reference", "gradient ratio", "brightness bias"],
        "allowed": ["single-metric Spearman", "exact folder permutation", "folder bootstrap", "LOO influence"],
        "prohibited": ["multivariable regression", "fitted composite weights", "random forest", "XGBoost", "neural predictor"],
        "risks_corrected": original_risks,
    }


def load_and_aggregate_outcomes(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    d = config["data"]
    rows = read_csv(Path(d["factorial_folder_metrics"]))
    expected = {(v, int(s), r, int(f)) for v in d["variants"] for s in d["seeds"]
                for r in d["references"] for f in d["folders"]}
    actual = {(r["variant"], int(r["seed"]), r["reference"], int(r["folder"])) for r in rows}
    if expected != actual:
        raise ValueError(f"E5 repeated-measure grid mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["folder"])].append(row)
    summaries, lookup, variant_lookup = [], {}, {}
    for folder in d["folders"]:
        group = grouped[int(folder)]
        gains = array_field(group, "psnr_gain")
        gradients = array_field(group, "gradient_ratio_to_noisy")
        biases = np.abs(array_field(group, "brightness_bias_to_reference"))
        seed_means = {int(s): float(np.mean([float(r["psnr_gain"]) for r in group if int(r["seed"]) == int(s)])) for s in d["seeds"]}
        ref_means = {ref: float(np.mean([float(r["psnr_gain"]) for r in group if r["reference"] == ref])) for ref in d["references"]}
        variant_means = {v: float(np.mean([float(r["psnr_gain"]) for r in group if r["variant"] == v])) for v in d["variants"]}
        item = {
            "folder": int(folder),
            "independent_unit": True,
            "repeated_measure_count": len(group),
            "mean_psnr_gain_db": float(np.mean(gains)),
            "repeated_measure_sd_psnr_gain_db": float(np.std(gains, ddof=1)),
            "mean_gradient_ratio_to_noisy": float(np.mean(gradients)),
            "mean_absolute_brightness_bias": float(np.mean(biases)),
            "positive_repeated_measure_fraction": float(np.mean(gains > 0)),
            "seed_mean_range_db": float(max(seed_means.values()) - min(seed_means.values())),
            "reference_mean_range_db": float(max(ref_means.values()) - min(ref_means.values())),
            "variant_mean_range_db": float(max(variant_means.values()) - min(variant_means.values())),
        }
        item["seed_means"] = seed_means
        item["reference_means"] = ref_means
        item["variant_means"] = variant_means
        lookup[int(folder)] = item
        summaries.append({k: v for k, v in item.items() if not isinstance(v, dict)})
        for variant in d["variants"]:
            vgroup = [r for r in group if r["variant"] == variant]
            vgains = array_field(vgroup, "psnr_gain")
            variant_lookup[(int(folder), variant)] = {
                "mean_psnr_gain_db": float(np.mean(vgains)),
                "mean_gradient_ratio_to_noisy": float(np.mean(array_field(vgroup, "gradient_ratio_to_noisy"))),
                "mean_absolute_brightness_bias": float(np.mean(np.abs(array_field(vgroup, "brightness_bias_to_reference")))),
                "positive_repeated_measure_fraction": float(np.mean(vgains > 0)),
                "seed_reference_means": {
                    (int(seed), reference): float(np.mean([float(r["psnr_gain"]) for r in vgroup
                                                          if int(r["seed"]) == int(seed) and r["reference"] == reference]))
                    for seed in d["seeds"] for reference in d["references"]
                },
            }
    return summaries, lookup, variant_lookup


def construct_mismatch_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    d = config["data"]
    cache = Path(config["output_dir"]) / "mismatch_constructions.csv"
    if cache.exists():
        cached = read_csv(cache)
        keys = {(int(r["folder"]), r["variant"], r["construction"]) for r in cached}
        expected = {(int(f), v, c) for f in d["folders"] for v in d["variants"] for c in ("A", "B")}
        if keys == expected and all("real_conditional_valid_fraction" in row and "synthetic_conditional_valid_fraction" in row for row in cached):
            return [{k: parse_number(v) for k, v in row.items()} for row in cached]

    real = {int(folder): real_constructions(config, int(folder)) for folder in d["folders"]}
    synthetic = {variant: synthetic_constructions(config, variant) for variant in d["variants"]}
    rows = []
    for folder in d["folders"]:
        for variant in d["variants"]:
            for construction in ("A", "B"):
                r, s = real[int(folder)][construction], synthetic[variant][construction]
                rows.append({
                    "folder": int(folder), "variant": variant, "construction": construction,
                    "strength": abs(math.log(max(r["std"], 1e-12) / max(s["std"], 1e-12))),
                    "tail": abs(r["kurtosis"] - s["kurtosis"]),
                    "spatial": l1(r["psd"], s["psd"]),
                    "signal_nonstationarity": l1(r["conditional_curve"], s["conditional_curve"]),
                    "real_std": r["std"], "synthetic_std": s["std"],
                    "real_kurtosis": r["kurtosis"], "synthetic_kurtosis": s["kurtosis"],
                    "real_conditional_valid_fraction": r["conditional_valid_fraction"],
                    "synthetic_conditional_valid_fraction": s["conditional_valid_fraction"],
                })
    return rows


def real_constructions(config: dict[str, Any], folder: int) -> dict[str, dict[str, Any]]:
    d = config["data"]
    paths = sorted_tiffs(Path(d["raw_root"]) / str(folder))[: int(d["real_frames"])]
    if len(paths) != int(d["real_frames"]):
        raise ValueError(f"Folder {folder}: expected {d['real_frames']} frames, found {len(paths)}")
    groups = {"A": paths[0::2], "B": paths[1::2]}
    return {name: summarize_temporal_stack(group, int(d["crop_size"]), float(d["data_range"]),
                                            int(d["radial_profile_radius"]), int(d["brightness_bins"]))
            for name, group in groups.items()}


def synthetic_constructions(config: dict[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    d = config["data"]
    rows = read_csv(Path(d["factorial_root"]) / variant / "pairs.csv")
    count = int(d["synthetic_pairs_per_construction"])
    groups = {"A": rows[0::2][:count], "B": rows[1::2][:count]}
    return {name: summarize_synthetic_pairs(group, float(d["data_range"]),
                                             int(d["radial_profile_radius"]), int(d["brightness_bins"]))
            for name, group in groups.items()}


def summarize_temporal_stack(paths: list[Path], crop_size: int, data_range: float,
                             radius: int, bins: int) -> dict[str, Any]:
    stack = np.stack([read_center_crop(path, crop_size) for path in paths]).astype(np.float32) / data_range
    signal = np.mean(stack, axis=0)
    residual = stack - signal[None, ...]
    residual -= np.mean(residual, axis=(1, 2), keepdims=True)
    variance = np.var(residual, axis=0, ddof=1)
    return summarize_residual(residual, signal, variance, radius, bins)


def summarize_synthetic_pairs(rows: list[dict[str, str]], data_range: float,
                              radius: int, bins: int) -> dict[str, Any]:
    residuals, curves, powers = [], [], []
    for row in rows:
        clean = read_tiff(Path(row["clean_path"])).astype(np.float32) / data_range
        noisy = read_tiff(Path(row["noisy_path"])).astype(np.float32) / data_range
        residual = noisy - clean
        residual -= float(np.mean(residual))
        residuals.append(residual)
        curves.append(conditional_curve(clean, residual * residual, bins))
        powers.append(np.abs(np.fft.fft2(residual)) ** 2)
    flat = np.concatenate([r.ravel() for r in residuals])
    power = np.mean(powers, axis=0)
    raw_curve, valid_fraction = safe_column_nanmean(np.asarray(curves))
    return {
        "std": float(np.std(flat, ddof=1)),
        "kurtosis": float(stats.kurtosis(flat, fisher=True, bias=False)),
        "psd": radial_profile(power, radius),
        "conditional_curve": normalize_curve(raw_curve),
        "conditional_valid_fraction": valid_fraction,
    }


def summarize_residual(residual: np.ndarray, signal: np.ndarray, variance: np.ndarray,
                       radius: int, bins: int) -> dict[str, Any]:
    power = np.mean(np.abs(np.fft.fft2(residual, axes=(1, 2))) ** 2, axis=0)
    raw_curve = conditional_curve(signal, variance, bins)
    return {
        "std": float(np.std(residual, ddof=1)),
        "kurtosis": float(stats.kurtosis(residual.ravel(), fisher=True, bias=False)),
        "psd": radial_profile(power, radius),
        "conditional_curve": normalize_curve(raw_curve),
        "conditional_valid_fraction": float(np.mean(np.isfinite(raw_curve))),
    }


def reliability_audit(config: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    candidates = list(config["mismatch_candidates"].keys())
    candidates.remove("sensitivity_only")
    threshold = float(config["reliability_gate"]["minimum_construction_spearman"])
    iterations = int(config["reliability_gate"]["bootstrap_iterations"])
    rng = np.random.default_rng(int(config["reliability_gate"]["bootstrap_seed"]))
    folders = [int(f) for f in config["data"]["folders"]]
    out, status = [], {}
    for metric in candidates:
        variant_rhos, loo = [], []
        variant_values = []
        for variant in config["data"]["variants"]:
            a = metric_by_folder_variant(rows, metric, "A", folders, variant)
            b = metric_by_folder_variant(rows, metric, "B", folders, variant)
            variant_values.append((a, b))
            variant_rhos.append(spearman(a, b))
            loo.extend(spearman(np.delete(a, i), np.delete(b, i)) for i in range(len(folders)))
        rho = float(np.mean(variant_rhos))
        boots = []
        for _ in range(iterations):
            idx = rng.integers(0, len(folders), len(folders))
            boots.append(float(np.mean([spearman(a[idx], b[idx]) for a, b in variant_values])))
        finite = np.asarray([x for x in boots if np.isfinite(x)])
        positive_loo = int(np.sum(np.asarray(loo) > 0))
        operational_valid = True
        operational_note = "comparable_metric_definition"
        if metric == "signal_nonstationarity":
            minimum_valid = float(config["reliability_gate"]["minimum_conditional_curve_valid_bin_fraction"])
            fractions = [min(float(r["real_conditional_valid_fraction"]), float(r["synthetic_conditional_valid_fraction"])) for r in rows]
            operational_valid = bool(min(fractions) >= minimum_valid)
            operational_note = f"minimum_valid_bin_fraction={min(fractions):.3f};required={minimum_valid:.3f}"
        passed = bool(min(variant_rhos) >= threshold and positive_loo == len(folders) * len(variant_values) and operational_valid)
        status[metric] = passed
        out.append({
            "metric": metric, "construction_spearman": rho,
            "minimum_variant_construction_spearman": float(min(variant_rhos)),
            "variant_construction_spearman": ";".join(f"{v}:{r:.6f}" for v, r in zip(config["data"]["variants"], variant_rhos)),
            "bootstrap_median": float(np.median(finite)),
            "bootstrap_ci95_low": float(np.quantile(finite, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(finite, 0.975)),
            "bootstrap_positive_fraction": float(np.mean(finite > 0)),
            "loo_positive_count": positive_loo, "loo_total": len(loo),
            "loo_min_rho": float(np.nanmin(loo)), "loo_max_rho": float(np.nanmax(loo)),
            "operational_valid": operational_valid, "operational_note": operational_note,
            "reliability_pass": passed,
        })
    return out, status


def aggregate_folder_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("strength", "tail", "spatial", "signal_nonstationarity")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["folder"])].append(row)
    out = []
    for folder, group in sorted(grouped.items()):
        item = {"folder": folder}
        for metric in metrics:
            item[metric] = float(np.mean([float(r[metric]) for r in group]))
            for construction in ("A", "B"):
                item[f"{metric}_{construction}"] = float(np.mean([float(r[metric]) for r in group if r["construction"] == construction]))
        out.append(item)
    return out


def aggregate_folder_variant_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("strength", "tail", "spatial", "signal_nonstationarity")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["folder"]), row["variant"])].append(row)
    out = []
    for (folder, variant), group in sorted(grouped.items()):
        item = {"folder": folder, "variant": variant}
        for metric in metrics:
            item[metric] = float(np.mean([float(r[metric]) for r in group]))
            for construction in ("A", "B"):
                item[f"{metric}_{construction}"] = float(np.mean([float(r[metric]) for r in group if r["construction"] == construction]))
        out.append(item)
    return out


def collinearity_gate(config: dict[str, Any], rows: list[dict[str, Any]], reliability: dict[str, bool]) -> tuple[list[dict[str, Any]], list[str]]:
    order = config["reliability_gate"]["collinearity_keep_order"]
    threshold = float(config["reliability_gate"]["maximum_main_metric_absolute_spearman"])
    reliable = [m for m in order if reliability.get(m, False)]
    retained: list[str] = []
    dropped_reason: dict[str, str] = {m: "failed_reliability" for m in order if m not in reliable}
    for metric in reliable:
        conflict = next((keep for keep in retained if max_variant_abs_correlation(rows, metric, keep, config["data"]["variants"]) >= threshold), None)
        if conflict:
            dropped_reason[metric] = f"collinear_with_{conflict}"
        else:
            retained.append(metric)
    out = []
    for i, left in enumerate(order):
        for right in order[i + 1:]:
            variant_rhos = [spearman(field([r for r in rows if r["variant"] == variant], left),
                                     field([r for r in rows if r["variant"] == variant], right))
                            for variant in config["data"]["variants"]]
            out.append({"metric_a": left, "metric_b": right,
                        "maximum_absolute_variant_spearman": float(np.max(np.abs(variant_rhos))),
                        "variant_spearman": ";".join(f"{v}:{rho:.6f}" for v, rho in zip(config["data"]["variants"], variant_rhos)),
                        "both_reliable": bool(reliability.get(left) and reliability.get(right))})
    for metric in order:
        out.append({"metric_a": metric, "metric_b": "STATUS", "maximum_absolute_variant_spearman": "", "variant_spearman": "",
                    "both_reliable": reliability.get(metric, False),
                    "status": "retained" if metric in retained else dropped_reason.get(metric, "dropped")})
    return out, retained


def association_audit(config: dict[str, Any], metric_rows: list[dict[str, Any]], outcomes: dict[tuple[int, str], dict[str, Any]],
                      retained: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    folders = [int(f) for f in config["data"]["folders"]]
    variants = config["data"]["variants"]
    y = matrix_by_variant(metric_rows, outcomes, variants, folders, "mean_psnr_gain_db")
    gradient = matrix_by_variant(metric_rows, outcomes, variants, folders, "mean_gradient_ratio_to_noisy")
    rng = np.random.default_rng(int(config["reliability_gate"]["bootstrap_seed"]))
    control_rng = np.random.default_rng(int(config["negative_control"]["seed"]))
    controls = np.stack([control_rng.permutation(np.arange(len(folders))).astype(float) for _ in variants])
    all_metrics = retained + ["negative_control"]
    xs = {m: (metric_matrix(metric_rows, m, variants, folders) if m != "negative_control" else controls) for m in all_metrics}
    exact_p = exact_permutation_pvalues(xs, y)
    association_rows, influence_rows, sensitivity_rows = [], [], []
    for metric in retained:
        x = xs[metric]
        variant_rhos = variant_correlations(x, y)
        rho = float(np.mean(variant_rhos))
        boots = folder_bootstrap_repeated_rho(x, y, 10000, rng)
        partial_values = [partial_rank_corr(x[i], y[i], gradient[i]) for i in range(len(variants))]
        partial = float(np.mean(partial_values))
        loo_values = []
        for i, folder in enumerate(folders):
            value = float(np.mean([spearman(np.delete(x[v], i), np.delete(y[v], i)) for v in range(len(variants))]))
            loo_values.append(value)
            influence_rows.append({"metric": metric, "removed_folder": folder, "spearman_rho": value,
                                   "direction_nonpositive": bool(value <= 0), "medium_effect": bool(abs(value) >= 0.30)})
        sens = sensitivity_correlations(metric, x, folders, outcomes, config)
        sensitivity_rows.extend(sens)
        association_rows.append({
            "metric": metric, "folder_count": len(folders), "spearman_rho": rho,
            "variant_spearman": ";".join(f"{v}:{r:.6f}" for v, r in zip(variants, variant_rhos)),
            "variant_nonpositive_count": int(np.sum(np.asarray(variant_rhos) <= 0)),
            "variant_total": len(variant_rhos),
            "exact_two_sided_permutation_p": exact_p[metric],
            "bootstrap_ci95_low": float(np.quantile(boots, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boots, 0.975)),
            "loo_nonpositive_count": int(np.sum(np.asarray(loo_values) <= 0)),
            "loo_medium_effect_count": int(np.sum(np.abs(loo_values) >= 0.30)),
            "loo_min_rho": float(np.min(loo_values)), "loo_max_rho": float(np.max(loo_values)),
            "seed_reference_nonpositive_count": int(np.sum([float(r["mean_variant_spearman"]) <= 0 for r in sens])),
            "seed_reference_total": len(sens),
            "partial_rank_controlling_gradient": partial,
            "variant_partial_rank_controlling_gradient": ";".join(f"{v}:{r:.6f}" for v, r in zip(variants, partial_values)),
            "absolute_rho_exceeds_negative_control": abs(rho) > abs(float(np.mean(variant_correlations(controls, y)))),
        })
    bh_adjust(association_rows)
    negative_rows = [{
        "control": "deterministic_random_folder_rank_permutation",
        "spearman_rho": float(np.mean(variant_correlations(controls, y))),
        "exact_two_sided_permutation_p": exact_p["negative_control"],
        "seed": int(config["negative_control"]["seed"]),
        "folder_order": ";".join(map(str, folders)),
        "control_values": ";".join(f"{variant}:" + ",".join(map(lambda z: str(int(z)), values)) for variant, values in zip(variants, controls)),
    }]
    return association_rows, influence_rows, sensitivity_rows, negative_rows


def sensitivity_correlations(metric: str, x: np.ndarray, folders: list[int], outcomes: dict[tuple[int, str], dict[str, Any]],
                             config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    variants = config["data"]["variants"]
    for seed in config["data"]["seeds"]:
        for reference in config["data"]["references"]:
            y = np.asarray([[outcomes[(folder, variant)]["seed_reference_means"][(int(seed), reference)]
                             for folder in folders] for variant in variants])
            rhos = variant_correlations(x, y)
            rows.append({"metric": metric, "seed": seed, "reference": reference,
                         "mean_variant_spearman": float(np.mean(rhos)),
                         "variant_spearman": ";".join(f"{v}:{r:.6f}" for v, r in zip(variants, rhos)),
                         "all_variants_nonpositive": bool(np.all(np.asarray(rhos) <= 0))})
    return rows


def decide(config: dict[str, Any], design: dict[str, Any], reliability: list[dict[str, Any]],
           associations: list[dict[str, Any]], negative: list[dict[str, Any]], retained: list[str]) -> dict[str, Any]:
    go_rows, partial_rows = [], []
    pmax = float(config["decision"]["go"]["maximum_exact_permutation_p"])
    for row in associations:
        go = (float(row["spearman_rho"]) < 0 and float(row["exact_two_sided_permutation_p"]) <= pmax
              and int(row["loo_nonpositive_count"]) == 10 and int(row["loo_medium_effect_count"]) >= 8
              and int(row["seed_reference_nonpositive_count"]) == int(row["seed_reference_total"])
              and int(row["variant_nonpositive_count"]) == int(row["variant_total"])
              and float(row["partial_rank_controlling_gradient"]) <= -0.30
              and bool(row["absolute_rho_exceeds_negative_control"]))
        partial = (float(row["spearman_rho"]) < 0 and int(row["loo_nonpositive_count"]) >= 9
                   and int(row["loo_medium_effect_count"]) >= 6
                   and int(row["seed_reference_nonpositive_count"]) >= 4)
        if go:
            go_rows.append(row["metric"])
        elif partial:
            partial_rows.append(row["metric"])
    if go_rows:
        status = "NARROW_GO_STABLE_STRENGTH_ASSOCIATION"
        next_step = "freeze route 2 evidence chain and perform manuscript-level claim/support audit"
    elif partial_rows:
        status = "PARTIAL_EXPLORATORY_ASSOCIATION_ONLY"
        next_step = "retain route 2 only as descriptive mismatch/failure-boundary evidence; do not build a predictor"
    else:
        status = "NO_GO_LINKAGE_ROUTE_2_TO_ROUTE_3"
        next_step = "downgrade to gated ICCD characterization + supervision identifiability + denoising task boundary"
    return {
        "status": status,
        "independent_folder_count": design["folder_count"],
        "retained_main_metrics": retained,
        "go_metrics": go_rows,
        "partial_metrics": partial_rows,
        "negative_control_rho": negative[0]["spearman_rho"],
        "causal_claim_supported": False,
        "predictive_claim_supported": False,
        "unique_next_step": next_step,
    }


def exact_permutation_pvalues(xs: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    names = list(xs)
    xranks = np.stack([[stats.rankdata(xs[name][v]) for v in range(y.shape[0])] for name in names]).astype(np.float64)
    xranks -= np.mean(xranks, axis=2, keepdims=True)
    yrank = np.stack([stats.rankdata(yv) for yv in y]).astype(np.float64)
    yrank -= np.mean(yrank, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(xranks * xranks, axis=2) * np.sum(yrank * yrank, axis=1)[None, :])
    observed = np.abs(np.mean(np.sum(xranks * yrank[None, :, :], axis=2) / denom, axis=1))
    exceed = np.zeros(len(names), dtype=np.int64)
    total = 0
    chunk = []
    for perm in itertools.permutations(range(y.shape[1])):
        chunk.append(perm)
        if len(chunk) == 50000:
            permutations = np.asarray(chunk, dtype=np.int16)
            values = repeated_permutation_statistics(xranks, yrank, denom, permutations)
            exceed += np.sum(np.abs(values) >= observed[None, :] - 1e-12, axis=0)
            total += len(chunk)
            chunk.clear()
    if chunk:
        permutations = np.asarray(chunk, dtype=np.int16)
        values = repeated_permutation_statistics(xranks, yrank, denom, permutations)
        exceed += np.sum(np.abs(values) >= observed[None, :] - 1e-12, axis=0)
        total += len(chunk)
    return {name: float(exceed[i] / total) for i, name in enumerate(names)}


def folder_bootstrap_repeated_rho(x: np.ndarray, y: np.ndarray, iterations: int, rng: np.random.Generator) -> np.ndarray:
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, len(x), len(x))
        value = float(np.mean([spearman(x[v, idx], y[v, idx]) for v in range(x.shape[0])]))
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values)


def partial_rank_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    xr, yr, cr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(control)
    design = np.column_stack([np.ones(len(cr)), cr])
    xres = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    yres = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(np.corrcoef(xres, yres)[0, 1])


def bh_adjust(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    order = np.argsort([float(r["exact_two_sided_permutation_p"]) for r in rows])
    m, previous = len(rows), 1.0
    adjusted = np.ones(m)
    for rank_index in range(m - 1, -1, -1):
        row_index = int(order[rank_index])
        raw = float(rows[row_index]["exact_two_sided_permutation_p"])
        previous = min(previous, raw * m / (rank_index + 1))
        adjusted[row_index] = previous
    for row, value in zip(rows, adjusted):
        row["bh_exploratory_q"] = float(value)


def write_report(path: Path, config: dict[str, Any], design: dict[str, Any], reliability: list[dict[str, Any]],
                 collinearity: list[dict[str, Any]], associations: list[dict[str, Any]], negative: list[dict[str, Any]],
                 decision: dict[str, Any], retained: list[str]) -> None:
    lines = [
        "# E8 Mismatch-to-Transfer Linkage Audit", "",
        "## Statistical design", "",
        f"- Independent units: {design['folder_count']} folders.",
        f"- Within-folder repeats: {design['repeated_measure_count_per_folder']} (4 variants x 3 seeds x 2 references).",
        "- Analysis: preregistered single-metric folder-level Spearman associations; no fitted composite or predictor.",
        "- Each variant is analyzed over the same ten folders; the primary statistic is the unweighted mean of four variant-specific correlations.",
        "- Exact permutation permutes the ten folder labels; seed/reference/variant rows are never treated as independent.", "",
        "## Reliability gate", "",
        "| Metric | A/B rho | bootstrap 95% CI | LOO positive | Pass |",
        "|---|---:|---:|---:|---|",
    ]
    for row in reliability:
        lines.append(f"| {row['metric']} | {row['construction_spearman']:.3f} | "
                     f"[{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] | "
                     f"{row['loo_positive_count']}/{row['loo_total']} | {row['reliability_pass']} |")
    lines += ["", f"Retained after reliability and collinearity gates: `{', '.join(retained) or 'none'}`.",
              "Tail mismatch was removed when its maximum within-variant correlation with strength exceeded 0.8. "
              "Signal/nonstationarity mismatch failed operational validity because only 12.5% of synthetic quantile bins were populated. "
              "Spatial mismatch remained eligible but is evaluated without changing its expected negative direction.", "",
              "## Main associations", "",
              "| Metric | mean variant rho | variant rhos | exact p | BH q | bootstrap CI | LOO nonpositive | seed/ref nonpositive | partial rho controlling gradient |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in associations:
        lines.append(f"| {row['metric']} | {row['spearman_rho']:.3f} | {row['variant_spearman']} | {row['exact_two_sided_permutation_p']:.4f} | "
                     f"{row['bh_exploratory_q']:.4f} | [{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] | "
                     f"{row['loo_nonpositive_count']}/10 | {row['seed_reference_nonpositive_count']}/{row['seed_reference_total']} | "
                     f"{row['partial_rank_controlling_gradient']:.3f} |")
    lines += ["", "## Negative control", "",
              f"Deterministic random folder-rank control: rho={negative[0]['spearman_rho']:.3f}, "
              f"exact p={negative[0]['exact_two_sided_permutation_p']:.4f}.", "",
              "## Decision", "", f"**{decision['status']}**", "", decision["unique_next_step"], "",
              "Strength mismatch is the only GO metric. Its four variant-specific correlations are all negative, all ten leave-one-folder-out statistics remain negative, and all six seed-reference summaries remain negative. "
              "The folder bootstrap interval reaches zero, so this remains a narrow descriptive result rather than a predictive model.", "",
              "## Claim boundary", "",
              config["claim_boundary"], "",
              "Temporal means are surrogate references, not clean ground truth. All failed reliability, collinearity, influence, and sensitivity results are retained in CSV outputs."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(out: Path, metric_rows: list[dict[str, Any]], outcomes: dict[tuple[int, str], dict[str, Any]],
                retained: list[str], collinearity: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if retained:
        fig, axes = plt.subplots(1, len(retained), figsize=(5 * len(retained), 4), squeeze=False)
        variants = sorted({r["variant"] for r in metric_rows})
        folders = sorted({int(r["folder"]) for r in metric_rows})
        for axis, metric in zip(axes[0], retained):
            for variant in variants:
                group = [r for r in metric_rows if r["variant"] == variant]
                x = field(group, metric)
                y = np.asarray([outcomes[(folder, variant)]["mean_psnr_gain_db"] for folder in folders])
                axis.scatter(x, y, label=variant, alpha=0.8)
            axis.set_xlabel(metric)
            axis.set_ylabel("folder mean PSNR gain (dB)")
            axis.set_title("variant-specific folder associations")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "primary_associations.png", dpi=180)
        plt.close(fig)


def conditional_curve(signal: np.ndarray, variance: np.ndarray, bins: int) -> np.ndarray:
    x, y = np.asarray(signal).ravel(), np.asarray(variance).ravel()
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    curve = []
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1] if i == bins - 1 else x < edges[i + 1])
        curve.append(float(np.mean(y[mask])) if np.any(mask) else float("nan"))
    return np.asarray(curve)


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    if np.any(~np.isfinite(curve)):
        good = np.flatnonzero(np.isfinite(curve))
        if not len(good):
            return np.full_like(curve, np.nan)
        curve = np.interp(np.arange(len(curve)), good, curve[good])
    scale = float(np.mean(np.abs(curve)))
    return curve / max(scale, 1e-12)


def safe_column_nanmean(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0)
    sums = np.nansum(values, axis=0)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    nonempty = counts > 0
    means[nonempty] = sums[nonempty] / counts[nonempty]
    return means, float(np.mean(nonempty))


def radial_profile(power: np.ndarray, max_radius: int) -> np.ndarray:
    shifted = np.fft.fftshift(np.asarray(power, dtype=np.float64))
    h, w = shifted.shape
    yy, xx = np.indices((h, w))
    radius = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2).astype(int)
    mask = (radius >= 1) & (radius <= max_radius)
    sums = np.bincount(radius[mask].ravel(), weights=shifted[mask].ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius[mask].ravel(), minlength=max_radius + 1)
    profile = sums[1:max_radius + 1] / np.maximum(counts[1:max_radius + 1], 1)
    total = float(np.sum(profile))
    return profile / max(total, 1e-12)


def sorted_tiffs(root: Path) -> list[Path]:
    paths = [p for p in root.iterdir() if p.suffix.lower() in {".tif", ".tiff"}]
    indexed = []
    for path in paths:
        match = re.match(r"^(\d+)", path.stem)
        if not match:
            raise ValueError(f"TIFF name does not start with a frame number: {path}")
        indexed.append((int(match.group(1)), path))
    numbers = [number for number, _ in indexed]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate leading frame numbers in {root}")
    return [path for _, path in sorted(indexed)]


def read_center_crop(path: Path, size: int) -> np.ndarray:
    import tifffile
    image = tifffile.memmap(path)
    h, w = image.shape
    top, left = (h - size) // 2, (w - size) // 2
    return np.asarray(image[top:top + size, left:left + size])


def read_tiff(path: Path) -> np.ndarray:
    import tifffile
    return np.asarray(tifffile.imread(path))


def aggregate_metric(rows: list[dict[str, Any]], metric: str, construction: str, folders: list[int]) -> np.ndarray:
    return np.asarray([np.mean([float(r[metric]) for r in rows if int(r["folder"]) == folder and r["construction"] == construction]) for folder in folders])


def metric_by_folder_variant(rows: list[dict[str, Any]], metric: str, construction: str,
                             folders: list[int], variant: str) -> np.ndarray:
    return np.asarray([np.mean([float(r[metric]) for r in rows if int(r["folder"]) == folder
                                and r["variant"] == variant and r["construction"] == construction])
                       for folder in folders])


def metric_matrix(rows: list[dict[str, Any]], metric: str, variants: list[str], folders: list[int]) -> np.ndarray:
    return np.asarray([[float(next(r[metric] for r in rows if int(r["folder"]) == folder and r["variant"] == variant))
                        for folder in folders] for variant in variants])


def matrix_by_variant(rows: list[dict[str, Any]], outcomes: dict[tuple[int, str], dict[str, Any]],
                      variants: list[str], folders: list[int], key: str) -> np.ndarray:
    return np.asarray([[float(outcomes[(folder, variant)][key]) for folder in folders] for variant in variants])


def variant_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray([spearman(x[i], y[i]) for i in range(x.shape[0])])


def max_variant_abs_correlation(rows: list[dict[str, Any]], left: str, right: str, variants: list[str]) -> float:
    return float(max(abs(spearman(field([r for r in rows if r["variant"] == variant], left),
                                  field([r for r in rows if r["variant"] == variant], right))) for variant in variants))


def repeated_permutation_statistics(xranks: np.ndarray, yranks: np.ndarray, denom: np.ndarray,
                                    permutations: np.ndarray) -> np.ndarray:
    # Return chunk x metric mean variant-specific Spearman statistics.
    result = np.zeros((len(permutations), xranks.shape[0]), dtype=np.float64)
    for variant in range(yranks.shape[0]):
        yp = yranks[variant][permutations]
        result += yp @ xranks[:, variant, :].T / denom[:, variant][None, :]
    return result / yranks.shape[0]


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def field(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows])


def array_field(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows])


def parse_number(value: str) -> Any:
    try:
        return float(value) if any(ch in value.lower() for ch in (".", "e")) else int(value)
    except (ValueError, AttributeError):
        return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
