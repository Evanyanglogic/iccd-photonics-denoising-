"""Aggregate three-seed 2x2 probe results and estimate causal factor effects."""

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
    training_root = root / "training"
    metric_paths = sorted(training_root.glob("*_seed*/real_eval/probe_metrics.csv"))
    expected = len(config["training"]["seeds"]) * len(config["construction"]["variants"])
    if len(metric_paths) != expected:
        raise ValueError(f"Expected {expected} probe metric files, found {len(metric_paths)}")
    rows = [row for path in metric_paths for row in read_csv(path)]
    folder_rows = summarize_folders(rows)
    cell_rows = summarize_cells(folder_rows)
    variant_rows = summarize_three_seed_variants(rows, folder_rows)
    effect_rows = calculate_effects(folder_rows)
    effect_summary = summarize_effects(effect_rows, config)
    uncertainty = uncertainty_budget(cell_rows, config, root)
    decision = decide(effect_summary, uncertainty)

    output_dir = root / "factor_analysis"
    output_dir.mkdir(exist_ok=True)
    write_csv(rows, output_dir / "all_probe_metrics.csv")
    write_csv(folder_rows, output_dir / "folder_metrics.csv")
    write_csv(cell_rows, output_dir / "cell_seed_summary.csv")
    write_csv(variant_rows, output_dir / "variant_three_seed_summary.csv")
    write_csv(effect_rows, output_dir / "factor_effects_by_folder_seed_reference.csv")
    write_csv(effect_summary, output_dir / "factor_effect_summary.csv")
    write_json(output_dir / "uncertainty_budget.json", uncertainty)
    write_json(output_dir / "factorial_decision.json", decision)
    save_plot(output_dir / "factor_effects.png", effect_summary)
    write_report(output_dir / "factorial_effect_report.md", decision, uncertainty, variant_rows, effect_summary)
    print(json.dumps(decision, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5_noise_factorial.yaml")
    parser.add_argument("--input-root", default="")
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


def summarize_folders(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], int(row["seed"]), row["reference"], int(row["folder"]))].append(row)
    fields = ["psnr_gain", "ssim", "ssim_gain", "gradient_ratio_to_noisy", "residual_std", "brightness_bias_to_reference"]
    out: list[dict[str, Any]] = []
    for (variant, seed, reference, folder), group in groups.items():
        item: dict[str, Any] = {
            "variant": variant,
            "seed": seed,
            "reference": reference,
            "folder": folder,
            "pair_count": len(group),
            "positive_pair_fraction": float(np.mean([float(row["psnr_gain"]) > 0.0 for row in group])),
        }
        for field in fields:
            item[field] = float(np.mean([float(row[field]) for row in group]))
        out.append(item)
    return sorted(out, key=lambda row: (row["variant"], row["seed"], row["reference"], row["folder"]))


def summarize_cells(folder_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in folder_rows:
        groups[(row["variant"], int(row["seed"]), row["reference"])].append(row)
    out: list[dict[str, Any]] = []
    for (variant, seed, reference), group in groups.items():
        gains = np.asarray([float(row["psnr_gain"]) for row in group])
        out.append(
            {
                "variant": variant,
                "seed": seed,
                "reference": reference,
                "mean_folder_psnr_gain": float(np.mean(gains)),
                "std_folder_psnr_gain": float(np.std(gains, ddof=1)),
                "positive_folder_count": int(np.sum(gains > 0.0)),
                "worst_folder_psnr_gain": float(np.min(gains)),
                "mean_ssim": mean_field(group, "ssim"),
                "mean_gradient_ratio_to_noisy": mean_field(group, "gradient_ratio_to_noisy"),
                "mean_residual_std": mean_field(group, "residual_std"),
                "mean_brightness_bias": mean_field(group, "brightness_bias_to_reference"),
            }
        )
    return sorted(out, key=lambda row: (row["variant"], row["seed"], row["reference"]))


def summarize_three_seed_variants(
    pair_rows: list[dict[str, str]], folder_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    t_critical_df2 = 4.3026527297
    out: list[dict[str, Any]] = []
    for variant in ("P-L", "P-H", "H-L", "H-H"):
        variant_pairs = [row for row in pair_rows if row["variant"] == variant]
        seeds = sorted({int(row["seed"]) for row in variant_pairs})
        seed_gains = np.asarray([
            np.mean([float(row["psnr_gain"]) for row in variant_pairs if int(row["seed"]) == seed])
            for seed in seeds
        ])
        seed_ssim = np.asarray([
            np.mean([float(row["ssim"]) for row in variant_pairs if int(row["seed"]) == seed])
            for seed in seeds
        ])
        seed_gradient = np.asarray([
            np.mean([float(row["gradient_ratio_to_noisy"]) for row in variant_pairs if int(row["seed"]) == seed])
            for seed in seeds
        ])
        mean_gain = float(np.mean(seed_gains))
        seed_std = float(np.std(seed_gains, ddof=1))
        half_width = t_critical_df2 * seed_std / np.sqrt(len(seed_gains))
        folder_means: dict[int, list[float]] = defaultdict(list)
        reference_folder_means: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in folder_rows:
            if row["variant"] != variant:
                continue
            folder = int(row["folder"])
            value = float(row["psnr_gain"])
            folder_means[folder].append(value)
            reference_folder_means[row["reference"]][folder].append(value)
        final_folder_gains = {folder: float(np.mean(values)) for folder, values in folder_means.items()}
        refs = sorted(reference_folder_means)
        gains_a = np.asarray([np.mean(reference_folder_means[refs[0]][folder]) for folder in sorted(final_folder_gains)])
        gains_b = np.asarray([np.mean(reference_folder_means[refs[1]][folder]) for folder in sorted(final_folder_gains)])
        out.append(
            {
                "variant": variant,
                "seed_count": len(seeds),
                "mean_folder_psnr_gain_db": mean_gain,
                "seed_std_psnr_gain_db": seed_std,
                "seed_t_ci95_low_db": mean_gain - half_width,
                "seed_t_ci95_high_db": mean_gain + half_width,
                "mean_ssim": float(np.mean(seed_ssim)),
                "seed_std_ssim": float(np.std(seed_ssim, ddof=1)),
                "mean_gradient_ratio_to_noisy": float(np.mean(seed_gradient)),
                "seed_std_gradient_ratio": float(np.std(seed_gradient, ddof=1)),
                "positive_folder_count": int(sum(value > 0.0 for value in final_folder_gains.values())),
                "worst_folder_psnr_gain_db": min(final_folder_gains.values()),
                "positive_pair_fraction": float(np.mean([float(row["psnr_gain"]) > 0.0 for row in variant_pairs])),
                "mean_residual_std": float(np.mean([float(row["residual_std"]) for row in variant_pairs])),
                "mean_brightness_bias": float(np.mean([float(row["brightness_bias_to_reference"]) for row in variant_pairs])),
                "reference_folder_sign_agreement": float(np.mean(np.sign(gains_a) == np.sign(gains_b))),
                "reference_folder_gain_correlation": float(np.corrcoef(gains_a, gains_b)[0, 1]),
            }
        )
    return out


def calculate_effects(folder_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["psnr_gain", "ssim", "gradient_ratio_to_noisy", "residual_std", "brightness_bias_to_reference"]
    lookup = {(row["variant"], int(row["seed"]), row["reference"], int(row["folder"])): row for row in folder_rows}
    combinations = sorted({(int(row["seed"]), row["reference"], int(row["folder"])) for row in folder_rows})
    out: list[dict[str, Any]] = []
    for seed, reference, folder in combinations:
        cells = {variant: lookup[(variant, seed, reference, folder)] for variant in ("P-L", "P-H", "H-L", "H-H")}
        definitions = {
            "strength_main": lambda v: 0.5 * ((v["P-H"] + v["H-H"]) - (v["P-L"] + v["H-L"])),
            "structure_main": lambda v: 0.5 * ((v["H-L"] + v["H-H"]) - (v["P-L"] + v["P-H"])),
            "interaction": lambda v: (v["H-H"] - v["H-L"]) - (v["P-H"] - v["P-L"]),
            "P-H_minus_P-L": lambda v: v["P-H"] - v["P-L"],
            "H-H_minus_H-L": lambda v: v["H-H"] - v["H-L"],
            "H-L_minus_P-L": lambda v: v["H-L"] - v["P-L"],
            "H-H_minus_P-H": lambda v: v["H-H"] - v["P-H"],
        }
        for effect, function in definitions.items():
            row: dict[str, Any] = {"effect": effect, "seed": seed, "reference": reference, "folder": folder}
            for field in fields:
                values = {variant: float(cell[field]) for variant, cell in cells.items()}
                row[field] = float(function(values))
            out.append(row)
    return out


def summarize_effects(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["effect"]].append(row)
    rng = np.random.default_rng(int(config["source"]["seed"]))
    iterations = int(config["decision"]["bootstrap_iterations"])
    out: list[dict[str, Any]] = []
    for effect, group in groups.items():
        folder_values: dict[int, list[float]] = defaultdict(list)
        seed_values: dict[int, list[float]] = defaultdict(list)
        reference_values: dict[str, list[float]] = defaultdict(list)
        for row in group:
            value = float(row["psnr_gain"])
            folder_values[int(row["folder"])].append(value)
            seed_values[int(row["seed"])].append(value)
            reference_values[row["reference"]].append(value)
        folder_means = np.asarray([np.mean(values) for values in folder_values.values()])
        bootstrap = np.asarray([np.mean(rng.choice(folder_means, len(folder_means), replace=True)) for _ in range(iterations)])
        seed_means = np.asarray([np.mean(values) for values in seed_values.values()])
        reference_means = np.asarray([np.mean(values) for values in reference_values.values()])
        mean = float(np.mean(folder_means))
        ci_low, ci_high = float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))
        seed_std = float(np.std(seed_means, ddof=1))
        reference_half_difference = float(abs(reference_means[0] - reference_means[1]) / 2.0)
        uncertainty = max(seed_std, reference_half_difference)
        out.append(
            {
                "effect": effect,
                "mean_psnr_gain_effect_db": mean,
                "folder_bootstrap_ci95_low": ci_low,
                "folder_bootstrap_ci95_high": ci_high,
                "seed_std_db": seed_std,
                "surrogate_reference_half_difference_db": reference_half_difference,
                "effect_to_uncertainty_ratio": abs(mean) / max(uncertainty, 1e-12),
                "stable_nonzero": int((ci_low > 0.0 or ci_high < 0.0) and abs(mean) > uncertainty),
                "mean_ssim_effect": mean_field(group, "ssim"),
                "mean_gradient_ratio_effect": mean_field(group, "gradient_ratio_to_noisy"),
                "mean_residual_std_effect": mean_field(group, "residual_std"),
                "mean_brightness_bias_effect": mean_field(group, "brightness_bias_to_reference"),
            }
        )
    return sorted(out, key=lambda row: row["effect"])


def uncertainty_budget(cell_rows: list[dict[str, Any]], config: dict[str, Any], root: Path) -> dict[str, Any]:
    by_variant_reference: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_variant_seed: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in cell_rows:
        by_variant_reference[(row["variant"], row["reference"])].append(float(row["mean_folder_psnr_gain"]))
        by_variant_seed[(row["variant"], int(row["seed"]))][row["reference"]] = float(row["mean_folder_psnr_gain"])
    seed_stds = []
    for variant in ("P-L", "P-H", "H-L", "H-H"):
        seed_means = []
        for seed in config["training"]["seeds"]:
            values = by_variant_seed[(variant, int(seed))]
            seed_means.append(float(np.mean(list(values.values()))))
        seed_stds.append(float(np.std(seed_means, ddof=1)))
    reference_differences = []
    for variant in ("P-L", "P-H", "H-L", "H-H"):
        a = np.mean(by_variant_reference[(variant, "reference_a_odd")])
        b = np.mean(by_variant_reference[(variant, "reference_b_even")])
        reference_differences.append(float(abs(a - b) / 2.0))
    validation = {row["variant"]: row for row in read_csv(root / "validation_variant_summary.csv")}
    low_ratio = float(validation["H-L"]["residual_std_mean"]) / float(validation["P-L"]["residual_std_mean"])
    high_ratio = float(validation["H-H"]["residual_std_mean"]) / float(validation["P-H"]["residual_std_mean"])
    construction_error_db = max(abs(20.0 * np.log10(low_ratio)), abs(20.0 * np.log10(high_ratio)))
    seed_uncertainty = max(seed_stds)
    reference_uncertainty = max(reference_differences)
    maximum = max(seed_uncertainty, reference_uncertainty, construction_error_db)
    condition_gain = 0.036
    return {
        "maximum_cell_seed_std_db": seed_uncertainty,
        "maximum_cell_reference_half_difference_db": reference_uncertainty,
        "matched_strength_construction_error_db": float(construction_error_db),
        "maximum_uncertainty_db": maximum,
        "prior_condition_strategy_gain_db": condition_gain,
        "prior_gain_exceeds_maximum_uncertainty": bool(condition_gain > maximum),
    }


def decide(effect_rows: list[dict[str, Any]], uncertainty: dict[str, Any]) -> dict[str, Any]:
    effects = {row["effect"]: row for row in effect_rows}
    strength = bool(effects["strength_main"]["stable_nonzero"])
    structure = bool(effects["structure_main"]["stable_nonzero"])
    interaction = bool(effects["interaction"]["stable_nonzero"])
    if interaction:
        case = "C_INTERACTION"
    elif strength and not structure:
        case = "A_STRENGTH_ONLY"
    elif structure and not strength:
        case = "B_STRUCTURE_ONLY"
    else:
        case = "D_UNSTABLE_OR_UNEXPLAINED"
    return {
        "case": case,
        "strength_main_stable": strength,
        "structure_main_stable": structure,
        "interaction_stable": interaction,
        "prior_condition_gain_exceeds_uncertainty": uncertainty["prior_gain_exceeds_maximum_uncertainty"],
        "claim_downgrade_required": bool(case == "D_UNSTABLE_OR_UNEXPLAINED" or not uncertainty["prior_gain_exceeds_maximum_uncertainty"]),
    }


def mean_field(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def save_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    primary = [row for row in summaries if row["effect"] in {"strength_main", "structure_main", "interaction"}]
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    means = np.asarray([float(row["mean_psnr_gain_effect_db"]) for row in primary])
    lows = means - np.asarray([float(row["folder_bootstrap_ci95_low"]) for row in primary])
    highs = np.asarray([float(row["folder_bootstrap_ci95_high"]) for row in primary]) - means
    axis.errorbar([row["effect"] for row in primary], means, yerr=np.vstack([lows, highs]), fmt="o", capsize=5)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Folder-level PSNR gain effect (dB)")
    axis.set_title("2x2 factorial effects with folder bootstrap CI")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    decision: dict[str, Any],
    uncertainty: dict[str, Any],
    variants: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> None:
    lines = [
        "# E5 Noise Structure-by-Strength Factorial Result",
        "",
        f"Decision: **{decision['case']}**",
        "",
        "## Three-Seed Cell Results", "",
        "| Variant | Folder gain | Seed SD | Seed 95% t-CI | Positive folders | Positive pairs | Grad/noisy | Worst folder | Reference sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variants:
        lines.append(
            f"| {row['variant']} | {float(row['mean_folder_psnr_gain_db']):.6f} | "
            f"{float(row['seed_std_psnr_gain_db']):.6f} | [{float(row['seed_t_ci95_low_db']):.6f}, "
            f"{float(row['seed_t_ci95_high_db']):.6f}] | {int(row['positive_folder_count'])}/10 | "
            f"{float(row['positive_pair_fraction']):.3f} | {float(row['mean_gradient_ratio_to_noisy']):.4f} | "
            f"{float(row['worst_folder_psnr_gain_db']):.6f} | {float(row['reference_folder_sign_agreement']):.3f} |"
        )
    lines.extend([
        "", "## Factor Effects", "",
        "| Effect | PSNR effect | Folder 95% CI | Seed SD | Reference half-diff | Stable |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in effects:
        lines.append(
            f"| {row['effect']} | {float(row['mean_psnr_gain_effect_db']):.6f} | "
            f"[{float(row['folder_bootstrap_ci95_low']):.6f}, {float(row['folder_bootstrap_ci95_high']):.6f}] | "
            f"{float(row['seed_std_db']):.6f} | {float(row['surrogate_reference_half_difference_db']):.6f} | "
            f"{bool(int(row['stable_nonzero']))} |"
        )
    lines.extend([
        "", "## Uncertainty Budget", "",
        f"- Maximum seed SD: {uncertainty['maximum_cell_seed_std_db']:.6f} dB",
        f"- Maximum reference half-difference: {uncertainty['maximum_cell_reference_half_difference_db']:.6f} dB",
        f"- Matched-strength construction error: {uncertainty['matched_strength_construction_error_db']:.6f} dB",
        f"- Prior 0.036 dB condition gain exceeds uncertainty: {uncertainty['prior_gain_exceeds_maximum_uncertainty']}",
        "", "## Claim Boundary", "",
        "- Results use temporal-mean surrogate references, not clean ground truth.",
        "- A factor is called stable only when its folder bootstrap CI excludes zero and its mean exceeds seed/reference variability.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
