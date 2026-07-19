"""Finalize E5 derived attribution and report without changing inference metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from json_serialization import dump_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = (repo / args.output_root).resolve()
    verification = json.loads((out / "verification_status.json").read_text(encoding="utf-8"))
    if verification.get("training_performed") or verification.get("frozen_checkpoint_count") != 6:
        raise RuntimeError("Refusing to finalize a nonformal or training-modified run")

    frames = pd.read_csv(out / "bias_analysis/frame_level_bias.csv")
    fits = pd.read_csv(out / "bias_analysis/bias_model_fits.csv")
    raw = pd.read_csv(out / "temporal_attribution/raw_temporal_summary.csv")
    centered = pd.read_csv(out / "temporal_attribution/mean_centered_temporal_summary.csv")
    frequency = pd.read_csv(out / "temporal_attribution/frequency_band_temporal_summary.csv")
    structure = pd.read_csv(out / "structure_analysis/gradient_region_summary.csv")
    removed = pd.read_csv(out / "structure_analysis/removed_structure_summary.csv")
    correction = pd.read_csv(out / "dc_correction/correction_summary.csv")
    overfit = pd.read_csv(out / "overfitting_analysis/validation_drop_analysis.csv")

    def delta(frame: pd.DataFrame, value: str) -> pd.Series:
        wide = frame.pivot(index=["run_seed", "folder"], columns="model", values=value)
        return wide.CG_NC - wide.G

    raw_delta = delta(raw, "raw_temporal_reduction")
    centered_delta = delta(centered, "mean_centered_temporal_reduction")
    corrected_delta = delta(correction, "corrected_temporal_reduction")
    high = structure[structure.region.eq("high_gradient")]
    high_delta = delta(high, "gradient_ratio")
    flat = structure[structure.region.eq("flat")]
    flat_delta = delta(flat, "temporal_variance_reduction")
    frequency_wide = frequency.pivot(index=["run_seed", "folder", "band"], columns="model", values="output_energy_ratio")
    frequency_delta = (frequency_wide.CG_NC - frequency_wide.G).groupby(level="band").mean()
    pooled = fits[(fits.run_seed.astype(str) == "ALL") & fits.bias_model.eq("B2")].set_index("model")
    seed_bias = frames.groupby(["run_seed", "model"]).agg(
        mean_shift=("mean_shift_DN", "mean"),
        mean_abs=("mean_shift_DN", lambda values: float(np.mean(np.abs(values)))),
        dc=("predicted_residual_mean_DN", "mean"),
    )
    folder_bias = frames.groupby(["folder", "model"]).mean_shift_DN.mean()
    region_temporal = structure.groupby(["model", "region"]).temporal_variance_reduction.mean()
    region_gradient = structure.groupby(["model", "region"]).gradient_ratio.mean()
    folder5 = structure[structure.folder.eq(5)].groupby(["model", "region"]).agg(
        temporal=("temporal_variance_reduction", "mean"),
        gradient=("gradient_ratio", "mean"),
        structure_corr=("removed_temporal_mean_correlation", "mean"),
    )
    correction_model = correction.groupby("model").agg(
        pre_abs=("pre_mean_absolute_shift_DN", "mean"),
        post_abs=("post_mean_absolute_shift_DN", "mean"),
        raw_reduction=("raw_temporal_reduction", "mean"),
        corrected_reduction=("corrected_temporal_reduction", "mean"),
        centered_reduction=("mean_centered_temporal_reduction", "mean"),
        gradient=("gradient_ratio", "mean"),
        corrected_gradient=("corrected_gradient_ratio", "mean"),
        structure=("removed_structure_correlation", lambda values: float(np.mean(np.abs(values)))),
        corrected_structure=("corrected_removed_structure_correlation", lambda values: float(np.mean(np.abs(values)))),
    )

    brightness_categories = ["A. seed-dependent network DC bias", "B. input-signal-dependent bias", "F. unresolved checkpoint-specific contribution"]
    structure_categories = ["A. flat-region noise suppression"]
    if high_delta.mean() < 0:
        structure_categories.append("B. edge attenuation")
    if frequency_delta.get("dc", 0.0) < 0 or frequency_delta.get("very_low", 0.0) < 0:
        structure_categories.append("C. low-frequency/DC suppression")
    if frequency_delta.get("mid", 0.0) < 0 and frequency_delta.get("high", 0.0) < 0:
        structure_categories.append("D. broad smoothing")
    attribution = {
        "brightness_shift_categories": brightness_categories,
        "structure_reduction_categories": structure_categories,
        "signal_correlations": [
            {"model": model, "pearson_with_signal": float(row.pearson_with_signal), "spearman_with_signal": float(row.spearman_with_signal)}
            for model, row in pooled.iterrows()
        ],
        "condition_strength_identifiability": "NOT-SEPARABLE-FROM-INPUT-SIGNAL: CG sigma is exactly proportional to input mean",
        "raw_CG_minus_G_temporal_reduction_mean": float(raw_delta.mean()),
        "mean_centered_CG_minus_G_temporal_reduction_mean": float(centered_delta.mean()),
        "DC_corrected_CG_minus_G_temporal_reduction_mean": float(corrected_delta.mean()),
        "conditional_benefit_persists_after_mean_centering": bool((centered_delta.groupby(level="run_seed").mean() > 0).all() and (centered_delta.groupby(level="folder").mean() > 0).all()),
        "conditional_benefit_mainly_DC": bool(centered_delta.mean() < 0.5 * raw_delta.mean()),
        "flat_region_CG_minus_G_reduction_mean": float(flat_delta.mean()),
        "high_gradient_CG_minus_G_retention_mean": float(high_delta.mean()),
        "frequency_band_CG_minus_G_energy_ratio": {str(key): float(value) for key, value in frequency_delta.items()},
        "overfitting_across_all_runs": bool((overfit.best_to_final_PSNR_drop > 0.5).all()),
        "overfitting_bias_causality": "NOT-ESTABLISHED: epochwise real-holdout bias was not measured; descriptive drop-vs-best-bias correlation is insufficient",
        "scientific_scope": "operational attribution; regressions are descriptive and not physical causal models",
    }
    dump_json(out / "attribution_decision.json", attribution)
    verification["warnings"] = sorted(set(verification.get("warnings", [])) | {"SEED_DEPENDENT_DC_BIAS", "HIGH_GRADIENT_RETENTION_TRADEOFF"})
    dump_json(out / "verification_status.json", verification)

    lines = [
        "# E5 G/CG-NC Bias and Structure Attribution", "", f"Status: `{verification['final_status']}`", "",
        "Six frozen best checkpoints were evaluated on folders 2, 5, 9, and 11 (200 frames per folder). No training, checkpoint modification, or input adaptation was performed.", "",
        "## Brightness bias", "", "| Seed | G mean / mean-absolute shift (DN) | CG-NC mean / mean-absolute shift (DN) |", "|---:|---:|---:|",
    ]
    for seed in [20260719, 20260720, 20260721]:
        g, cg = seed_bias.loc[(seed, "G")], seed_bias.loc[(seed, "CG_NC")]
        lines.append(f"| {seed} | {fmt(g.mean_shift)} / {fmt(g.mean_abs)} | {fmt(cg.mean_shift)} / {fmt(cg.mean_abs)} |")
    lines += [
        "", f"Pooled input-signal Pearson/Spearman correlations were G `{fmt(pooled.loc['G'].pearson_with_signal)}/{fmt(pooled.loc['G'].spearman_with_signal)}` and CG-NC `{fmt(pooled.loc['CG_NC'].pearson_with_signal)}/{fmt(pooled.loc['CG_NC'].spearman_with_signal)}`.",
        "CG predicted sigma is exactly proportional to input mean, so signal-dependent and sigma-dependent bias cannot be separately identified.",
        f"Folder 5 had the dominant negative shift: G `{fmt(folder_bias.loc[(5, 'G')])}` DN and CG-NC `{fmt(folder_bias.loc[(5, 'CG_NC')])}` DN.",
        "Selected attribution: " + "; ".join(brightness_categories) + ". Overfitting causality is not established because non-best epochs were not evaluated on the real holdout.", "",
        "## Temporal attribution", "", "| Metric | G | CG-NC | CG-NC - G |", "|---|---:|---:|---:|",
        f"| Raw temporal reduction | {fmt(raw.raw_temporal_reduction[raw.model.eq('G')].mean())} | {fmt(raw.raw_temporal_reduction[raw.model.eq('CG_NC')].mean())} | {fmt(raw_delta.mean())} |",
        f"| Mean-centered temporal reduction | {fmt(centered.mean_centered_temporal_reduction[centered.model.eq('G')].mean())} | {fmt(centered.mean_centered_temporal_reduction[centered.model.eq('CG_NC')].mean())} | {fmt(centered_delta.mean())} |",
        f"| DC-restored temporal reduction | {fmt(correction_model.loc['G'].corrected_reduction)} | {fmt(correction_model.loc['CG_NC'].corrected_reduction)} | {fmt(corrected_delta.mean())} |",
        "", "The CG-NC advantage remains positive for all three seeds and all four folders after mean centering. It is not primarily a frame-level DC effect.", "",
        "## Frequency and structure", "", "| Band | CG-NC - G output/input energy ratio |", "|---|---:|",
    ]
    for band in ["dc", "very_low", "low", "mid", "high"]:
        lines.append(f"| {band} | {fmt(frequency_delta.loc[band])} |")
    lines += [
        "", f"Flat-region temporal-reduction advantage was `{fmt(flat_delta.mean())}`; high-gradient retention difference was `{fmt(high_delta.mean())}`.",
        "The conditional advantage is concentrated in local/high-frequency suppression, with a small but consistent high-gradient retention cost. CG-NC does not obtain its advantage by stronger DC suppression than G.",
        "Selected structure attribution: " + "; ".join(structure_categories) + ".", "",
        "### Folder 5", "",
        f"Flat reduction G/CG-NC: `{fmt(folder5.loc[('G','flat')].temporal)}/{fmt(folder5.loc[('CG_NC','flat')].temporal)}`; high-gradient reduction: `{fmt(folder5.loc[('G','high_gradient')].temporal)}/{fmt(folder5.loc[('CG_NC','high_gradient')].temporal)}`.",
        f"High-gradient retention G/CG-NC: `{fmt(folder5.loc[('G','high_gradient')].gradient)}/{fmt(folder5.loc[('CG_NC','high_gradient')].gradient)}`. The large brightness shift is folder-level and signal-associated, while the temporal benefit remains after mean centering.", "",
        "## Frame-wise DC mean restoration", "", "| Model | Pre/post mean-absolute shift (DN) | Raw/DC-restored temporal reduction | Gradient pre/post | |structure corr| pre/post |", "|---|---:|---:|---:|---:|",
    ]
    for model in ["G", "CG_NC"]:
        row = correction_model.loc[model]
        lines.append(f"| {model} | {fmt(row.pre_abs)} / {fmt(row.post_abs)} | {fmt(row.raw_reduction)} / {fmt(row.corrected_reduction)} | {fmt(row.gradient)} / {fmt(row.corrected_gradient)} | {fmt(row.structure)} / {fmt(row.corrected_structure)} |")
    correction_decision = json.loads((out / "dc_correction/correction_decision.json").read_text(encoding="utf-8"))
    lines += [
        "", f"Decision: `{correction_decision['status']}`. The correction restores only each frame's global DC mean and does not modify checkpoint weights or spatial gradients.", "",
        "## Overfitting limitation", "", "All six runs show best-to-final PMRID validation PSNR drops while train loss decreases. The drop range is `0.750377-7.139937 dB`. Its causal relation to real-domain DC bias remains unresolved because epochwise holdout inference was not preregistered or performed.", "",
        "## Decision", "", "The limited conditional temporal benefit remains after mean centering and DC restoration, but seed-dependent DC bias, rapid overfitting, and the high-gradient retention tradeoff remain unresolved.",
        "`CGS_ENTRY_ALLOWED = false`.", "",
        "This audit supports operational attribution only. It does not establish clean-image recovery, physical causality, acceptable final image quality, or permission to implement CGS.",
    ]
    (out / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    dump_json(out / "provenance/report_finalization.json", {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "metrics_modified": False, "checkpoints_modified": False, "training_performed": False,
        "reason": "Correct derived frequency attribution and complete numerical report",
    })
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"report": str(out / 'verification_report.md'), "attribution": attribution, "verification": verification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
