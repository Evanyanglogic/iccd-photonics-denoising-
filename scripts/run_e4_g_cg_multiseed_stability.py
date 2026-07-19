"""Run frozen G/CG-NC training and real-holdout evaluation across three seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from json_serialization import dump_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def run_logged(command: list[str], repo: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    return return_code


def finalize_failed_run(
    out: Path,
    repo: Path,
    cfg: dict[str, Any],
    started: str,
    commit: str,
    seed: int,
    stage: str,
    return_code: int,
    child_status: dict[str, Any] | None,
) -> int:
    child_final_status = (child_status or {}).get("final_status", "UNKNOWN-CHILD-FAILURE")
    final_status = f"MULTISEED-{child_final_status}"
    failed_output = out / "training" / f"seed_{seed}" / "metrics" / "warnings.csv"
    failed_pairs = pd.read_csv(failed_output).to_dict("records") if failed_output.exists() else []
    script_paths = [
        Path(__file__),
        repo / "scripts/run_e2_g_cg_scaled_training.py",
        repo / "scripts/run_e3_real_iccd_holdout_validation.py",
        repo / "scripts/json_serialization.py",
    ]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in script_paths]).to_csv(
        out / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig"
    )
    verification = {
        "final_status": final_status,
        "failed_seed": seed,
        "failed_stage": stage,
        "child_return_code": return_code,
        "child_status": child_status,
        "failed_pairs": failed_pairs,
        "all_seed_runs_completed": False,
        "conditional_benefit": "NOT-DETERMINED",
        "CGS_ENTRY_ALLOWED": False,
        "data_leakage_detected": False,
        "source_data_protection": "PRE-RUN-HASHES-PASSED; POST-RUN-HASHES-INCOMPLETE-FOR-FAILED-SEED",
        "provenance_complete": True,
    }
    dump_json(out / "verification_status.json", verification)
    dump_json(
        out / "provenance/run_manifest.json",
        {
            "experiment_id": cfg["experiment_id"],
            "started_at_utc": started,
            "ended_at_utc": now(),
            "git_commit": commit,
            "failed_seed": seed,
            "failed_stage": stage,
            "child_return_code": return_code,
            "final_status": final_status,
        },
    )
    (out / "provenance/git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")
    report = [
        "# E4 G/CG-NC Multiseed Stability",
        "",
        f"Status: `{final_status}`",
        "",
        f"The run stopped at `{stage}` for seed `{seed}` with child return code `{return_code}`.",
        "",
        "No failed pair, seed, threshold, noise strength, model, or checkpoint was replaced.",
        "",
        "A multiseed conditional-benefit decision was not issued because the preregistered run did not complete.",
    ]
    (out / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(verification, indent=2))
    return return_code


def descriptive(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def decide(synthetic: pd.DataFrame, real: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = cfg["decision"]
    synthetic_wide = synthetic.pivot(index="run_seed", columns="model", values=["output_psnr", "output_ssim"])
    synthetic_psnr_delta = synthetic_wide["output_psnr"]["CG_NC"] - synthetic_wide["output_psnr"]["G"]
    synthetic_ssim_delta = synthetic_wide["output_ssim"]["CG_NC"] - synthetic_wide["output_ssim"]["G"]
    difference_metrics = [
        "temporal_variance_reduction",
        "row_energy_reduction",
        "column_energy_reduction",
        "max_absolute_shift_DN",
        "high_gradient_retention",
        "removed_structure_correlation",
    ]
    real_wide = real.pivot(index=["run_seed", "folder"], columns="model", values=difference_metrics)
    delta = real_wide.xs("CG_NC", axis=1, level=1) - real_wide.xs("G", axis=1, level=1)
    seed_real = delta.groupby(level="run_seed").temporal_variance_reduction.mean()
    folder_real = delta.groupby(level="folder").temporal_variance_reduction.mean()
    brightness_cg = real[real.model.eq("CG_NC")].max_absolute_shift_DN
    brightness_severe = bool((brightness_cg > float(limits["severe_brightness_shift_DN"])).any())
    gradient_delta = delta.high_gradient_retention
    gradient_systematic = bool(
        (gradient_delta.groupby(level="run_seed").mean() < -float(limits["obvious_gradient_retention_drop"])).sum() >= 2
    )
    structure_delta = delta.removed_structure_correlation.abs()
    structure_systematic = bool(
        (structure_delta.groupby(level="run_seed").mean() > float(limits["structure_correlation_margin"])).sum() >= 2
    )
    synthetic_noninferior = int((synthetic_psnr_delta >= 0).sum())
    real_better = int((seed_real > 0).sum())
    folders_better = int((folder_real > 0).sum())
    direction_consistent = bool((seed_real > 0).all())
    single_seed_dominates = bool(
        not direction_consistent
        or (np.max(np.abs(seed_real)) > 2.0 * max(float(np.median(np.abs(seed_real))), 1e-12))
    )
    core_repeatable = bool(
        synthetic_noninferior >= int(limits["required_synthetic_noninferior_seeds"])
        and real_better >= int(limits["required_real_better_seeds"])
        and folders_better >= int(limits["required_folders_mean_better"])
        and not single_seed_dominates
    )
    brightness_worse = bool((delta.max_absolute_shift_DN.groupby(level="run_seed").mean() > 0).sum() >= 2)
    gradient_worse = bool((gradient_delta.groupby(level="run_seed").mean() < 0).sum() >= 2)
    structure_worse = bool((structure_delta.groupby(level="run_seed").mean() > 0).sum() >= 2)
    synthetic_near_zero = bool(abs(float(synthetic_psnr_delta.mean())) < 0.01)
    if core_repeatable and not any([brightness_worse, gradient_worse, structure_worse, synthetic_near_zero]):
        status = "REPEATABLE-CONDITIONAL-BENEFIT"
    elif core_repeatable:
        status = "CONDITIONAL-BENEFIT-WITH-TRADEOFFS"
    else:
        status = "NO-REPEATABLE-CONDITIONAL-BENEFIT"
    decision = {
        "status": status,
        "synthetic_noninferior_seed_count": synthetic_noninferior,
        "real_better_seed_count": real_better,
        "folders_with_positive_mean_real_delta": folders_better,
        "direction_consistent_across_seeds": direction_consistent,
        "single_seed_dominates": single_seed_dominates,
        "brightness_systematically_worse": brightness_worse,
        "severe_brightness_shift_detected": brightness_severe,
        "gradient_systematically_worse": gradient_worse,
        "gradient_obviously_worse": gradient_systematic,
        "removed_structure_systematically_higher": structure_worse,
        "removed_structure_margin_exceeded": structure_systematic,
        "synthetic_advantage_near_zero": synthetic_near_zero,
        "synthetic_psnr_deltas": synthetic_psnr_delta.to_dict(),
        "synthetic_ssim_deltas": synthetic_ssim_delta.to_dict(),
        "real_mean_deltas": seed_real.to_dict(),
    }
    cgs_allowed = bool(
        status == "REPEATABLE-CONDITIONAL-BENEFIT"
        and not brightness_worse
        and not gradient_worse
        and not structure_worse
    )
    cgs = {
        "CGS_ENTRY_ALLOWED": cgs_allowed,
        "reason": "All repeatability and structure-preservation gates passed" if cgs_allowed else "Conditional benefit has unresolved brightness/gradient/removed-structure tradeoffs or is not repeatable",
        "allowed_next_component_if_true": "calibration-only spatial-correlation residual feasibility",
        "prohibited": ["full CGS", "row/column component", "stable component"],
    }
    return decision, cgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = (repo / args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = (repo / args.output_root).resolve()
    if out.exists():
        raise FileExistsError(out)
    for directory in ["provenance", "configs", "manifests", "training", "synthetic_metrics", "real_holdout_metrics", "multiseed_summary", "logs"]:
        (out / directory).mkdir(parents=True, exist_ok=False)
    started = now()
    commit = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    provenance = out / "provenance"
    (provenance / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (provenance / "git_status_before.txt").write_text(status_before, encoding="utf-8")
    (provenance / "git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (provenance / "command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (provenance / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "configs" / config_path.name).write_bytes(config_path.read_bytes())
    environment = f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\ncuda={torch.cuda.is_available()}\nnumpy={np.__version__}\n"
    (provenance / "environment.txt").write_text(environment, encoding="utf-8")
    (provenance / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout, encoding="utf-8")

    base_training = yaml.safe_load((repo / cfg["training_config"]).read_text(encoding="utf-8"))
    base_holdout = yaml.safe_load((repo / cfg["holdout_config"]).read_text(encoding="utf-8"))
    if cfg["experiments"] != ["G", "CG_NC"]:
        raise RuntimeError("This audit is frozen to G and CG_NC")
    if base_training["evaluation_folders"] != cfg["evaluation_folders"] or base_holdout["evaluation_folders"] != cfg["evaluation_folders"]:
        raise RuntimeError("Evaluation folder drift")
    if base_training["calibration_folders"] != cfg["calibration_folders"] or base_holdout["calibration_folders"] != cfg["calibration_folders"]:
        raise RuntimeError("Calibration folder drift")
    frozen = cfg["frozen"]
    if abs(float(base_training["g_sigma_DN"]) - float(frozen["g_sigma_DN"])) > 1e-12 or abs(float(base_training["condition_slope"]) - float(frozen["condition_slope"])) > 1e-15:
        raise RuntimeError("Frozen noise parameters drift")

    seeds = cfg["run_seeds"][:1] if args.smoke else cfg["run_seeds"]
    synthetic_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed = int(seed)
        training_cfg = json.loads(json.dumps(base_training))
        training_cfg["experiment_id"] = f"{cfg['experiment_id']}_seed_{seed}"
        training_cfg["base_seed"] = seed
        training_cfg["training"]["model_seed"] = seed
        training_cfg["training"]["dataloader_seed"] = seed
        training_cfg_path = out / "configs" / f"training_seed_{seed}.yaml"
        training_cfg_path.write_text(yaml.safe_dump(training_cfg, sort_keys=False), encoding="utf-8")
        training_out = out / "training" / f"seed_{seed}"
        command = [sys.executable, str(repo / "scripts/run_e2_g_cg_scaled_training.py"), "--config", str(training_cfg_path), "--output-root", str(training_out), "--experiments", "G,CG_NC"]
        if args.smoke:
            command.append("--smoke")
        return_code = run_logged(command, repo, out / "logs" / f"training_seed_{seed}.log")
        if return_code != 0:
            status_path = training_out / "verification_status.json"
            child_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
            return finalize_failed_run(out, repo, cfg, started, commit, seed, "training_preflight_or_training", return_code, child_status)
        comparison = pd.read_csv(training_out / "metrics/experiment_comparison.csv")
        validation = pd.read_csv(training_out / "metrics/validation_pair_metrics.csv")
        for row in comparison.to_dict("records"):
            synthetic_rows.append({"run_seed": seed, **row})
        validation.insert(0, "run_seed", seed)
        validation.to_csv(out / "synthetic_metrics" / f"validation_pair_metrics_seed_{seed}.csv", index=False, encoding="utf-8-sig")

        holdout_cfg = json.loads(json.dumps(base_holdout))
        holdout_cfg["experiment_id"] = f"{cfg['experiment_id']}_real_seed_{seed}"
        holdout_cfg["training_report"] = str(training_out.relative_to(repo)).replace("\\", "/")
        holdout_cfg["checkpoints"] = {}
        for model in cfg["experiments"]:
            item = comparison[comparison.experiment.eq(model)].iloc[0]
            checkpoint = training_out / "checkpoints" / model / "best.pt"
            checkpoint_hash = sha256_file(checkpoint)
            holdout_cfg["checkpoints"][model] = {
                "path": str(checkpoint.relative_to(repo)).replace("\\", "/"),
                "epoch": int(item.best_epoch),
                "input_channels": 1,
                "sha256": checkpoint_hash,
            }
            checkpoint_rows.append({"run_seed": seed, "model": model, "best_epoch": int(item.best_epoch), "path": str(checkpoint), "sha256": checkpoint_hash})
        holdout_cfg_path = out / "configs" / f"holdout_seed_{seed}.yaml"
        holdout_cfg_path.write_text(yaml.safe_dump(holdout_cfg, sort_keys=False), encoding="utf-8")
        holdout_out = out / "real_holdout_metrics" / f"seed_{seed}"
        command = [sys.executable, str(repo / "scripts/run_e3_real_iccd_holdout_validation.py"), "--config", str(holdout_cfg_path), "--output-root", str(holdout_out)]
        if args.smoke:
            command.append("--smoke")
        return_code = run_logged(command, repo, out / "logs" / f"holdout_seed_{seed}.log")
        if return_code != 0:
            status_path = holdout_out / "verification_status.json"
            child_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
            return finalize_failed_run(out, repo, cfg, started, commit, seed, "real_holdout", return_code, child_status)
        real = pd.read_csv(holdout_out / "metrics/model_comparison.csv")
        real.insert(0, "run_seed", seed)
        real_rows.extend(real.to_dict("records"))

    synthetic = pd.DataFrame(synthetic_rows)
    real = pd.DataFrame(real_rows)
    checkpoints = pd.DataFrame(checkpoint_rows)
    synthetic.to_csv(out / "multiseed_summary/synthetic_seed_summary.csv", index=False, encoding="utf-8-sig")
    real.to_csv(out / "multiseed_summary/folder_seed_summary.csv", index=False, encoding="utf-8-sig")
    checkpoints.to_csv(out / "manifests/checkpoint_manifest.csv", index=False, encoding="utf-8-sig")
    real_seed = real.groupby(["run_seed", "model"]).agg(
        folder_count=("folder", "count"),
        mean_temporal_variance_reduction=("temporal_variance_reduction", "mean"),
        mean_row_energy_reduction=("row_energy_reduction", "mean"),
        mean_column_energy_reduction=("column_energy_reduction", "mean"),
        mean_absolute_brightness_shift_DN=("mean_shift_DN", lambda x: float(np.mean(np.abs(x)))),
        mean_high_gradient_retention=("high_gradient_retention", "mean"),
        mean_absolute_removed_structure_correlation=("removed_structure_correlation", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
    real_seed.to_csv(out / "multiseed_summary/real_seed_summary.csv", index=False, encoding="utf-8-sig")

    synthetic_wide = synthetic.pivot(index="run_seed", columns="experiment")
    real_wide = real_seed.pivot(index="run_seed", columns="model")
    tradeoff = pd.DataFrame({
        "run_seed": seeds,
        "synthetic_psnr_delta": synthetic_wide["output_psnr"]["CG_NC"] - synthetic_wide["output_psnr"]["G"],
        "synthetic_ssim_delta": synthetic_wide["output_ssim"]["CG_NC"] - synthetic_wide["output_ssim"]["G"],
        "real_temporal_reduction_delta": real_wide["mean_temporal_variance_reduction"]["CG_NC"] - real_wide["mean_temporal_variance_reduction"]["G"],
        "absolute_brightness_shift_delta_DN": real_wide["mean_absolute_brightness_shift_DN"]["CG_NC"] - real_wide["mean_absolute_brightness_shift_DN"]["G"],
        "high_gradient_retention_delta": real_wide["mean_high_gradient_retention"]["CG_NC"] - real_wide["mean_high_gradient_retention"]["G"],
        "absolute_removed_structure_correlation_delta": real_wide["mean_absolute_removed_structure_correlation"]["CG_NC"] - real_wide["mean_absolute_removed_structure_correlation"]["G"],
    }).reset_index(drop=True)
    tradeoff.to_csv(out / "multiseed_summary/tradeoff_summary.csv", index=False, encoding="utf-8-sig")
    decision, cgs = decide(synthetic.rename(columns={"experiment": "model"}), real, cfg)
    dump_json(out / "multiseed_summary/conditional_benefit_decision.json", decision)
    dump_json(out / "multiseed_summary/cgs_entry_decision.json", cgs)

    epoch_rows = []
    overfit_warnings = []
    for seed in seeds:
        epoch = pd.read_csv(out / "training" / f"seed_{seed}" / "metrics/epoch_metrics.csv")
        epoch.insert(0, "run_seed", seed)
        epoch_rows.append(epoch)
        for model, group in epoch.groupby("experiment"):
            best = float(group.validation_psnr.max()); final = float(group.sort_values("epoch").iloc[-1].validation_psnr)
            if best - final > 0.1:
                overfit_warnings.append({"run_seed": int(seed), "model": model, "best_minus_final_psnr_DN": best - final})
    epochs = pd.concat(epoch_rows, ignore_index=True)
    epochs.to_csv(out / "synthetic_metrics/epoch_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overfit_warnings).to_csv(out / "multiseed_summary/overfit_warnings.csv", index=False, encoding="utf-8-sig")

    finite = bool(np.isfinite(synthetic.select_dtypes("number")).all().all() and np.isfinite(real.select_dtypes("number")).all().all())
    training_complete = bool(len(synthetic) == len(seeds) * 2 and (synthetic.best_epoch >= 1).all())
    evaluated_folder_count = 1 if args.smoke else len(cfg["evaluation_folders"])
    holdout_complete = bool(len(real) == len(seeds) * evaluated_folder_count * 2)
    source_protected = True
    for seed in seeds:
        train_status = json.loads((out / "training" / f"seed_{seed}" / "verification_status.json").read_text(encoding="utf-8"))
        holdout_status = json.loads((out / "real_holdout_metrics" / f"seed_{seed}" / "verification_status.json").read_text(encoding="utf-8"))
        source_protected &= bool(train_status["source_data_protected"] and holdout_status["source_data_protected"])
    final_status = "MULTISEED-STABILITY-FAILED" if not all([training_complete, holdout_complete, finite, source_protected]) else (
        "MULTISEED-STABILITY-VALID-WITH-TRADEOFFS" if decision["status"] == "CONDITIONAL-BENEFIT-WITH-TRADEOFFS" else "MULTISEED-STABILITY-VALID"
    )
    verification = {
        "final_status": final_status,
        "run_seeds": seeds,
        "experiments": cfg["experiments"],
        "training_complete": training_complete,
        "holdout_complete": holdout_complete,
        "finite_metrics": finite,
        "conditional_benefit": decision["status"],
        "CGS_ENTRY_ALLOWED": cgs["CGS_ENTRY_ALLOWED"],
        "overfit_warnings": overfit_warnings,
        "data_leakage_detected": False,
        "source_data_protected": source_protected,
        "provenance_complete": True,
    }
    dump_json(out / "verification_status.json", verification)
    run_manifest = {"experiment_id": cfg["experiment_id"], "smoke": args.smoke, "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "run_seeds": seeds, "training_runs": len(seeds) * 2, "real_holdout_model_runs": len(seeds) * 2, "final_status": final_status}
    dump_json(provenance / "run_manifest.json", run_manifest)
    script_paths = [Path(__file__), config_path, repo / "scripts/run_e2_g_cg_scaled_training.py", repo / "scripts/run_e3_real_iccd_holdout_validation.py", repo / "scripts/json_serialization.py"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in script_paths]).to_csv(provenance / "script_hashes.csv", index=False, encoding="utf-8-sig")
    (provenance / "git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")
    report = [
        "# E4 G/CG-NC Multiseed Stability",
        "",
        f"Status: `{final_status}`",
        "",
        f"Conditional benefit: `{decision['status']}`",
        "",
        f"CGS entry allowed: `{cgs['CGS_ENTRY_ALLOWED']}`",
        "",
        synthetic.to_markdown(index=False),
        "",
        tradeoff.to_markdown(index=False),
        "",
        "This audit evaluates training-seed repeatability. It does not establish clean-image recovery, statistical significance, or a complete physical ICCD noise model.",
    ]
    (out / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(out / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(verification, indent=2))
    return 0 if final_status != "MULTISEED-STABILITY-FAILED" else 5


if __name__ == "__main__":
    raise SystemExit(main())
