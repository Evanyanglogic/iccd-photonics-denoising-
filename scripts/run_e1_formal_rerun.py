"""Single auditable entry point for the E1 formal ICCD characterization rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from e1_formal_common import load_config, read_csv, sha256_file, write_json


STEP_COMMANDS = [
    ("input_audit", ["scripts/e1_formal_analysis.py", "--module", "input_audit"]),
    ("noise_summary", ["scripts/e1_formal_analysis.py", "--module", "noise_summary"]),
    ("mean_variance", ["scripts/e1_formal_analysis.py", "--module", "mean_variance"]),
    ("robustness", ["scripts/e1_formal_analysis.py", "--module", "robustness"]),
    ("temporal_stability", ["scripts/audit_iccd_temporal_stability.py"]),
    ("stable_component", ["scripts/analyze_repeatable_stable_component.py"]),
    ("row_column", ["scripts/analyze_row_column_structure.py"]),
    ("spatial", ["scripts/e1_formal_analysis.py", "--module", "spatial"]),
    ("combined", ["scripts/e1_formal_analysis.py", "--module", "combined"]),
]

REQUIRED_CSVS = [
    "input_audit/input_manifest.csv",
    "input_audit/data_integrity_report.csv",
    "input_audit/frame_level_statistics.csv",
    "noise_summary/folder_noise_summary.csv",
    "mean_variance/mean_variance_bins.csv",
    "mean_variance/fano_like_summary.csv",
    "robustness/robustness_by_crop_and_frames.csv",
    "temporal_stability/temporal_drift_summary.csv",
    "stable_component/stable_component_summary.csv",
    "row_column/row_column_summary.csv",
    "spatial/spatial_correlation_summary.csv",
    "combined/folder_eligibility_summary.csv",
]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_config = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(source_config)
    apply_overrides(config, args)
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    config["output_root"] = str(output_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_root}")
    output_root.mkdir(parents=True)
    create_output_tree(output_root)
    provenance = output_root / "provenance"
    started = utc_now()
    initial_status = git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    commit = git(repo_root, ["rev-parse", "HEAD"]).strip()
    write_text(provenance / "git_commit.txt", commit + "\n")
    write_text(provenance / "git_status.txt", initial_status)
    write_text(provenance / "git_diff.patch", git(repo_root, ["diff", "--binary", "HEAD"]))
    write_text(provenance / "command.txt", subprocess.list2cmdline(sys.argv) + "\n")
    save_environment(provenance, repo_root)

    import yaml

    snapshot = provenance / "config.resolved.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    hash_paths = source_paths(repo_root)
    write_hashes(hash_paths, provenance / "script_hashes.csv", repo_root)
    manifest: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "started_at_utc": started,
        "ended_at_utc": None,
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_worktree_clean_at_start": initial_status.strip() == "",
        "source_config": str(source_config),
        "resolved_config": str(snapshot),
        "command": subprocess.list2cmdline(sys.argv),
        "steps": [],
        "status": "RUNNING",
    }
    write_json(manifest, provenance / "run_manifest.json")

    try:
        for name, command in STEP_COMMANDS:
            run_step(name, command, snapshot, repo_root, output_root, manifest)
        verification = verify_run(config, output_root, repo_root, initial_status, args.smoke)
        manifest["status"] = verification["status"]
        exit_code = 0 if verification["status"] != "FAILED" else 2
    except Exception as exc:
        verification = {"status": "FAILED", "checks": [], "error": f"{type(exc).__name__}: {exc}"}
        write_json(verification, output_root / "verification_status.json")
        write_text(output_root / "verification_report.md", f"# E1 Verification Report\n\n- Status: **FAILED**\n- Error: `{verification['error']}`\n")
        manifest["status"] = "FAILED"
        exit_code = 1
        raise
    finally:
        manifest["ended_at_utc"] = utc_now()
        manifest["duration_seconds"] = elapsed_seconds(started, manifest["ended_at_utc"])
        manifest["outputs"] = output_hashes(output_root)
        write_json(manifest, provenance / "run_manifest.json")
    print(json.dumps(verification, indent=2), flush=True)
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1_formal_rerun_20260717.yaml")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--folders", nargs="*", type=int)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_root:
        config["output_root"] = args.output_root
    if args.folders:
        config["folders"] = args.folders
    if args.max_frames > 0:
        config["max_frames"] = args.max_frames
    if args.smoke:
        max_frames = args.max_frames or 16
        config["max_frames"] = max_frames
        config["experiment_id"] = f"{config['experiment_id']}_smoke"
        config["folders"] = args.folders or [int(config["folders"][0])]
        config["crop_sizes"] = [64, 128, 256]
        config["primary_crop_size"] = 256
        config["frame_counts"] = [4, 8, 12, max_frames]
        config["integrity"]["expected_frame_count"] = 200
        config["integrity"]["pixel_statistics_stride"] = 32
        for module in ["noise_summary", "mean_variance", "temporal_stability", "stable_component", "row_column", "spatial"]:
            config[module]["frame_count"] = max_frames
        config["stable_component"]["block_size"] = 4
        config["row_column"]["block_size"] = 4
        config["spatial"]["max_radius"] = 64


def create_output_tree(output_root: Path) -> None:
    for name in [
        "provenance",
        "input_audit",
        "noise_summary",
        "mean_variance",
        "robustness",
        "temporal_stability",
        "stable_component",
        "row_column",
        "spatial",
        "combined",
        "logs",
    ]:
        (output_root / name).mkdir()


def source_paths(repo_root: Path) -> list[Path]:
    names = [
        "scripts/summarize_single_condition_noise.py",
        "scripts/fit_mean_variance_curve.py",
        "scripts/evaluate_noise_robustness.py",
        "scripts/analyze_iccd_spatial_correlation.py",
        "scripts/e1_formal_common.py",
        "scripts/e1_formal_analysis.py",
        "scripts/audit_iccd_temporal_stability.py",
        "scripts/analyze_repeatable_stable_component.py",
        "scripts/analyze_row_column_structure.py",
        "scripts/run_e1_formal_rerun.py",
        "configs/e1_formal_rerun_20260717.yaml",
    ]
    return [repo_root / name for name in names]


def run_step(
    name: str,
    command: list[str],
    config_path: Path,
    repo_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
) -> None:
    full_command = [sys.executable, *command, "--config", str(config_path)]
    started = utc_now()
    log_path = output_root / "logs" / f"{name}.log"
    result = subprocess.run(full_command, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    step = {
        "name": name,
        "command": subprocess.list2cmdline(full_command),
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "exit_code": result.returncode,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    manifest["steps"].append(step)
    write_json(manifest, output_root / "provenance" / "run_manifest.json")
    print(f"[{name}] exit={result.returncode}", flush=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"E1 step {name} failed with exit code {result.returncode}")


def verify_run(
    config: dict[str, Any], output_root: Path, repo_root: Path, initial_status: str, smoke: bool
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for relative in REQUIRED_CSVS:
        path = output_root / relative
        check(f"required_csv:{relative}", path.is_file() and path.stat().st_size > 0, str(path))

    integrity = read_csv(output_root / "input_audit" / "data_integrity_report.csv")
    expected_folders = {int(value) for value in config["folders"]}
    observed_folders = {int(float(row["folder"])) for row in integrity}
    check("all_configured_folders_present", observed_folders == expected_folders, f"expected={sorted(expected_folders)} observed={sorted(observed_folders)}")
    check("integrity_all_pass", all(row["status"] == "PASS" for row in integrity), f"statuses={[row['status'] for row in integrity]}")

    frame_rows = read_csv(output_root / "input_audit" / "frame_level_statistics.csv")
    check("dtype_uint16", all(row["dtype"] == str(config["dtype_expected"]) for row in frame_rows), f"rows={len(frame_rows)}")
    expected_shape = "x".join(str(value) for value in config["image_shape_expected"])
    check("shape_expected", all(row["shape"] == expected_shape for row in frame_rows), expected_shape)

    robustness = read_csv(output_root / "robustness" / "robustness_by_crop_and_frames.csv")
    expected_robustness = len(expected_folders) * len(set(config["crop_sizes"])) * len(set(config["frame_counts"]))
    combinations = {(int(float(row["folder"])), int(float(row["crop_size"])), int(float(row["frame_count"]))) for row in robustness}
    check("robustness_full_factorial", len(robustness) == expected_robustness and len(combinations) == expected_robustness, f"expected={expected_robustness} observed={len(robustness)}")

    finite_failures = []
    principal = {
        "noise_summary/folder_noise_summary.csv": ["value", "temporal_std_mean", "temporal_var_mean"],
        "mean_variance/fano_like_summary.csv": ["value", "fano_like_dn"],
        "temporal_stability/temporal_drift_summary.csv": ["value", "lag1_residual_correlation"],
        "stable_component/stable_component_summary.csv": ["value", "minimum_split_map_correlation"],
        "row_column/row_column_summary.csv": ["value", "row_pattern_energy_dn", "column_pattern_energy_dn"],
        "spatial/spatial_correlation_summary.csv": ["value", "radial_autocorr_r1"],
    }
    for relative, fields in principal.items():
        for index, row in enumerate(read_csv(output_root / relative)):
            for field in fields:
                try:
                    value = float(row[field])
                except Exception:
                    value = float("nan")
                if not math.isfinite(value):
                    finite_failures.append(f"{relative}:{index}:{field}")
    check("principal_metrics_finite", not finite_failures, ";".join(finite_failures[:20]))

    noise = {int(float(row["folder"])): row for row in read_csv(output_root / "noise_summary" / "folder_noise_summary.csv")}
    matching = {
        int(float(row["folder"])): row
        for row in robustness
        if int(float(row["crop_size"])) == int(config["primary_crop_size"])
        and int(float(row["frame_count"])) == int(config["noise_summary"]["frame_count"])
    }
    tolerance = float(config["verification"]["robustness_relative_tolerance"])
    differences = {
        folder: abs(float(noise[folder]["temporal_std_mean"]) - float(matching[folder]["temporal_std_mean"]))
        / max(abs(float(noise[folder]["temporal_std_mean"])), 1e-12)
        for folder in expected_folders
    }
    check("cross_output_recompute_consistent", all(value <= tolerance for value in differences.values()), json.dumps(differences, sort_keys=True))

    provenance_files = [
        "git_commit.txt",
        "git_status.txt",
        "git_diff.patch",
        "environment.txt",
        "pip_freeze.txt",
        "gpu_info.txt",
        "script_hashes.csv",
        "run_manifest.json",
        "config.resolved.yaml",
        "command.txt",
    ]
    check("provenance_complete", all((output_root / "provenance" / name).is_file() for name in provenance_files), ",".join(provenance_files))
    final_status = git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    check("worktree_unchanged_by_run", final_status == initial_status, f"before={initial_status!r} after={final_status!r}")
    check("committed_clean_code", initial_status.strip() == "", "formal VERIFIED-RUN requires a clean worktree at start")
    manifest_hash = sha256_file(output_root / "input_audit" / "input_manifest.csv")
    check("input_manifest_hashed", len(manifest_hash) == 64, manifest_hash)

    hard_fail = any(not item["passed"] for item in checks if item["name"] not in {"committed_clean_code"})
    status = "FAILED" if hard_fail else ("PARTIAL-RUN" if smoke or initial_status.strip() else "VERIFIED-RUN")
    payload = {"status": status, "smoke": smoke, "checks": checks, "input_manifest_sha256": manifest_hash}
    write_json(payload, output_root / "verification_status.json")
    report_lines = ["# E1 Verification Report", "", f"- Status: **{status}**", "", "## Checks", ""]
    for item in checks:
        report_lines.append(f"- {'PASS' if item['passed'] else 'FAIL'} `{item['name']}`: {item['detail']}")
    report_lines.extend(["", "## Recomputed Folder Values", "", "| folder | temporal std (summary) | temporal std (robustness) | relative difference |", "|---:|---:|---:|---:|"])
    for folder in sorted(expected_folders):
        report_lines.append(
            f"| {folder} | {float(noise[folder]['temporal_std_mean']):.9g} | "
            f"{float(matching[folder]['temporal_std_mean']):.9g} | {differences[folder]:.3g} |"
        )
    report_lines.extend(["", "All values above are recomputed from the cited bottom-level CSV files.", ""])
    write_text(output_root / "verification_report.md", "\n".join(report_lines))
    return payload


def save_environment(provenance: Path, repo_root: Path) -> None:
    environment_lines = [
        f"timestamp_utc={utc_now()}",
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"cwd={repo_root}",
    ]
    for module_name in ["numpy", "scipy", "PIL", "tifffile", "torch", "yaml", "skimage", "matplotlib"]:
        try:
            module = __import__(module_name)
            environment_lines.append(f"{module_name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            environment_lines.append(f"{module_name}=unavailable:{type(exc).__name__}")
    try:
        import torch

        environment_lines.extend(
            [
                f"torch_cuda_available={torch.cuda.is_available()}",
                f"torch_cuda_version={torch.version.cuda}",
                f"torch_cudnn_version={torch.backends.cudnn.version()}",
                f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
            ]
        )
    except Exception:
        pass
    write_text(provenance / "environment.txt", "\n".join(environment_lines) + "\n")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    write_text(provenance / "pip_freeze.txt", freeze.stdout or "")
    try:
        gpu = subprocess.run(["nvidia-smi"], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        gpu_text = gpu.stdout or ""
    except Exception as exc:
        gpu_text = f"nvidia-smi unavailable: {type(exc).__name__}: {exc}\n"
    write_text(provenance / "gpu_info.txt", gpu_text)


def write_hashes(paths: list[Path], output_path: Path, repo_root: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in paths:
            writer.writerow({"relative_path": str(path.relative_to(repo_root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})


def output_hashes(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            rows.append({"relative_path": str(path.relative_to(output_root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stdout}")
    return result.stdout or ""


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


if __name__ == "__main__":
    raise SystemExit(main())
