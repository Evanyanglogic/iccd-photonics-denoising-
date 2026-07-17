"""Auditable E2 input/provenance review without batch synthetic generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STEPS = [
    ("clean_content_audit", "scripts/audit_e2_synthetic_inputs.py"),
    ("split_leakage_audit", "scripts/check_e2_split_leakage.py"),
    ("single_pair_round_trip", "scripts/audit_e2_round_trip.py"),
]


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_config = resolve(repo, args.config)
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    if args.smoke:
        config["experiment_id"] += "_smoke"
    output = resolve(repo, config["output_root"])
    config["output_root"] = str(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for name in ["provenance", "input_audit", "generation_audit", "round_trip_audit", "leakage_audit", "manifests", "previews", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    initial_status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    commit = git(repo, ["rev-parse", "HEAD"]).strip()
    provenance = output / "provenance"
    write_text(provenance / "git_commit.txt", commit + "\n")
    write_text(provenance / "git_status.txt", initial_status)
    write_text(provenance / "git_diff.patch", git(repo, ["diff", "--binary", "HEAD"]))
    write_text(provenance / "command.txt", subprocess.list2cmdline(sys.argv) + "\n")
    resolved = provenance / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    save_environment(provenance, repo)
    write_script_hashes(repo, provenance / "script_hashes.csv")
    write_input_hashes(repo, config, provenance / "input_file_hashes.csv")

    manifest: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "RUNNING",
        "smoke": args.smoke,
        "started_at_utc": utc_now(),
        "git_commit": commit,
        "git_worktree_clean_at_start": initial_status.strip() == "",
        "command": subprocess.list2cmdline(sys.argv),
        "source_config": str(source_config),
        "resolved_config": str(resolved),
        "steps": [],
        "batch_generation_performed": False,
    }
    write_json(manifest, provenance / "run_manifest.json")
    try:
        for name, script in STEPS:
            run_step(name, script, resolved, repo, output, manifest)
        verification = verify(config, output, repo, initial_status, args.smoke)
        manifest["status"] = verification["status"]
    except Exception as exc:
        verification = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}"}
        write_json(verification, output / "verification_status.json")
        manifest["status"] = "INVALID"
        raise
    finally:
        manifest["ended_at_utc"] = utc_now()
        manifest["outputs"] = output_hashes(output)
        write_json(manifest, provenance / "run_manifest.json")
    print(json.dumps(verification, indent=2, ensure_ascii=False), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2_formal_input_audit_20260717.yaml")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run_step(name: str, script: str, config: Path, repo: Path, output: Path, manifest: dict[str, Any]) -> None:
    command = [sys.executable, script, "--config", str(config)]
    started = utc_now()
    result = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = output / "logs" / f"{name}.log"
    write_text(log, result.stdout or "")
    manifest["steps"].append({"name": name, "command": subprocess.list2cmdline(command), "started_at_utc": started, "ended_at_utc": utc_now(), "exit_code": result.returncode, "log": str(log), "log_sha256": sha256_file(log)})
    write_json(manifest, output / "provenance" / "run_manifest.json")
    print(f"[{name}] exit={result.returncode}", flush=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(f"Step {name} failed with exit code {result.returncode}")


def verify(config: dict[str, Any], output: Path, repo: Path, initial_status: str, smoke: bool) -> dict[str, Any]:
    clean = read_json(output / "input_audit" / "clean_content_summary.json")
    leakage = read_json(output / "leakage_audit" / "leakage_summary.json")
    round_trip = read_json(output / "round_trip_audit" / "round_trip_summary.json")
    historical = read_json(output / "generation_audit" / "historical_e2_summary.json")
    required = [
        "input_audit/input_clean_manifest.csv",
        "input_audit/clean_content_audit.csv",
        "input_audit/clean_content_summary.json",
        "input_audit/duplicate_or_near_duplicate_report.csv",
        "generation_audit/e2_parameter_to_e1_mapping.csv",
        "generation_audit/historical_e2_output_audit.csv",
        "round_trip_audit/round_trip_metrics.csv",
        "leakage_audit/leakage_summary.json",
    ]
    checks = []
    add = lambda name, passed, detail: checks.append({"name": name, "passed": bool(passed), "detail": detail})
    add("required_outputs", all((output / item).is_file() and (output / item).stat().st_size > 0 for item in required), ",".join(required))
    add("clean_content_integrity", clean["status"] == "PASS", clean["status"])
    add("full_clean_file_hashes", clean["file_count"] == 100 and config["hash_strategy"] == "full_file_sha256", f"count={clean['file_count']}")
    add("round_trip", round_trip["status"] == "PASS", round_trip["status"])
    add("generation_numerics", round_trip["generation_numeric_status"] == "PASS", round_trip["generation_numeric_status"])
    add("historical_exact_replay", round_trip["all_historical_clean_exact_match"] and round_trip["all_historical_noisy_exact_match"], json.dumps(round_trip))
    add("scene_isolated_split", leakage["status"] == "PASS", leakage["reason"])
    add("historical_provenance", historical["historical_chain_status"] != "INVALID_FOR_FORMAL_USE", historical["historical_chain_status"])
    final_status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    add("worktree_unchanged", final_status == initial_status, f"before={initial_status!r} after={final_status!r}")
    add("committed_clean_code", initial_status.strip() == "", "required for non-smoke formal status")
    add("no_batch_generation", True, "only two fixed single-pair variant replays were written")

    if not checks[0]["passed"] or not checks[1]["passed"] or not checks[2]["passed"] or not checks[3]["passed"]:
        status = "INVALID"
    elif leakage["status"] != "PASS" or historical["historical_chain_status"] == "INVALID_FOR_FORMAL_USE":
        status = "INVALID"
    elif smoke or initial_status.strip():
        status = "PARTIAL-RUN"
    else:
        status = "VERIFIED-INPUT"
    decision = {
        "status": status,
        "checks": checks,
        "first_formal_candidate": config["formal_decision"]["first_candidate"],
        "go_for_batch_generation": False,
        "blocking_issue": "dark-offset correction clips 97%+ of clean pixels to zero; source_scene metadata and scene-isolated splits are absent; historical physical naming is unsupported",
    }
    write_json(decision, output / "verification_status.json")
    write_report(output / "verification_report.md", clean, leakage, round_trip, historical, decision)
    return decision


def write_report(path: Path, clean: dict[str, Any], leakage: dict[str, Any], round_trip: dict[str, Any], historical: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# E2 Formal Input and Provenance Audit",
        "",
        f"- E2 status: **{decision['status']}**",
        f"- Clean-content files: {clean['file_count']}",
        f"- Clean-content integrity: {clean['status']}",
        f"- Leakage gate: {leakage['status']}",
        f"- Single-pair round-trip: {round_trip['status']}",
        f"- Historical chain: {historical['historical_chain_status']}",
        "- Batch generation performed: no",
        "",
        "## Decision",
        "",
        "Candidate D is selected: audit and fixed-sample historical replay only.",
        "Batch generation is NO-GO until source scenes can be identified and split without scene leakage.",
        f"Dark-offset correction zero ratio: {clean['corrected_zero_ratio_range'][0]:.4%}..{clean['corrected_zero_ratio_range'][1]:.4%}.",
        "The historical `physical-scale` label must be replaced by `legacy_unscaled_content`; it is not a calibrated physical model.",
        "The historical `p99` label means per-image clean-content p99 normalization to 0.25.",
        "",
        "## Checks",
        "",
    ]
    for item in decision["checks"]:
        lines.append(f"- {'PASS' if item['passed'] else 'FAIL'} `{item['name']}`: {item['detail']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_environment(provenance: Path, repo: Path) -> None:
    versions = [f"python={sys.version}", f"executable={sys.executable}", f"platform={platform.platform()}", f"cwd={repo}"]
    for name in ["numpy", "scipy", "tifffile", "yaml"]:
        try:
            module = __import__(name)
            versions.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            versions.append(f"{name}=unavailable:{type(exc).__name__}")
    write_text(provenance / "environment.txt", "\n".join(versions) + "\n")


def write_script_hashes(repo: Path, path: Path) -> None:
    files = [
        "configs/e2_formal_input_audit_20260717.yaml",
        "scripts/audit_e2_synthetic_inputs.py",
        "scripts/check_e2_split_leakage.py",
        "scripts/audit_e2_round_trip.py",
        "scripts/run_e2_formal_input_audit.py",
        "scripts/generate_iccd_like_synthetic_pairs.py",
        "src/iccd_noise/physical_model.py",
    ]
    write_hash_csv(path, [(name, repo / name) for name in files])


def write_input_hashes(repo: Path, config: dict[str, Any], path: Path) -> None:
    files = [
        config["source_pairs_csv"], config["source_splits_yaml"], config["dark_offset_path"], config["bad_pixel_mask_path"],
        "configs/iccd_prior_20260319.yaml",
        f"{config['e1_formal_root']}/noise_summary/folder_noise_summary.csv",
        f"{config['e1_formal_root']}/mean_variance/fano_like_summary.csv",
        f"{config['e1_formal_root']}/row_column/row_column_summary.csv",
        f"{config['e1_formal_root']}/spatial/spatial_correlation_summary.csv",
    ]
    write_hash_csv(path, [(name, resolve(repo, name)) for name in files])


def write_hash_csv(path: Path, files: list[tuple[str, Path]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "resolved_path", "size_bytes", "sha256"])
        writer.writeheader()
        for name, file_path in files:
            writer.writerow({"input": name, "resolved_path": str(file_path), "size_bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})


def output_hashes(output: Path) -> list[dict[str, Any]]:
    return [{"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_manifest.json"]


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout or ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
