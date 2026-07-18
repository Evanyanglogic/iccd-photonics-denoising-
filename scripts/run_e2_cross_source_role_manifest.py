"""Formal runner for the E2 cross-source role and isolation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from build_e2_cross_source_role_manifest import make_isolation_matrix, make_pmrid_scene_manifest, make_role_rows, prohibited_transitions, sha256
from json_serialization import dump_json
from validate_e2_source_role_isolation import validate_manifests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = repo / args.output_root
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest_outputs = {name: repo / path for name, path in config["outputs"].items()}
    existing = [str(path) for path in manifest_outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite formal manifests: {existing}")
    for name in ("provenance", "logs"):
        (output / name).mkdir(parents=True, exist_ok=False if name == "provenance" else True)
    started = now(); commit = git(repo, "rev-parse", "HEAD").strip(); status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status_before.strip():
        raise RuntimeError("Formal role-manifest run requires a clean code worktree")

    inputs = config["inputs"]
    e1 = json.loads((repo / inputs["e1_status"]).read_text(encoding="utf-8"))
    smoke = json.loads((repo / inputs["candidate_a_smoke_status"]).read_text(encoding="utf-8"))
    stability = json.loads((repo / inputs["candidate_a_stability_status"]).read_text(encoding="utf-8"))
    validation_root = repo / inputs["validation_audit_root"]
    validation = json.loads((validation_root / "verification_status.json").read_text(encoding="utf-8"))
    if e1.get("status") != "VERIFIED-RUN" or smoke.get("status") != "GO-SMOKE" or stability.get("final_status") != "GO-STABILITY" or validation.get("status") != "VALIDATION-READY-FOUND":
        raise RuntimeError("Frozen prerequisite status mismatch")
    report_commit = git(repo, "log", "-1", "--format=%H", "--", str(validation_root.relative_to(repo))).strip()
    if report_commit != inputs["validation_audit_expected_commit"]:
        raise RuntimeError(f"Validation audit commit drift: {report_commit}")

    scmos_manifest = pd.read_csv(repo / inputs["scmos_manifest"])
    if len(scmos_manifest) != 100 or set(scmos_manifest.allowed_role) != {"debug_only"}:
        raise RuntimeError("sCMOS role drift: all 100 rows must remain debug_only")
    source_summary = pd.read_csv(validation_root / "candidate_source_summary.csv")
    inventory = pd.read_csv(validation_root / "candidate_directory_inventory.csv")
    prior_hashes = pd.read_csv(validation_root / "provenance/input_hashes.csv")

    pmrid_root = Path(inputs["pmrid_root"]); benchmark = Path(inputs["pmrid_benchmark"])
    pmrid_files_before = sorted(path for path in pmrid_root.rglob("*") if path.is_file())
    scmos_dir = Path(scmos_manifest.absolute_path.iloc[0]).parent
    scmos_count_before = len(list(scmos_dir.glob("*.tif*")))
    scmos_paths = [Path(path) for path in scmos_manifest.sort_values("sha256").head(8).absolute_path]
    prior_pmrid_gt = [Path(path) for path in prior_hashes.path if str(path).lower().endswith("gt.raw") and str(pmrid_root).lower() in str(path).lower()]
    if len(prior_pmrid_gt) != 39:
        raise RuntimeError(f"Expected 39 PMRID GT hashes from validation audit, found {len(prior_pmrid_gt)}")
    protected = [benchmark] + prior_pmrid_gt + scmos_paths
    before = {str(path): {"sha256": file_hash(path), "mtime_ns": path.stat().st_mtime_ns} for path in protected}
    pmrid_count_before = len(pmrid_files_before)

    roles = make_role_rows(config, source_summary, inventory)
    isolation = make_isolation_matrix(roles)
    pmrid = make_pmrid_scene_manifest(config, prior_hashes)
    conflicts, checks = validate_manifests(roles, isolation, pmrid)
    transitions = prohibited_transitions()

    output_names = {
        "role_manifest": "e2_cross_source_role_manifest_20260718.csv",
        "isolation_matrix": "e2_cross_source_isolation_matrix_20260718.csv",
        "pmrid_scene_manifest": "e2_pmrid_validation_scene_manifest_20260718.csv",
    }
    frames = {"role_manifest": roles, "isolation_matrix": isolation, "pmrid_scene_manifest": pmrid}
    for key, frame in frames.items():
        report_path = output / output_names[key]
        frame.to_csv(report_path, index=False, encoding="utf-8-sig")
        manifest_outputs[key].parent.mkdir(parents=True, exist_ok=True)
        manifest_outputs[key].write_bytes(report_path.read_bytes())
        if file_hash(report_path) != file_hash(manifest_outputs[key]):
            raise RuntimeError(f"Manifest copy hash mismatch: {key}")
    conflicts.to_csv(output / "source_role_conflicts.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(output / "prohibited_role_transitions.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "e1_noise_characterization": "VERIFIED-RUN", "candidate_a_numeric_smoke": "GO-SMOKE",
        "candidate_a_numeric_stability": "GO-STABILITY", "debug_content_source": "AVAILABLE",
        "validation_content_source": "AVAILABLE-WITH-DOMAIN-LIMITATION", "training_content_source": "MISSING",
        "formal_train_validation_split": "NOT-ALLOWED", "formal_synthetic_generation": "NOT-ALLOWED",
        "model_training": "NOT-ALLOWED", "real_iccd_evaluation": "NOT-READY",
    }
    dump_json(output / "pipeline_readiness.json", readiness)
    limitations = ["PMRID preprocessing plan is NOT-FROZEN", "PMRID is not ICCD-domain content", "formal training content source is MISSING"]
    final_status = "INVALID-ROLE-MANIFEST" if len(conflicts) else "ROLE-MANIFEST-VERIFIED-WITH-LIMITATIONS"
    verification = {"experiment_id": config["experiment_id"], "status": final_status, "source_count": len(roles), "role_counts": roles.allowed_role.value_counts().to_dict(), "source_role_conflict_count": len(conflicts), "prohibited_role_transition_count": len(transitions), "checks": checks, "limitations": limitations, "pipeline_readiness": readiness, "provenance_complete": False, "source_data_protected": False, "synthetic_pairs_generated": False, "training_or_split_performed": False, "next_task": "Audit and determine a formal training-content-source acquisition path using existing-data review, new acquisition, or a legally usable public source; do not generate synthetic pairs."}
    dump_json(output / "verification_status.json", verification)

    resolved = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    (output / "provenance/git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "provenance/git_status_before.txt").write_text(status_before, encoding="utf-8")
    (output / "provenance/git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (output / "provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (output / "provenance/environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\npandas={pd.__version__}\n", encoding="utf-8")
    (output / "provenance/resolved_config.yaml").write_text(resolved, encoding="utf-8")
    scripts = [Path(__file__), repo / "scripts/build_e2_cross_source_role_manifest.py", repo / "scripts/validate_e2_source_role_isolation.py", repo / "scripts/json_serialization.py", config_path]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": file_hash(path)} for path in scripts]).to_csv(output / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")

    pmrid_files_after = sorted(path for path in pmrid_root.rglob("*") if path.is_file())
    protection_rows = []
    for path_text, snapshot in before.items():
        path = Path(path_text); after_hash = file_hash(path); after_mtime = path.stat().st_mtime_ns
        protection_rows.append({"path": path_text, "sha256_before": snapshot["sha256"], "sha256_after": after_hash, "mtime_ns_before": snapshot["mtime_ns"], "mtime_ns_after": after_mtime, "unchanged": snapshot["sha256"] == after_hash and snapshot["mtime_ns"] == after_mtime})
    protection = pd.DataFrame(protection_rows)
    protection.to_csv(output / "provenance/source_protection.csv", index=False, encoding="utf-8-sig")
    source_protected = bool(len(pmrid_files_after) == pmrid_count_before and len(list(scmos_dir.glob("*.tif*"))) == scmos_count_before == 100 and protection.unchanged.all())
    run = {"experiment_id": config["experiment_id"], "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "status": final_status, "source_data_protected": source_protected, "source_write_performed": False, "synthetic_pairs_generated": False, "training_or_split_performed": False, "pmrid_file_count_before": pmrid_count_before, "pmrid_file_count_after": len(pmrid_files_after), "scmos_file_count_before": scmos_count_before, "scmos_file_count_after": len(list(scmos_dir.glob("*.tif*")))}
    dump_json(output / "provenance/run_manifest.json", run)
    dump_json(output / "logs/run.log", {"verification": verification, "source_protected": source_protected})
    (output / "provenance/git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")

    report = f"""# E2 Cross-Source Role Manifest\n\nStatus: `{final_status}`\n\nThe manifest freezes {len(roles)} audited sources or risk-reference entries. The current 100-image sCMOS source remains `debug_only`; all 39 PMRID benchmark GT RAW entries remain `validation_content_only` in four official scene blocks; and the formal training-content placeholder remains `MISSING`. No source is assigned `training_content_only`.\n\nPMRID preprocessing remains `NOT-FROZEN`, and PMRID content is mobile Bayer RAW rather than ICCD-domain ground truth. Therefore formal split construction, synthetic generation, model training, checkpoint selection, and real ICCD evaluation remain not allowed.\n"""
    (output / "verification_report.md").write_text(report, encoding="utf-8")

    required = ["e2_cross_source_role_manifest_20260718.csv", "e2_cross_source_isolation_matrix_20260718.csv", "e2_pmrid_validation_scene_manifest_20260718.csv", "source_role_conflicts.csv", "prohibited_role_transitions.csv", "pipeline_readiness.json", "verification_status.json", "verification_report.md", "provenance/run_manifest.json", "provenance/source_protection.csv"]
    copies_match = all(file_hash(output / output_names[key]) == file_hash(manifest_outputs[key]) for key in frames)
    verification["provenance_complete"] = all((output / name).is_file() for name in required) and copies_match
    verification["report_manifest_copies_match"] = copies_match
    verification["source_data_protected"] = source_protected
    if not source_protected or not verification["provenance_complete"]:
        verification["status"] = "FAILED"
        run["status"] = "FAILED"
    dump_json(output / "verification_status.json", verification)
    dump_json(output / "provenance/run_manifest.json", run)
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": file_hash(path)})
    pd.DataFrame(hashes).to_csv(output / "output_hashes.csv", index=False, encoding="utf-8-sig")
    json.loads((output / "pipeline_readiness.json").read_text(encoding="utf-8")); json.loads((output / "verification_status.json").read_text(encoding="utf-8")); json.loads((output / "provenance/run_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": verification["status"], "source_count": len(roles), "role_counts": verification["role_counts"], "conflicts": len(conflicts), "pmrid_rows": len(pmrid), "readiness": readiness, "source_protected": source_protected}, ensure_ascii=False, indent=2))
    return 0 if verification["status"].startswith("ROLE-MANIFEST-VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
