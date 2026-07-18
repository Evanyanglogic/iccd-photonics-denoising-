"""Formal, non-materializing audit of E2 training-content-source strategies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from audit_local_training_content_candidates import build_local_candidate_audit
from audit_public_training_content_sources import build_public_candidates
from build_training_acquisition_plan import build_metadata_template, build_minimum_plan, build_recommended_plan
from json_serialization import dump_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_files(root: Path) -> int:
    if not root.exists():
        return -1
    return sum(len(files) for _, _, files in os.walk(root))


def route_comparison() -> pd.DataFrame:
    values = {
        "source_traceability": ["low-medium: histories are mixed", "high: captured under a frozen protocol", "high for official sources"],
        "pmrid_isolation": ["mixed; each source requires recovery", "high by design", "high at dataset level; verify after download"],
        "scene_independence": ["not defensible for available candidates", "high if 20-40 independent scenes are recorded", "dataset dependent; RAISE/FiveK broad, SIDD only 10 base scenes"],
        "data_scale": ["large only for contaminated derived PMRID tree", "60 minimum / 200 recommended files", "5,000-8,156 for leading candidates"],
        "bit_depth": ["mostly 8-bit derived PNG or role-ineligible sources", "controllable uint16 or camera-native high bit depth", "RAW/DNG or normalized Raw-RGB depending on dataset"],
        "processing_status": ["unknown or derived", "recordable from acquisition start", "documented but camera/dataset specific"],
        "acquisition_time": ["short audit, but no source passes", "moderate; operator and scene setup required", "short planning, long download/input audit"],
        "implementation_cost": ["low but unusable", "moderate", "low-medium monetary; high storage/preprocessing"],
        "license_risk": ["source records incomplete", "owned/project-governed capture; consent/privacy still required", "low for confirmed research licenses; dataset-specific restrictions"],
        "domain_match": ["mixed and not defensible", "best controllable match to grayscale/high-bit pipeline; still not ICCD clean", "camera RAW content, not ICCD domain"],
        "preprocessing_complexity": ["high due unknown derivation", "medium and auditable", "high for Bayer/proprietary RAW"],
        "leakage_risk": ["high", "low if compositions and groups are frozen", "low vs PMRID but historical-project overlap varies"],
        "reproducibility": ["low", "high with protocol, hashes and metadata", "high for official dataset; preprocessing must be frozen"],
        "recommendation": ["NO", "PRIMARY", "BACKUP"],
    }
    return pd.DataFrame([{"dimension": key, "route_a_local": val[0], "route_b_new_acquisition": val[1], "route_c_public": val[2]} for key, val in values.items()])


def leakage_analysis() -> pd.DataFrame:
    return pd.DataFrame([
        {"route": "Route A", "risk": "HIGH", "evidence": "No local candidate combines original-source provenance, defensible groups, unpolluted history and PMRID isolation.", "mitigation": "Do not upgrade any frozen local role."},
        {"route": "Route B", "risk": "LOW-CONDITIONAL", "evidence": "A new dataset can exclude PMRID and ICCD evaluation compositions and bind repeats to scene/acquisition blocks.", "mitigation": "Freeze role before capture and perform post-capture hash, duplicate, metadata and overlap audit."},
        {"route": "Route C/RAISE", "risk": "LOW-CONDITIONAL", "evidence": "Independent official dataset and cameras; no current local-project occurrence detected.", "mitigation": "Use training-only subset, preserve official metadata, audit hashes/content after approved download."},
        {"route": "Route C/FiveK", "risk": "LOW-CONDITIONAL", "evidence": "Independent photography corpus; no current local-project occurrence detected.", "mitigation": "Resolve file-scoped license mapping and freeze DNG/TIFF choice before download."},
        {"route": "Route C/SID", "risk": "HIGH", "evidence": "Local PMRID parent contains SID Sony lists and 10,666 historical derived training PNG files.", "mitigation": "Do not select without a dedicated history and license-scope audit."},
        {"route": "Route C/SIDD", "risk": "MEDIUM-HIGH", "evidence": "SIDD samples exist in the PNGAN parent and the released Raw-RGB is already normalized/black-level corrected.", "mitigation": "Not a preferred backup; require separate history and preprocessing audit."},
        {"route": "Route C/RENOIR", "risk": "UNACCEPTABLE", "evidence": "Official page does not state an explicit dataset license.", "mitigation": "Exclude until written license terms are confirmed."},
    ])


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
    output.mkdir(parents=True)
    (output / "provenance").mkdir()
    (output / "logs").mkdir()
    started = now()
    commit = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status_before.strip():
        raise RuntimeError("Formal strategy audit requires a clean worktree")

    inputs = config["inputs"]
    role_report = repo / inputs["role_audit_root"]
    role_status = json.loads((role_report / "verification_status.json").read_text(encoding="utf-8"))
    readiness_before = json.loads((role_report / "pipeline_readiness.json").read_text(encoding="utf-8"))
    role_commit = git(repo, "log", "-1", "--format=%H", "--", str(role_report.relative_to(repo))).strip()
    if role_status.get("status") != "ROLE-MANIFEST-VERIFIED-WITH-LIMITATIONS":
        raise RuntimeError("Frozen role-manifest status mismatch")
    if role_commit != inputs["role_audit_expected_commit"]:
        raise RuntimeError(f"Frozen role audit commit drift: {role_commit}")
    roles = pd.read_csv(repo / inputs["role_manifest"])
    scmos = roles.loc[roles.source_id == "scmos_500ms_current_100"].iloc[0]
    pmrid = roles.loc[roles.source_id == "pmrid_official_benchmark_gt_raw"].iloc[0]
    missing = roles.loc[roles.source_id == "formal_training_content_placeholder"].iloc[0]
    if scmos.allowed_role != "debug_only" or pmrid.allowed_role != "validation_content_only" or missing.final_status != "MISSING":
        raise RuntimeError("Frozen source-role drift")
    pmrid_scenes = pd.read_csv(repo / inputs["validation_scene_manifest"])
    if len(pmrid_scenes) != 39 or set(pmrid_scenes.scene_id.astype(str)) != {"Scene1", "Scene2", "Scene3", "Scene4"}:
        raise RuntimeError("PMRID validation scene manifest drift")

    scmos_manifest = pd.read_csv(repo / inputs["scmos_manifest"])
    protected_files = [Path(path) for path in scmos_manifest.sort_values("sha256").head(8).absolute_path]
    protected_files += [Path(path) for path in pmrid_scenes.sort_values("SHA256").head(4).paired_gt_path]
    file_snapshot = {str(path): {"sha256": sha256(path), "mtime_ns": path.stat().st_mtime_ns} for path in protected_files}
    roots = [Path(path) for path in config["protection_roots"]]
    root_counts_before = {str(path): count_files(path) for path in roots}

    local = build_local_candidate_audit(repo, config)
    minimum = build_minimum_plan(config)
    recommended = build_recommended_plan(config)
    metadata = build_metadata_template()
    public, evidence = build_public_candidates(config["public_audit"]["accessed_on"])
    comparison = route_comparison()
    leakage = leakage_analysis()
    primary = {
        "primary_strategy": "NEW-CONTENT-ACQUISITION",
        "formal_source_name": config["acquisition"]["formal_name"],
        "future_role_after_successful_input_audit": "training_content_only",
        "current_materialization_status": "NO",
        "current_training_source_status": "MISSING",
        "selection_basis": ["lowest controllable leakage risk", "source and role can be frozen before capture", "true scene/acquisition grouping can be recorded", "high-bit-depth grayscale compatibility can be controlled", "dataset-level isolation from PMRID validation"],
        "primary_limitations": ["capture has not occurred", "device metadata capability requires operator confirmation", "captured sCMOS content will retain sensor noise/pedestal until separately audited", "time and storage require manual confirmation"],
        "backup_strategies": [
            {"strategy": "LEGAL-PUBLIC-DATASET", "candidate": "RAISE-1k training-only subset", "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS"},
            {"strategy": "LEGAL-PUBLIC-DATASET", "candidate": "MIT-Adobe FiveK DNG", "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS"},
        ],
        "next_task": "Output and obtain human confirmation of the formal acquisition protocol, directory structure and metadata table; do not acquire data automatically.",
    }
    pipeline = {
        "training_source_strategy": "VERIFIED",
        "training_source_materialized": "NO",
        "training_source_audited": "NO",
        "training_content_source": "MISSING",
        "formal_training_content_manifest": "NOT-AVAILABLE",
        "formal_train_validation_split": "NOT-ALLOWED",
        "formal_synthetic_generation": "NOT-ALLOWED",
        "model_training": "NOT-ALLOWED",
        "validation_content_source": readiness_before["validation_content_source"],
    }

    outputs = {
        "local_training_candidate_audit.csv": local,
        "acquisition_strategy_minimum.csv": minimum,
        "acquisition_strategy_recommended.csv": recommended,
        "acquisition_metadata_template.csv": metadata,
        "public_dataset_candidates.csv": public,
        "public_source_evidence.csv": evidence,
        "strategy_comparison.csv": comparison,
        "leakage_risk_analysis.csv": leakage,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    dump_json(output / "primary_training_strategy.json", primary)
    dump_json(output / "pipeline_readiness.json", pipeline)

    limitations = ["training source is not materialized", "capture device metadata capability and storage/time require human confirmation", "public backups require post-download input and license-scope audit"]
    verification = {
        "experiment_id": config["experiment_id"], "status": "TRAINING-STRATEGY-VERIFIED-WITH-LIMITATIONS",
        "routes_audited": ["LOCAL-EXISTING-SOURCE-AUDIT", "NEW-CONTENT-ACQUISITION", "LEGAL-PUBLIC-DATASET"],
        "local_candidate_count": len(local), "local_training_ready_count": int(local.training_ready.sum()),
        "public_candidate_count": len(public), "primary_strategy": primary["primary_strategy"], "backup_strategy_count": len(primary["backup_strategies"]),
        "limitations": limitations, "data_downloaded": False, "data_acquired": False, "synthetic_pairs_generated": False,
        "training_manifest_created": False, "split_created": False, "model_training_performed": False,
        "provenance_complete": False, "source_data_protected": False, "pipeline_readiness": pipeline,
        "next_task": primary["next_task"],
    }
    dump_json(output / "verification_status.json", verification)

    (output / "provenance/git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "provenance/git_status_before.txt").write_text(status_before, encoding="utf-8")
    (output / "provenance/git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (output / "provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (output / "provenance/environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\npandas={pd.__version__}\n", encoding="utf-8")
    (output / "provenance/resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scripts = [Path(__file__), repo / "scripts/audit_local_training_content_candidates.py", repo / "scripts/build_training_acquisition_plan.py", repo / "scripts/audit_public_training_content_sources.py", repo / "scripts/json_serialization.py", config_path, repo / "docs/training_content_acquisition_protocol.md"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256(path)} for path in scripts]).to_csv(output / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")

    root_counts_after = {str(path): count_files(path) for path in roots}
    protection_rows = []
    for path_text, before in file_snapshot.items():
        path = Path(path_text)
        protection_rows.append({"path": path_text, "sha256_before": before["sha256"], "sha256_after": sha256(path), "mtime_ns_before": before["mtime_ns"], "mtime_ns_after": path.stat().st_mtime_ns})
    protection = pd.DataFrame(protection_rows)
    protection["unchanged"] = (protection.sha256_before == protection.sha256_after) & (protection.mtime_ns_before == protection.mtime_ns_after)
    protection.to_csv(output / "provenance/source_protection.csv", index=False, encoding="utf-8-sig")
    root_counts = pd.DataFrame([{"root": root, "file_count_before": count, "file_count_after": root_counts_after[root], "unchanged": count == root_counts_after[root]} for root, count in root_counts_before.items()])
    root_counts.to_csv(output / "provenance/source_root_counts.csv", index=False, encoding="utf-8-sig")
    source_protected = bool(protection.unchanged.all() and root_counts.unchanged.all())
    status_after = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    (output / "provenance/git_status_after.txt").write_text(status_after, encoding="utf-8")
    run = {"experiment_id": config["experiment_id"], "started_at_utc": started, "ended_at_utc": now(), "git_commit": commit, "status": verification["status"], "source_data_protected": source_protected, "source_write_performed": False, "network_use": "official-source metadata verification only; no dataset download", "data_downloaded": False, "data_acquired": False, "synthetic_pairs_generated": False, "split_or_training_performed": False}
    dump_json(output / "provenance/run_manifest.json", run)

    report = f"""# E2 Training Source Strategy Audit

Status: `TRAINING-STRATEGY-VERIFIED-WITH-LIMITATIONS`

## Decision

The unique primary strategy is `NEW-CONTENT-ACQUISITION`. The source remains unmaterialized and the formal training-content status remains `MISSING`. A minimum acquisition contains 20 independent scenes and 3 files per scene (60 files); the recommended acquisition contains 40 scenes and 5 files per scene (200 files). Every repeated capture remains blocked by its scene and acquisition group.

## Route A

The bounded review retained {len(local)} local data candidates from the frozen audit. None is training-ready. The strongest local lead, `E:/PMRID-Pytorch-main/Code/data`, contains 10,666 historical 8-bit RGB PNG patches in input/groundtruth directories and SID-style list files, but not a sufficiently traceable, untouched content source. Existing sCMOS, PMRID validation, ICCD evaluation/calibration data, previews, caches, sparse public samples and model outputs keep their frozen non-training roles.

## Route B

New acquisition has the lowest controllable leakage risk because role, scene, acquisition group, device settings and hashes can be recorded before any generator use. It remains conditional on manual confirmation, actual acquisition and a separate formal input audit. The data must be called `newly acquired operational training content`, not clean ground truth.

## Route C

Five official candidates were reviewed without downloading data. RAISE and MIT-Adobe FiveK are the two backups. RAISE has explicit non-commercial research/education terms and camera-native RAW subsets; FiveK has 5,000 DNG files under file-list-specific research licenses. SID and SIDD have higher project-history/preprocessing risk. RENOIR is excluded because its official page did not expose an explicit dataset license.

## Readiness

No formal synthetic generation, split construction or model training is allowed. The next task is human confirmation of the acquisition protocol, directory layout and metadata table; acquisition must not start automatically.
"""
    (output / "verification_report.md").write_text(report, encoding="utf-8")
    dump_json(output / "logs/run.log", {"verification": verification, "source_protected": source_protected})

    required = list(outputs) + ["primary_training_strategy.json", "pipeline_readiness.json", "verification_status.json", "verification_report.md", "provenance/run_manifest.json", "provenance/source_protection.csv", "provenance/source_root_counts.csv"]
    verification["provenance_complete"] = all((output / name).is_file() for name in required)
    verification["source_data_protected"] = source_protected
    if not verification["provenance_complete"] or not source_protected:
        verification["status"] = "FAILED"
        run["status"] = "FAILED"
    dump_json(output / "verification_status.json", verification)
    dump_json(output / "provenance/run_manifest.json", run)
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(hashes).to_csv(output / "output_hashes.csv", index=False, encoding="utf-8-sig")
    for json_path in ["primary_training_strategy.json", "pipeline_readiness.json", "verification_status.json", "provenance/run_manifest.json"]:
        json.loads((output / json_path).read_text(encoding="utf-8"))
    print(json.dumps({"status": verification["status"], "local_candidates": len(local), "local_ready": int(local.training_ready.sum()), "public_candidates": len(public), "primary": primary["primary_strategy"], "source_protected": source_protected}, ensure_ascii=False, indent=2))
    return 0 if verification["status"].startswith("TRAINING-STRATEGY-VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
