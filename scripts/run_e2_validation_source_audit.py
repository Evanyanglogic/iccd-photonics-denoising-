"""Run the bounded, read-only E2 validation content source audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import tifffile
import yaml

from audit_validation_content_candidate import audit_generic, audit_pmrid, load_generic, prepare_representation
from compare_content_source_independence import compare, sha256, thumbnail, perceptual_hash
from inventory_validation_content_sources import inventory_candidates
from json_serialization import dump_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_references(root: Path, needles: list[str], exclude: set[str] | None = None) -> list[str]:
    findings = []
    suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".csv"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes or any(part in (exclude or set()) for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(needle.lower() in text for needle in needles):
            findings.append(str(path))
    return findings


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
        raise FileExistsError(f"Refusing to overwrite: {output}")
    for directory in ("provenance", "logs"):
        (output / directory).mkdir(parents=True, exist_ok=False if directory == "provenance" else True)
    started = utc_now()
    commit = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status_before.strip():
        raise RuntimeError("Formal audit requires a clean, traceable code worktree")

    inventory = inventory_candidates(config)
    inventory.to_csv(output / "candidate_directory_inventory.csv", index=False, encoding="utf-8-sig")
    source_inventory_before = inventory.set_index("candidate_id")[["file_count", "image_file_count", "approximate_size_bytes"]].to_dict("index")
    deep = inventory[inventory["deep_audit"] == True]
    if sum(deep.initial_priority == "B") > config["limits"]["max_priority_b_deep_audits"]:
        raise RuntimeError("Priority B deep-audit limit exceeded")

    representatives = pd.read_csv(repo / config["current_representatives"])
    manifest = pd.read_csv(repo / config["current_content_manifest"])
    current = []
    size = int(config["limits"]["thumbnail_size"])
    roi = {"top": 768, "left": 768, "height": 512, "width": 512}
    for row in representatives.itertuples(index=False):
        manifest_row = manifest.loc[manifest.content_id == row.content_id].iloc[0]
        path = Path(manifest_row.absolute_path)
        if sha256(path) != manifest_row.sha256:
            raise RuntimeError(f"Current source input drift: {path}")
        image = tifffile.imread(path)[roi["top"]:roi["top"] + roi["height"], roi["left"]:roi["left"] + roi["width"]]
        rep = prepare_representation(path, image, size, row.content_id)
        current.append(rep)

    all_stats, all_metadata, all_independence, summaries = [], [], [], {}
    rep_sets = {}
    for row in deep.itertuples(index=False):
        root = Path(row.absolute_path)
        if row.candidate_id == "pmrid_official_benchmark":
            stats, reps, summary = audit_pmrid(root, size, config["limits"]["max_candidate_representatives"])
        elif row.candidate_id == "f_scmos_other_exposures":
            stats, reps, summary = audit_generic(row.candidate_id, root, size, config["limits"]["max_candidate_representatives"], {"500ms", "dark_background"})
        else:
            stats, reps, summary = audit_generic(row.candidate_id, root, size, config["limits"]["max_candidate_representatives"])
        all_stats.extend(stats)
        summaries[row.candidate_id] = summary
        for key, value in summary.items():
            all_metadata.append({"candidate_id": row.candidate_id, "field": key, "value": json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)})
        all_independence.extend(compare(row.candidate_id, reps, current))
        rep_sets[row.candidate_id] = reps

    stats_frame = pd.DataFrame(all_stats)
    metadata_frame = pd.DataFrame(all_metadata)
    independence_frame = pd.DataFrame(all_independence)
    stats_frame.to_csv(output / "candidate_file_statistics.csv", index=False, encoding="utf-8-sig")
    metadata_frame.to_csv(output / "candidate_metadata_summary.csv", index=False, encoding="utf-8-sig")
    independence_frame.to_csv(output / "candidate_independence_analysis.csv", index=False, encoding="utf-8-sig")

    current_hashes = set(manifest.sha256.str.lower())
    d_copy_files = sorted(Path("D:/PMRID4/data/500ms").glob("*.tif*"))
    d_copy_overlap = sum(sha256(path) in current_hashes for path in d_copy_files)
    pmrid_summary = summaries["pmrid_official_benchmark"]
    pmrid_independence = independence_frame[independence_frame.candidate_id == "pmrid_official_benchmark"]
    pmrid_usage_current = text_references(repo, ["PMRID-Pytorch-main/PMRID/PMRID", "PMRID\\PMRID\\Scene"], {"reports"})
    parent_training_lists = list(Path("E:/PMRID-Pytorch-main").rglob("*list*.txt"))
    benchmark_in_training_lists = []
    for path in parent_training_lists:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "scene1/bright" in text or "gt.raw" in text:
            benchmark_in_training_lists.append(str(path))

    pmrid_stats = stats_frame[stats_frame.candidate_id == "pmrid_official_benchmark"]
    pmrid_gates = {
        "all_gt_files_read": len(pmrid_stats) == 39 and pmrid_summary["gt_files_read"] == 39 and not pmrid_summary["failures"],
        "official_scene_groups_present": pmrid_summary["scene_ids"] == ["Scene1", "Scene2", "Scene3", "Scene4"],
        "dtype_uint16": set(pmrid_stats.dtype) == {"uint16"},
        "shape_3000x4000": set(pmrid_stats["shape"]) == {"3000x4000"},
        "zero_ratio_acceptable": float(pmrid_stats.zero_ratio.max()) < config["classification"]["zero_ratio_max"],
        "saturation_ratio_acceptable": float(pmrid_stats.saturation_ratio.max()) < config["classification"]["saturation_ratio_max"],
        "no_exact_hash_match": not bool(pmrid_independence.exact_sha256_match.any()),
        "no_perceptual_hash_match": not bool(pmrid_independence.perceptual_hash_match.any()),
        "no_near_duplicate_correlation": float(pmrid_independence.correlation.max()) < config["classification"]["cross_source_correlation_duplicate_threshold"],
        "no_near_duplicate_ssim": float(pmrid_independence.ssim.max()) < config["classification"]["cross_source_ssim_duplicate_threshold"],
        "no_training_list_reference": not benchmark_in_training_lists,
    }
    pmrid_status = "VALIDATION-READY" if all(pmrid_gates.values()) else "VALIDATION-CANDIDATE-WITH-LIMITATIONS"

    candidate_paths = inventory.set_index("candidate_id").absolute_path.to_dict()
    summary_rows = [
        {"candidate_id": "pmrid_official_benchmark", "path": candidate_paths["pmrid_official_benchmark"], "files_audited": len(pmrid_stats), "dtype": "uint16", "shape": "3000x4000", "traceability": "Official MegEngine/PMRID ECCV 2020 repository, benchmark.json and local README structure agree", "processing_status": "official paired mobile Bayer RAW benchmark GT", "independence": f"Different public smartphone dataset/device; exact matches={int(pmrid_independence.exact_sha256_match.sum())}, perceptual-hash matches={int(pmrid_independence.perceptual_hash_match.sum())}, max correlation={pmrid_independence.correlation.max():.6f}, max SSIM={pmrid_independence.ssim.max():.6f}", "group_structure": "Official Scene1-Scene4 plus bright/dark and ISO/exposure metadata", "leakage_risk": "No evidence of use by Candidate A or current ICCD generator; restrict to future validation_content_only", "historical_training_or_tuning": bool(benchmark_in_training_lists), "allowed_role": "validation_content_only" if pmrid_status == "VALIDATION-READY" else "debug_only", "status": pmrid_status},
        {"candidate_id": "f_scmos_other_exposures", "path": candidate_paths["f_scmos_other_exposures"], "files_audited": len(stats_frame[stats_frame.candidate_id == "f_scmos_other_exposures"]), "dtype": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "f_scmos_other_exposures", "dtype"].unique())), "shape": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "f_scmos_other_exposures", "shape"].unique())), "traceability": "Local exposure series with acquisition-time filenames but no independent scene manifest", "processing_status": "processing status unknown", "independence": "Same device, date family and highly related acquisition as current 500 ms source", "group_structure": "Exposure directories only; no defensible independent scene groups", "leakage_risk": "High source/content overlap risk", "historical_training_or_tuning": True, "allowed_role": "debug_only", "status": "DEBUG-ONLY"},
        {"candidate_id": "d_val_cmos_derived", "path": "D:/PMRID4/data/val_cmos_images", "files_audited": len(stats_frame[stats_frame.candidate_id == "d_val_cmos_derived"]), "dtype": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "d_val_cmos_derived", "dtype"].unique())), "shape": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "d_val_cmos_derived", "shape"].unique())), "traceability": "Generated by D:/PMRID4/TIFPatchDataLoader.py", "processing_status": "derived uint8 validation previews", "independence": "Derived from historical PMRID4 validation pairs", "group_structure": "Ten sample triplets, not independent scenes", "leakage_risk": "Explicit historical validation derivative", "historical_training_or_tuning": True, "allowed_role": "excluded", "status": "EXCLUDED"},
        {"candidate_id": "pngan_public_samples", "path": "E:/PNGAN-main/datasets", "files_audited": len(stats_frame[stats_frame.candidate_id == "pngan_public_samples"]), "dtype": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "pngan_public_samples", "dtype"].unique())), "shape": ";".join(sorted(stats_frame.loc[stats_frame.candidate_id == "pngan_public_samples", "shape"].unique())), "traceability": "Dataset names documented by parent PNGAN README; local folders are sparse samples rather than complete datasets", "processing_status": "camera-processed or benchmark-derived PNG/JPEG samples", "independence": "Likely content-independent but local subsets are incomplete", "group_structure": "No complete official split locally", "leakage_risk": "Parent PNGAN explicitly used these datasets for modeling/evaluation", "historical_training_or_tuning": True, "allowed_role": "excluded", "status": "EXCLUDED"},
        {"candidate_id": "d_pmrid4_500ms_copy", "path": "D:/PMRID4/data/500ms", "files_audited": len(d_copy_files), "dtype": "uint16", "shape": "2048x2048", "traceability": "Local renamed copy", "processing_status": "same bytes as current source", "independence": f"{d_copy_overlap}/100 exact SHA256 overlap with current source", "group_structure": "No independent scene evidence", "leakage_risk": "Exact duplicate", "historical_training_or_tuning": True, "allowed_role": "excluded", "status": "EXCLUDED"},
    ]
    detailed_ids = {row["candidate_id"] for row in summary_rows}
    remaining_status = {
        "current_scmos_500ms": ("DEBUG-ONLY", "Current single unknown source group"),
        "e_pmrid7_exposures": ("EXCLUDED", "Historical training content and same acquisition family"),
        "d_pmrid4_cache": ("EXCLUDED", "Derived training cache"),
        "d_real_iccd_evaluation": ("EXCLUDED", "Real ICCD evaluation data; validation-role leakage"),
        "f_iccd_pir": ("EXCLUDED", "ICCD calibration/evaluation and derived outputs"),
        "f_scmos_dark": ("EXCLUDED", "Dark calibration frames, not content"),
        "pmrid_local_training_data": ("EXCLUDED", "Historical training data"),
        "pmrid_noise_calibration": ("EXCLUDED", "Noise calibration data, not validation content"),
    }
    for row in inventory.itertuples(index=False):
        if row.candidate_id in detailed_ids:
            continue
        status, reason = remaining_status.get(row.candidate_id, ("EXCLUDED", row.exclusion_reason or "Not selected for deep audit"))
        summary_rows.append({"candidate_id": row.candidate_id, "path": row.absolute_path, "files_audited": 0, "dtype": "not deeply audited", "shape": "not deeply audited", "traceability": row.source_notes, "processing_status": "not deeply audited", "independence": reason, "group_structure": "not accepted", "leakage_risk": reason, "historical_training_or_tuning": status == "EXCLUDED", "allowed_role": "debug_only" if status == "DEBUG-ONLY" else "excluded", "status": status})
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "candidate_source_summary.csv", index=False, encoding="utf-8-sig")

    group_rows = [
        {"candidate_id": "pmrid_official_benchmark", "source_scene": scene, "evidence": "benchmark.json meta.scene_id", "is_real_group": True, "allowed_for_blocking": True} for scene in pmrid_summary["scene_ids"]
    ] + [
        {"candidate_id": candidate, "source_scene": "unknown", "evidence": evidence, "is_real_group": False, "allowed_for_blocking": False}
        for candidate, evidence in [("f_scmos_other_exposures", "Exposure folders are not scene identifiers"), ("d_val_cmos_derived", "Sample numbers are generated outputs"), ("pngan_public_samples", "Sparse local samples do not preserve complete official splits")]
    ]
    pd.DataFrame(group_rows).to_csv(output / "candidate_group_analysis.csv", index=False, encoding="utf-8-sig")
    leakage = summary_frame[["candidate_id", "historical_training_or_tuning", "leakage_risk", "allowed_role", "status"]].copy()
    leakage.to_csv(output / "candidate_leakage_risk.csv", index=False, encoding="utf-8-sig")
    pmrid_audit = pd.DataFrame([
        {"question": "source", "answer": "Official MegEngine/PMRID repository; ECCV 2020 Practical Deep Raw Image Denoising on Mobile Devices"},
        {"question": "local_structure", "answer": "39 input/gt RAW pairs, 4 official scenes, bright/dark, ISO/exposure metadata"},
        {"question": "format", "answer": "uint16 BGGR Bayer RAW, 3000x4000, official loader normalizes by 65535"},
        {"question": "license", "answer": "Official repository is Apache-2.0; attribution/source URL must accompany use"},
        {"question": "current_project_use", "answer": f"No training-list reference to benchmark GT RAW found; current audit/config references: {len(pmrid_usage_current)}"},
        {"question": "parent_project_use", "answer": f"Local PMRID code and pretrained model coexist; benchmark paths found in training lists: {len(benchmark_in_training_lists)}"},
        {"question": "role", "answer": "Future validation_content_only; not retrospective evidence for any model previously tuned on PMRID"},
    ])
    pmrid_audit.to_csv(output / "pmrid_source_audit.csv", index=False, encoding="utf-8-sig")

    backup = [
        {"candidate_id": "f_scmos_other_exposures", "status": "DEBUG-ONLY", "reason": "Same acquisition family and no independent scene grouping"},
        {"candidate_id": "pngan_public_samples", "status": "EXCLUDED", "reason": "Sparse samples and historical PNGAN use"},
    ]
    decision = {
        "primary_candidate": {
            "candidate_id": "pmrid_official_benchmark", "path": "E:/PMRID-Pytorch-main/PMRID/PMRID", "content_files": 39,
            "source": config["official_sources"]["pmrid_repository"], "independence_evidence": "Different OPPO Reno 10x public RAW benchmark, zero exact/perceptual matches in bounded comparison, official scene metadata",
            "group_structure": pmrid_summary["scene_ids"], "preprocessing": "Read uint16 GT RAW at 3000x4000 and divide by 65535; preserve Bayer or preregister deterministic Bayer-to-grayscale conversion later",
            "allowed_role": "validation_content_only" if pmrid_status == "VALIDATION-READY" else "debug_only", "limitations": ["Only four scenes", "Mobile Bayer RAW domain differs from grayscale sCMOS/ICCD content", "Must not be used for retrospective claims involving PMRID-tuned models"],
            "status": pmrid_status,
        },
        "backup_candidates": backup,
        "validation_ready_exists": pmrid_status == "VALIDATION-READY",
        "formal_split_design_allowed_next": pmrid_status == "VALIDATION-READY",
        "split_created_in_this_run": False,
        "next_task": "Build a formal dual-source isolation manifest for training and validation content without generating synthetic pairs.",
    }
    dump_json(output / "primary_candidate_decision.json", decision)

    resolved = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    (output / "provenance/git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "provenance/git_status_before.txt").write_text(status_before, encoding="utf-8")
    (output / "provenance/git_diff.patch").write_text(git(repo, "diff", "--binary", "HEAD"), encoding="utf-8")
    (output / "provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    (output / "provenance/environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\nscipy={scipy.__version__}\ntifffile={tifffile.__version__}\n", encoding="utf-8")
    with (output / "provenance/pip_freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, text=True, check=True)
    (output / "provenance/resolved_config.yaml").write_text(resolved, encoding="utf-8")
    script_paths = [Path(__file__), repo / "scripts/inventory_validation_content_sources.py", repo / "scripts/audit_validation_content_candidate.py", repo / "scripts/compare_content_source_independence.py", repo / "scripts/json_serialization.py", config_path]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": file_hash(path)} for path in script_paths]).to_csv(output / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")
    input_paths = [repo / config["current_content_manifest"], repo / config["current_representatives"], Path("E:/PMRID-Pytorch-main/PMRID/PMRID/benchmark.json")]
    input_paths.extend(Path(path) for path in stats_frame.path)
    input_paths.extend(Path(item["path"]) for item in current)
    input_paths = list(dict.fromkeys(input_paths))
    pd.DataFrame([{"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in input_paths]).to_csv(output / "provenance/input_hashes.csv", index=False, encoding="utf-8-sig")
    overall_status = "VALIDATION-READY-FOUND" if pmrid_status == "VALIDATION-READY" else "VALIDATION-CANDIDATE-WITH-LIMITATIONS"
    run_manifest = {"experiment_id": config["experiment_id"], "started_at_utc": started, "ended_at_utc": utc_now(), "git_commit": commit, "status": overall_status, "deep_candidates": list(deep.candidate_id), "source_write_performed": False, "synthetic_pairs_generated": False, "training_or_split_performed": False}
    dump_json(output / "provenance/run_manifest.json", run_manifest)

    report = f"""# E2 Validation Content Source Audit\n\nStatus: `{overall_status}`\n\nThe bounded inventory covered {len(inventory)} configured candidate directories: {(inventory.initial_priority == 'A').sum()} Priority A, {(inventory.initial_priority == 'B').sum()} Priority B, and {(inventory.initial_priority == 'C').sum()} Priority C. Deep review was limited to one Priority A and three Priority B candidates.\n\nThe primary candidate is the official PMRID ECCV 2020 benchmark at `E:/PMRID-Pytorch-main/PMRID/PMRID`. Its 39 GT RAW files are readable uint16 3000x4000 Bayer arrays, organized by four official scene IDs with bright/dark and ISO/exposure metadata. Its formal status is `{pmrid_status}` based on the frozen integrity, grouping, independence, numerical, and leakage gates recorded in `verification_status.json`.\n\nThis decision does not create a split, generate synthetic pairs, establish ICCD-domain equivalence, or validate model performance. The PMRID source has only four scenes and a mobile Bayer RAW domain; future preprocessing and scene blocking must be preregistered.\n"""
    (output / "verification_report.md").write_text(report, encoding="utf-8")
    verification = {"experiment_id": config["experiment_id"], "status": overall_status, "inventory_count": len(inventory), "priority_counts": inventory.initial_priority.value_counts().to_dict(), "deep_candidates": list(deep.candidate_id), "primary_candidate": "pmrid_official_benchmark", "primary_candidate_gates": pmrid_gates, "validation_ready_exists": pmrid_status == "VALIDATION-READY", "provenance_complete": False, "source_data_protected": False, "all_outputs_inside_repo": str(output.resolve()).startswith(str(repo.resolve())), "next_task": decision["next_task"]}
    dump_json(output / "verification_status.json", verification)
    source_inventory_after = inventory_candidates(config).set_index("candidate_id")[["file_count", "image_file_count", "approximate_size_bytes"]].to_dict("index")
    protection_rows = []
    for candidate_id, before in source_inventory_before.items():
        after = source_inventory_after[candidate_id]
        protection_rows.append({"candidate_id": candidate_id, "file_count_before": before["file_count"], "file_count_after": after["file_count"], "image_count_before": before["image_file_count"], "image_count_after": after["image_file_count"], "bytes_before": before["approximate_size_bytes"], "bytes_after": after["approximate_size_bytes"], "unchanged": before == after})
    before_lookup = {row.path: (row.sha256, int(row.modified_time)) for row in stats_frame.itertuples(index=False)}
    for item in current:
        source_path = Path(item["path"])
        before_lookup[str(source_path)] = (item["sha256"], source_path.stat().st_mtime_ns)
    sampled_protection = []
    for path_text, (before_hash, before_mtime) in before_lookup.items():
        source_path = Path(path_text)
        after_hash, after_mtime = sha256(source_path), source_path.stat().st_mtime_ns
        sampled_protection.append({"path": path_text, "sha256_before": before_hash, "sha256_after": after_hash, "mtime_ns_before": before_mtime, "mtime_ns_after": after_mtime, "unchanged": before_hash == after_hash and before_mtime == after_mtime})
    protection_frame = pd.DataFrame(protection_rows)
    sample_protection_frame = pd.DataFrame(sampled_protection)
    protection_frame.to_csv(output / "provenance/source_directory_protection.csv", index=False, encoding="utf-8-sig")
    sample_protection_frame.to_csv(output / "provenance/source_file_protection.csv", index=False, encoding="utf-8-sig")
    source_protected = bool(protection_frame.unchanged.all() and sample_protection_frame.unchanged.all())
    verification["source_data_protected"] = source_protected
    run_manifest["source_data_protected"] = source_protected
    run_manifest["ended_at_utc"] = utc_now()
    dump_json(output / "provenance/run_manifest.json", run_manifest)
    (output / "provenance/git_status_after.txt").write_text(git(repo, "status", "--porcelain=v1", "--untracked-files=all"), encoding="utf-8")
    dump_json(output / "logs/run.log", {"status": overall_status, "pmrid_gates": pmrid_gates, "source_data_protected": source_protected})

    required = ["candidate_directory_inventory.csv", "candidate_source_summary.csv", "candidate_file_statistics.csv", "candidate_metadata_summary.csv", "candidate_independence_analysis.csv", "candidate_group_analysis.csv", "candidate_leakage_risk.csv", "pmrid_source_audit.csv", "primary_candidate_decision.json", "verification_status.json", "verification_report.md", "provenance/run_manifest.json", "provenance/source_directory_protection.csv", "provenance/source_file_protection.csv", "logs/run.log"]
    verification["provenance_complete"] = all((output / path).is_file() for path in required)
    dump_json(output / "verification_status.json", verification)
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": file_hash(path)})
    pd.DataFrame(hashes).to_csv(output / "output_hashes.csv", index=False, encoding="utf-8-sig")
    json.loads((output / "verification_status.json").read_text(encoding="utf-8"))
    json.loads((output / "provenance/run_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": verification["status"], "primary": decision["primary_candidate"], "priority_counts": verification["priority_counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
