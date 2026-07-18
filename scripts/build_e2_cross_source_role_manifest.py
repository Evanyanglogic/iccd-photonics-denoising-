"""Build frozen E2 source-role, isolation, and PMRID scene manifests."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd


ROLE_COLUMNS = [
    "source_id", "source_name", "absolute_path", "disk", "source_type", "dataset_name",
    "source_provenance", "provenance_status", "official_reference", "license_status",
    "file_count", "image_count", "dtype", "shape", "channel_layout", "bit_depth",
    "processing_status", "source_scene_available", "source_group_available",
    "official_split_available", "isolation_level", "independence_from_scmos", "used_in_e1",
    "used_in_generator_design", "used_in_training", "used_in_checkpoint_selection",
    "allowed_role", "prohibited_roles", "allowed_operations", "prohibited_operations",
    "preprocessing_status", "source_audit_report", "source_audit_commit", "leakage_risk",
    "domain_limitation", "final_status", "notes",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_role_rows(config: dict, source_summary: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    summaries = source_summary.set_index("candidate_id").to_dict("index")
    inventories = inventory.set_index("candidate_id").to_dict("index")
    audit_report = "reports/e2_validation_source_audit_20260718_v2/verification_report.md"
    audit_commit = config["inputs"]["validation_audit_expected_commit"]

    def base(source_id: str, candidate_id: str, source_name: str, source_type: str, dataset_name: str, role: str, final_status: str) -> dict:
        summary = summaries[candidate_id]
        item = inventories[candidate_id]
        return {
            "source_id": source_id, "source_name": source_name, "absolute_path": summary["path"],
            "disk": Path(summary["path"]).drive, "source_type": source_type, "dataset_name": dataset_name,
            "source_provenance": summary["traceability"], "provenance_status": "AUDITED",
            "official_reference": "", "license_status": "unknown", "file_count": item["file_count"],
            "image_count": item["image_file_count"], "dtype": summary["dtype"], "shape": summary["shape"],
            "channel_layout": "single-channel" if "x3" not in str(summary["shape"]) else "mixed RGB/RGBA",
            "bit_depth": "16" if "uint16" in str(summary["dtype"]) else ("8" if "uint8" in str(summary["dtype"]) else "unknown"),
            "processing_status": summary["processing_status"], "source_scene_available": False,
            "source_group_available": False, "official_split_available": False,
            "isolation_level": "source-level only", "independence_from_scmos": summary["independence"],
            "used_in_e1": False, "used_in_generator_design": False,
            "used_in_training": False, "used_in_checkpoint_selection": False,
            "allowed_role": role, "prohibited_roles": "training_content_only;validation_content_only" if role in {"debug_only", "excluded"} else "",
            "allowed_operations": "metadata audit;numerical audit" if role != "excluded" else "provenance audit only",
            "prohibited_operations": "formal training;checkpoint selection;cross-role reuse",
            "preprocessing_status": "NOT-APPLICABLE", "source_audit_report": audit_report,
            "source_audit_commit": audit_commit, "leakage_risk": summary["leakage_risk"],
            "domain_limitation": "Not accepted as independent formal training/validation content",
            "final_status": final_status, "notes": summary["group_structure"],
        }

    rows = []
    scmos = base("scmos_500ms_current_100", "current_scmos_500ms", "sCMOS-derived operational content source", "operational content", "local sCMOS 500 ms", "debug_only", "DEBUG-ONLY")
    scmos.update({"file_count": 100, "image_count": 100, "dtype": "uint16", "shape": "2048x2048", "channel_layout": "single-channel", "bit_depth": 16, "processing_status": "unknown", "isolation_level": "single_unknown_source_group", "used_in_generator_design": True, "used_in_training": False, "used_in_checkpoint_selection": False, "allowed_operations": "generator smoke;numerical audit;debug;non-independent preview", "prohibited_operations": "formal training;validation;checkpoint selection;internal train-validation split", "preprocessing_status": "FROZEN-DEBUG-ONLY", "domain_limitation": "Highly homogeneous; scene and acquisition group unknown", "notes": "All 100 entries remain debug_only."})
    rows.append(scmos)

    pmrid = base("pmrid_official_benchmark_gt_raw", "pmrid_official_benchmark", "Independent PMRID validation content source", "benchmark reference RAW content", "PMRID ECCV 2020", "validation_content_only", "VALIDATION-READY")
    pmrid.update({"official_reference": config["pmrid"]["official_repository"], "license_status": config["pmrid"]["license"], "file_count": 39, "image_count": 39, "dtype": "uint16", "shape": "3000x4000", "channel_layout": "BGGR Bayer mosaic", "bit_depth": 16, "source_scene_available": True, "source_group_available": True, "official_split_available": True, "isolation_level": "dataset-level and official scene-level", "used_in_training": False, "used_in_checkpoint_selection": False, "prohibited_roles": "training_content_only;debug_only", "allowed_operations": "generator numerical check;content-isolation validation;scene-blocked validation;fixed-residual numerical check", "prohibited_operations": "training;Candidate A tuning;ICCD performance claim;ICCD physical-noise claim;file-level random split", "preprocessing_status": config["pmrid"]["preprocessing_plan_status"], "leakage_risk": "Low for Candidate A; high if a PMRID-tuned checkpoint is treated as independent", "domain_limitation": "Mobile Bayer RAW content domain, not ICCD content or ICCD ground truth", "notes": "GT denotes PMRID benchmark reference only."})
    rows.append(pmrid)

    other = base("scmos_other_exposures", "f_scmos_other_exposures", "Other local sCMOS exposures", "operational content", "local sCMOS exposure family", "debug_only", "DEBUG-ONLY")
    other.update({"used_in_generator_design": True, "used_in_training": True, "used_in_checkpoint_selection": True, "allowed_operations": "debug;numerical audit", "preprocessing_status": "NOT-FROZEN", "domain_limitation": "Same device/acquisition family; independent scenes unverified"})
    rows.append(other)
    previews = base("pmrid4_validation_previews", "d_val_cmos_derived", "PMRID4 validation previews", "derived preview", "local PMRID4", "excluded", "EXCLUDED")
    previews.update({"used_in_checkpoint_selection": True, "notes": "Historical derived validation previews; no evidence they were training content."})
    rows.append(previews)
    pngan = base("pangan_local_public_samples", "pngan_public_samples", "PNGAN local public samples", "incomplete benchmark samples", "PNGAN bundled samples", "excluded", "EXCLUDED")
    pngan.update({"used_in_generator_design": True, "used_in_training": True, "used_in_checkpoint_selection": True, "notes": "Parent PNGAN project used these named datasets for modeling/evaluation; local subsets are incomplete."})
    rows.append(pngan)

    mappings = [
        ("pmrid4_500ms_exact_copy", "d_pmrid4_500ms_copy"), ("pmrid7_historical_exposures", "e_pmrid7_exposures"),
        ("pmrid4_training_cache", "d_pmrid4_cache"), ("real_iccd_evaluation_frames", "d_real_iccd_evaluation"),
        ("iccd_pir_calibration_outputs", "f_iccd_pir"), ("scmos_dark_frames", "f_scmos_dark"),
        ("pmrid_local_historical_training_data", "pmrid_local_training_data"), ("pmrid_noise_calibration_data", "pmrid_noise_calibration"),
    ]
    explicit_history = {
        "pmrid4_500ms_exact_copy": (False, True, True),
        "pmrid7_historical_exposures": (False, True, True),
        "pmrid4_training_cache": (False, True, True),
        "real_iccd_evaluation_frames": (True, False, False),
        "iccd_pir_calibration_outputs": (True, False, False),
        "scmos_dark_frames": (True, False, False),
        "pmrid_local_historical_training_data": (False, True, True),
        "pmrid_noise_calibration_data": (True, False, False),
    }
    for source_id, candidate_id in mappings:
        row = base(source_id, candidate_id, source_id.replace("_", " "), "excluded audited source", candidate_id, "excluded", "EXCLUDED")
        generator_design, training, checkpoint = explicit_history[source_id]
        row.update({"used_in_generator_design": generator_design, "used_in_training": training, "used_in_checkpoint_selection": checkpoint})
        if source_id == "real_iccd_evaluation_frames":
            row.update({"used_in_e1": True, "notes": "Real ICCD repeated frames used for E1 characterization/evaluation; excluded from content roles."})
        rows.append(row)

    parent_models_path = Path(config["inputs"]["pmrid_parent_models"])
    parent_files = [path for path in parent_models_path.rglob("*") if path.is_file()]
    rows.append({
        "source_id": "pmrid_parent_project_models", "source_name": "PMRID parent-project models and checkpoints",
        "absolute_path": str(parent_models_path), "disk": parent_models_path.drive, "source_type": "model artifact risk reference",
        "dataset_name": "PMRID", "source_provenance": "Local parent PMRID project containing code and pretrained checkpoint",
        "provenance_status": "AUDITED-RISK-REFERENCE", "official_reference": config["pmrid"]["official_repository"],
        "license_status": config["pmrid"]["license"], "file_count": len(parent_files), "image_count": 0,
        "dtype": "not applicable", "shape": "not applicable", "channel_layout": "not applicable", "bit_depth": "not applicable",
        "processing_status": "model artifacts", "source_scene_available": False, "source_group_available": False,
        "official_split_available": False, "isolation_level": "model-history relationship", "independence_from_scmos": "not an image source",
        "used_in_e1": False, "used_in_generator_design": False, "used_in_training": True, "used_in_checkpoint_selection": True,
        "allowed_role": "excluded", "prohibited_roles": "training_content_only;validation_content_only;debug_only",
        "allowed_operations": "external reference audit only", "prohibited_operations": "independent ICCD model claim;checkpoint selection for PMRID validation",
        "preprocessing_status": "NOT-APPLICABLE", "source_audit_report": audit_report, "source_audit_commit": audit_commit,
        "leakage_risk": "High if PMRID-tuned checkpoints are evaluated on PMRID as independent evidence",
        "domain_limitation": "Model artifact, not content", "final_status": "EXCLUDED", "notes": "Current Candidate A did not use these models.",
    })
    rows.append({
        "source_id": "formal_training_content_placeholder", "source_name": "Formal training content source placeholder",
        "absolute_path": "", "disk": "", "source_type": "missing source placeholder", "dataset_name": "missing",
        "source_provenance": "No audited training-ready source exists", "provenance_status": "MISSING",
        "official_reference": "", "license_status": "not applicable", "file_count": 0, "image_count": 0,
        "dtype": "not available", "shape": "not available", "channel_layout": "not available", "bit_depth": "not available",
        "processing_status": "not available", "source_scene_available": False, "source_group_available": False,
        "official_split_available": False, "isolation_level": "none", "independence_from_scmos": "not assessable",
        "used_in_e1": False, "used_in_generator_design": False, "used_in_training": False, "used_in_checkpoint_selection": False,
        "allowed_role": "none_missing_source", "prohibited_roles": "training_content_only;validation_content_only;debug_only",
        "allowed_operations": "source acquisition planning only", "prohibited_operations": "training;generation;split assignment;placeholder substitution",
        "preprocessing_status": "NOT-AVAILABLE", "source_audit_report": "", "source_audit_commit": "",
        "leakage_risk": "Invalid if populated without a separate formal audit", "domain_limitation": "Training source absent",
        "final_status": "MISSING", "notes": "training_source_status=MISSING; sCMOS must not fill this row.",
    })
    return pd.DataFrame(rows, columns=ROLE_COLUMNS)


def make_isolation_matrix(roles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_a, source_b in itertools.combinations(roles.source_id, 2):
        exact, perceptual, scene, acquisition, device = "not assessed", "not assessed", "unknown", "unknown", "unknown"
        project_history, model_history, allowed, risk, decision, evidence = "unknown", "unknown", "no", "high", "DO-NOT-CROSS-USE", "No accepted cross-role evidence."
        pair = {source_a, source_b}
        if pair == {"scmos_500ms_current_100", "pmrid_official_benchmark_gt_raw"}:
            exact, perceptual, scene, acquisition, device = "0", "not detected", "no", "no", "no"
            project_history, model_history, allowed, risk, decision = "no", "no", "conditional", "low", "DATASET-LEVEL-ISOLATED"
            evidence = "Validation audit: zero SHA256/perceptual matches; max correlation 0.3364 and max SSIM 0.3403. Conditional does not promote sCMOS to training."
        elif pair == {"scmos_500ms_current_100", "scmos_other_exposures"}:
            exact, perceptual, scene, acquisition, device = "not required", "not sufficient", "unverified", "likely", "yes"
            project_history, model_history, allowed, risk, decision = "yes", "yes", "no", "high", "SAME-ACQUISITION-FAMILY-NOT-ISOLATED"
            evidence = "Same device/acquisition family; exposure folders are not independent scene identifiers."
        elif pair == {"pmrid_official_benchmark_gt_raw", "pmrid_parent_project_models"}:
            exact, perceptual, scene, acquisition, device = "not applicable", "not applicable", "benchmark overlap", "benchmark history", "same PMRID project"
            project_history, model_history, allowed, risk, decision = "yes", "yes", "no", "high", "MODEL-HISTORY-LEAKAGE-RISK"
            evidence = "Parent project contains PMRID code and pretrained model. PMRID-tuned checkpoints cannot provide independent PMRID validation evidence."
        elif "formal_training_content_placeholder" in pair:
            allowed, risk, decision = "no", "blocking", "TRAINING-SOURCE-MISSING"
            evidence = "Placeholder has no files and cannot be assigned a data role."
        rows.append({"source_a": source_a, "source_b": source_b, "exact_hash_overlap": exact, "perceptual_overlap": perceptual, "scene_overlap": scene, "acquisition_overlap": acquisition, "acquisition_family_overlap": acquisition, "device_overlap": device, "dataset_overlap": "no" if pair == {"scmos_500ms_current_100", "pmrid_official_benchmark_gt_raw"} else "unknown", "project_history_overlap": project_history, "model_history_overlap": model_history, "allowed_cross_role_use": allowed, "leakage_risk": risk, "isolation_decision": decision, "evidence": evidence})
    return pd.DataFrame(rows)


def make_pmrid_scene_manifest(config: dict, prior_hashes: pd.DataFrame) -> pd.DataFrame:
    benchmark_path = Path(config["inputs"]["pmrid_benchmark"])
    root = benchmark_path.parent
    records = json.loads(benchmark_path.read_text(encoding="utf-8"))
    hash_lookup = {str(Path(row.path).resolve()).lower(): row.sha256 for row in prior_hashes.itertuples(index=False)}
    rows = []
    for index, record in enumerate(records):
        gt_path = (root / record["gt"]).resolve()
        input_path = (root / record["input"]).resolve()
        meta = record["meta"]
        previous_hash = hash_lookup.get(str(gt_path).lower())
        if previous_hash is None:
            raise RuntimeError(f"PMRID GT missing from prior validation audit hashes: {gt_path}")
        current_hash = sha256(gt_path)
        if current_hash != previous_hash:
            raise RuntimeError(f"PMRID-SHA256-DRIFT: {gt_path}")
        scene_number = str(meta["scene_id"]).replace("Scene", "")
        rows.append({
            "pmrid_content_id": f"pmrid_gt_{index:03d}", "source_path": str(gt_path), "relative_path": record["gt"],
            "SHA256": current_hash, "paired_input_path": str(input_path), "paired_gt_path": str(gt_path),
            "scene_id": meta["scene_id"], "bright_dark_condition": meta["light"], "ISO": meta["ISO"],
            "exposure": meta["exp_time"], "Bayer_pattern": meta["bayer_pattern"], "dtype": "uint16",
            "shape": "x".join(map(str, meta["shape"])), "official_split": "PMRID benchmark",
            "benchmark_entry": index, "allowed_role": "validation_content_only", "isolation_block_id": f"pmrid_scene_{scene_number}",
            "preprocessing_status": "UNMODIFIED-RAW-REFERENCE", "preprocessing_plan_status": config["pmrid"]["preprocessing_plan_status"],
            "provenance_reference": config["pmrid"]["official_repository"],
        })
    return pd.DataFrame(rows)


def prohibited_transitions() -> pd.DataFrame:
    rules = [
        ("scmos_500ms_current_100", "debug_only", "training_content_only", "Highly homogeneous source; scene/group unknown"),
        ("scmos_500ms_current_100", "debug_only", "validation_content_only", "Not independent and no defensible split"),
        ("pmrid_official_benchmark_gt_raw", "validation_content_only", "training_content_only", "PMRID is frozen validation-only"),
        ("scmos_other_exposures", "debug_only", "training_content_only", "Same acquisition family and scenes unverified"),
        ("scmos_other_exposures", "debug_only", "validation_content_only", "Same acquisition family and scenes unverified"),
        ("pmrid4_validation_previews", "excluded", "validation_content_only", "Derived uint8 previews"),
        ("pangan_local_public_samples", "excluded", "validation_content_only", "Incomplete subset and historical PNGAN use"),
        ("formal_training_content_placeholder", "none_missing_source", "training_content_only", "Requires a separate formal source audit"),
        ("pmrid_official_benchmark_gt_raw", "scene-blocked", "file-random-split", "Official scene structure must be preserved"),
        ("any_source", "single-role", "training-and-validation", "Cross-role reuse is prohibited"),
        ("real_iccd_evaluation_frames", "excluded", "training_content_only", "Evaluation leakage"),
        ("pmrid_parent_project_models", "excluded", "independent-validation-model", "PMRID model-history leakage"),
        ("formal_training_content_placeholder", "missing", "substitute-scmos", "sCMOS debug source cannot fill missing training source"),
    ]
    return pd.DataFrame(rules, columns=["source_id", "from_role", "prohibited_to_role", "reason"])
