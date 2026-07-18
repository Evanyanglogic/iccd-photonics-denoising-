"""Validation gates for frozen cross-source role manifests."""
from __future__ import annotations

import pandas as pd


ALLOWED_ROLES = {"debug_only", "validation_content_only", "training_content_only", "excluded", "none_missing_source"}


def validate_manifests(roles: pd.DataFrame, isolation: pd.DataFrame, pmrid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    conflicts = []
    if roles.source_id.duplicated().any():
        conflicts.append({"conflict": "duplicate_source_id", "details": ";".join(roles.loc[roles.source_id.duplicated(), "source_id"])})
    invalid_roles = sorted(set(roles.allowed_role) - ALLOWED_ROLES)
    if invalid_roles:
        conflicts.append({"conflict": "invalid_role", "details": ";".join(invalid_roles)})
    expected = {
        "scmos_500ms_current_100": "debug_only", "pmrid_official_benchmark_gt_raw": "validation_content_only",
        "scmos_other_exposures": "debug_only", "pmrid4_validation_previews": "excluded",
        "pangan_local_public_samples": "excluded", "formal_training_content_placeholder": "none_missing_source",
    }
    role_lookup = roles.set_index("source_id").allowed_role.to_dict()
    for source_id, role in expected.items():
        if role_lookup.get(source_id) != role:
            conflicts.append({"conflict": "frozen_role_mismatch", "details": f"{source_id}: {role_lookup.get(source_id)} != {role}"})
    if (roles.allowed_role == "training_content_only").any():
        conflicts.append({"conflict": "unauthorized_training_source", "details": ";".join(roles.loc[roles.allowed_role == "training_content_only", "source_id"])})
    if len(pmrid) != 39 or sorted(pmrid.scene_id.unique()) != ["Scene1", "Scene2", "Scene3", "Scene4"]:
        conflicts.append({"conflict": "pmrid_scene_manifest_incomplete", "details": f"rows={len(pmrid)} scenes={sorted(pmrid.scene_id.unique())}"})
    if set(pmrid.allowed_role) != {"validation_content_only"} or set(pmrid.preprocessing_plan_status) != {"NOT-FROZEN"}:
        conflicts.append({"conflict": "pmrid_role_or_preprocessing_drift", "details": "PMRID role/preprocessing plan changed"})
    expected_pairs = len(roles) * (len(roles) - 1) // 2
    if len(isolation) != expected_pairs:
        conflicts.append({"conflict": "isolation_matrix_incomplete", "details": f"{len(isolation)} != {expected_pairs}"})
    checks = {
        "role_count": len(roles), "isolation_pair_count": len(isolation), "expected_isolation_pair_count": expected_pairs,
        "pmrid_entry_count": len(pmrid), "pmrid_scenes": sorted(pmrid.scene_id.unique()),
        "scmos_debug_only": role_lookup.get("scmos_500ms_current_100") == "debug_only",
        "pmrid_validation_only": role_lookup.get("pmrid_official_benchmark_gt_raw") == "validation_content_only",
        "training_source_missing": role_lookup.get("formal_training_content_placeholder") == "none_missing_source" and not (roles.allowed_role == "training_content_only").any(),
        "pmrid_preprocessing_not_frozen": set(pmrid.preprocessing_plan_status) == {"NOT-FROZEN"},
        "no_role_conflicts": not conflicts,
    }
    return pd.DataFrame(conflicts, columns=["conflict", "details"]), checks
