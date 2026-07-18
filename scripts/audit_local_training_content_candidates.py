"""Build the bounded Route A training-content candidate audit."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


def _count_files(root: Path, suffixes: set[str] | None = None) -> int:
    if not root.exists():
        return 0
    files = (path for path in root.rglob("*") if path.is_file())
    if suffixes is not None:
        files = (path for path in files if path.suffix.lower() in suffixes)
    return sum(1 for _ in files)


def build_local_candidate_audit(repo: Path, config: dict) -> pd.DataFrame:
    roles = pd.read_csv(repo / config["inputs"]["role_manifest"])
    inventory = pd.read_csv(repo / config["inputs"]["local_inventory"])
    inventory_by_path = {str(row.absolute_path).lower(): row for row in inventory.itertuples()}
    excluded_ids = {"pmrid_parent_project_models", "formal_training_content_placeholder"}
    rows = []
    for row in roles.itertuples():
        if row.source_id in excluded_ids:
            continue
        inv = inventory_by_path.get(str(row.absolute_path).lower())
        assessment = "EXCLUDED"
        reason = row.notes
        if row.source_id == "pmrid_local_historical_training_data":
            assessment = "EXCLUDED-HISTORICAL-DERIVED-TRAINING-DATA"
            reason = "10666 derived 8-bit RGB PNG patches in a historical training tree; source RAW linkage and clean provenance are not recoverable enough for formal reuse."
        elif row.source_id == "pangan_local_public_samples":
            assessment = "DEBUG-ONLY-INCOMPLETE-PUBLIC-SUBSETS"
            reason = "Only 39 sparse sample files across seven named datasets; incomplete source subsets and parent-project history prevent formal training use."
        rows.append({
            "candidate_id": row.source_id,
            "absolute_path": row.absolute_path,
            "previous_role": row.allowed_role,
            "previous_status": row.final_status,
            "previous_inventory_priority": getattr(inv, "initial_priority", "not_in_inventory"),
            "previous_image_count": getattr(inv, "image_file_count", ""),
            "source_traceability": row.provenance_status,
            "processing_status": row.processing_status,
            "pmrid_isolation": row.independence_from_scmos if row.source_id == "pmrid_official_benchmark_gt_raw" else "not sufficient for training role",
            "historical_training_or_tuning_risk": row.used_in_training,
            "route_a_assessment": assessment,
            "training_ready": False,
            "reason": reason,
        })

    historical = Path(config["inputs"]["historical_training_root"])
    gt = historical / "train" / "groundtruth"
    inp = historical / "train" / "input"
    png_counts = {"groundtruth": _count_files(gt, {".png"}), "input": _count_files(inp, {".png"})}
    sample = next(gt.glob("*.png"), None) if gt.exists() else None
    sample_meta = "unavailable"
    if sample:
        with Image.open(sample) as image:
            sample_meta = f"mode={image.mode};shape={image.height}x{image.width};bit_depth=8"
    for item in rows:
        if item["candidate_id"] == "pmrid_local_historical_training_data":
            item["local_detail"] = f"groundtruth_png={png_counts['groundtruth']};input_png={png_counts['input']};{sample_meta}"
        elif item["candidate_id"] == "pangan_local_public_samples":
            item["local_detail"] = f"local_sample_files={_count_files(Path(config['inputs']['pngan_samples_root']))}"
        else:
            item["local_detail"] = "reused from frozen role and validation-source audits"
    return pd.DataFrame(rows)
