"""Build a traceable content manifest from the verified E2 no-dark audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from e2_content_manifest_lib import read_config, resolve, sha256_file


def build(repo: Path, cfg: dict, output: Path) -> pd.DataFrame:
    source = resolve(repo, cfg["input_report"])
    status = pd.read_json(source / "verification_status.json", typ="series")["status"]
    if status != cfg["required_input_status"]:
        raise RuntimeError(f"Unexpected input status: {status}")
    files = pd.read_csv(source / "input_manifest.csv")
    full = pd.read_csv(source / "full_image_statistics.csv")
    roi = pd.read_csv(source / "center_roi_statistics.csv")
    metadata = pd.read_csv(source / "tiff_metadata.csv")
    if len(files) != cfg["expected_count"] or not files["sha256_match"].astype(bool).all():
        raise RuntimeError("Input count or SHA256 gate failed")
    joined = files.merge(full, on="source_pair_key", suffixes=("", "_full"), validate="one_to_one")
    joined = joined.merge(roi, on="source_pair_key", suffixes=("", "_roi"), validate="one_to_one")
    joined = joined.merge(metadata[["source_pair_key", "dtype", "shape"]], on="source_pair_key", suffixes=("", "_meta"), validate="one_to_one")
    rows = []
    for index, row in joined.sort_values("source_pair_key").reset_index(drop=True).iterrows():
        raw_path = Path(str(row["absolute_path"]))
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        current_sha256 = sha256_file(raw_path)
        if current_sha256 != row["actual_sha256"]:
            raise RuntimeError(f"INPUT-DRIFT: {raw_path}")
        rows.append({
            "content_id": f"content_{index:03d}", "source_pair_key": row["source_pair_key"],
            "absolute_path": str(raw_path), "relative_path": raw_path.name, "filename": raw_path.name,
            "sha256": current_sha256, "file_size": int(row["file_size_bytes"]),
            "modified_time": pd.to_datetime(int(row["mtime_ns"]), unit="ns").isoformat(),
            "dtype": row["dtype_meta"], "shape": row["shape_meta"],
            "roi_top": cfg["roi"]["top"], "roi_left": cfg["roi"]["left"],
            "roi_height": cfg["roi"]["height"], "roi_width": cfg["roi"]["width"],
            "full_mean_dn": row["raw_mean_dn"], "full_std_dn": row["raw_std_dn"],
            "full_p1_dn": row["raw_p1_dn"], "full_p50_dn": row["raw_p50_dn"], "full_p99_dn": row["raw_p99_dn"],
            "roi_mean_dn": row["raw_mean_dn_roi"], "roi_std_dn": row["raw_std_dn_roi"],
            "roi_p1_dn": row["raw_p1_dn_roi"], "roi_p50_dn": row["raw_p50_dn_roi"], "roi_p99_dn": row["raw_p99_dn_roi"],
            "zero_ratio": row["zero_ratio"], "saturation_ratio": row["saturation_ratio"],
            "source_scene": cfg["source_scene"], "source_group": cfg["source_group"],
            "isolation_block_id": cfg["isolation_block_id"], "acquisition_order": index,
            "processing_status": cfg["processing_status"], "preprocessing": cfg["preprocessing"],
            "allowed_role": cfg["allowed_role_without_independent_validation"],
            "exclusion_reason": "No independently validated content source exists for formal validation",
            "provenance_report": cfg["input_report"], "source_audit_status": status,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "e2_content_manifest_20260717.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg, _ = read_config(repo, args.config)
    output = resolve(repo, args.output_root); output.mkdir(parents=True, exist_ok=True)
    print(f"manifest_rows={len(build(repo, cfg, output))}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
