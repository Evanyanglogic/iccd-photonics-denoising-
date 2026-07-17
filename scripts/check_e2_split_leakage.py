"""Check E2 source splits for file, source-image, and scene-level leakage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(config)
    return 0


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    audit_dir = output_root / "leakage_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(Path(config["source_pairs_csv"]))
    clean_audit = {row["source_pair_key"]: row for row in read_csv(output_root / "input_audit" / "clean_content_audit.csv")}
    splits = yaml.safe_load(Path(config["source_splits_yaml"]).read_text(encoding="utf-8"))
    key_to_split = {key: split for split, keys in splits.items() for key in keys}
    pair_by_key = {row["pair_key"]: row for row in source_rows}

    duplicate_membership = []
    seen: dict[str, str] = {}
    for split, keys in splits.items():
        for key in keys:
            if key in seen:
                duplicate_membership.append({"pair_key": key, "split_a": seen[key], "split_b": split})
            seen[key] = split

    missing_keys = sorted(set(pair_by_key) - set(key_to_split))
    unknown_keys = sorted(set(key_to_split) - set(pair_by_key))
    sha_splits: dict[str, set[str]] = {}
    path_splits: dict[str, set[str]] = {}
    scene_field_present = all("source_scene" in row and row.get("source_scene", "").strip() for row in source_rows)
    scene_splits: dict[str, set[str]] = {}
    assignment_rows = []
    for key, row in pair_by_key.items():
        split = key_to_split.get(key, "UNASSIGNED")
        audit = clean_audit[key]
        sha_splits.setdefault(audit["sha256"], set()).add(split)
        path_splits.setdefault(str(Path(row[config["clean_column"]]).resolve()).lower(), set()).add(split)
        scene = row.get("source_scene", "")
        if scene:
            scene_splits.setdefault(scene, set()).add(split)
        assignment_rows.append(
            {
                "pair_key": key,
                "split": split,
                "clean_path": row[config["clean_column"]],
                "clean_sha256": audit["sha256"],
                "source_scene": scene,
                "scene_metadata_status": "present" if scene else "missing",
            }
        )

    cross_split_hashes = {key: sorted(value) for key, value in sha_splits.items() if len(value) > 1}
    cross_split_paths = {key: sorted(value) for key, value in path_splits.items() if len(value) > 1}
    cross_split_scenes = {key: sorted(value) for key, value in scene_splits.items() if len(value) > 1}
    near_rows = read_csv(output_root / "input_audit" / "duplicate_or_near_duplicate_report.csv")
    near_cross = []
    for row in near_rows:
        first = row["source_pair_key_a"]
        second = row["source_pair_key_b"]
        if key_to_split.get(first) != key_to_split.get(second):
            near_cross.append({**row, "split_a": key_to_split.get(first, ""), "split_b": key_to_split.get(second, "")})

    scene_gate = scene_field_present and not cross_split_scenes
    passed = not any([duplicate_membership, missing_keys, unknown_keys, cross_split_hashes, cross_split_paths, near_cross]) and scene_gate
    summary = {
        "status": "PASS" if passed else "FAIL",
        "pair_count": len(source_rows),
        "split_counts": {split: len(keys) for split, keys in splits.items()},
        "duplicate_split_membership_count": len(duplicate_membership),
        "missing_pair_key_count": len(missing_keys),
        "unknown_split_key_count": len(unknown_keys),
        "cross_split_exact_hash_count": len(cross_split_hashes),
        "cross_split_exact_path_count": len(cross_split_paths),
        "cross_split_near_duplicate_count": len(near_cross),
        "source_scene_field_present": scene_field_present,
        "cross_split_scene_count": len(cross_split_scenes),
        "scene_level_gate": "PASS" if scene_gate else "FAIL",
        "reason": "scene/source-group isolation cannot be verified because source_scene metadata is absent" if not scene_field_present else "",
    }
    write_csv(assignment_rows, audit_dir / "split_assignments.csv")
    write_csv_or_header(near_cross, audit_dir / "cross_split_near_duplicates.csv", ["source_pair_key_a", "source_pair_key_b", "correlation", "split_a", "split_b"])
    (audit_dir / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_or_header(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
