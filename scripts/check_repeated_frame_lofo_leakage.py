"""Verify source-frame isolation in repeated-frame LOFO role manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for fold_dir in sorted(root.glob("heldout_*")):
        roles = read_csv(fold_dir / "source_frame_roles.csv")
        manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
        errors = check_fold(roles, manifest)
        row = {
            "fold": fold_dir.name,
            "heldout_test_folder": manifest["heldout_test_folder"],
            "validation_folder": manifest["validation_folder"],
            "source_role_count": len(roles),
            "unique_source_count": len({item["source_key"] for item in roles}),
            "error_count": len(errors),
            "status": "PASS" if not errors else "FAIL",
            "errors": " | ".join(errors),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No heldout fold directories under {root}")
    write_csv(rows, root / "leakage_check.csv")
    decision = {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "fold_count": len(rows)}
    (root / "leakage_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["status"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/e6_repeated_frame_protocol")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_fold(rows: list[dict[str, str]], manifest: dict[str, Any]) -> list[str]:
    errors = []
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source_key"], []).append(row)
    duplicates = {key: items for key, items in by_source.items() if len(items) != 1}
    if duplicates:
        errors.append(f"source frame reused: {list(duplicates)[:5]}")
    test_folder = str(manifest["heldout_test_folder"])
    val_folder = str(manifest["validation_folder"])
    for row in rows:
        if row["folder"] == test_folder and row["split"] != "test":
            errors.append(f"heldout folder in {row['split']}: {row['source_key']}")
            break
        if row["folder"] == val_folder and row["split"] != "val":
            errors.append(f"validation folder in {row['split']}: {row['source_key']}")
            break
    train_folders = {row["folder"] for row in rows if row["split"] == "train"}
    val_folders = {row["folder"] for row in rows if row["split"] == "val"}
    test_folders = {row["folder"] for row in rows if row["split"] == "test"}
    if train_folders & val_folders or train_folders & test_folders or val_folders & test_folders:
        errors.append("folder overlap across train/val/test")
    test_inputs = {row["source_key"] for row in rows if row["split"] == "test" and row["role"] == "input"}
    references = {row["source_key"] for row in rows if row["split"] == "test" and row["role"] == "reference_member"}
    if test_inputs & references:
        errors.append("test input leaks into temporal reference")
    return errors


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
