"""Build leakage-auditable LOFO frame-role manifests without training models."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FRAME_NUMBER = re.compile(r"^(\d+)")


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    data_cfg = config["data"]
    protocol = config["blocked_candidate_protocol"]
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = [int(value) for value in data_cfg["folders"]]
    raw_root = Path(data_cfg["raw_root"])
    indexed = {folder: indexed_tiffs(raw_root / str(folder)) for folder in folders}
    expected = int(data_cfg["frames_per_folder"])
    for folder, paths in indexed.items():
        missing = sorted(set(range(1, expected + 1)).difference(paths))
        if missing:
            raise ValueError(f"Folder {folder} missing source frames: {missing[:10]}")

    summary_rows = []
    for test_position, test_folder in enumerate(folders):
        val_folder = folders[(test_position + 1) % len(folders)]
        train_folders = [folder for folder in folders if folder not in {test_folder, val_folder}]
        fold_dir = output_dir / f"heldout_{test_folder}"
        fold_dir.mkdir(exist_ok=True)
        pair_rows, role_rows = build_fold(indexed, train_folders, val_folder, test_folder, protocol)
        write_csv(pair_rows, fold_dir / "candidate_pairs.csv")
        write_csv(role_rows, fold_dir / "source_frame_roles.csv")
        write_splits(pair_rows, fold_dir / "splits.yaml")
        fold_manifest = {
            "heldout_test_folder": test_folder,
            "validation_folder": val_folder,
            "train_folders": train_folders,
            "training_permitted": bool(config["training_permitted"]),
            "audit_decision": config["audit_decision"],
            "train_pair_count": sum(row["split"] == "train" for row in pair_rows),
            "val_pair_count": sum(row["split"] == "val" for row in pair_rows),
            "test_input_count": len(protocol["test_input_frames"]),
            "reference_frame_count": int(protocol["reference_frames"][1]) - int(protocol["reference_frames"][0]) + 1,
        }
        (fold_dir / "fold_manifest.json").write_text(json.dumps(fold_manifest, indent=2), encoding="utf-8")
        summary_rows.append(fold_manifest)
    write_csv(summary_rows, output_dir / "lofo_fold_summary.csv")
    print(f"Wrote {len(summary_rows)} blocked LOFO fold manifests to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e6_repeated_frame_protocol.yaml")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def indexed_tiffs(folder: Path) -> dict[int, Path]:
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            match = FRAME_NUMBER.match(path.name)
            if match:
                result[int(match.group(1))] = path
    return result


def build_fold(
    indexed: dict[int, dict[int, Path]], train_folders: list[int], val_folder: int,
    test_folder: int, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    block_size = int(protocol["block_size"])
    pair_count = int(protocol["nonreused_pairs_per_folder"])
    target_offsets = [int(value) for value in protocol["target_offsets"]]
    for folder in train_folders + [val_folder]:
        split = "train" if folder in train_folders else "val"
        for pair_index in range(pair_count):
            block_start = pair_index * block_size + 1
            input_index = block_start + int(protocol["input_offset"])
            target_indices = [block_start + offset for offset in target_offsets]
            pair_key = f"folder_{folder}_block_{pair_index + 1:02d}"
            pair_rows.append(
                {
                    "pair_key": pair_key,
                    "folder": folder,
                    "split": split,
                    "input_frame_index": input_index,
                    "input_path": str(indexed[folder][input_index]),
                    "target_frame_indices": " ".join(map(str, target_indices)),
                    "target_frame_paths": "|".join(str(indexed[folder][index]) for index in target_indices),
                    "target_type": "disjoint_8frame_temporal_mean_surrogate",
                    "materialized": False,
                }
            )
            role_rows.append(frame_role(folder, input_index, indexed[folder][input_index], split, "input", pair_key))
            role_rows.extend(
                frame_role(folder, index, indexed[folder][index], split, "target_member", pair_key)
                for index in target_indices
            )
    reference_start, reference_end = [int(value) for value in protocol["reference_frames"]]
    for index in range(reference_start, reference_end + 1):
        role_rows.append(frame_role(test_folder, index, indexed[test_folder][index], "test", "reference_member", "test_reference"))
    for index in [int(value) for value in protocol["test_input_frames"]]:
        role_rows.append(frame_role(test_folder, index, indexed[test_folder][index], "test", "input", f"test_frame_{index}"))
    return pair_rows, role_rows


def frame_role(folder: int, index: int, path: Path, split: str, role: str, pair_key: str) -> dict[str, Any]:
    return {
        "source_key": f"folder_{folder}_frame_{index}",
        "folder": folder,
        "frame_index": index,
        "source_path": str(path),
        "split": split,
        "role": role,
        "pair_key": pair_key,
    }


def write_splits(rows: list[dict[str, Any]], path: Path) -> None:
    lines = []
    for split in ("train", "val"):
        lines.append(f"{split}:")
        lines.extend(f"  - {row['pair_key']}" for row in rows if row["split"] == split)
    lines.extend(["test:", "  # Test inputs and reference members are recorded in source_frame_roles.csv."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
