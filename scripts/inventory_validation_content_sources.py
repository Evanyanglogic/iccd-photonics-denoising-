"""Bounded, read-only inventory for configured validation content candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".raw", ".npy"}


def inventory_candidates(config: dict) -> pd.DataFrame:
    rows = []
    for item in config["candidate_directories"]:
        root = Path(item["path"])
        files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
        image_files = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
        names = {p.name.lower() for p in files}
        dirs = {p.name.lower() for p in root.rglob("*") if p.is_dir()} if root.is_dir() else set()
        rows.append({
            **item,
            "absolute_path": str(root),
            "disk": root.drive,
            "parent_project": root.parts[1] if len(root.parts) > 1 else "",
            "exists": root.is_dir(),
            "file_count": len(files),
            "image_file_count": len(image_files),
            "extensions": ";".join(sorted({p.suffix.lower() for p in files if p.suffix})),
            "approximate_size_bytes": sum(p.stat().st_size for p in files),
            "directory_depth": max((len(p.relative_to(root).parts) for p in files), default=0),
            "README_present": any(n.startswith("readme") for n in names),
            "manifest_present": any("manifest" in n or n == "benchmark.json" for n in names),
            "pair_table_present": any("pair" in n and Path(n).suffix in {".csv", ".txt", ".json"} for n in names),
            "train_dir_present": "train" in dirs,
            "val_dir_present": bool({"val", "validation"} & dirs),
            "test_dir_present": "test" in dirs,
            "scene_dirs_present": any(n.startswith("scene") for n in dirs),
            "source_notes": "Configured bounded candidate; no source files modified.",
            "initial_priority": item["priority"],
            "exclusion_reason": "" if item["priority"] != "C" else "Pre-identified as current, derived, calibration, evaluation, cache, duplicate, or historical training data.",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    inventory_candidates(config).to_csv(output / "candidate_directory_inventory.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
