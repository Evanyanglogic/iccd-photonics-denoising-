from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config(repo: Path, path: str) -> tuple[dict[str, Any], Path]:
    source = resolve(repo, path)
    return yaml.safe_load(source.read_text(encoding="utf-8")), source


def resolve(repo: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo / candidate


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout


def connected_components(ids: list[str], pairs: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for row in pairs.loc[mask].itertuples(index=False):
        union(row.source_pair_key_a, row.source_pair_key_b)
    roots = sorted({find(item) for item in ids})
    labels = {root: index + 1 for index, root in enumerate(roots)}
    return {item: labels[find(item)] for item in ids}


def output_hashes(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            rows.append({"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def image_inventory(path: Path, limit: int = 50000) -> tuple[int, list[str]]:
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".raw"}
    count, samples = 0, []
    if not path.is_dir():
        return 0, samples
    for root, _, files in os.walk(path):
        for name in files:
            if Path(name).suffix.lower() in extensions:
                count += 1
                if len(samples) < 5:
                    samples.append(str(Path(root) / name))
                if count >= limit:
                    return count, samples
    return count, samples


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

