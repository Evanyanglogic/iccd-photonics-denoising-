"""Audit ICCD/sCMOS paired TIFF data before training.

This script is deliberately conservative: it checks pairing, numeric range,
basic calibration statistics, and split readiness before any model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSummary:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    p001: float
    p01: float
    p50: float
    p99: float
    p999: float
    saturated_fraction: float
    zero_fraction: float


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    clean_dir = Path(args.clean_dir or config.get("clean_dir", "")).expanduser()
    noisy_dir = Path(args.noisy_dir or config.get("noisy_dir", "")).expanduser()
    dark_dir = optional_path(args.dark_dir or config.get("dark_dir", ""))
    flat_dir = optional_path(args.flat_dir or config.get("flat_dir", ""))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = Path(args.pairs_out)
    splits_out = Path(args.splits_out)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    audit = run_audit(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        dark_dir=dark_dir,
        flat_dir=flat_dir,
        config=config,
        max_sample_pairs=int(config.get("max_sample_pairs", 32)),
        max_calibration_files=int(config.get("max_calibration_files", 64)),
    )

    write_pairs_csv(audit["pairs"], pairs_out)
    write_splits(audit["pairs"], config, splits_out)
    report_path = output_dir / "data_audit.md"
    write_markdown_report(audit, config, pairs_out, splits_out, report_path)
    print(f"Wrote audit report: {report_path}")
    print(f"Wrote pair manifest: {pairs_out}")
    print(f"Wrote split manifest: {splits_out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_iccd.yaml")
    parser.add_argument("--clean-dir", default="")
    parser.add_argument("--noisy-dir", default="")
    parser.add_argument("--dark-dir", default="")
    parser.add_argument("--flat-dir", default="")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--pairs-out", default="data_manifest/pairs.csv")
    parser.add_argument("--splits-out", default="data_manifest/splits.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for simple top-level YAML scalars/lists."""

    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        current_key = key
        if value == "":
            result[key] = []
        elif value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            result[key] = coerce_scalar(value)
    return result


def coerce_scalar(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser()


def run_audit(
    clean_dir: Path,
    noisy_dir: Path,
    dark_dir: Path | None,
    flat_dir: Path | None,
    config: dict[str, Any],
    max_sample_pairs: int,
    max_calibration_files: int,
) -> dict[str, Any]:
    if not clean_dir.exists():
        raise FileNotFoundError(f"clean_dir does not exist: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"noisy_dir does not exist: {noisy_dir}")

    metadata = load_metadata(optional_path(config.get("metadata_csv", "")))
    pair_regex = str(config.get("pair_key_regex", "") or "")
    expected_max = float(config.get("expected_range_max", 65535))

    clean_files = list_tiffs(clean_dir)
    noisy_files = list_tiffs(noisy_dir)
    clean_by_key = index_by_pair_key(clean_files, pair_regex)
    noisy_by_key = index_by_pair_key(noisy_files, pair_regex)
    common_keys = sorted(set(clean_by_key).intersection(noisy_by_key))
    missing_noisy = sorted(set(clean_by_key).difference(noisy_by_key))
    missing_clean = sorted(set(noisy_by_key).difference(clean_by_key))

    pairs = []
    summaries = []
    for key in common_keys:
        clean_path = clean_by_key[key]
        noisy_path = noisy_by_key[key]
        pair_row = build_pair_row(key, clean_path, noisy_path, metadata)
        pairs.append(pair_row)
        if len(summaries) < max_sample_pairs:
            clean_summary = summarize_image(clean_path, expected_max=expected_max)
            noisy_summary = summarize_image(noisy_path, expected_max=expected_max)
            summaries.append(
                {
                    "key": key,
                    "clean": clean_summary,
                    "noisy": noisy_summary,
                    "shape_match": clean_summary.shape == noisy_summary.shape,
                    "brightness_ratio": safe_ratio(noisy_summary.p50, clean_summary.p50),
                }
            )

    dark_summaries = summarize_collection(dark_dir, expected_max, max_calibration_files)
    flat_summaries = summarize_collection(flat_dir, expected_max, max_calibration_files)

    return {
        "clean_dir": clean_dir,
        "noisy_dir": noisy_dir,
        "clean_count": len(clean_files),
        "noisy_count": len(noisy_files),
        "pair_count": len(common_keys),
        "missing_noisy": missing_noisy,
        "missing_clean": missing_clean,
        "pairs": pairs,
        "sample_summaries": summaries,
        "dark_summaries": dark_summaries,
        "flat_summaries": flat_summaries,
        "metadata_rows": len(metadata),
        "warnings": build_warnings(common_keys, missing_noisy, missing_clean, summaries, config),
    }


def list_tiffs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in TIFF_SUFFIXES)


def index_by_pair_key(paths: list[Path], pair_regex: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        key = pair_key(path, pair_regex)
        if key in result:
            duplicates.append(key)
            continue
        result[key] = path
    if duplicates:
        raise ValueError(f"Duplicate pair keys found: {duplicates[:10]}")
    return result


def pair_key(path: Path, pair_regex: str) -> str:
    if pair_regex:
        match = re.search(pair_regex, path.stem)
        if not match:
            return path.stem
        if "key" in match.groupdict():
            return match.group("key")
        return match.group(0)
    return path.stem


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keys = [row.get("filename", ""), row.get("stem", "")]
            for key in keys:
                if key:
                    rows[Path(key).stem] = row
                    rows[Path(key).name] = row
    return rows


def build_pair_row(
    key: str,
    clean_path: Path,
    noisy_path: Path,
    metadata: dict[str, dict[str, str]],
) -> dict[str, str]:
    clean_meta = metadata.get(clean_path.stem, metadata.get(clean_path.name, {}))
    noisy_meta = metadata.get(noisy_path.stem, metadata.get(noisy_path.name, {}))
    row = {
        "pair_key": key,
        "clean_path": str(clean_path),
        "noisy_path": str(noisy_path),
    }
    for field in sorted(set(clean_meta).union(noisy_meta)):
        clean_value = clean_meta.get(field, "")
        noisy_value = noisy_meta.get(field, "")
        if clean_value == noisy_value:
            row[field] = clean_value
        else:
            row[f"clean_{field}"] = clean_value
            row[f"noisy_{field}"] = noisy_value
    return row


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def summarize_image(path: Path, expected_max: float) -> ImageSummary:
    arr = read_tiff(path)
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError(f"Empty image: {path}")
    return ImageSummary(
        path=path,
        dtype=str(arr.dtype),
        shape=tuple(int(x) for x in arr.shape),
        minimum=float(np.min(flat)),
        maximum=float(np.max(flat)),
        p001=float(np.percentile(flat, 0.1)),
        p01=float(np.percentile(flat, 1)),
        p50=float(np.percentile(flat, 50)),
        p99=float(np.percentile(flat, 99)),
        p999=float(np.percentile(flat, 99.9)),
        saturated_fraction=float(np.mean(flat >= expected_max)),
        zero_fraction=float(np.mean(flat <= 0)),
    )


def summarize_collection(root: Path | None, expected_max: float, limit: int) -> list[ImageSummary]:
    if root is None or not root.exists():
        return []
    return [summarize_image(path, expected_max) for path in list_tiffs(root)[:limit]]


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def build_warnings(
    common_keys: list[str],
    missing_noisy: list[str],
    missing_clean: list[str],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not common_keys:
        warnings.append("No paired TIFF files were found.")
    if missing_noisy or missing_clean:
        warnings.append("Pair manifest is incomplete: some keys exist in only one directory.")
    if not config.get("metadata_csv"):
        warnings.append("No metadata_csv configured; scene/condition split cannot be verified.")
    if not config.get("dark_dir"):
        warnings.append("No dark_dir configured; dark-field distribution is not checked.")
    if not config.get("flat_dir"):
        warnings.append("No flat_dir configured; mean-variance/flat-field behavior is not checked.")
    if any(not item["shape_match"] for item in summaries):
        warnings.append("At least one sampled clean/noisy pair has a shape mismatch.")
    if any(item["clean"].saturated_fraction > 0.001 for item in summaries):
        warnings.append("Some sampled clean frames contain saturated pixels above 0.1%.")
    return warnings


def write_pairs_csv(pairs: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = sorted({field for row in pairs for field in row})
    preferred = ["pair_key", "clean_path", "noisy_path"]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def write_splits(pairs: list[dict[str, str]], config: dict[str, Any], output_path: Path) -> None:
    split_cfg = config.get("split", {}) if isinstance(config.get("split", {}), dict) else {}
    train_fraction = float(split_cfg.get("train_fraction", 0.7))
    val_fraction = float(split_cfg.get("val_fraction", 0.15))
    seed = int(split_cfg.get("seed", 0))
    rows = {"train": [], "val": [], "test": []}
    for pair in pairs:
        key = split_group_key(pair, config)
        bucket = stable_bucket(str(key), seed=seed)
        if bucket < train_fraction:
            split = "train"
        elif bucket < train_fraction + val_fraction:
            split = "val"
        else:
            split = "test"
        rows[split].append(pair["pair_key"])

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Auto-generated by scripts/audit_iccd_dataset.py\n")
        handle.write("# Splits are hashed by scene/condition fields when metadata is available.\n")
        for split, keys in rows.items():
            handle.write(f"{split}:\n")
            for key in keys:
                handle.write(f"  - {key}\n")


def split_group_key(pair: dict[str, str], config: dict[str, Any]) -> str:
    """Build the split unit from scene and condition metadata when available."""

    fields = []
    fields.extend(config.get("scene_key_fields", []) or [])
    fields.extend(config.get("condition_key_fields", []) or [])
    parts = [f"{field}={pair[field]}" for field in fields if pair.get(field)]
    if not parts:
        return pair["pair_key"]
    return "|".join(parts)


def stable_bucket(value: str, seed: int = 0) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def write_markdown_report(
    audit: dict[str, Any],
    config: dict[str, Any],
    pairs_out: Path,
    splits_out: Path,
    report_path: Path,
) -> None:
    lines = [
        "# ICCD Dataset Audit",
        "",
        "## Summary",
        "",
        f"- Dataset: {config.get('dataset_name', 'unknown')}",
        f"- Device: {config.get('device', 'unknown')}",
        f"- Clean dir: `{audit['clean_dir']}`",
        f"- Noisy dir: `{audit['noisy_dir']}`",
        f"- Clean TIFF count: {audit['clean_count']}",
        f"- Noisy TIFF count: {audit['noisy_count']}",
        f"- Paired count: {audit['pair_count']}",
        f"- Metadata rows indexed: {audit['metadata_rows']}",
        f"- Pair manifest: `{pairs_out}`",
        f"- Split manifest: `{splits_out}`",
        "",
        "## Warnings",
        "",
    ]
    if audit["warnings"]:
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    else:
        lines.append("- No blocking warning detected in the sampled audit.")

    lines.extend(["", "## Sample Pair Range Check", ""])
    lines.append("| key | shape ok | clean dtype | noisy dtype | clean p50 | noisy p50 | noisy/clean p50 | clean max | noisy max | clean sat frac | noisy sat frac |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in audit["sample_summaries"]:
        clean = item["clean"]
        noisy = item["noisy"]
        lines.append(
            "| "
            f"{item['key']} | {item['shape_match']} | {clean.dtype} | {noisy.dtype} | "
            f"{clean.p50:.4g} | {noisy.p50:.4g} | {format_float(item['brightness_ratio'])} | "
            f"{clean.maximum:.4g} | {noisy.maximum:.4g} | {clean.saturated_fraction:.4g} | {noisy.saturated_fraction:.4g} |"
        )

    lines.extend(["", "## Calibration Coverage", ""])
    lines.append(f"- Dark frames sampled: {len(audit['dark_summaries'])}")
    lines.append(f"- Flat frames sampled: {len(audit['flat_summaries'])}")
    if audit["dark_summaries"]:
        lines.append(f"- Dark median range: {summary_range(audit['dark_summaries'], 'p50')}")
    if audit["flat_summaries"]:
        lines.append(f"- Flat median range: {summary_range(audit['flat_summaries'], 'p50')}")

    lines.extend(["", "## Missing Pairs", ""])
    lines.append(f"- Clean without noisy: {len(audit['missing_noisy'])}")
    lines.append(f"- Noisy without clean: {len(audit['missing_clean'])}")
    if audit["missing_noisy"][:10]:
        lines.append(f"- First clean-only keys: {', '.join(audit['missing_noisy'][:10])}")
    if audit["missing_clean"][:10]:
        lines.append(f"- First noisy-only keys: {', '.join(audit['missing_clean'][:10])}")

    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Do not start model changes until pair integrity, metadata coverage, float-domain metrics, and held-out splits are confirmed.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4g}"


def summary_range(summaries: list[ImageSummary], attr: str) -> str:
    values = [float(getattr(item, attr)) for item in summaries]
    return f"{min(values):.4g} to {max(values):.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
