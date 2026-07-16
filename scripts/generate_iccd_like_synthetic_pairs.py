"""Generate ICCD-like synthetic noisy pairs from sCMOS content frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_noise import ICCDNoiseConfig, ICCDNoiseModel


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "clean"
    noisy_dir = output_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    rows = read_pairs(Path(args.pairs_csv))
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    if not rows:
        raise ValueError(f"No rows found in pair manifest: {args.pairs_csv}")

    config = load_config(Path(args.config))
    iccd_config_values = dict(config.get("iccd", {}))
    base_seed = int(args.seed if args.seed is not None else config.get("seed", 20260715))

    output_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        clean_source_path = Path(row[args.clean_column])
        dark_offset = load_optional_npy(Path(row.get("dark_offset_path", "")))
        bad_mask = load_optional_npy(Path(row.get("bad_pixel_mask_path", "")))

        clean_raw = center_crop(load_tiff(clean_source_path), args.crop_size).astype(np.float32)
        dark_crop = align_optional_map(dark_offset, clean_raw.shape)
        mask_crop = align_optional_map(bad_mask, clean_raw.shape)
        valid = valid_mask(clean_raw, mask_crop, args.range_max)
        valid_fraction = float(np.mean(valid))
        if valid_fraction < args.min_valid_fraction:
            raise ValueError(
                f"{row['pair_key']} valid fraction {valid_fraction:.6g} is below "
                f"--min-valid-fraction {args.min_valid_fraction:.6g}"
            )

        clean = correct_normalize_and_fill(clean_raw, dark_crop, valid, args.range_max)
        clean, content_scale = maybe_scale_content(clean, valid, args.content_p99_target)

        pair_key = f"iccd_like_{row['pair_key']}"
        prior = ICCDNoiseModel(ICCDNoiseConfig(**{**iccd_config_values, "seed": base_seed + index}))
        noisy = prior.add_noise(clean)

        clean_path = clean_dir / f"{pair_key}.tif"
        noisy_path = noisy_dir / f"{pair_key}.tif"
        save_tiff_uint16(clean_path, clean, args.range_max)
        save_tiff_uint16(noisy_path, noisy, args.range_max)

        output_rows.append(
            {
                "pair_key": pair_key,
                "clean_path": str(clean_path.resolve()),
                "noisy_path": str(noisy_path.resolve()),
                "source_pair_key": row["pair_key"],
                "source_device": row.get("source_device", "sCMOS"),
                "clean_source_path": str(clean_source_path),
                "synthetic_noise_model": "iccd_prior",
                "noise_config": str(Path(args.config)),
                "crop_size": args.crop_size,
                "range_max": args.range_max,
                "valid_fraction": valid_fraction,
                "content_scale": content_scale,
                "claim_boundary": "synthetic ICCD-like noisy data from sCMOS content, not real ICCD paired data",
            }
        )
        metric_rows.append(summarize_pair(pair_key, clean, noisy, valid, valid_fraction, content_scale))

    pairs_out = output_dir / "pairs.csv"
    splits_out = output_dir / "splits.yaml"
    metrics_out = output_dir / "synthetic_pair_metrics.csv"
    report_out = output_dir / "synthetic_pair_report.md"
    write_csv(output_rows, pairs_out)
    write_csv(metric_rows, metrics_out)
    write_splits(
        output_rows,
        splits_out,
        source_splits=load_optional_splits(args.source_splits),
        source_to_new={row["source_pair_key"]: row["pair_key"] for row in output_rows},
    )
    write_report(report_out, pairs_out, splits_out, metrics_out, args, config, metric_rows, output_rows)

    print(f"Wrote pairs: {pairs_out}")
    print(f"Wrote splits: {splits_out}")
    print(f"Wrote metrics: {metrics_out}")
    print(f"Wrote report: {report_out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--config", default="configs/iccd_prior_20260319.yaml")
    parser.add_argument("--output-dir", default="reports/iccd_like_synthetic_pairs")
    parser.add_argument("--source-splits", default="")
    parser.add_argument("--clean-column", default="clean_path")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument(
        "--content-p99-target",
        type=float,
        default=0.0,
        help="Optional target p99 after correction. Keep 0 to preserve physical scale.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                result[key] = {}
                current = result[key]
            else:
                result[key] = coerce_scalar(value.strip('"').strip("'"))
                current = None
            continue
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = coerce_scalar(value.strip().strip('"').strip("'"))
    return result


def coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def save_tiff_uint16(path: Path, image: np.ndarray, range_max: float) -> None:
    try:
        import tifffile

        arr = np.rint(np.clip(image, 0.0, 1.0) * range_max).astype(np.uint16)
        tifffile.imwrite(path, arr)
    except Exception as exc:
        raise RuntimeError(f"Failed to write TIFF {path}: {exc}") from exc


def load_optional_npy(path: Path) -> np.ndarray | None:
    if not str(path) or not path.exists():
        return None
    return np.load(path)


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {arr.shape}")
    h, w = arr.shape
    size = min(crop_size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top : top + size, left : left + size]


def align_optional_map(arr: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.shape == shape:
        return arr
    return center_crop(np.asarray(arr), min(shape))


def valid_mask(raw: np.ndarray, bad_mask: np.ndarray | None, range_max: float) -> np.ndarray:
    mask = np.ones(raw.shape, dtype=bool)
    if bad_mask is not None:
        mask &= ~np.asarray(bad_mask, dtype=bool)
    mask &= raw > 0
    mask &= raw < range_max
    return mask


def correct_normalize_and_fill(
    raw: np.ndarray,
    dark_offset: np.ndarray | None,
    valid: np.ndarray,
    range_max: float,
) -> np.ndarray:
    corrected = raw.astype(np.float32)
    if dark_offset is not None:
        corrected = corrected - dark_offset.astype(np.float32)
    clean = np.clip(corrected / float(range_max), 0.0, 1.0).astype(np.float32)
    if not np.any(valid):
        raise ValueError("No valid pixels remain after masking")
    fill_value = float(np.median(clean[valid]))
    clean = clean.copy()
    clean[~valid] = fill_value
    return clean


def maybe_scale_content(clean: np.ndarray, valid: np.ndarray, target_p99: float) -> tuple[np.ndarray, float]:
    if target_p99 <= 0:
        return clean, 1.0
    current = float(np.percentile(clean[valid], 99))
    if current <= 1e-12:
        return clean, 1.0
    scale = float(target_p99 / current)
    return np.clip(clean * scale, 0.0, 1.0).astype(np.float32), scale


def summarize_pair(
    pair_key: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    valid: np.ndarray,
    valid_fraction: float,
    content_scale: float,
) -> dict[str, Any]:
    residual = noisy - clean
    return {
        "pair_key": pair_key,
        "valid_fraction": valid_fraction,
        "content_scale": content_scale,
        "clean_mean": float(np.mean(clean[valid])),
        "clean_std": float(np.std(clean[valid], ddof=1)),
        "noisy_mean": float(np.mean(noisy[valid])),
        "noisy_std": float(np.std(noisy[valid], ddof=1)),
        "residual_mean": float(np.mean(residual[valid])),
        "residual_std": float(np.std(residual[valid], ddof=1)),
        "clean_p99": float(np.percentile(clean[valid], 99)),
        "noisy_p99": float(np.percentile(noisy[valid], 99)),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_optional_splits(path_value: str) -> dict[str, list[str]]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    splits: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current = line[:-1]
            splits[current] = []
        elif current and line.startswith("- "):
            splits[current].append(line[2:].strip())
    return splits


def write_splits(
    rows: list[dict[str, Any]],
    path: Path,
    source_splits: dict[str, list[str]],
    source_to_new: dict[str, str],
) -> None:
    if source_splits:
        splits = {
            split: [source_to_new[key] for key in keys if key in source_to_new]
            for split, keys in source_splits.items()
        }
    else:
        keys = [str(row["pair_key"]) for row in rows]
        train_end = int(round(len(keys) * 0.85))
        val_end = train_end + int(round(len(keys) * 0.08))
        splits = {"train": keys[:train_end], "val": keys[train_end:val_end], "test": keys[val_end:]}

    lines = ["# Auto-generated by scripts/generate_iccd_like_synthetic_pairs.py"]
    for split in ["train", "val", "test"]:
        lines.append(f"{split}:")
        for key in splits.get(split, []):
            lines.append(f"  - {key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    pairs_out: Path,
    splits_out: Path,
    metrics_out: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    summary = summarize_metrics(metrics)
    config_snapshot = config.get("iccd", asdict(ICCDNoiseConfig()))
    lines = [
        "# ICCD-Like Synthetic Pair Generation",
        "",
        "## Inputs",
        "",
        f"- Source pairs CSV: `{args.pairs_csv}`",
        f"- Source splits: `{args.source_splits or 'not provided'}`",
        f"- ICCD prior config: `{args.config}`",
        f"- Clean source column: `{args.clean_column}`",
        f"- Crop size: {args.crop_size}",
        f"- Content p99 target: {args.content_p99_target:g}",
        "",
        "## Outputs",
        "",
        f"- Pairs CSV: `{pairs_out}`",
        f"- Splits YAML: `{splits_out}`",
        f"- Metrics CSV: `{metrics_out}`",
        f"- Clean TIFF directory: `{pairs_out.parent / 'clean'}`",
        f"- Noisy TIFF directory: `{pairs_out.parent / 'noisy'}`",
        "",
        "## Summary",
        "",
        f"- Pair count: {len(rows)}",
        f"- Valid fraction mean: {summary['valid_fraction_mean']:.6g}",
        f"- Clean mean/std: {summary['clean_mean_mean']:.6g} / {summary['clean_std_mean']:.6g}",
        f"- Noisy mean/std: {summary['noisy_mean_mean']:.6g} / {summary['noisy_std_mean']:.6g}",
        f"- Residual mean/std: {summary['residual_mean_mean']:.6g} / {summary['residual_std_mean']:.6g}",
        f"- Clean p99 / noisy p99: {summary['clean_p99_mean']:.6g} / {summary['noisy_p99_mean']:.6g}",
        "",
        "## ICCD Prior Snapshot",
        "",
        "```json",
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        "```",
        "",
        "## Claim Boundary",
        "",
        "- Clean images are offset-corrected sCMOS content/reference crops.",
        "- Noisy images are synthetic ICCD-like samples generated from the E1-derived prior.",
        "- This output is not real ICCD paired ground truth and should be used for controlled synthetic-data experiments only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "valid_fraction",
        "clean_mean",
        "clean_std",
        "noisy_mean",
        "noisy_std",
        "residual_mean",
        "residual_std",
        "clean_p99",
        "noisy_p99",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
