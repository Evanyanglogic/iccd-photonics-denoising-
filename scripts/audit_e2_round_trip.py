"""Reproduce one historical E2 pair and audit float-to-uint16 round trips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_iccd_like_synthetic_pairs import (
    align_optional_map,
    center_crop,
    correct_normalize_and_fill,
    load_optional_npy,
    load_tiff,
    maybe_scale_content,
    valid_mask,
)
from src.iccd_noise import ICCDNoiseConfig, ICCDNoiseModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(config)
    return 0


def run(config: dict[str, Any]) -> dict[str, Any]:
    import tifffile

    output_dir = Path(config["output_root"]) / "round_trip_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(Path(config["source_pairs_csv"]))
    index = int(config["round_trip"]["source_row_index"])
    source = source_rows[index]
    if source["pair_key"] != config["round_trip"]["source_pair_key"]:
        raise ValueError("Configured round-trip row and pair key do not agree")
    raw = center_crop(load_tiff(Path(source[config["clean_column"]])), int(config["crop_size"])).astype(np.float32)
    dark = align_optional_map(load_optional_npy(Path(config["dark_offset_path"])), raw.shape)
    bad = align_optional_map(load_optional_npy(Path(config["bad_pixel_mask_path"])), raw.shape)
    valid = valid_mask(raw, bad, float(config["range_max"]))
    base_clean = correct_normalize_and_fill(raw, dark, valid, float(config["range_max"]))
    rows = []
    for variant in config["round_trip"]["variants"]:
        variant_cfg = config["historical_variants"][variant]
        prior_cfg = yaml.safe_load(Path(variant_cfg["config"]).read_text(encoding="utf-8"))
        clean, content_scale = maybe_scale_content(base_clean, valid, float(variant_cfg["content_p99_target"]))
        model_values = dict(prior_cfg["iccd"])
        model_values["seed"] = int(config["random_seed"]) + index
        noisy_float = ICCDNoiseModel(ICCDNoiseConfig(**model_values)).add_noise(clean)
        unclipped_values = {**model_values, "clip": False}
        noisy_unclipped = ICCDNoiseModel(ICCDNoiseConfig(**unclipped_values)).add_noise(clean)
        variant_dir = output_dir / variant
        variant_dir.mkdir()
        clean_path = variant_dir / "clean_round_trip.tif"
        noisy_path = variant_dir / "noisy_round_trip.tif"
        save_uint16(clean_path, clean, float(config["range_max"]))
        save_uint16(noisy_path, noisy_float, float(config["range_max"]))
        clean_reload = tifffile.imread(clean_path).astype(np.float32) / float(config["range_max"])
        noisy_reload = tifffile.imread(noisy_path).astype(np.float32) / float(config["range_max"])
        residual_float = noisy_float - clean
        residual_reload = noisy_reload - clean_reload
        historical_root = Path(variant_cfg["output_root"])
        pair_key = f"iccd_like_{source['pair_key']}"
        historical_clean = tifffile.imread(historical_root / "clean" / f"{pair_key}.tif")
        historical_noisy = tifffile.imread(historical_root / "noisy" / f"{pair_key}.tif")
        generated_clean_dn = np.rint(np.clip(clean, 0.0, 1.0) * float(config["range_max"])).astype(np.uint16)
        generated_noisy_dn = np.rint(np.clip(noisy_float, 0.0, 1.0) * float(config["range_max"])).astype(np.uint16)
        row = {
            "variant": variant,
            "source_pair_key": source["pair_key"],
            "seed": model_values["seed"],
            "content_p99_target": variant_cfg["content_p99_target"],
            "content_scale": content_scale,
            "float_clean_mean": float(np.mean(clean[valid])),
            "float_noisy_mean": float(np.mean(noisy_float[valid])),
            "float_residual_mean": float(np.mean(residual_float[valid])),
            "float_residual_std": float(np.std(residual_float[valid], ddof=1)),
            "preclip_low_ratio": float(np.mean(noisy_unclipped < 0.0)),
            "preclip_high_ratio": float(np.mean(noisy_unclipped > 1.0)),
            "saved_zero_ratio": float(np.mean(generated_noisy_dn == 0)),
            "saved_saturation_ratio": float(np.mean(generated_noisy_dn == int(config["range_max"]))),
            "clean_float_round_trip_max_error_dn": float(np.max(np.abs(clean_reload - clean)) * float(config["range_max"])),
            "noisy_float_round_trip_max_error_dn": float(np.max(np.abs(noisy_reload - noisy_float)) * float(config["range_max"])),
            "residual_round_trip_max_error_dn": float(np.max(np.abs(residual_reload - residual_float)) * float(config["range_max"])),
            "residual_std_after_reload": float(np.std(residual_reload[valid], ddof=1)),
            "historical_clean_exact_match": bool(np.array_equal(historical_clean, generated_clean_dn)),
            "historical_noisy_exact_match": bool(np.array_equal(historical_noisy, generated_noisy_dn)),
            "clean_output_sha256": sha256_file(clean_path),
            "noisy_output_sha256": sha256_file(noisy_path),
            "output_dtype": str(generated_noisy_dn.dtype),
        }
        row["round_trip_quantization_pass"] = bool(
            row["clean_float_round_trip_max_error_dn"] <= float(config["round_trip"]["max_float_quantization_error_dn"])
            and row["noisy_float_round_trip_max_error_dn"] <= float(config["round_trip"]["max_float_quantization_error_dn"])
            and row["residual_round_trip_max_error_dn"] <= float(config["round_trip"]["max_residual_round_trip_error_dn"])
        )
        row["clipping_gate_pass"] = bool(
            row["preclip_low_ratio"] + row["preclip_high_ratio"] <= float(config["round_trip"]["max_preclip_ratio"])
        )
        row["brightness_gate_pass"] = bool(
            abs(row["float_residual_mean"] * float(config["range_max"]))
            <= float(config["round_trip"]["max_absolute_brightness_bias_dn"])
        )
        row["generation_numeric_pass"] = bool(row["clipping_gate_pass"] and row["brightness_gate_pass"])
        rows.append(row)
        print(
            f"round_trip variant={variant} quantization={row['round_trip_quantization_pass']} "
            f"generation_numeric={row['generation_numeric_pass']}",
            flush=True,
        )
    write_csv(rows, output_dir / "round_trip_metrics.csv")
    summary = {
        "status": "PASS" if all(row["round_trip_quantization_pass"] for row in rows) else "FAIL",
        "generation_numeric_status": "PASS" if all(row["generation_numeric_pass"] for row in rows) else "FAIL",
        "variant_count": len(rows),
        "all_historical_clean_exact_match": all(row["historical_clean_exact_match"] for row in rows),
        "all_historical_noisy_exact_match": all(row["historical_noisy_exact_match"] for row in rows),
        "note": "This is a fixed single-pair replay, not batch synthetic generation.",
    }
    (output_dir / "round_trip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_uint16(path: Path, image: np.ndarray, range_max: float) -> None:
    import tifffile

    tifffile.imwrite(path, np.rint(np.clip(image, 0.0, 1.0) * range_max).astype(np.uint16))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
