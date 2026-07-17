"""Build a controlled 2x2 synthetic noise structure-by-strength dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_noise import ICCDNoiseConfig, ICCDNoiseModel


def main() -> int:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    source_cfg = config["source"]
    construction = config["construction"]
    output_root = Path(args.output_root or config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    clean_dir = output_root / "shared_clean"
    clean_dir.mkdir(exist_ok=True)

    p_rows = read_csv(resolve_path(source_cfg["p99_pairs"]))
    h_rows = read_csv(resolve_path(source_cfg["physical_pairs"]))
    h_by_source = {row["source_pair_key"]: row for row in h_rows}
    if args.max_pairs > 0:
        p_rows = p_rows[: args.max_pairs]
    prior_values = load_yaml(resolve_path(source_cfg["noise_config"]))["iccd"]
    prior_values = {**prior_values, "clip": False}
    base_seed = int(source_cfg["seed"])
    range_max = float(source_cfg["range_max"])
    variants = dict(construction["variants"])
    output_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    metric_rows: list[dict[str, Any]] = []

    for index, p_row in enumerate(p_rows):
        source_key = p_row["source_pair_key"]
        if source_key not in h_by_source:
            raise KeyError(f"Physical source missing {source_key}")
        h_row = h_by_source[source_key]
        p_clean = load_tiff_float(Path(p_row["clean_path"]), range_max)
        h_clean = load_tiff_float(Path(h_row["clean_path"]), range_max)
        if p_clean.shape != h_clean.shape:
            raise ValueError(f"Shape mismatch for {source_key}: {p_clean.shape} vs {h_clean.shape}")

        pedestal = float(construction["common_clean_pedestal_dn"]) / range_max
        common_clean = pedestal + (1.0 - pedestal) * p_clean
        seed = base_seed + index
        p_unit, p_source_std = generate_unit_residual(p_clean, prior_values, seed)
        h_unit, h_source_std = generate_unit_residual(h_clean, prior_values, seed)
        common_std = safe_std(common_clean)
        low_target = p_source_std / safe_std(p_clean) * common_std
        high_target = h_source_std / safe_std(h_clean) * common_std
        clean_path = clean_dir / f"factorial_{source_key}.tif"
        save_uint16(clean_path, common_clean, range_max)

        units = {"p99": p_unit, "physical": h_unit}
        targets = {"low": low_target, "high": high_target}
        for variant, factors in variants.items():
            variant_dir = output_root / variant
            noisy_dir = variant_dir / "noisy"
            noisy_dir.mkdir(parents=True, exist_ok=True)
            unit = units[str(factors["structure"])]
            target_std = float(targets[str(factors["strength"])])
            noisy_float, preclip_ratio = project_residual(
                common_clean,
                unit,
                target_std,
                int(construction["projection_iterations"]),
            )
            noisy_path = noisy_dir / f"factorial_{source_key}.tif"
            save_uint16(noisy_path, noisy_float, range_max)
            clean_saved = load_tiff_float(clean_path, range_max)
            noisy_saved = load_tiff_float(noisy_path, range_max)
            residual = noisy_saved - clean_saved
            pair_key = f"factorial_{source_key}"
            output_rows[variant].append(
                {
                    "pair_key": pair_key,
                    "clean_path": str(clean_path.resolve()),
                    "noisy_path": str(noisy_path.resolve()),
                    "source_pair_key": source_key,
                    "structure_source": factors["structure"],
                    "strength_level": factors["strength"],
                    "target_residual_std": target_std,
                    "construction_seed": seed,
                    "claim_boundary": "controlled synthetic factorial data; not real ICCD ground truth",
                }
            )
            metric_rows.append(
                {
                    "variant": variant,
                    "pair_key": pair_key,
                    "source_pair_key": source_key,
                    "structure_source": factors["structure"],
                    "strength_level": factors["strength"],
                    "seed": seed,
                    "p99_source_residual_std": p_source_std,
                    "physical_source_residual_std": h_source_std,
                    "p99_source_clean_std": safe_std(p_clean),
                    "physical_source_clean_std": safe_std(h_clean),
                    "common_clean_std": common_std,
                    "target_residual_std": target_std,
                    "realized_residual_mean": float(np.mean(residual)),
                    "realized_residual_std": safe_std(residual),
                    "preclip_pixel_ratio": preclip_ratio,
                    "saved_zero_pixel_ratio": float(np.mean(noisy_saved <= 0.0)),
                    "saved_one_pixel_ratio": float(np.mean(noisy_saved >= 1.0)),
                    "signal_residual_correlation": safe_corr(clean_saved, residual),
                }
            )

    source_splits = load_yaml(resolve_path(source_cfg["source_splits"]))
    key_map = {row["source_pair_key"]: row["pair_key"] for row in output_rows[next(iter(variants))]}
    mapped_splits = {
        split: [key_map[key.removeprefix("iccd_like_")] for key in keys if key.removeprefix("iccd_like_") in key_map]
        for split, keys in source_splits.items()
    }
    for variant, rows in output_rows.items():
        write_csv(rows, output_root / variant / "pairs.csv")
        write_yaml(mapped_splits, output_root / variant / "splits.yaml")
    write_csv(metric_rows, output_root / "construction_metrics.csv")
    write_json(
        output_root / "construction_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "pair_count_per_variant": len(p_rows),
            "variants": variants,
            "source": source_cfg,
            "construction": construction,
            "note": (
                "Original absolute p99 residual std is larger because its clean content was rescaled. "
                "Strength levels use source residual-to-clean std transferred to the shared clean domain. "
                "A common 1024-DN pedestal is applied identically to every variant because zero-valued clean "
                "pixels cannot support zero-mean signed residuals without one-sided clipping."
            ),
        },
    )
    print(f"Wrote factorial dataset: {output_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5_noise_factorial.yaml")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(value: dict[str, Any], path: Path) -> None:
    import yaml

    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_tiff_float(path: Path, range_max: float) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path), dtype=np.float32) / float(range_max)


def save_uint16(path: Path, image: np.ndarray, range_max: float) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.rint(np.clip(image, 0.0, 1.0) * range_max).astype(np.uint16))


def generate_unit_residual(clean: np.ndarray, prior_values: dict[str, Any], seed: int) -> tuple[np.ndarray, float]:
    prior = ICCDNoiseModel(ICCDNoiseConfig(**{**prior_values, "seed": seed}))
    residual = prior.add_noise(clean) - clean
    residual = residual.astype(np.float64)
    residual -= float(np.mean(residual))
    std = safe_std(residual)
    return (residual / std).astype(np.float32), std


def project_residual(clean: np.ndarray, unit: np.ndarray, target_std: float, iterations: int) -> tuple[np.ndarray, float]:
    scale = float(target_std)
    bias = 0.0
    preclip_ratio = 0.0
    noisy = clean.copy()
    for _ in range(iterations):
        preclip = clean + scale * unit + bias
        preclip_ratio = float(np.mean((preclip < 0.0) | (preclip > 1.0)))
        noisy = np.clip(preclip, 0.0, 1.0)
        realized = noisy - clean
        bias -= float(np.mean(realized))
        realized_std = safe_std(realized)
        scale *= float(target_std / realized_std)
    return noisy.astype(np.float32), preclip_ratio


def safe_std(array: np.ndarray) -> float:
    return max(float(np.std(np.asarray(array, dtype=np.float64), ddof=1)), 1e-12)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == "__main__":
    raise SystemExit(main())
