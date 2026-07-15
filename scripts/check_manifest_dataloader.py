"""Check manifest-backed ICCD dataloader output before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iccd_data import make_iccd_dataset
from src.iccd_data.manifest import load_pair_manifest, load_split_manifest


def main() -> int:
    args = parse_args()
    pairs_csv = Path(args.pairs_csv)
    splits_yaml = Path(args.splits_yaml)

    records = load_pair_manifest(pairs_csv, base_dir=args.base_dir)
    splits = load_split_manifest(splits_yaml)
    print(f"Pair records: {len(records)}")
    print(f"Splits: {', '.join(f'{name}={len(keys)}' for name, keys in splits.items())}")

    dataset = make_iccd_dataset(
        pairs_csv=pairs_csv,
        splits_yaml=splits_yaml,
        split=args.split,
        range_max=args.range_max,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        crop_mode=args.crop_mode,
        augment=args.augment,
        seed=args.seed,
        base_dir=args.base_dir,
        return_tensors=not args.numpy,
    )
    print(f"Dataset split: {args.split}")
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("No sample available for this split.")
        return 0

    sample = dataset[0]
    noisy = as_numpy(sample["noisy"])
    clean = as_numpy(sample["clean"])
    print(f"First pair key: {sample['pair_key']}")
    print(f"Noisy shape/range: {tuple(noisy.shape)} / {float(noisy.min()):.6g}..{float(noisy.max()):.6g}")
    print(f"Clean shape/range: {tuple(clean.shape)} / {float(clean.min()):.6g}..{float(clean.max()):.6g}")
    print(f"Metadata keys: {', '.join(sorted(sample['metadata'])) if sample['metadata'] else '(none)'}")
    if noisy.shape != clean.shape:
        raise ValueError(f"Shape mismatch in dataloader output: noisy {noisy.shape}, clean {clean.shape}")
    if noisy.ndim != 3:
        raise ValueError(f"Expected CHW output, got shape {noisy.shape}")
    if np.any(noisy < 0) or np.any(noisy > 1) or np.any(clean < 0) or np.any(clean > 1):
        raise ValueError("Dataloader output is outside [0, 1]")
    print("Manifest dataloader check passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="data_manifest/pairs.csv")
    parser.add_argument("--splits-yaml", default="data_manifest/splits.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--range-max", type=float, default=65535.0)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--crop-mode", choices=["none", "center", "random"], default="none")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--numpy", action="store_true", help="Return numpy arrays instead of torch tensors.")
    return parser.parse_args()


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
