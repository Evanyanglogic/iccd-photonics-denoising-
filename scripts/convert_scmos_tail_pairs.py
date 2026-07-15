"""Create a pair manifest for multi-exposure sCMOS data by tail frame index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any


TIFF_SUFFIXES = {".tif", ".tiff"}
TRAILING_INDEX = re.compile(r"(?P<index>\d{3})$")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    noisy_dir = root / args.noisy_exposure
    clean_dir = root / args.clean_exposure
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy exposure folder does not exist: {noisy_dir}")
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean exposure folder does not exist: {clean_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = Path(args.pairs_out) if args.pairs_out else output_dir / "pairs.csv"
    splits_out = Path(args.splits_out) if args.splits_out else output_dir / "splits.yaml"
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    noisy_by_index = index_tiffs(noisy_dir)
    clean_by_index = index_tiffs(clean_dir)
    common_indices = sorted(set(noisy_by_index).intersection(clean_by_index))
    if args.max_pairs > 0:
        common_indices = common_indices[: args.max_pairs]
    rows = [
        make_pair_row(
            idx,
            noisy_by_index[idx],
            clean_by_index[idx],
            args,
        )
        for idx in common_indices
    ]
    if not rows:
        raise ValueError(f"No common tail indices found for {args.noisy_exposure} -> {args.clean_exposure}")

    write_csv(rows, pairs_out)
    splits = make_splits(rows, args.val_fraction, args.test_fraction)
    write_splits(splits, splits_out)
    report_path = output_dir / "manifest_report.md"
    write_report(args, rows, noisy_by_index, clean_by_index, pairs_out, splits_out, report_path, splits)
    print(f"Wrote pair manifest: {pairs_out}")
    print(f"Wrote split manifest: {splits_out}")
    print(f"Wrote manifest report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--noisy-exposure", required=True)
    parser.add_argument("--clean-exposure", required=True)
    parser.add_argument("--output-dir", default="reports/scmos_tail_pair_manifest")
    parser.add_argument("--pairs-out", default="")
    parser.add_argument("--splits-out", default="")
    parser.add_argument("--source-device", default="sCMOS")
    parser.add_argument("--use-case", default="scmos_proxy_pair_or_iccd_like_content_source")
    parser.add_argument("--dark-offset-path", default="")
    parser.add_argument("--bad-pixel-mask-path", default="")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def index_tiffs(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in sorted([item for item in folder.iterdir() if item.is_file() and item.suffix.lower() in TIFF_SUFFIXES], key=natural_file_key):
        idx = trailing_index(path)
        if idx is None:
            continue
        if idx in result:
            duplicates.append(idx)
            continue
        result[idx] = path
    if duplicates:
        print(f"Warning: ignored duplicate tail indices in {folder}: {duplicates[:12]}")
    return result


def natural_file_key(path: Path) -> tuple[int, str]:
    idx = trailing_index(path)
    return (idx if idx is not None else 10**12, path.name)


def trailing_index(path: Path) -> int | None:
    match = TRAILING_INDEX.search(path.stem)
    if not match:
        return None
    return int(match.group("index"))


def make_pair_row(idx: int, noisy_path: Path, clean_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    pair_key = f"{args.noisy_exposure}_to_{args.clean_exposure}_{idx:03d}"
    return {
        "pair_key": pair_key,
        "clean_path": str(clean_path),
        "noisy_path": str(noisy_path),
        "source_device": args.source_device,
        "use_case": args.use_case,
        "pairing_method": "tail_index",
        "tail_index": f"{idx:03d}",
        "noisy_exposure": args.noisy_exposure,
        "clean_exposure": args.clean_exposure,
        "dark_offset_path": args.dark_offset_path,
        "bad_pixel_mask_path": args.bad_pixel_mask_path,
        "claim_boundary": "sCMOS proxy/content source, not real ICCD paired data",
    }


def make_splits(rows: list[dict[str, Any]], val_fraction: float, test_fraction: float) -> dict[str, list[str]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be non-negative and sum to less than 1")
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        key = row["pair_key"]
        bucket = stable_bucket(key)
        if bucket < test_fraction:
            split = "test"
        elif bucket < test_fraction + val_fraction:
            split = "val"
        else:
            split = "train"
        splits[split].append(key)
    return splits


def stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_splits(splits: dict[str, list[str]], output_path: Path) -> None:
    lines = ["# Auto-generated by scripts/convert_scmos_tail_pairs.py"]
    for name in ["train", "val", "test"]:
        lines.append(f"{name}:")
        for key in splits[name]:
            lines.append(f"  - {key}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    noisy_by_index: dict[int, Path],
    clean_by_index: dict[int, Path],
    pairs_out: Path,
    splits_out: Path,
    report_path: Path,
    splits: dict[str, list[str]],
) -> None:
    common_indices = [int(row["tail_index"]) for row in rows]
    lines = [
        "# sCMOS Tail-Index Pair Manifest",
        "",
        f"- Root: `{args.root}`",
        f"- Noisy exposure: `{args.noisy_exposure}`",
        f"- Clean exposure: `{args.clean_exposure}`",
        f"- Source device: `{args.source_device}`",
        f"- Use case: `{args.use_case}`",
        f"- Pair manifest: `{pairs_out}`",
        f"- Split manifest: `{splits_out}`",
        "",
        "## Pairing Summary",
        "",
        f"- Noisy indexed TIFFs: {len(noisy_by_index)}",
        f"- Clean indexed TIFFs: {len(clean_by_index)}",
        f"- Common tail-index pairs: {len(rows)}",
        f"- Tail-index range: {min(common_indices):03d}..{max(common_indices):03d}",
        f"- Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
        "",
        "## Metadata Columns",
        "",
        "- `source_device`: sCMOS",
        "- `use_case`: proxy pair or ICCD-like synthetic content source",
        "- `dark_offset_path`: derived dark/offset artifact, if provided",
        "- `bad_pixel_mask_path`: derived bad-pixel mask artifact, if provided",
        "- `claim_boundary`: prevents accidental interpretation as real ICCD paired data",
        "",
        "## Next Gate",
        "",
        "Run the no-model baseline on this manifest, then inspect brightness, alignment, and mask-aware metrics before using these images as reference content.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
