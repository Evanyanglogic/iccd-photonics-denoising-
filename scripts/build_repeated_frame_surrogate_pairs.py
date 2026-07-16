"""Build surrogate clean/noisy pairs from repeated ICCD frames."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
LEADING_NUMBER = re.compile(r"^(?P<number>\d+)")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "clean_surrogate"
    noisy_dir = output_dir / "noisy_heldout"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for folder in select_folders(root, args.folders):
        paths = list_tiffs(folder)
        if len(paths) < args.train_frames + 1:
            print(f"Skipping {folder.name}: not enough frames ({len(paths)})")
            continue
        train_paths = paths[: args.train_frames]
        heldout_paths = paths[args.train_frames : args.train_frames + args.max_heldout_frames * args.heldout_stride : args.heldout_stride]
        if not heldout_paths:
            continue

        print(f"Processing {folder.name}: {len(train_paths)} clean-surrogate frames, {len(heldout_paths)} held-out frames")
        clean = mean_crop(train_paths, args.crop_size)
        clean_path = clean_dir / f"folder_{folder.name}_mean_clean.tif"
        write_tiff(clean_path, np.clip(np.rint(clean), 0, args.range_max).astype(np.uint16))
        summary_rows.append(summarize_clean(folder.name, clean, heldout_paths, args.crop_size))

        for noisy_path_in in heldout_paths:
            frame_index = natural_file_key(noisy_path_in)[0]
            noisy_crop = center_crop(read_tiff(noisy_path_in), args.crop_size)
            noisy_path = noisy_dir / f"folder_{folder.name}_frame_{frame_index}_noisy.tif"
            write_tiff(noisy_path, noisy_crop.astype(np.uint16))
            pair_rows.append(
                {
                    "pair_key": f"folder_{folder.name}_frame_{frame_index}",
                    "clean_path": str(clean_path),
                    "noisy_path": str(noisy_path),
                    "folder": folder.name,
                    "heldout_frame_index": frame_index,
                    "surrogate_type": "train_mean_vs_heldout_frame",
                    "train_frame_count": args.train_frames,
                    "crop_size": args.crop_size,
                }
            )

    if not pair_rows:
        raise ValueError("No surrogate pairs were produced.")

    pairs_csv = output_dir / "pairs.csv"
    summary_csv = output_dir / "surrogate_summary.csv"
    report_path = output_dir / "surrogate_pair_report.md"
    write_csv(pair_rows, pairs_csv)
    write_csv(summary_rows, summary_csv)
    write_report(root, pair_rows, summary_rows, pairs_csv, summary_csv, report_path, args)
    print(f"Wrote pairs CSV: {pairs_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/gated_iccd_surrogate_pairs")
    parser.add_argument("--folders", nargs="*", default=[])
    parser.add_argument("--train-frames", type=int, default=100)
    parser.add_argument("--max-heldout-frames", type=int, default=8)
    parser.add_argument("--heldout-stride", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--range-max", type=int, default=65535)
    return parser.parse_args()


def select_folders(root: Path, names: list[str]) -> list[Path]:
    if names:
        return [root / name for name in names]
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: natural_folder_key(path.name))


def natural_folder_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_", 1)[0]), name
    except ValueError:
        return 10**12, name


def list_tiffs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES], key=natural_file_key)


def natural_file_key(path: Path) -> tuple[int, str]:
    match = LEADING_NUMBER.match(path.name)
    if match:
        return int(match.group("number")), path.name
    return 10**12, path.name


def mean_crop(paths: list[Path], crop_size: int) -> np.ndarray:
    acc = None
    for path in paths:
        crop = center_crop(read_tiff(path), crop_size).astype(np.float64)
        if acc is None:
            acc = np.zeros_like(crop, dtype=np.float64)
        acc += crop
    if acc is None:
        raise ValueError("No paths provided")
    return (acc / len(paths)).astype(np.float32)


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF {path}: {exc}") from exc


def write_tiff(path: Path, arr: np.ndarray) -> None:
    try:
        import tifffile

        tifffile.imwrite(path, arr)
    except Exception as exc:
        raise RuntimeError(f"Failed to write TIFF {path}: {exc}") from exc


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale TIFF, got shape {arr.shape}")
    h, w = arr.shape
    size = min(crop_size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top : top + size, left : left + size]


def summarize_clean(folder: str, clean: np.ndarray, heldout_paths: list[Path], crop_size: int) -> dict[str, Any]:
    return {
        "folder": folder,
        "heldout_pair_count": len(heldout_paths),
        "crop_size": crop_size,
        "clean_mean": float(np.mean(clean)),
        "clean_std": float(np.std(clean, ddof=1)),
        "clean_p01": float(np.percentile(clean, 1)),
        "clean_p50": float(np.percentile(clean, 50)),
        "clean_p99": float(np.percentile(clean, 99)),
        "heldout_indices": " ".join(str(natural_file_key(path)[0]) for path in heldout_paths),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    root: Path,
    pair_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    pairs_csv: Path,
    summary_csv: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Repeated-Frame Surrogate Pair Report",
        "",
        f"- Root: `{root}`",
        f"- Folders: {len(summary_rows)}",
        f"- Pairs: {len(pair_rows)}",
        f"- Train frames for mean surrogate: {args.train_frames}",
        f"- Held-out frames per folder: up to {args.max_heldout_frames}",
        f"- Held-out stride: {args.heldout_stride}",
        f"- Crop size: {args.crop_size}",
        f"- Pairs CSV: `{pairs_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## Folder Summary",
        "",
        "| folder | pairs | clean mean | clean std | clean p50 | held-out indices |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['folder']} | {row['heldout_pair_count']} | {row['clean_mean']:.6g} | "
            f"{row['clean_std']:.6g} | {row['clean_p50']:.6g} | {row['heldout_indices']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- The clean image is a temporal mean surrogate from repeated frames, not a true clean ground truth.",
            "- The pairs are suitable for E2.2 synthetic-noise fidelity checks against held-out temporal residuals.",
            "- Do not use these pairs as supervised denoising ground truth without explicitly labeling the surrogate assumption.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
