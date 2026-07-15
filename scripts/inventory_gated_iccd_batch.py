"""Inventory a batch of gated ICCD exposure folders.

Scans immediate subdirectories under a root such as `F:/20260319`, parses
PictureInfo.txt when available, counts TIFF frames, and samples basic TIFF range
statistics from the first frame in each folder.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
LEADING_NUMBER = re.compile(r"^(?P<number>\d+)")
PICTURE_INFO_PATTERN = re.compile(
    r"^(?P<filename>.+?\.tiff?)\s+"
    r"Exposure.*?delay:(?P<exposure_delay_ms>[-+0-9.]+)ms,\s*width:(?P<exposure_width_ms>[-+0-9.]+)ms\s+"
    r"Sync\.A.*?delay:(?P<sync_a_delay_ns>[-+0-9.]+)ns,\s*width:(?P<sync_a_width_us>[-+0-9.]+)us\s+"
    r"Sync\.B.*?delay:(?P<sync_b_delay_ns>[-+0-9.]+)ns,\s*width:(?P<sync_b_width_us>[-+0-9.]+)us\s+"
    r"gain.*?(?P<gain>[-+0-9.]+)"
)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [inventory_folder(path, args.picture_info) for path in sorted(root.iterdir()) if path.is_dir()]
    if args.include_root:
        rows.insert(0, inventory_folder(root, args.picture_info))
    rows = [row for row in rows if int(row["tiff_count"]) > 0 or int(row["picture_info_rows"]) > 0]
    if not rows:
        raise ValueError(f"No TIFF/PictureInfo folders found under {root}")

    csv_path = output_dir / "gated_iccd_batch_inventory.csv"
    report_path = output_dir / "gated_iccd_batch_inventory.md"
    write_csv(rows, csv_path)
    write_report(root, rows, csv_path, report_path)
    print(f"Wrote inventory CSV: {csv_path}")
    print(f"Wrote inventory report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="reports/gated_iccd_batch_inventory")
    parser.add_argument("--picture-info", default="PictureInfo.txt")
    parser.add_argument("--include-root", action="store_true")
    return parser.parse_args()


def inventory_folder(folder: Path, picture_info_name: str) -> dict[str, Any]:
    tiffs = list_tiffs(folder)
    metadata = read_picture_info(folder / picture_info_name)
    first_meta = next(iter(metadata.values()), {})
    first_stats = summarize_first_frame(tiffs[0]) if tiffs else {}
    numbers = [leading_number(path) for path in tiffs]
    numbers = [number for number in numbers if number is not None]
    return {
        "folder": str(folder),
        "folder_name": folder.name,
        "tiff_count": len(tiffs),
        "picture_info_rows": len(metadata),
        "frame_min_index": min(numbers) if numbers else "",
        "frame_max_index": max(numbers) if numbers else "",
        "missing_index_count": missing_count(numbers),
        "exposure_width_ms": first_meta.get("exposure_width_ms", ""),
        "exposure_delay_ms": first_meta.get("exposure_delay_ms", ""),
        "sync_a_width_us": first_meta.get("sync_a_width_us", ""),
        "sync_b_width_us": first_meta.get("sync_b_width_us", ""),
        "gain": first_meta.get("gain", ""),
        **first_stats,
    }


def list_tiffs(root: Path) -> list[Path]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_key)


def natural_key(path: Path) -> tuple[int, str]:
    number = leading_number(path)
    return (number if number is not None else 10**12, path.name)


def leading_number(path: Path) -> int | None:
    match = LEADING_NUMBER.match(path.name)
    if not match:
        return None
    return int(match.group("number"))


def missing_count(numbers: list[int]) -> int:
    if not numbers:
        return 0
    values = set(numbers)
    return sum(1 for item in range(min(values), max(values) + 1) if item not in values)


def read_picture_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = PICTURE_INFO_PATTERN.match(line)
            if not match:
                continue
            data = match.groupdict()
            filename = data.pop("filename")
            rows[filename] = data
    return rows


def summarize_first_frame(path: Path) -> dict[str, Any]:
    try:
        import tifffile

        arr = np.asarray(tifffile.imread(path))
    except Exception as exc:
        return {"sample_error": str(exc)}
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return {
        "sample_shape": "x".join(str(item) for item in arr.shape),
        "sample_dtype": str(arr.dtype),
        "sample_min": float(np.min(flat)),
        "sample_max": float(np.max(flat)),
        "sample_p50": float(np.percentile(flat, 50)),
        "sample_p99": float(np.percentile(flat, 99)),
        "sample_saturated_fraction": float(np.mean(flat >= 65535)),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, rows: list[dict[str, Any]], csv_path: Path, report_path: Path) -> None:
    lines = [
        "# Gated ICCD Batch Inventory",
        "",
        f"- Root: `{root}`",
        f"- Folders with TIFF/PictureInfo: {len(rows)}",
        f"- CSV: `{csv_path}`",
        "",
        "## Folders",
        "",
        "| folder | tiffs | metadata rows | exposure width ms | sync A width us | sync B width us | gain | shape | dtype | p50 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['folder_name']} | {row['tiff_count']} | {row['picture_info_rows']} | "
            f"{row['exposure_width_ms']} | {row['sync_a_width_us']} | {row['sync_b_width_us']} | {row['gain']} | "
            f"{row.get('sample_shape', '')} | {row.get('sample_dtype', '')} | "
            f"{format_value(row.get('sample_p50', ''))} | {format_value(row.get('sample_p99', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Download at least one additional exposure/gate folder with matching frame indices, then rerun this script to identify candidate clean/noisy pairs.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
