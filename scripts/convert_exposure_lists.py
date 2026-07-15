"""Convert legacy exposure pair lists into pairs.csv and splits.yaml.

Legacy PMRID lists contain two relative TIFF paths per row. In `exposure` mode,
the longer exposure is treated as the clean/reference image and the shorter
exposure is treated as the noisy/input image.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


EXPOSURE_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)$", re.IGNORECASE)
FRAME_PATTERN = re.compile(r"^(?P<scene>.+)_(?P<frame>\d+)$")


def main() -> int:
    args = parse_args()
    path_root = Path(args.path_root)
    train_rows = parse_list(
        Path(args.train_list),
        path_root=path_root,
        split="train",
        mode=args.mode,
        allow_missing=args.allow_missing,
    )
    val_rows = (
        parse_list(
            Path(args.val_list),
            path_root=path_root,
            split="val",
            mode=args.mode,
            allow_missing=args.allow_missing,
        )
        if args.val_list
        else []
    )
    all_rows = train_rows + val_rows
    if not all_rows:
        raise ValueError("No rows were parsed from the supplied list files.")

    pairs_out = Path(args.pairs_out)
    splits_out = Path(args.splits_out)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    write_pairs(all_rows, pairs_out)
    write_splits(train_rows, val_rows, splits_out)
    print(f"Wrote pair manifest: {pairs_out}")
    print(f"Wrote split manifest: {splits_out}")
    print(f"Rows: train={len(train_rows)}, val={len(val_rows)}, total={len(all_rows)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", default="")
    parser.add_argument("--path-root", required=True, help="Root used to resolve relative paths such as data/15ms/a.tif.")
    parser.add_argument("--pairs-out", default="data_manifest/pairs.csv")
    parser.add_argument("--splits-out", default="data_manifest/splits.yaml")
    parser.add_argument(
        "--mode",
        choices=["exposure", "first-clean", "second-clean"],
        default="exposure",
        help="How to decide clean/noisy columns.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if referenced TIFF files are absent.")
    return parser.parse_args()


def parse_list(path: Path, path_root: Path, split: str, mode: str, allow_missing: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number} expected 2 paths, got {len(parts)}")
            left = PairPath.from_value(parts[0], path_root)
            right = PairPath.from_value(parts[1], path_root)
            if not allow_missing:
                check_exists(left, path, line_number)
                check_exists(right, path, line_number)
            clean, noisy = orient_pair(left, right, mode)
            rows.append(build_row(clean=clean, noisy=noisy, split=split, source_list=path))
    return rows


def check_exists(pair_path: "PairPath", list_path: Path, line_number: int) -> None:
    if not pair_path.absolute.exists():
        raise FileNotFoundError(f"{list_path}:{line_number} referenced file does not exist: {pair_path.absolute}")


def orient_pair(left: "PairPath", right: "PairPath", mode: str) -> tuple["PairPath", "PairPath"]:
    if mode == "first-clean":
        return left, right
    if mode == "second-clean":
        return right, left

    left_ms = left.exposure_ms
    right_ms = right.exposure_ms
    if left_ms is None or right_ms is None:
        raise ValueError(f"Cannot infer exposure order from {left.original!r} and {right.original!r}")
    if left_ms >= right_ms:
        return left, right
    return right, left


def build_row(clean: "PairPath", noisy: "PairPath", split: str, source_list: Path) -> dict[str, Any]:
    if clean.stem != noisy.stem:
        raise ValueError(f"Stem mismatch: clean {clean.original}, noisy {noisy.original}")
    scene_id, frame_id = split_scene_frame(clean.stem)
    pair_key = f"{clean.stem}__noisy_{noisy.exposure_label}__clean_{clean.exposure_label}"
    return {
        "pair_key": pair_key,
        "clean_path": str(clean.absolute),
        "noisy_path": str(noisy.absolute),
        "scene_id": scene_id,
        "frame_id": frame_id,
        "clean_exposure": clean.exposure_label,
        "noisy_exposure": noisy.exposure_label,
        "clean_exposure_ms": clean.exposure_ms,
        "noisy_exposure_ms": noisy.exposure_ms,
        "exposure_ratio": clean.exposure_ms / noisy.exposure_ms if clean.exposure_ms and noisy.exposure_ms else "",
        "split_source": split,
        "source_list": str(source_list),
    }


def split_scene_frame(stem: str) -> tuple[str, str]:
    match = FRAME_PATTERN.match(stem)
    if not match:
        return stem, ""
    return match.group("scene"), match.group("frame")


def write_pairs(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "pair_key",
        "clean_path",
        "noisy_path",
        "scene_id",
        "frame_id",
        "clean_exposure",
        "noisy_exposure",
        "clean_exposure_ms",
        "noisy_exposure_ms",
        "exposure_ratio",
        "split_source",
        "source_list",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_splits(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Auto-generated by scripts/convert_exposure_lists.py\n")
        handle.write("train:\n")
        for row in train_rows:
            handle.write(f"  - {row['pair_key']}\n")
        handle.write("val:\n")
        for row in val_rows:
            handle.write(f"  - {row['pair_key']}\n")
        handle.write("test:\n")


class PairPath:
    """Parsed list-entry path with exposure metadata."""

    def __init__(
        self,
        original: str,
        absolute: Path,
        exposure_label: str,
        exposure_ms: float,
        stem: str,
    ) -> None:
        self.original = original
        self.absolute = absolute
        self.exposure_label = exposure_label
        self.exposure_ms = exposure_ms
        self.stem = stem

    @classmethod
    def from_value(cls, value: str, path_root: Path) -> "PairPath":
        relative = Path(value)
        absolute = relative if relative.is_absolute() else (path_root / relative)
        exposure_label = infer_exposure_label(relative)
        exposure_ms = parse_exposure_ms(exposure_label)
        return cls(
            original=value,
            absolute=absolute.resolve(),
            exposure_label=exposure_label,
            exposure_ms=exposure_ms,
            stem=relative.stem,
        )


def infer_exposure_label(path: Path) -> str:
    for part in path.parts:
        if EXPOSURE_PATTERN.match(part):
            return part
    raise ValueError(f"Could not infer exposure directory from path: {path}")


def parse_exposure_ms(label: str) -> float:
    match = EXPOSURE_PATTERN.match(label)
    if not match:
        raise ValueError(f"Invalid exposure label: {label}")
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    return value * 1000.0 if unit == "s" else value


if __name__ == "__main__":
    raise SystemExit(main())
