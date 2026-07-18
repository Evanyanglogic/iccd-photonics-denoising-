"""Build exactly one debug-only E1-strength Gaussian residual smoke pair."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(repo: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest = pd.read_csv(repo / cfg["content_manifest"])
    selected = manifest.sort_values("sha256", kind="stable").iloc[0]
    if selected.allowed_role != cfg["required_content_role"]:
        raise RuntimeError("Selected content is not debug_only")
    source = Path(selected.absolute_path)
    if sha256_file(source) != selected.sha256:
        raise RuntimeError("INPUT-DRIFT in selected content")
    e1 = pd.read_csv(repo / cfg["e1_strength_csv"])
    field = cfg["residual_baseline"]["e1_field"]
    values = e1[field].astype(float).to_numpy()
    sigma_dn = float(np.median(values))
    divisor = float(cfg["content_preprocessing"]["normalization_divisor"])
    sigma_norm = sigma_dn / divisor
    raw = tifffile.imread(source)
    roi = cfg["roi"]; top,left,height,width = (int(roi[k]) for k in ("top","left","height","width"))
    raw_roi = raw[top:top+height, left:left+width]
    if raw_roi.shape != (height, width) or raw_roi.dtype != np.uint16:
        raise RuntimeError(f"Unexpected selected ROI: {raw_roi.shape} {raw_roi.dtype}")
    content = (raw_roi.astype(np.float32) / np.float32(divisor)).astype(np.float32)
    rng = np.random.default_rng(int(cfg["residual_baseline"]["seed"]))
    residual = rng.normal(0.0, sigma_norm, size=content.shape).astype(np.float32)
    noisy_unclipped = (content + residual).astype(np.float32)
    noisy = np.clip(noisy_unclipped, 0.0, 1.0).astype(np.float32)
    arrays = output / "float_arrays"; tiffs = output / "tiff"; previews = output / "previews"
    arrays.mkdir(parents=True, exist_ok=True); tiffs.mkdir(exist_ok=True); previews.mkdir(exist_ok=True)
    np.save(arrays / "content_float.npy", content)
    np.save(arrays / "residual_float.npy", residual)
    np.save(arrays / "noisy_unclipped_float.npy", noisy_unclipped)
    content_u16 = np.rint(content * divisor).astype(np.uint16)
    noisy_u16 = np.rint(noisy * divisor).astype(np.uint16)
    tifffile.imwrite(tiffs / "content_uint16.tiff", content_u16, compression=None)
    tifffile.imwrite(tiffs / "noisy_uint16.tiff", noisy_u16, compression=None)
    Image.fromarray((content_u16 / 257).astype(np.uint8)).save(previews / "content_preview.png")
    Image.fromarray((noisy_u16 / 257).astype(np.uint8)).save(previews / "noisy_preview.png")
    residual_preview = np.clip(127.5 + residual / max(sigma_norm, 1e-12) * 24.0, 0, 255).astype(np.uint8)
    Image.fromarray(residual_preview).save(previews / "residual_preview.png")
    selection = {
        "content_id": selected.content_id, "source_pair_key": selected.source_pair_key,
        "source_path": str(source), "source_sha256": selected.sha256,
        "selection_rule": cfg["content_selection"], "allowed_role": selected.allowed_role,
        "roi_top": top, "roi_left": left, "roi_height": height, "roi_width": width,
        "raw_roi_min_dn": int(raw_roi.min()), "raw_roi_max_dn": int(raw_roi.max()),
        "raw_roi_mean_dn": float(raw_roi.mean()), "raw_roi_std_dn": float(raw_roi.std()),
    }
    pd.DataFrame([selection]).to_csv(output / "input" / "selected_content.csv", index=False, encoding="utf-8-sig")
    strength = pd.DataFrame({"folder": e1["folder"], "temporal_std_dn": values})
    strength["selected_median_sigma_dn"] = sigma_dn
    strength["selected_sigma_norm"] = sigma_norm
    strength["metric_definition"] = "Median across folder-level mean per-pixel temporal std in raw DN at fixed E1 ROI"
    strength.to_csv(output / "input" / "e1_strength_source.csv", index=False, encoding="utf-8-sig")
    return {"selection": selection, "sigma_dn": sigma_dn, "sigma_norm": sigma_norm, "folder_count": len(values), "content": content, "residual": residual, "noisy_unclipped": noisy_unclipped, "noisy": noisy}
