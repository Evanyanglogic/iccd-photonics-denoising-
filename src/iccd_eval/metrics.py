"""Float-domain metrics for ICCD low-light denoising.

The functions in this module avoid uint8 conversion. Inputs are expected to be
linear-domain arrays normalized to [0, 1] unless a different data_range is
provided explicitly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def to_numpy_image(image: Any) -> np.ndarray:
    """Convert tensors/arrays to a float32 numpy image without quantization."""

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected one image or call batch code, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected HxW, HxWxC, CxHxW, or 1xCxHxW image, got {arr.shape}")
    return arr


def psnr(pred: Any, target: Any, data_range: float = 1.0, eps: float = 1e-12) -> float:
    """Compute PSNR in float domain."""

    pred_arr = to_numpy_image(pred)
    target_arr = to_numpy_image(target)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch: pred {pred_arr.shape}, target {target_arr.shape}")
    mse = float(np.mean((pred_arr - target_arr) ** 2))
    if mse <= eps:
        return float("inf")
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse)


def ssim(pred: Any, target: Any, data_range: float = 1.0) -> float:
    """Compute SSIM without uint8 conversion.

    Uses scikit-image when available. The fallback is a global SSIM estimate,
    intended for smoke tests rather than final paper numbers.
    """

    pred_arr = to_numpy_image(pred)
    target_arr = to_numpy_image(target)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch: pred {pred_arr.shape}, target {target_arr.shape}")

    try:
        from skimage.metrics import structural_similarity

        kwargs: dict[str, Any] = {"data_range": data_range}
        if pred_arr.ndim == 3:
            kwargs["channel_axis"] = -1
        return float(structural_similarity(target_arr, pred_arr, **kwargs))
    except Exception:
        return _global_ssim(pred_arr, target_arr, data_range=data_range)


def _global_ssim(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    x = target.astype(np.float64)
    y = pred.astype(np.float64)
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
    return float(numerator / denominator)


def residual_statistics(pred: Any, target: Any) -> dict[str, float]:
    """Summarize residual distribution for noise/statistical checks."""

    residual = to_numpy_image(pred) - to_numpy_image(target)
    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "var": float(np.var(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "p01": float(np.percentile(residual, 1)),
        "p50": float(np.percentile(residual, 50)),
        "p99": float(np.percentile(residual, 99)),
    }


def brightness_bin_psnr(
    pred: Any,
    target: Any,
    bins: int = 8,
    data_range: float = 1.0,
) -> list[dict[str, float]]:
    """Compute PSNR grouped by target brightness bins."""

    pred_arr = to_numpy_image(pred)
    target_arr = to_numpy_image(target)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Shape mismatch: pred {pred_arr.shape}, target {target_arr.shape}")

    edges = np.linspace(0.0, float(data_range), bins + 1)
    rows: list[dict[str, float]] = []
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (target_arr >= lo) & (target_arr <= hi)
        else:
            mask = (target_arr >= lo) & (target_arr < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            value = float("nan")
        else:
            mse = float(np.mean((pred_arr[mask] - target_arr[mask]) ** 2))
            value = float("inf") if mse <= 1e-12 else 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": float(count), "psnr": value})
    return rows


def image_quality(pred: Any, target: Any, data_range: float = 1.0) -> dict[str, float]:
    """Return the core paper-facing full-reference metrics."""

    stats = residual_statistics(pred, target)
    return {
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
        "residual_mean": stats["mean"],
        "residual_std": stats["std"],
        "residual_mae": stats["mae"],
    }
