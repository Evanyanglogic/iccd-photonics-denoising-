"""Noise-statistical comparison helpers."""

from __future__ import annotations

import numpy as np


def mean_variance_by_intensity(
    clean: np.ndarray,
    noisy: np.ndarray,
    bins: int = 16,
) -> list[dict[str, float]]:
    """Compute residual mean/variance grouped by clean-image intensity."""

    clean_arr = np.asarray(clean, dtype=np.float32)
    residual = np.asarray(noisy, dtype=np.float32) - clean_arr
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []

    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (clean_arr >= lo) & (clean_arr <= hi)
        else:
            mask = (clean_arr >= lo) & (clean_arr < hi)
        values = residual[mask]
        if values.size == 0:
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0.0, "mean": float("nan"), "var": float("nan")})
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": float(values.size),
                "mean": float(np.mean(values)),
                "var": float(np.var(values)),
            }
        )
    return rows


def radial_power_spectrum(image: np.ndarray) -> np.ndarray:
    """Return a compact radial average of the 2D power spectrum."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    h, w = spectrum.shape
    y, x = np.indices((h, w))
    radius = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2).astype(np.int32)
    radial_sum = np.bincount(radius.ravel(), weights=spectrum.ravel())
    radial_count = np.bincount(radius.ravel())
    return (radial_sum / np.maximum(radial_count, 1)).astype(np.float32)

