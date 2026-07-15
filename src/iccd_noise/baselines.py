"""Baseline synthetic noise priors for ICCD comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class NoisePrior(Protocol):
    """Common interface for synthetic noise priors."""

    def add_noise(self, clean_image: np.ndarray) -> np.ndarray:
        """Return a noisy image with the same shape as the clean input."""


@dataclass(frozen=True)
class PoissonGaussianConfig:
    """Generic signal-dependent Poisson-Gaussian baseline."""

    peak_photons: float = 120.0
    read_noise_sigma: float = 0.01
    clip: bool = True
    seed: int | None = None


class PoissonGaussianNoiseModel:
    """Generate noisy images with a generic Poisson-Gaussian model."""

    def __init__(self, config: PoissonGaussianConfig | None = None) -> None:
        self.config = config or PoissonGaussianConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def add_noise(self, clean_image: np.ndarray) -> np.ndarray:
        clean = np.asarray(clean_image, dtype=np.float32)
        clean = np.clip(clean, 0.0, 1.0)
        peak = max(float(self.config.peak_photons), 1e-6)
        shot = self.rng.poisson(clean * peak).astype(np.float32) / peak
        read = self.rng.normal(0.0, self.config.read_noise_sigma, size=clean.shape).astype(np.float32)
        noisy = shot + read
        if self.config.clip:
            noisy = np.clip(noisy, 0.0, 1.0)
        return noisy.astype(np.float32)


@dataclass(frozen=True)
class SCMOSLikeConfig:
    """Simplified sCMOS-like prior used as a device-family baseline."""

    signal_gain: float = 0.20
    read_noise_sigma: float = 0.015
    row_noise_sigma: float = 0.004
    column_noise_sigma: float = 0.0
    offset: float = 0.0
    clip: bool = True
    seed: int | None = None


class SCMOSLikeNoiseModel:
    """Approximate signal-dependent readout noise with optional row/column terms."""

    def __init__(self, config: SCMOSLikeConfig | None = None) -> None:
        self.config = config or SCMOSLikeConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def add_noise(self, clean_image: np.ndarray) -> np.ndarray:
        clean = np.asarray(clean_image, dtype=np.float32)
        clean = np.clip(clean, 0.0, 1.0)
        signal_var = np.maximum(clean, 0.0) * max(self.config.signal_gain, 0.0)
        read_var = max(self.config.read_noise_sigma, 0.0) ** 2
        pixel_sigma = np.sqrt(signal_var + read_var)
        pixel_noise = self.rng.normal(0.0, pixel_sigma).astype(np.float32)
        structured = self._structured_noise(clean.shape)
        noisy = clean + pixel_noise + structured + self.config.offset
        if self.config.clip:
            noisy = np.clip(noisy, 0.0, 1.0)
        return noisy.astype(np.float32)

    def _structured_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) == 2:
            h, w = shape
            row_shape = (h, 1)
            col_shape = (1, w)
        elif len(shape) == 3:
            c, h, w = shape if shape[0] <= 4 else (shape[2], shape[0], shape[1])
            if shape[0] <= 4:
                row_shape = (c, h, 1)
                col_shape = (c, 1, w)
            else:
                row_shape = (h, 1, c)
                col_shape = (1, w, c)
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {shape}")

        noise = np.zeros(shape, dtype=np.float32)
        if self.config.row_noise_sigma > 0:
            noise = noise + self.rng.normal(0.0, self.config.row_noise_sigma, size=row_shape).astype(np.float32)
        if self.config.column_noise_sigma > 0:
            noise = noise + self.rng.normal(0.0, self.config.column_noise_sigma, size=col_shape).astype(np.float32)
        return noise
