"""First-pass ICCD physical-prior noise model.

The model is intentionally small and explicit. It is meant to become the
front-end prior for ICCD-aware PNGAN training, not the final claim by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICCDNoiseConfig:
    """Parameters for a simplified ICCD imaging chain.

    All values operate on normalized images in [0, 1]. They should later be
    calibrated from dark/flat sequences instead of kept as defaults.
    """

    photon_scale: float = 120.0
    photocathode_qe: float = 0.18
    mcp_gain_mean: float = 1.0
    mcp_gain_var: float = 0.18
    dark_count_rate: float = 0.002
    phosphor_sigma: float = 0.8
    read_noise_sigma: float = 0.01
    offset: float = 0.0
    clip: bool = True
    seed: int | None = None


class ICCDNoiseModel:
    """Simulate an ICCD-like noisy observation from a clean image."""

    def __init__(self, config: ICCDNoiseConfig | None = None) -> None:
        self.config = config or ICCDNoiseConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def add_noise(self, clean_image: np.ndarray) -> np.ndarray:
        """Return an ICCD-like noisy image with the same shape as input."""

        clean = np.asarray(clean_image, dtype=np.float32)
        clean = np.clip(clean, 0.0, 1.0)

        photons = self._sample_photons(clean)
        photoelectrons = self._apply_photocathode(photons)
        intensified = self._apply_mcp_gain(photoelectrons)
        phosphor = self._apply_phosphor_diffusion(intensified)
        readout = self._apply_readout(phosphor)

        noisy = readout + self.config.offset
        if self.config.clip:
            noisy = np.clip(noisy, 0.0, 1.0)
        return noisy.astype(np.float32)

    def _sample_photons(self, clean: np.ndarray) -> np.ndarray:
        lam = clean * self.config.photon_scale
        signal = self.rng.poisson(lam).astype(np.float32)
        dark = self.rng.poisson(
            self.config.dark_count_rate * self.config.photon_scale,
            size=clean.shape,
        ).astype(np.float32)
        return signal + dark

    def _apply_photocathode(self, photons: np.ndarray) -> np.ndarray:
        qe = np.clip(self.config.photocathode_qe, 0.0, 1.0)
        return self.rng.binomial(photons.astype(np.int64), qe).astype(np.float32)

    def _apply_mcp_gain(self, electrons: np.ndarray) -> np.ndarray:
        mean = max(self.config.mcp_gain_mean, 1e-6)
        var = max(self.config.mcp_gain_var, 1e-6)
        shape = mean * mean / var
        scale = var / mean
        gain = self.rng.gamma(shape=shape, scale=scale, size=electrons.shape)
        return electrons * gain.astype(np.float32)

    def _apply_phosphor_diffusion(self, image: np.ndarray) -> np.ndarray:
        sigma = self.config.phosphor_sigma
        if sigma <= 0:
            return image
        return _gaussian_blur_numpy(image, sigma)

    def _apply_readout(self, image: np.ndarray) -> np.ndarray:
        normalized = image / max(self.config.photon_scale * self.config.photocathode_qe, 1e-6)
        read_noise = self.rng.normal(
            loc=0.0,
            scale=self.config.read_noise_sigma,
            size=image.shape,
        ).astype(np.float32)
        return normalized.astype(np.float32) + read_noise


def _gaussian_blur_numpy(image: np.ndarray, sigma: float) -> np.ndarray:
    """Small separable Gaussian blur without scipy dependency."""

    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)

    if image.ndim == 2:
        return _blur_2d(image, kernel)
    if image.ndim == 3:
        channels = [_blur_2d(channel, kernel) for channel in image]
        return np.stack(channels, axis=0).astype(np.float32)
    raise ValueError(f"Expected 2D or CHW image, got shape {image.shape}")


def _blur_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    padded_h = np.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_h)
    padded_v = np.pad(horizontal, ((pad, pad), (0, 0)), mode="reflect")
    vertical = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_v)
    return vertical.astype(np.float32)

