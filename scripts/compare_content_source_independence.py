"""Small, bounded image-source independence diagnostics."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thumbnail(array: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    lo, hi = np.percentile(image, [1, 99])
    image = np.clip((image - lo) / max(float(hi - lo), 1e-12), 0, 1)
    source = Image.fromarray(np.ascontiguousarray(image, dtype=np.float32))
    result = np.asarray(source.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Thumbnail conversion produced non-finite values")
    return result


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    x -= x.mean(); y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    ux, uy = x.mean(), y.mean(); vx, vy = x.var(), y.var(); covariance = ((x - ux) * (y - uy)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * ux * uy + c1) * (2 * covariance + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))


def perceptual_hash(image: np.ndarray) -> str:
    if not np.isfinite(image).all():
        raise ValueError("Perceptual hash input contains non-finite values")
    small = np.asarray(Image.fromarray(np.ascontiguousarray(image, dtype=np.float32)).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    coefficients = dctn(small, type=2, norm="ortho")[:8, :8]
    bits = coefficients > np.median(coefficients[1:])
    return f"{int(''.join('1' if x else '0' for x in bits.ravel()), 2):016x}"


def compare(candidate_id: str, candidate_images: list[dict], reference_images: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidate_images:
        a = candidate["thumbnail"]
        low_a = gaussian_filter(a, 3.0)
        high_a = a - low_a
        grad_a = np.hypot(*np.gradient(a))
        for reference in reference_images:
            b = reference["thumbnail"]
            low_b = gaussian_filter(b, 3.0)
            high_b = b - low_b
            grad_b = np.hypot(*np.gradient(b))
            metrics = {
                "candidate_id": candidate_id,
                "candidate_file": candidate["path"],
                "reference_content_id": reference["content_id"],
                "reference_file": reference["path"],
                "exact_sha256_match": candidate["sha256"] == reference["sha256"],
                "perceptual_hash_match": candidate["phash"] == reference["phash"],
                "correlation": corr(a, b),
                "ssim": global_ssim(a, b),
                "low_frequency_correlation": corr(low_a, low_b),
                "high_frequency_correlation": corr(high_a, high_b),
                "gradient_similarity": corr(grad_a, grad_b),
            }
            numeric_values = [value for key, value in metrics.items() if key in {"correlation", "ssim", "low_frequency_correlation", "high_frequency_correlation", "gradient_similarity"}]
            if not np.isfinite(numeric_values).all():
                raise ValueError("Independence comparison produced non-finite metrics")
            rows.append(metrics)
    return rows
