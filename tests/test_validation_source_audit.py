from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_content_source_independence import compare, perceptual_hash, thumbnail


def test_float_thumbnail_is_finite_and_bounded() -> None:
    image = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    result = thumbnail(image, 128)
    assert result.shape == (128, 128)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_independence_metrics_are_standard_finite_values() -> None:
    first = thumbnail(np.tile(np.arange(512, dtype=np.uint16), (512, 1)), 128)
    second = thumbnail(np.tile(np.arange(512, dtype=np.uint16)[:, None], (1, 512)), 128)
    candidate = [{"path": "candidate", "sha256": "a", "phash": perceptual_hash(first), "thumbnail": first}]
    reference = [{"path": "reference", "content_id": "reference", "sha256": "b", "phash": perceptual_hash(second), "thumbnail": second}]
    row = compare("candidate", candidate, reference)[0]
    for key in ("correlation", "ssim", "low_frequency_correlation", "high_frequency_correlation", "gradient_similarity"):
        assert isinstance(row[key], float)
        assert np.isfinite(row[key])
