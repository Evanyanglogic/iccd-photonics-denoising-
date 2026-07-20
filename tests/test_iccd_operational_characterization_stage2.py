from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_iccd_operational_characterization_stage2 import pearson  # noqa: E402


def test_pearson_does_not_mutate_input_views() -> None:
    first = np.arange(12, dtype=np.float64).reshape(3, 4)
    second = first * 2.0 + 3.0
    first_before = first.copy()
    second_before = second.copy()

    assert pearson(first.ravel(), second.ravel()) == 1.0
    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)
