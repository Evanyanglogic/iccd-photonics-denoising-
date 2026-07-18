"""Strict conversion of scientific Python values to standard JSON values."""
from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def dump_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    safe_payload = to_json_safe(payload)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")

