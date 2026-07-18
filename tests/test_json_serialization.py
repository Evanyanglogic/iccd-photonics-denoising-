from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from json_serialization import dump_json, to_json_safe


class Example(Enum):
    VALUE = np.int64(7)


def test_scientific_payload_round_trip(tmp_path: Path) -> None:
    payload = {
        "bool": np.bool_(True),
        "integer": np.int64(3),
        "unsigned": np.uint16(4),
        "floating": np.float32(0.5),
        "array": np.array([1, 2, 3], dtype=np.int32),
        "nested": {"items": [np.bool_(False), (np.float64(1.25),)]},
        "path": Path("reports/example"),
        "datetime": datetime(2026, 7, 18, tzinfo=timezone.utc),
        "enum": Example.VALUE,
        "nan": np.float64(np.nan),
        "positive_inf": np.float32(np.inf),
        "negative_inf": float("-inf"),
    }
    destination = tmp_path / "payload.json"
    dump_json(destination, payload)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["bool"] is True and isinstance(loaded["bool"], bool)
    assert loaded["integer"] == 3 and isinstance(loaded["integer"], int)
    assert loaded["unsigned"] == 4
    assert loaded["floating"] == 0.5 and isinstance(loaded["floating"], float)
    assert loaded["array"] == [1, 2, 3]
    assert loaded["nested"] == {"items": [False, [1.25]]}
    assert loaded["path"] == "reports\\example" or loaded["path"] == "reports/example"
    assert loaded["datetime"] == "2026-07-18T00:00:00+00:00"
    assert loaded["enum"] == 7
    assert loaded["nan"] is None and loaded["positive_inf"] is None and loaded["negative_inf"] is None


def test_standard_json_rejects_no_fields_and_preserves_gate_bool() -> None:
    safe = to_json_safe({"gate": np.bool_(True), "value": np.float64(np.nan)})
    encoded = json.dumps(safe, allow_nan=False)
    loaded = json.loads(encoded)
    assert set(loaded) == {"gate", "value"}
    assert loaded["gate"] is True and isinstance(loaded["gate"], bool)
    assert loaded["value"] is None

