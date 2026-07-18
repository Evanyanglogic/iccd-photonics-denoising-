"""Write and reload strict JSON smoke artifacts without running image pairs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from json_serialization import dump_json


class SmokeState(Enum):
    PASS = "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    payload = {
        "gate": np.bool_(True), "count": np.int64(3), "ratio": np.float32(0.5),
        "values": np.array([1, 2, 3], dtype=np.uint16), "nested": [np.bool_(False), {"nan": np.nan, "inf": np.inf}],
        "path": output, "started_at": datetime.now(timezone.utc), "state": SmokeState.PASS,
    }
    verification = output / "verification_status.json"
    manifest = output / "run_manifest.json"
    dump_json(verification, payload)
    dump_json(manifest, {"status": SmokeState.PASS, "verification": payload})
    loaded_verification = json.loads(verification.read_text(encoding="utf-8"))
    loaded_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    checks = {
        "verification_parse": isinstance(loaded_verification["gate"], bool),
        "manifest_parse": loaded_manifest["status"] == "PASS",
        "number_types": isinstance(loaded_verification["count"], int) and isinstance(loaded_verification["ratio"], float),
        "nonfinite_to_null": loaded_verification["nested"][1] == {"inf": None, "nan": None},
    }
    dump_json(output / "smoke_log.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
