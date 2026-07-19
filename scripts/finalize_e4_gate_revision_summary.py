"""Refresh derived E4 multiseed decisions after a verified summary-code correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from json_serialization import dump_json
from run_e4_g_cg_multiseed_stability import decide


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = (repo / args.output_root).resolve()
    cfg = yaml.safe_load((repo / args.config).read_text(encoding="utf-8"))
    verification_path = root / "verification_status.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("final_status") != "PAIR-GATE-REVISION-VERIFIED":
        raise RuntimeError("Cannot finalize an incomplete gate-revision run")
    synthetic = pd.read_csv(root / "multiseed_summary/synthetic_seed_summary.csv")
    real = pd.read_csv(root / "multiseed_summary/folder_seed_summary.csv")
    if len(synthetic) != 6 or len(real) != 24:
        raise RuntimeError("Multiseed metric count drift")
    decision, cgs = decide(synthetic.rename(columns={"experiment": "model"}), real, cfg)
    dump_json(root / "multiseed_summary/conditional_benefit_decision.json", decision)
    dump_json(root / "multiseed_summary/cgs_entry_decision.json", cgs)
    verification["conditional_benefit"] = decision["status"]
    verification["CGS_ENTRY_ALLOWED"] = cgs["CGS_ENTRY_ALLOWED"]
    dump_json(verification_path, verification)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    dump_json(root / "provenance/summary_correction.json", {
        "summary_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc),
        "reason": "Compare abs(CG structure correlation) minus abs(G structure correlation), not abs(CG-G)",
        "training_rerun": False,
        "checkpoint_changed": False,
        "metric_values_changed": False,
    })
    scripts = [repo / "scripts/run_e4_g_cg_gate_revision.py", repo / "scripts/run_e4_g_cg_multiseed_stability.py", Path(__file__), repo / "scripts/run_e2_g_cg_scaled_training.py", repo / "scripts/run_e3_real_iccd_holdout_validation.py", repo / "scripts/json_serialization.py"]
    pd.DataFrame([{"path": str(path.relative_to(repo)), "sha256": sha256_file(path)} for path in scripts]).to_csv(root / "provenance/script_hashes.csv", index=False, encoding="utf-8-sig")
    hashes = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            hashes.append({"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(root / "output_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"conditional_benefit": decision["status"], "CGS_ENTRY_ALLOWED": cgs["CGS_ENTRY_ALLOWED"], "summary_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
