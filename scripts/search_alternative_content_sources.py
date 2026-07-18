"""Inventory local alternative content sources without declaring them validation-ready."""
from __future__ import annotations

import argparse
from pathlib import Path

from e2_content_manifest_lib import image_inventory, read_config, resolve, write_csv_rows


def run(cfg: dict, output: Path) -> list[dict]:
    rows=[]
    for candidate in cfg["alternative_sources"]:
        path=Path(candidate["path"]); count,samples=image_inventory(path)
        relation=candidate["relation"]
        if relation in {"same_acquisition_family", "historical_copy_or_same_acquisition_family"}:
            reason="Not demonstrably independent from the audited 500 ms source"
        elif relation in {"real_iccd_evaluation_data", "iccd_calibration_or_background"}:
            reason="Reserved real ICCD evaluation/calibration data; invalid for synthetic checkpoint validation"
        else:
            reason="Potential independent source, but traceability, preprocessing, grouping, and scale are not audited"
        rows.append({"path":str(path),"exists":path.is_dir(),"image_like_file_count":count,"relation":relation,"sample_paths":" | ".join(samples),"independent_scene_established":False,"validation_ready":False,"decision":"candidate_requires_separate_audit" if count else "unavailable","reason":reason})
    write_csv_rows(output/"alternative_content_sources.csv",rows)
    return rows


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-root",required=True); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]; cfg,_=read_config(repo,args.config); output=resolve(repo,args.output_root)
    print(f"candidate_sources={len(run(cfg,output))}"); return 0


if __name__ == "__main__": raise SystemExit(main())

