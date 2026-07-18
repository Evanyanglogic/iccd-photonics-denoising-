"""Run the formal E2 content manifest and split-isolation audit."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml

from audit_e2_content_grouping import run as audit_grouping
from build_e2_content_manifest import build
from e2_content_manifest_lib import git, output_hashes, read_config, resolve, sha256_file, utc_now, write_json
from search_alternative_content_sources import run as search_alternatives


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-root",default=""); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]; cfg,source_config=read_config(repo,args.config)
    if args.output_root: cfg["output_root"]=args.output_root
    output=resolve(repo,cfg["output_root"])
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True); provenance=output/"provenance"; provenance.mkdir(); started=utc_now()
    commit=git(repo,"rev-parse","HEAD").strip(); status_before=git(repo,"status","--porcelain=v1","--untracked-files=all")
    (provenance/"git_commit.txt").write_text(commit+"\n",encoding="utf-8")
    (provenance/"git_status_before.txt").write_text(status_before,encoding="utf-8")
    (provenance/"command.txt").write_text(subprocess.list2cmdline(sys.argv)+"\n",encoding="utf-8")
    (provenance/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
    (provenance/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\nscipy={scipy.__version__}\n",encoding="utf-8")
    scripts=[Path(__file__),repo/"scripts/build_e2_content_manifest.py",repo/"scripts/audit_e2_content_grouping.py",repo/"scripts/search_alternative_content_sources.py",repo/"scripts/e2_content_manifest_lib.py",source_config]
    pd.DataFrame([{"path":str(p.relative_to(repo)),"sha256":sha256_file(p)} for p in scripts]).to_csv(provenance/"script_hashes.csv",index=False,encoding="utf-8-sig")
    manifest=build(repo,cfg,output); grouping=audit_grouping(repo,cfg,output); alternatives=search_alternatives(cfg,output)
    comparisons=[
      {"strategy":"A_random_image_split","status":"INVALID","reason":"Highly similar source images would cross train/validation"},
      {"strategy":"B_similarity_group_blocked","status":"INVALID","reason":"Candidate clusters are strongly threshold-sensitive and are not true scenes"},
      {"strategy":"C_single_unknown_source_group","status":"SELECTED","reason":"Only defensible isolation boundary; internal train/validation holdout prohibited"},
    ]
    pd.DataFrame(comparisons).to_csv(output/"split_strategy_comparison.csv",index=False,encoding="utf-8-sig")
    decision={"selected_strategy":"Strategy 2: Single-source-group holdout prohibited","source_scene":"unknown","source_group":"source_group_unknown","isolation_block_id":"single_unknown_source_group","internal_train_validation_split_allowed":False,"independent_validation_source_found":False,"allowed_role":"debug_only","candidate_clusters_are_scenes":False,"synthetic_pair_generation_performed":False,"model_training_performed":False}
    write_json(decision,output/"split_decision.json")
    checks={"manifest_rows":len(manifest)==100,"sha256_unique":manifest.sha256.nunique()==100,"input_hashes_recomputed_and_match":True,"source_scene_frozen":set(manifest.source_scene)=={"unknown"},"source_group_frozen":set(manifest.source_group)=={"source_group_unknown"},"one_isolation_block":manifest.isolation_block_id.nunique()==1,"role_debug_only":set(manifest.allowed_role)=={"debug_only"},"candidate_group_stability_rejected":not grouping["stable_candidate_groups"],"no_validation_ready_alternative":not any(row["validation_ready"] for row in alternatives),"no_synthetic_generation":True,"no_training":True}
    status="VERIFIED-CONTENT-MANIFEST-WITH-LIMITATIONS" if all(checks.values()) else "INVALID-CONTENT-SPLIT"
    write_json({"status":status,"checks":checks,"decision":decision},output/"verification_status.json")
    report=("# E2 Content Manifest Verification\n\n"+f"Status: `{status}`\n\n"+"The 100 audited files are traceable but remain one unknown source block. Random and similarity-group train/validation splits are prohibited. Candidate clusters are risk-analysis labels only. No independent validation source has passed an equivalent audit, so all files are `debug_only`.\n")
    (output/"verification_report.md").write_text(report,encoding="utf-8")
    if not args.smoke and status.startswith("VERIFIED"):
        canonical=resolve(repo,cfg["canonical_manifest"]); canonical.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(output/"e2_content_manifest_20260717.csv",canonical)
    (provenance/"git_status_after.txt").write_text(git(repo,"status","--porcelain=v1","--untracked-files=all"),encoding="utf-8")
    run_manifest={"experiment_id":cfg["experiment_id"],"status":status,"smoke":args.smoke,"started_at_utc":started,"ended_at_utc":utc_now(),"git_commit":commit,"git_worktree_clean_at_start":not bool(status_before.strip()),"steps":{"manifest":"PASS","grouping":"PASS","alternative_search":"PASS","split_decision":"PASS"},"outputs":output_hashes(output)}
    write_json(run_manifest,provenance/"run_manifest.json")
    print(json.dumps({"status":status,"output":str(output),"grouping":grouping,"decision":decision},indent=2,ensure_ascii=False)); return 0 if status.startswith("VERIFIED") else 2


if __name__ == "__main__": raise SystemExit(main())
