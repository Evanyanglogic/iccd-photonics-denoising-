"""Formal runner for one E1-strength Gaussian synthetic residual smoke pair."""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, scipy, tifffile, yaml

from audit_e2_residual_smoke import audit
from build_e2_residual_baseline import build, sha256_file


def now(): return datetime.now(timezone.utc).isoformat()
def git(repo,*args): return subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=True).stdout
def write_json(path,payload): path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--self-check",action="store_true"); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]; config_path=repo/args.config; cfg=yaml.safe_load(config_path.read_text(encoding="utf-8")); output=repo/args.output_root
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    for name in ("provenance","config","input","float_arrays","tiff","previews","metrics","logs"): (output/name).mkdir(parents=True,exist_ok=True)
    started=now(); commit=git(repo,"rev-parse","HEAD").strip(); status_before=git(repo,"status","--porcelain=v1","--untracked-files=all")
    (output/"provenance/git_commit.txt").write_text(commit+"\n",encoding="utf-8"); (output/"provenance/git_status_before.txt").write_text(status_before,encoding="utf-8")
    (output/"provenance/git_diff.patch").write_text(git(repo,"diff","--binary","HEAD"),encoding="utf-8"); (output/"provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv)+"\n",encoding="utf-8")
    (output/"config/resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
    env=f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\nscipy={scipy.__version__}\npandas={pd.__version__}\ntifffile={tifffile.__version__}\n"
    (output/"provenance/environment.txt").write_text(env,encoding="utf-8")
    subprocess.run([sys.executable,"-m","pip","freeze"],text=True,stdout=(output/"provenance/pip_freeze.txt").open("w",encoding="utf-8"),check=True)
    scripts=[Path(__file__),repo/"scripts/build_e2_residual_baseline.py",repo/"scripts/audit_e2_residual_smoke.py",config_path,repo/cfg["content_manifest"],repo/cfg["e1_strength_csv"],repo/cfg["e1_status"]]
    pd.DataFrame([{"path":str(p.relative_to(repo)),"sha256":sha256_file(p)} for p in scripts]).to_csv(output/"provenance/script_hashes.csv",index=False,encoding="utf-8-sig")
    e1_status=json.loads((repo/cfg["e1_status"]).read_text(encoding="utf-8"))["status"]
    if e1_status!="VERIFIED-RUN": raise RuntimeError(f"E1 is not VERIFIED-RUN: {e1_status}")
    built=build(repo,cfg,output); audited=audit(cfg,output,built)
    input_hash_paths=[Path(built["selection"]["source_path"]),repo/cfg["content_manifest"],repo/cfg["e1_strength_csv"],repo/cfg["e1_status"]]
    pd.DataFrame([{"path":str(p),"sha256":sha256_file(p)} for p in input_hash_paths]).to_csv(output/"provenance/input_hashes.csv",index=False,encoding="utf-8-sig")
    definitions=[
      {"candidate":"A strength-only Gaussian","status":"GO" if all(audited["checks"].values()) else "NO-GO","reason":"Minimal zero-mean Gaussian with strength from formal E1 median temporal std; only candidate executed"},
      {"candidate":"B empirical residual","status":"NO-GO","reason":"No frozen ICCD calibration/test isolation; residual scene/stable-component leakage is unresolved"},
      {"candidate":"C structured operational","status":"NO-GO","reason":"E1 summary energies and correlations do not uniquely identify independent generator parameters"},]
    pd.DataFrame(definitions).to_csv(output/"metrics/candidate_definition_audit.csv",index=False,encoding="utf-8-sig")
    definition_checks={"candidate_a_only":True,"single_pair_only":True,"e1_verified":True,"strength_from_e1_median":True,"no_dark":not cfg["content_preprocessing"]["dark_subtraction"],"no_pedestal":not cfg["content_preprocessing"]["scalar_pedestal_subtraction"],"no_p99":not cfg["content_preprocessing"]["per_image_p99_scaling"],"content_debug_only":built["selection"]["allowed_role"]=="debug_only","no_training_or_split":True,"clean_worktree_at_start":args.self_check or not bool(status_before.strip())}
    final="GO-SMOKE" if all(definition_checks.values()) and all(audited["checks"].values()) else "NUMERIC-NO-GO"
    verification={"status":final,"definition_checks":definition_checks,"numeric_checks":audited["checks"],"candidate_executed":"A strength-only Gaussian","pair_count":1,"synthetic_dataset_generated":False,"model_training_performed":False}
    write_json(output/"verification_status.json",verification)
    report=f"# E2 Residual Baseline Smoke\n\nStatus: `{final}`\n\nCandidate A was executed once with sigma={built['sigma_dn']:.9f} DN ({built['sigma_norm']:.12g} normalized). It is an operational synthetic residual baseline, not an ICCD physical or calibrated noise model. Candidate B and C were definition-audited only.\n"
    (output/"verification_report.md").write_text(report,encoding="utf-8")
    (output/"provenance/git_status_after.txt").write_text(git(repo,"status","--porcelain=v1","--untracked-files=all"),encoding="utf-8")
    (output/"logs/run.log").write_text(json.dumps({"verification":verification,"sigma_dn":built["sigma_dn"],"selection":built["selection"]},indent=2,ensure_ascii=False),encoding="utf-8")
    hashes=[]
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name not in {"output_hashes.csv","run_manifest.json"}: hashes.append({"relative_path":str(p.relative_to(output)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    pd.DataFrame(hashes).to_csv(output/"output_hashes.csv",index=False,encoding="utf-8-sig")
    run={"experiment_id":cfg["experiment_id"],"status":final,"self_check":args.self_check,"started_at_utc":started,"ended_at_utc":now(),"exit_code":0 if final=="GO-SMOKE" else 2,"git_commit":commit,"git_worktree_clean_at_start":not bool(status_before.strip()),"seed":cfg["residual_baseline"]["seed"],"pair_count":1}
    write_json(output/"provenance/run_manifest.json",run)
    print(json.dumps({"status":final,"sigma_dn":built["sigma_dn"],"sigma_norm":built["sigma_norm"],"selection":built["selection"],"metrics":audited},indent=2,ensure_ascii=False)); return run["exit_code"]


if __name__=="__main__": raise SystemExit(main())
