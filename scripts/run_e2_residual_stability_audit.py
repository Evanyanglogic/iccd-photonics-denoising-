"""Run the preregistered 8-content x 3-seed Candidate A stability audit."""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, scipy, tifffile, yaml

from audit_e2_residual_stability import aggregate, pair_metrics, save_representative, select_contents


def now(): return datetime.now(timezone.utc).isoformat()
def git(repo,*args): return subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=True).stdout
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def write_json(path,payload): Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output-root",required=True); p.add_argument("--self-check",action="store_true"); p.add_argument("--smoke-contents",type=int,default=2); p.add_argument("--smoke-seeds",type=int,default=1); args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]; cfg_path=repo/args.config; cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8")); output=repo/args.output_root
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    for d in ("provenance","config","selected_contents","float_arrays","tiff","previews","metrics","logs"): (output/d).mkdir(parents=True,exist_ok=True)
    started=now(); commit=git(repo,"rev-parse","HEAD").strip(); status_before=git(repo,"status","--porcelain=v1","--untracked-files=all")
    manifest=pd.read_csv(repo/cfg["content_manifest"]); quantiles=cfg["content_selection"]["quantiles"]; selected=select_contents(manifest,quantiles)
    if args.self_check: selected=selected.head(args.smoke_contents).copy(); seeds=cfg["seeds"][:args.smoke_seeds]
    else: seeds=cfg["seeds"]
    if not (selected.allowed_role==cfg["required_content_role"]).all() or not (selected.isolation_block_id==cfg["isolation_block_id"]).all(): raise RuntimeError("Content role or isolation block changed")
    e1=pd.read_csv(repo/cfg["e1_strength_csv"]); source_sigma=float(np.median(e1.temporal_std_mean.astype(float)))
    if abs(source_sigma-float(cfg["sigma_dn"]))>1e-9: raise RuntimeError(f"Frozen sigma differs from E1 source: {source_sigma}")
    if json.loads((repo/cfg["e1_status"]).read_text(encoding="utf-8"))["status"]!="VERIFIED-RUN": raise RuntimeError("E1 not verified")
    if json.loads((repo/cfg["baseline_report"]).read_text(encoding="utf-8"))["status"]!="GO-SMOKE": raise RuntimeError("Candidate A baseline not GO-SMOKE")
    selected.to_csv(output/"selected_contents/selected_debug_contents.csv",index=False,encoding="utf-8-sig")
    (output/"provenance/git_commit.txt").write_text(commit+"\n",encoding="utf-8"); (output/"provenance/git_status_before.txt").write_text(status_before,encoding="utf-8"); (output/"provenance/git_diff.patch").write_text(git(repo,"diff","--binary","HEAD"),encoding="utf-8")
    (output/"provenance/command.txt").write_text(subprocess.list2cmdline(sys.argv)+"\n",encoding="utf-8"); (output/"config/resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
    (output/"provenance/environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\nscipy={scipy.__version__}\npandas={pd.__version__}\ntifffile={tifffile.__version__}\n",encoding="utf-8")
    subprocess.run([sys.executable,"-m","pip","freeze"],stdout=(output/"provenance/pip_freeze.txt").open("w",encoding="utf-8"),text=True,check=True)
    script_paths=[Path(__file__),repo/"scripts/audit_e2_residual_stability.py",cfg_path]
    pd.DataFrame([{"path":str(x.relative_to(repo)),"sha256":sha(x)} for x in script_paths]).to_csv(output/"provenance/script_hashes.csv",index=False,encoding="utf-8-sig")
    input_paths=[repo/cfg["content_manifest"],repo/cfg["e1_strength_csv"],repo/cfg["e1_status"],repo/cfg["baseline_report"]]+[Path(x) for x in selected.absolute_path]
    input_hashes=pd.DataFrame([{"path":str(x),"sha256":sha(x),"size_bytes":x.stat().st_size,"mtime_ns":x.stat().st_mtime_ns} for x in input_paths]); input_hashes.to_csv(output/"provenance/input_hashes_before.csv",index=False,encoding="utf-8-sig")
    source_dir=Path(selected.absolute_path.iloc[0]).parent; source_names_before=sorted(x.name for x in source_dir.glob("*.tif*")); source_count_before=len(source_names_before)
    divisor=float(cfg["normalization_divisor"]); sigma_dn=float(cfg["sigma_dn"]); sigma_norm=sigma_dn/divisor
    selected_means=selected.set_index("content_id").roi_mean_dn; median_target=float(selected.roi_mean_dn.median()); median_id=selected.assign(d=(selected.roi_mean_dn-median_target).abs()).sort_values(["d","sha256"]).iloc[0].content_id
    rep_roles={selected.sort_values("roi_mean_dn").iloc[0].content_id:"lowest_mean",median_id:"closest_selected_median",selected.sort_values("roi_mean_dn").iloc[-1].content_id:"highest_mean"}
    rows=[]
    for item in selected.itertuples(index=False):
        path=Path(item.absolute_path)
        if sha(path)!=item.sha256: raise RuntimeError(f"INPUT-DRIFT: {path}")
        raw=tifffile.imread(path); roi=cfg["roi"]; content=(raw[roi["top"]:roi["top"]+roi["height"],roi["left"]:roi["left"]+roi["width"]].astype(np.float32)/divisor).astype(np.float32)
        for seed in seeds:
            residual=np.random.default_rng(seed).normal(0.0,sigma_norm,size=(512,512)).astype(np.float32)
            metrics,noisy,content_u16,noisy_u16=pair_metrics(content,residual,sigma_dn,divisor)
            row={"content_id":item.content_id,"content_sha256":item.sha256,"seed":seed,"sigma_dn":sigma_dn,"sigma_norm":sigma_norm,"allowed_role":item.allowed_role,"isolation_block_id":item.isolation_block_id}; row.update(metrics); rows.append(row)
            if seed==cfg["representative_save"]["seed"] and item.content_id in rep_roles:
                save_representative(output,rep_roles[item.content_id],item.content_id,seed,content,residual,(content+residual).astype(np.float32),noisy,content_u16,noisy_u16,sigma_norm)
    pairs=pd.DataFrame(rows); pairs.to_csv(output/"metrics/pair_level_metrics.csv",index=False,encoding="utf-8-sig")
    agg=aggregate(pairs,selected,cfg); agg["content"].to_csv(output/"metrics/content_level_seed_summary.csv",index=False,encoding="utf-8-sig"); agg["seed"].to_csv(output/"metrics/seed_level_summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame([agg["cross"]]).to_csv(output/"metrics/cross_content_stability.csv",index=False,encoding="utf-8-sig"); agg["correlations"].to_csv(output/"metrics/correlation_analysis.csv",index=False,encoding="utf-8-sig")
    clipping_cols=["content_id","seed","negative_before_clipping_ratio","above_one_before_clipping_ratio","added_zero_clipping_ratio","added_one_clipping_ratio","source_zero_ratio","source_saturation_ratio"]; pairs[clipping_cols].to_csv(output/"metrics/clipping_summary.csv",index=False,encoding="utf-8-sig")
    rt_cols=["content_id","seed","content_round_trip_max_error_dn","noisy_round_trip_max_error_dn","residual_reconstruction_max_error_dn","residual_reconstruction_std_relative_error","residual_reconstruction_mean_error_dn","round_trip_zero_ratio_change","round_trip_one_ratio_change","round_trip_dtype","round_trip_shape","tiff_compression","tiff_readable"]; pairs[rt_cols].to_csv(output/"metrics/round_trip_summary.csv",index=False,encoding="utf-8-sig")
    gates=cfg["gates"]; warnings=[]
    for row in pairs.itertuples(index=False):
        reasons=[]
        if row.residual_std_relative_error>=gates["pair_residual_std_relative_error_max"]: reasons.append("residual_std")
        if abs(row.brightness_difference_dn)>=gates["pair_abs_brightness_difference_dn_max"]: reasons.append("brightness")
        if row.negative_before_clipping_ratio>=gates["negative_before_clipping_max"] or row.above_one_before_clipping_ratio>=gates["above_one_before_clipping_max"] or row.added_zero_clipping_ratio>=gates["added_zero_clipping_max"] or row.added_one_clipping_ratio>=gates["added_one_clipping_max"]: reasons.append("clipping")
        if not gates["gradient_energy_ratio_min"]<=row.gradient_energy_ratio<=gates["gradient_energy_ratio_max"]: reasons.append("gradient")
        if row.noisy_round_trip_max_error_dn>gates["noisy_round_trip_max_error_dn"] or row.residual_reconstruction_max_error_dn>gates["residual_reconstruction_max_error_dn"]: reasons.append("round_trip")
        if reasons: warnings.append({"content_id":row.content_id,"seed":row.seed,"status":"WARN","reasons":";".join(reasons)})
    pd.DataFrame(warnings,columns=["content_id","seed","status","reasons"]).to_csv(output/"metrics/warning_pairs.csv",index=False,encoding="utf-8-sig")
    source_names_after=sorted(x.name for x in source_dir.glob("*.tif*")); after_rows=[]
    for before in input_hashes[input_hashes.path.isin(selected.absolute_path) ].itertuples(index=False):
        path=Path(before.path); after_rows.append({"path":before.path,"sha256_before":before.sha256,"sha256_after":sha(path),"mtime_ns_before":before.mtime_ns,"mtime_ns_after":path.stat().st_mtime_ns,"unchanged":before.sha256==sha(path) and before.mtime_ns==path.stat().st_mtime_ns})
    protection={"source_directory":str(source_dir),"count_before":source_count_before,"count_after":len(source_names_after),"file_names_unchanged":source_names_before==source_names_after,"selected_hashes_and_mtimes_unchanged":all(x["unchanged"] for x in after_rows),"source_write_performed":False,"all_outputs_inside_repo":str(output.resolve()).startswith(str(repo.resolve()))}
    pd.DataFrame(after_rows).to_csv(output/"provenance/source_protection_check.csv",index=False,encoding="utf-8-sig"); write_json(output/"provenance/source_protection_status.json",protection)
    if args.self_check:
        status="SELF-CHECK-PASS" if len(pairs)==len(selected)*len(seeds) and not warnings else "SELF-CHECK-FAIL"
        checks={"pair_count":len(pairs)==len(selected)*len(seeds),"warning_free":not warnings,"source_protected":protection["count_before"]==protection["count_after"] and protection["file_names_unchanged"] and protection["selected_hashes_and_mtimes_unchanged"] and not protection["source_write_performed"] and protection["all_outputs_inside_repo"]}
    else:
        c=agg["cross"]; checks={"pair_count_24":len(pairs)==24,"content_count_8":len(selected)==8,"seeds_exact":sorted(pairs.seed.unique())==sorted(cfg["seeds"]),"sigma_frozen":pairs.sigma_dn.nunique()==1 and float(pairs.sigma_dn.iloc[0])==sigma_dn,"pair_std_error":c["pair_residual_std_relative_error_max"]<gates["pair_residual_std_relative_error_max"],"median_std_error":c["pair_residual_std_relative_error_median"]<gates["median_residual_std_relative_error_max"],"content_std_cv":c["content_level_residual_std_mean_cv"]<gates["content_level_residual_std_mean_cv_max"],"clipping":c["worst_negative_clipping"]<gates["negative_before_clipping_max"] and c["worst_above_one_clipping"]<gates["above_one_before_clipping_max"] and c["worst_added_zero_clipping"]<gates["added_zero_clipping_max"] and c["worst_added_one_clipping"]<gates["added_one_clipping_max"],"pair_brightness":c["brightness_difference_max_abs_dn"]<gates["pair_abs_brightness_difference_dn_max"],"overall_brightness":abs(c["brightness_difference_mean_dn"])<gates["overall_abs_mean_brightness_difference_dn_max"] and c["absolute_brightness_difference_mean_dn"]<gates["overall_mean_abs_brightness_difference_dn_max"],"gradient":c["gradient_ratio_min"]>=gates["gradient_energy_ratio_min"] and c["gradient_ratio_max"]<=gates["gradient_energy_ratio_max"],"round_trip":c["noisy_round_trip_worst_dn"]<=gates["noisy_round_trip_max_error_dn"] and c["residual_reconstruction_worst_dn"]<=gates["residual_reconstruction_max_error_dn"],"no_warning_pairs":not warnings,"no_content_anomaly":not agg["content"].seed_anomaly.any(),"no_seed_anomaly":not agg["seed"].systematic_anomaly.any(),"provenance":not bool(status_before.strip()),"source_protected":protection["file_names_unchanged"] and protection["selected_hashes_and_mtimes_unchanged"],"debug_only":set(pairs.allowed_role)=={"debug_only"},"no_split_or_training":True,"candidate_a_only":True,"no_real_iccd_tuning":True}
        status="GO-STABILITY" if all(checks.values()) else ("PARTIAL-STABILITY" if checks["pair_count_24"] and checks["source_protected"] else "NUMERIC-NO-GO")
    verification={"status":status,"checks":checks,"pair_count":len(pairs),"content_count":len(selected),"seeds":seeds,"sigma_dn":sigma_dn,"source_group":"single_unknown_source_group","training_performed":False,"train_validation_split_created":False,"candidate_b_or_c_called":False}; write_json(output/"verification_status.json",verification)
    report=f"# E2 Candidate A Residual Stability Audit\n\nStatus: `{status}`\n\nThis audit covers {len(selected)} debug-only contents and {len(seeds)} preregistered seeds within one unknown source group. It tests numerical stability only and does not establish ICCD realism, synthetic-real transfer, scene generalization, or training readiness.\n"; (output/"verification_report.md").write_text(report,encoding="utf-8")
    (output/"provenance/git_status_after.txt").write_text(git(repo,"status","--porcelain=v1","--untracked-files=all"),encoding="utf-8"); (output/"logs/run.log").write_text(json.dumps({"verification":verification,"warnings":warnings,"source_protection":protection},indent=2,ensure_ascii=False),encoding="utf-8")
    hashes=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"output_hashes.csv","run_manifest.json"}: hashes.append({"relative_path":str(path.relative_to(output)),"size_bytes":path.stat().st_size,"sha256":sha(path)})
    pd.DataFrame(hashes).to_csv(output/"output_hashes.csv",index=False,encoding="utf-8-sig")
    run={"experiment_id":cfg["experiment_id"],"status":status,"self_check":args.self_check,"started_at_utc":started,"ended_at_utc":now(),"exit_code":0 if status in {"GO-STABILITY","SELF-CHECK-PASS"} else 2,"git_commit":commit,"git_worktree_clean_at_start":not bool(status_before.strip()),"steps":{"selection":0,"generation":0,"aggregation":0,"source_protection":0}}; write_json(output/"provenance/run_manifest.json",run)
    print(json.dumps({"status":status,"checks":checks,"cross":agg["cross"],"warnings":warnings,"source_protection":protection},indent=2,ensure_ascii=False)); return run["exit_code"]


if __name__=="__main__": raise SystemExit(main())
