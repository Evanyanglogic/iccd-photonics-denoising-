"""Fit the calibration-only signal model and stop before training if pair safety fails."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,yaml
from scipy.optimize import nnls
from json_serialization import dump_json

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def git(repo,*a): return subprocess.run(['git',*a],cwd=repo,text=True,capture_output=True,check=True).stdout
def fit(name,x,std,var):
 if name=='variance_unconstrained': b,a=np.polyfit(x,var,1); return float(a),float(b),'variance'
 if name=='variance_nonnegative': c,_=nnls(np.c_[np.ones(len(x)),x],var); return float(c[0]),float(c[1]),'variance'
 if name=='variance_zero_intercept': return 0.,float(x@var/(x@x)),'variance'
 c,_=nnls(np.c_[np.ones(len(x)),x],std); return float(c[0]),float(c[1]),'std'
def predict(a,b,kind,s,eps=1e-12):
 raw=a+b*np.asarray(s,float)
 return raw,np.sqrt(np.maximum(raw,eps)) if kind=='variance' else raw
def model_audit(cal):
 x=cal.mean_signal.to_numpy(float); sd=cal.temporal_std_mean.to_numpy(float); var=cal.temporal_var_mean.to_numpy(float); rows=[]; loo=[]
 for name in ['variance_unconstrained','variance_nonnegative','variance_zero_intercept','std_nonnegative']:
  a,b,k=fit(name,x,sd,var); raw,pred=predict(a,b,k,x); errs=[]; slopes=[]
  for i in range(len(x)):
   keep=np.arange(len(x))!=i; aa,bb,kk=fit(name,x[keep],sd[keep],var[keep]); rr,pp=predict(aa,bb,kk,[x[i]]); err=float(sd[i]-pp[0]); errs.append(err); slopes.append(bb); loo.append({'model':name,'excluded_folder':int(cal.iloc[i].folder),'train_a':aa,'train_b':bb,'predicted_sigma_DN':float(pp[0]),'observed_sigma_DN':float(sd[i]),'error_DN':err,'raw_variance_or_sigma':float(rr[0]),'finite':bool(np.isfinite(pp[0])),'positive':bool(pp[0]>0)})
  rows.append({'model':name,'a':a,'b':b,'prediction_domain':k,'calibration_predictions_finite':bool(np.isfinite(pred).all()),'calibration_variance_nonnegative':bool((raw>=0).all()) if k=='variance' else True,'full_rmse_DN':float(np.sqrt(np.mean((sd-pred)**2))),'loocv_rmse_DN':float(np.sqrt(np.mean(np.square(errs)))),'loocv_mae_DN':float(np.mean(np.abs(errs))),'loo_slope_sign_stability':float(np.mean(np.sign(slopes)==np.sign(b))),'maximum_slope_relative_change':float(np.max(np.abs(np.asarray(slopes)-b))/max(abs(b),1e-12))})
 out=pd.DataFrame(rows).sort_values('loocv_rmse_DN'); out['selection_rank']=np.arange(1,len(out)+1); return out,pd.DataFrame(loo)
def patch_coords(hexhash,h,w,ph=512,pw=512):
 return 2*(int(hexhash[:8],16)%((h-ph)//2+1)),2*(int(hexhash[8:16],16)%((w-pw)//2+1))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output-root',required=True); args=ap.parse_args(); repo=Path(__file__).resolve().parents[1]; cfgp=(repo/args.config).resolve(); cfg=yaml.safe_load(cfgp.read_text(encoding='utf-8')); out=(repo/args.output_root).resolve()
 if out.exists(): raise FileExistsError(out)
 started=now(); commit=git(repo,'rev-parse','HEAD').strip(); status_before=git(repo,'status','--porcelain=v1','--untracked-files=all')
 for d in ['provenance','configs','manifests','condition_model','generated_data/G','generated_data/CG','generated_data/limited_previews','training/G','training/CG_NC','training/CG_C','metrics','checkpoints','logs']: (out/d).mkdir(parents=True,exist_ok=False)
 split=pd.read_csv(repo/cfg['folder_split']); cal_ids=split.loc[split.role.eq('calibration'),'folder'].astype(int).tolist(); eval_ids=split.loc[split.role.eq('evaluation'),'folder'].astype(int).tolist()
 if cal_ids!=[1,4,7,8,10,13] or eval_ids!=[2,5,9,11]: raise RuntimeError('folder split drift')
 e1=pd.read_csv(repo/cfg['e1_statistics']); cal=e1[e1.folder.isin(cal_ids)].copy(); fits,loo=model_audit(cal); selected=fits.iloc[0]
 model=yaml.safe_load((repo/cfg['condition_model']).read_text(encoding='utf-8'))
 if selected.model!='std_nonnegative' or abs(selected.a-model['a'])>1e-12 or abs(selected.b-model['b'])>1e-12: raise RuntimeError('condition model config drift')
 sc=pd.read_csv(repo/cfg['scmos_manifest']); required_role=set(sc.allowed_role); expected_hash={str(r.absolute_path):r.sha256 for _,r in sc.iterrows()}; source_before={p:{'sha256':sha256(p),'mtime_ns':Path(p).stat().st_mtime_ns} for p in expected_hash}
 if any(source_before[p]['sha256']!=expected_hash[p] for p in expected_hash): raise RuntimeError('sCMOS input hash drift')
 pair_rows=[]
 for _,r in sc.sort_values('content_id').iterrows():
  signal=float(r.roi_mean_dn); raw,pred=predict(model['a'],model['b'],'std',[signal]); sigma=float(pred[0]); inrange=model['calibration_signal_range_DN'][0]<=signal<=model['calibration_signal_range_DN'][1]; valid=math.isfinite(sigma) and sigma>0 and sigma<=cfg['safe_sigma_max_DN']
  for seed in cfg['training_noise_seeds']: pair_rows.append({'pair_id':f"{r.content_id}_{seed}",'content_id':r.content_id,'absolute_path':r.absolute_path,'content_sha256':r.sha256,'seed':seed,'roi_top':768,'roi_left':768,'roi_height':512,'roi_width':512,'allowed_role':'exploratory_training_only','patch_mean_DN':signal,'signal_in_calibration_range':inrange,'predicted_sigma_raw_DN':float(raw[0]),'predicted_sigma_used_DN':sigma if valid else float('nan'),'extrapolation_flag':not inrange,'g_sigma_DN':cfg['g_sigma_DN'],'cg_pair_valid':valid,'failure_reason':'' if valid else 'SIGMA_SAFETY_LIMIT_EXCEEDED','synthetic_generated':False})
 pairs=pd.DataFrame(pair_rows)
 pm_manifest=pd.read_csv(repo/cfg['pmrid_scene_manifest']); benchmark=json.loads(Path(cfg['pmrid_benchmark']).read_text(encoding='utf-8')); pmrows=[]; pm_before={}
 for i,item in enumerate(benchmark):
  p=Path(cfg['pmrid_root'])/item['gt']; expected=str(pm_manifest.iloc[i].SHA256); bh=sha256(p); pm_before[str(p)]={'sha256':bh,'mtime_ns':p.stat().st_mtime_ns};
  if bh!=expected: raise RuntimeError(f'PMRID hash drift: {p}')
  H,W=item['meta']['shape']; top,left=patch_coords(bh,H,W); arr=np.memmap(p,dtype=np.uint16,mode='r',shape=(H,W)); signal=float(np.mean(arr[top:top+512,left:left+512],dtype=np.float64)); raw,pred=predict(model['a'],model['b'],'std',[signal]); sigma=float(pred[0]); inrange=model['calibration_signal_range_DN'][0]<=signal<=model['calibration_signal_range_DN'][1]; valid=math.isfinite(sigma) and sigma>0 and sigma<=cfg['safe_sigma_max_DN']; pmrows.append({'pmrid_content_id':pm_manifest.iloc[i].pmrid_content_id,'benchmark_entry':i,'scene_id':item['meta']['scene_id'],'condition':item['meta']['light'],'ISO':item['meta']['ISO'],'exposure':item['meta']['exp_time'],'gt_path':str(p),'gt_sha256':bh,'dtype':'uint16','shape':f'{H}x{W}','Bayer_pattern':'BGGR','patch_top':top,'patch_left':left,'patch_height':512,'patch_width':512,'patch_mean_DN':signal,'normalization_divisor':65535.0,'black_level_subtraction':False,'signal_in_calibration_range':inrange,'predicted_sigma_raw_DN':float(raw[0]),'predicted_sigma_used_DN':sigma if valid else float('nan'),'extrapolation_flag':not inrange,'cg_pair_valid':valid,'failure_reason':'' if valid else 'SIGMA_SAFETY_LIMIT_EXCEEDED','synthetic_generated':False,'allowed_role':'validation_content_only'})
 pm=pd.DataFrame(pmrows)
 pairs.to_csv(out/'manifests/scmos_training_pairs.csv',index=False,encoding='utf-8-sig'); pm.to_csv(out/'manifests/pmrid_validation_patches.csv',index=False,encoding='utf-8-sig'); split.to_csv(out/'manifests/iccd_calibration_evaluation_split.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'source':'sCMOS_100','role':'exploratory_training_only','count':100,'used':False,'reason':'CG preflight failed before generation'},{'source':'PMRID_GT_RAW','role':'validation_content_only','count':39,'used':False,'reason':'CG preflight failed before generation'},{'source':'ICCD_calibration_folders','role':'condition_parameter_calibration_only','count':6,'used':True,'reason':'read existing E1 CSV only'},{'source':'ICCD_evaluation_folders','role':'holdout','count':4,'used':False,'reason':'not read'}]).to_csv(out/'manifests/data_role_manifest.csv',index=False,encoding='utf-8-sig')
 fits.to_csv(out/'condition_model/calibration_fit.csv',index=False,encoding='utf-8-sig'); loo.to_csv(out/'condition_model/loocv.csv',index=False,encoding='utf-8-sig'); dump_json(out/'condition_model/signal_condition_model.json',model); dump_json(out/'condition_model/extrapolation_policy.json',{'policy':'allow_and_flag_no_clamp','failure_conditions':['NaN','Inf','negative variance','sigma<=0','sigma>300 DN'],'safe_sigma_max_DN':300.0})
 for p in [cfgp,repo/cfg['condition_model']]: (out/'configs'/p.name).write_bytes(p.read_bytes())
 blockers={'scmos_cg_pairs_total':len(pairs),'scmos_cg_pairs_invalid':int((~pairs.cg_pair_valid).sum()),'pmrid_patches_total':len(pm),'pmrid_cg_patches_invalid':int((~pm.cg_pair_valid).sum()),'training_started':False,'reason':'All 300 sCMOS CG pairs exceed the pre-registered 300 DN sigma safety limit.'}
 pd.DataFrame([blockers]).to_csv(out/'metrics/warnings.csv',index=False,encoding='utf-8-sig')
 verification={'experiment_id':cfg['experiment_id'],'status':'TRAINING-PREFLIGHT-NO-GO','condition_model_fit_completed':True,'condition_model_selected':'std_nonnegative','synthetic_pairs_generated':False,'training_started':False,'experiments_completed':{'G':False,'CG_NC':False,'CG_C':False},'blocking_gate':'CG_SIGMA_DOMAIN_MISMATCH','blockers':blockers,'evaluation_iccd_holdout_preserved':True,'provenance_complete':False,'source_data_protected':False,'next_task':'Resolve the single condition-scale/domain mismatch without changing network, loss, split, or using validation outcomes.'}; dump_json(out/'verification_status.json',verification)
 (out/'verification_report.md').write_text(f"""# E2 G/CG Initial Training Preflight\n\nStatus: `TRAINING-PREFLIGHT-NO-GO`\n\nCalibration-only fitting selected `sigma_DN = {model['a']:.12g} + {model['b']:.12g} * reference_patch_mean_DN` with LOOCV RMSE {selected.loocv_rmse_DN:.6f} DN. The model is operational and has no physical gain interpretation.\n\nThe frozen sCMOS ROI means span {sc.roi_mean_dn.min():.3f}-{sc.roi_mean_dn.max():.3f} DN, entirely outside the ICCD calibration range {model['calibration_signal_range_DN'][0]:.3f}-{model['calibration_signal_range_DN'][1]:.3f} DN. Consequently all {len(pairs)} CG training pairs predict sigma above 300 DN. PMRID deterministic patch means span {pm.patch_mean_DN.min():.3f}-{pm.patch_mean_DN.max():.3f} DN; {(~pm.cg_pair_valid).sum()}/{len(pm)} exceed the same gate.\n\nNo clipping, brightness mapping, clamping, synthetic generation, model training, checkpoint selection, or real ICCD evaluation was performed. G-only training was not started because the pre-registered experiment requires all three arms and would not answer the requested G-vs-CG comparison.\n""",encoding='utf-8')
 prov=out/'provenance'; (prov/'git_commit.txt').write_text(commit+'\n',encoding='utf-8'); (prov/'git_status_before.txt').write_text(status_before,encoding='utf-8'); (prov/'git_diff.patch').write_text(git(repo,'diff','--binary','HEAD'),encoding='utf-8'); (prov/'command.txt').write_text(subprocess.list2cmdline(sys.argv)+'\n',encoding='utf-8'); (prov/'environment.txt').write_text(f'python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\n',encoding='utf-8'); (prov/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
 scripts=[Path(__file__),cfgp,repo/cfg['condition_model'],repo/'scripts/json_serialization.py']; pd.DataFrame([{'path':str(p.relative_to(repo)),'sha256':sha256(p)} for p in scripts]).to_csv(prov/'script_hashes.csv',index=False,encoding='utf-8-sig')
 source_rows=[]
 for p,b in {**source_before,**pm_before}.items(): source_rows.append({'path':p,'sha256_before':b['sha256'],'sha256_after':sha256(p),'mtime_ns_before':b['mtime_ns'],'mtime_ns_after':Path(p).stat().st_mtime_ns})
 source=pd.DataFrame(source_rows); source['unchanged']=(source.sha256_before==source.sha256_after)&(source.mtime_ns_before==source.mtime_ns_after); source.to_csv(prov/'source_protection.csv',index=False,encoding='utf-8-sig'); protected=bool(source.unchanged.all())
 run={'experiment_id':cfg['experiment_id'],'started_at_utc':started,'ended_at_utc':now(),'git_commit':commit,'status':'TRAINING-PREFLIGHT-NO-GO','source_data_protected':protected,'synthetic_pairs_generated':False,'training_started':False,'evaluation_iccd_folders_read':False}; dump_json(prov/'run_manifest.json',run); (prov/'git_status_after.txt').write_text(git(repo,'status','--porcelain=v1','--untracked-files=all'),encoding='utf-8')
 verification['source_data_protected']=protected; verification['provenance_complete']=all((out/p).exists() for p in ['condition_model/calibration_fit.csv','condition_model/loocv.csv','manifests/scmos_training_pairs.csv','manifests/pmrid_validation_patches.csv','provenance/run_manifest.json','provenance/source_protection.csv','verification_report.md']); dump_json(out/'verification_status.json',verification)
 hashes=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='output_hashes.csv': hashes.append({'relative_path':str(p.relative_to(out)),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
 pd.DataFrame(hashes).to_csv(out/'output_hashes.csv',index=False,encoding='utf-8-sig'); print(json.dumps(verification,indent=2)); return 2
if __name__=='__main__': raise SystemExit(main())
