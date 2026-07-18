"""Audit brightness-adjusted observed-state reliability and freeze an E1 folder split."""
from __future__ import annotations

import argparse, hashlib, json, math, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile, yaml
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr

from json_serialization import dump_json


def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def git(repo,*args): return subprocess.run(['git',*args],cwd=repo,text=True,capture_output=True,check=True).stdout
def rho(a,b):
    x=float(spearmanr(np.asarray(a,float),np.asarray(b,float)).statistic)
    return x if math.isfinite(x) else float('nan')
def pr(a,b):
    x=float(pearsonr(np.asarray(a,float),np.asarray(b,float)).statistic)
    return x if math.isfinite(x) else float('nan')
def list_tiffs(root):
    def key(p):
        m=re.match(r'^(\d+)',p.name); return (int(m.group(1)) if m else 10**9,p.name)
    return sorted([p for p in root.iterdir() if p.suffix.lower() in {'.tif','.tiff'}],key=key)
def read_roi_stack(paths,roi):
    t,l,h,w=[int(roi[k]) for k in ('top','left','height','width')]
    out=np.empty((len(paths),h,w),np.float32)
    for i,p in enumerate(paths): out[i]=np.asarray(tifffile.memmap(p)[t:t+h,l:l+w],np.float32)
    return out
def corr(x,y):
    x=np.asarray(x,np.float64).ravel(); y=np.asarray(y,np.float64).ravel(); x-=x.mean(); y-=y.mean()
    d=math.sqrt(float(x@x)*float(y@y)); return float(x@y/d) if d>1e-12 else float('nan')
def metrics(stack,sigma):
    n=len(stack); mean_map=np.mean(stack,axis=0,dtype=np.float64)
    var=np.var(stack,axis=0,ddof=1,dtype=np.float64)
    residual=stack.astype(np.float32)-mean_map.astype(np.float32)
    residual-=np.mean(residual,axis=(1,2),keepdims=True,dtype=np.float64).astype(np.float32)
    rows=np.mean(residual,axis=2,dtype=np.float64); cols=np.mean(residual,axis=1,dtype=np.float64)
    acfs=[corr(residual[:,:,:-1],residual[:,:,1:]),corr(residual[:,:-1,:],residual[:,1:,:]),corr(residual[:,:-1,:-1],residual[:,1:,1:]),corr(residual[:,:-1,1:],residual[:,1:,:-1])]
    half=n//2; first=float(np.mean(stack[:half],dtype=np.float64)); second=float(np.mean(stack[half:],dtype=np.float64))
    stable=mean_map-gaussian_filter(mean_map,sigma=sigma,mode='reflect')
    return {'frame_count':n,'mean_signal_DN':float(mean_map.mean()),'median_signal_DN':float(np.median(mean_map)),
      'temporal_std_DN':float(np.mean(np.sqrt(np.maximum(var,0)))),'temporal_variance_DN2':float(np.mean(var)),
      'row_energy_DN':float(np.sqrt(np.mean(rows**2))),'column_energy_DN':float(np.sqrt(np.mean(cols**2))),
      'radial_acf_lag1':float(np.mean(acfs)),'observed_stable_strength_DN':float(np.std(stable,ddof=1)),
      'drift_percent':100*(second-first)/first if first else float('nan')}
def subset_indices(n): return {
    'first_32':np.arange(0,32),'middle_32':np.arange((n-32)//2,(n-32)//2+32),'last_32':np.arange(n-32,n),
    'odd':np.arange(0,n,2),'even':np.arange(1,n,2),'first_half':np.arange(0,n//2),'second_half':np.arange(n//2,n)}

def fit_model(name,x,std,var):
    x=np.asarray(x,float); std=np.asarray(std,float); var=np.asarray(var,float)
    if name=='L1':
        b,a=np.polyfit(x,std,1); pred=a+b*x; native=std; equation='temporal_std = a + b*mean_signal'
    elif name=='L2':
        b,a=np.polyfit(x,var,1); pv=a+b*x; pred=np.sqrt(np.maximum(pv,0)); native=var; equation='temporal_variance = a + b*mean_signal'
    elif name=='P1':
        b,a=np.polyfit(np.log(x),np.log(std),1); pred=np.exp(a+b*np.log(x)); native=np.log(std); equation='log(temporal_std) = a + b*log(mean_signal)'
    else:
        coef,_=nnls(np.column_stack([np.ones(len(x)),x]),std); a,b=coef; pred=a+b*x; native=std; equation='temporal_std = nonnegative read_term + k*mean_signal'
    if name=='L2': native_pred=a+b*x
    elif name=='P1': native_pred=a+b*np.log(x)
    else: native_pred=pred
    return {'a':float(a),'b':float(b),'prediction_std':pred,'equation':equation,
      'r_squared_native':float(1-np.sum((native-native_pred)**2)/np.sum((native-native.mean())**2)),
      'rmse_std':float(np.sqrt(np.mean((std-pred)**2))),
      'nonphysical_warning':bool((name in {'L1','L2'} and a<0) or np.any(~np.isfinite(pred)) or np.any(pred<=0))}
def fit_all(df,models):
    x=df.mean_signal_DN.to_numpy(); s=df.temporal_std_DN.to_numpy(); v=df.temporal_variance_DN2.to_numpy()
    summaries=[]; loo=[]
    for name in models:
        full=fit_model(name,x,s,v); errors=[]; slopes=[]
        for i,row in df.reset_index(drop=True).iterrows():
            keep=np.arange(len(df))!=i; fit=fit_model(name,x[keep],s[keep],v[keep]); pred=fit_model_prediction(name,fit,float(x[i])); err=float(s[i]-pred)
            errors.append(err); slopes.append(fit['b']); loo.append({'model':name,'excluded_folder':int(row.folder),'train_a':fit['a'],'train_b':fit['b'],'predicted_std_DN':pred,'observed_std_DN':float(s[i]),'prediction_error_DN':err,'residual_sign':int(np.sign(err)),'slope_relative_change':abs(fit['b']-full['b'])/max(abs(full['b']),1e-12)})
        summaries.append({'model':name,'equation':full['equation'],'a':full['a'],'b':full['b'],'pearson_r_signal_vs_native_target':pr(x,v if name=='L2' else s),'r_squared_native':full['r_squared_native'],'full_rmse_std_DN':full['rmse_std'],'loocv_rmse_std_DN':float(np.sqrt(np.mean(np.square(errors)))),'loocv_mae_std_DN':float(np.mean(np.abs(errors))),'loo_slope_sign_consistency':float(np.mean(np.sign(slopes)==np.sign(full['b']))),'maximum_slope_relative_change':float(max(abs(np.asarray(slopes)-full['b'])/max(abs(full['b']),1e-12))),'nonphysical_warning':full['nonphysical_warning']})
    out=pd.DataFrame(summaries).sort_values('loocv_rmse_std_DN').reset_index(drop=True); out['loocv_rank']=np.arange(1,len(out)+1)
    lo=pd.DataFrame(loo); lo['high_influence']=lo.slope_relative_change>0.25
    return out,lo
def fit_model_prediction(name,fit,x):
    if name=='L2': return float(math.sqrt(max(fit['a']+fit['b']*x,0)))
    if name=='P1': return float(math.exp(fit['a']+fit['b']*math.log(x)))
    return float(fit['a']+fit['b']*x)
def add_adjusted(df,name):
    fit=fit_model(name,df.mean_signal_DN,df.temporal_std_DN,df.temporal_variance_DN2)
    out=df.copy(); out['adjustment_model']=name; out['predicted_temporal_std_DN']=fit['prediction_std']; out['brightness_adjusted_temporal_residual_DN']=out.temporal_std_DN-out.predicted_temporal_std_DN
    sd=float(out.brightness_adjusted_temporal_residual_DN.std(ddof=1)); out['standardized_adjusted_residual']=out.brightness_adjusted_temporal_residual_DN/sd if sd else 0
    return out,fit
def matrices(df,features):
    return df[features].corr(method='spearman'),df[features].corr(method='pearson')
def vif_and_pca(df,features):
    z=(df[features]-df[features].mean())/df[features].std(ddof=1); rows=[]
    for f in features:
        others=[x for x in features if x!=f]; X=np.column_stack([np.ones(len(z)),z[others]]); y=z[f].to_numpy(); pred=X@np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2); rows.append({'analysis':'VIF','feature':f,'component':'','value':float(1/max(1-r2,1e-12)),'note':'descriptive only; n=10'})
    u,s,vt=np.linalg.svd(z.to_numpy(),full_matrices=False); ev=s*s/np.sum(s*s)
    for i,val in enumerate(ev): rows.append({'analysis':'PCA_VARIANCE','feature':'','component':f'PC{i+1}','value':float(val),'note':'auxiliary, not a physical condition'})
    for i in range(min(3,len(features))):
        for j,f in enumerate(features): rows.append({'analysis':'PCA_LOADING','feature':f,'component':f'PC{i+1}','value':float(vt[i,j]),'note':'auxiliary, sign arbitrary'})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output-root',required=True); args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]; cfg_path=(repo/args.config).resolve(); cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')); output=(repo/args.output_root).resolve()
    if output.exists(): raise FileExistsError(f'Output exists: {output}')
    if not str(output).lower().startswith(str(repo).lower()): raise RuntimeError('Output must remain inside repository')
    status_before=git(repo,'status','--porcelain=v1','--untracked-files=all'); commit=git(repo,'rev-parse','HEAD').strip(); started=now()
    for p in ['provenance','logs']: (output/p).mkdir(parents=True,exist_ok=False)
    root=Path(cfg['data_root']); count_before=sum(len(fs) for _,_,fs in os.walk(root)); protected=[]; protected_before={}; center_rows=[]; subset_rows=[]; roi_rows=[]
    formal=repo/cfg['formal_e1_root']; v=json.loads((formal/'verification_status.json').read_text(encoding='utf-8'))
    if v['status']!=cfg['formal_e1_expected_status'] or git(repo,'log','-1','--format=%H','--',str((formal/'verification_status.json').relative_to(repo))).strip()!=cfg['formal_e1_report_commit']: raise RuntimeError('Formal E1 provenance drift')
    sigma=float(cfg['highpass_sigma_px']); subsets=subset_indices(int(cfg['frame_count']))
    for folder in cfg['folders']:
        paths=list_tiffs(root/str(folder))[:int(cfg['frame_count'])]
        if len(paths)!=cfg['frame_count']: raise RuntimeError(f'folder {folder} frame count drift')
        protected.extend([paths[0],paths[-1],root/str(folder)/'PictureInfo.txt'])
        for p in protected[-3:]: protected_before[str(p)]={'sha256':sha256(p),'mtime_ns':p.stat().st_mtime_ns}
        with tifffile.TiffFile(paths[0]) as tf:
            if str(tf.pages[0].dtype)!=cfg['dtype_expected'] or list(tf.pages[0].shape)!=cfg['shape_expected']: raise RuntimeError(f'folder {folder} input drift')
        stack=read_roi_stack(paths,cfg['frozen_roi']); m=metrics(stack,sigma); m.update({'folder':folder,'subset':'full_200',**{f'roi_{k}':v for k,v in cfg['frozen_roi'].items()}}); center_rows.append(m)
        for name,idx in subsets.items():
            q=metrics(stack[idx],sigma); q.update({'folder':folder,'subset':name,'frame_indices_1based':';'.join(map(str,(idx+1).tolist())),**{f'roi_{k}':v for k,v in cfg['frozen_roi'].items()}}); subset_rows.append(q)
        del stack
        for name,roi in cfg['roi_sensitivity'].items():
            rs=read_roi_stack(paths,roi); q=metrics(rs,sigma); q.update({'folder':folder,'roi_name':name,**{f'roi_{k}':v for k,v in roi.items()}}); roi_rows.append(q); del rs
        print(f'folder={folder} complete',flush=True)
    center=pd.DataFrame(center_rows).sort_values('folder'); subsets_df=pd.DataFrame(subset_rows); rois=pd.DataFrame(roi_rows)
    spatial=pd.read_csv(formal/'spatial/spatial_correlation_summary.csv'); center=center.merge(spatial[['folder','psd_low_fraction','psd_mid_fraction','psd_high_fraction']],on='folder',how='left')
    fits,loo=fit_all(center,cfg['models']); selected=str(fits.iloc[0].model); adjusted,selected_fit=add_adjusted(center,selected)
    adjusted['residual_rank']=adjusted.brightness_adjusted_temporal_residual_DN.rank(method='average'); adjusted['residual_sign']=np.sign(adjusted.brightness_adjusted_temporal_residual_DN).astype(int)
    sub_parts=[]
    for name,g in subsets_df.groupby('subset',sort=False): sub_parts.append(add_adjusted(g.sort_values('folder'),selected)[0])
    subsets_df=pd.concat(sub_parts,ignore_index=True); subsets_df['adjusted_residual_rank_within_subset']=subsets_df.groupby('subset').brightness_adjusted_temporal_residual_DN.rank()
    roi_parts=[]
    for name,g in rois.groupby('roi_name',sort=False): roi_parts.append(add_adjusted(g.sort_values('folder'),selected)[0])
    rois=pd.concat(roi_parts,ignore_index=True); rois['adjusted_residual_rank_within_roi']=rois.groupby('roi_name').brightness_adjusted_temporal_residual_DN.rank()
    full_res=adjusted.set_index('folder').brightness_adjusted_temporal_residual_DN
    repeat=[]
    for folder,g in subsets_df.groupby('folder'):
        vals=g.brightness_adjusted_temporal_residual_DN.to_numpy(); full=float(full_res.loc[folder]); full_rank=float(adjusted.set_index('folder').loc[folder].residual_rank); repeat.append({'folder':folder,'subset_count':len(g),'temporal_std_cv':float(g.temporal_std_DN.std(ddof=1)/g.temporal_std_DN.mean()),'adjusted_residual_min_DN':float(vals.min()),'adjusted_residual_max_DN':float(vals.max()),'adjusted_residual_sign_consistency_vs_full':float(np.mean(np.sign(vals)==np.sign(full))),'adjusted_residual_rank_mean_absolute_deviation':float(np.mean(np.abs(g.adjusted_residual_rank_within_subset-full_rank))),'row_energy_cv':float(g.row_energy_DN.std(ddof=1)/g.row_energy_DN.mean()),'column_energy_cv':float(g.column_energy_DN.std(ddof=1)/g.column_energy_DN.mean()),'radial_acf_cv_abs':float(g.radial_acf_lag1.std(ddof=1)/max(abs(g.radial_acf_lag1.mean()),1e-12)),'stable_strength_cv':float(g.observed_stable_strength_DN.std(ddof=1)/g.observed_stable_strength_DN.mean()),'first_middle_last_mean_range_DN':float(g[g.subset.isin(['first_32','middle_32','last_32'])].mean_signal_DN.max()-g[g.subset.isin(['first_32','middle_32','last_32'])].mean_signal_DN.min())})
    repeat=pd.DataFrame(repeat)
    # Brightness-adjust row/column/stable independently for redundancy diagnostics.
    for col in ['row_energy_DN','column_energy_DN','observed_stable_strength_DN']:
        b,a=np.polyfit(center.mean_signal_DN,center[col],1); adjusted[col.replace('_DN','_brightness_adjusted_DN')]=center[col]-(a+b*center.mean_signal_DN)
    features=['mean_signal_DN','temporal_std_DN','brightness_adjusted_temporal_residual_DN','row_energy_DN','column_energy_DN','radial_acf_lag1','observed_stable_strength_DN','drift_percent']
    sp,pe=matrices(adjusted,features); sp.index.name='feature'; pe.index.name='feature'; red=vif_and_pca(adjusted,features)
    roi_summary=[]
    for folder,g in rois.groupby('folder'):
        full=float(full_res.loc[folder]); roi_summary.append({'folder':folder,'raw_temporal_std_roi_cv':float(g.temporal_std_DN.std(ddof=1)/g.temporal_std_DN.mean()),'adjusted_residual_roi_sign_consistency':float(np.mean(np.sign(g.brightness_adjusted_temporal_residual_DN)==np.sign(full))),'adjusted_residual_roi_range_DN':float(g.brightness_adjusted_temporal_residual_DN.max()-g.brightness_adjusted_temporal_residual_DN.min()),'radial_acf_roi_cv_abs':float(g.radial_acf_lag1.std(ddof=1)/max(abs(g.radial_acf_lag1.mean()),1e-12))})
    roi_summary=pd.DataFrame(roi_summary)
    # Candidate ranking is rule-based and uses no denoising outcomes.
    subset_rank_rho=float(np.median([rho(adjusted.set_index('folder').loc[g.folder].temporal_std_DN,g.temporal_std_DN) for _,g in subsets_df.groupby('subset')]))
    roi_rank_rho=float(np.median([rho(adjusted.set_index('folder').loc[g.folder].temporal_std_DN,g.temporal_std_DN) for _,g in rois.groupby('roi_name')]))
    signal_ok=bool(fits.iloc[0].loocv_rmse_std_DN/center.temporal_std_DN.mean()<0.25 and fits.iloc[0].loo_slope_sign_consistency==1 and subset_rank_rho>=0.9 and roi_rank_rho>=0.7)
    state_ok=bool((repeat.adjusted_residual_sign_consistency_vs_full>=6/7).sum()>=4 and (roi_summary.adjusted_residual_roi_sign_consistency>=.8).sum()>=4)
    acf_subset_rhos=[rho(adjusted.set_index('folder').loc[g.folder].radial_acf_lag1,g.radial_acf_lag1) for _,g in subsets_df.groupby('subset')]
    acf_roi_rhos=[rho(adjusted.set_index('folder').loc[g.folder].radial_acf_lag1,g.radial_acf_lag1) for _,g in rois.groupby('roi_name')]
    acf_aux=bool(abs(rho(center.mean_signal_DN,center.radial_acf_lag1))<.5 and np.median(acf_subset_rhos)>=.7 and np.median(acf_roi_rhos)>=.7)
    candidates=pd.DataFrame([
      {'rank':1,'candidate':'predicted_sigma_from_signal_model','role':'primary','reliable':signal_ok,'brightness_relation':'explicitly modeled','subset_rank_rho_median':subset_rank_rho,'roi_rank_rho_median':roi_rank_rho,'decision':'SELECT' if signal_ok else 'REJECT'},
      {'rank':2,'candidate':'brightness_adjusted_temporal_residual','role':'folder-state alternative','reliable':state_ok,'brightness_relation':'residualized','subset_rank_rho_median':float(np.median([rho(adjusted.set_index('folder').loc[g.folder].brightness_adjusted_temporal_residual_DN,g.brightness_adjusted_temporal_residual_DN) for _,g in subsets_df.groupby('subset')])),'roi_rank_rho_median':float(np.median([rho(adjusted.set_index('folder').loc[g.folder].brightness_adjusted_temporal_residual_DN,g.brightness_adjusted_temporal_residual_DN) for _,g in rois.groupby('roi_name')])),'decision':'AUXILIARY' if state_ok else 'REJECT'},
      {'rank':3,'candidate':'radial_acf_lag1','role':'optional auxiliary','reliable':acf_aux,'brightness_relation':f'rho={rho(center.mean_signal_DN,center.radial_acf_lag1):.6g}','subset_rank_rho_median':float(np.median(acf_subset_rhos)),'roi_rank_rho_median':float(np.median(acf_roi_rhos)),'decision':'AUXILIARY' if acf_aux else 'NOT-YET'},
      {'rank':4,'candidate':'row_column_stable','role':'excluded from CG main condition','reliable':False,'brightness_relation':'highly confounded/redundant','subset_rank_rho_median':float('nan'),'roi_rank_rho_median':float('nan'),'decision':'REJECT'}])
    split_rows=[]
    for label,key in [('A','split_a'),('B','split_b')]:
        for role in ['calibration','evaluation']:
            ids=cfg[key][role]; g=center[center.folder.isin(ids)]; split_rows.append({'split':label,'role':role,'folders':';'.join(map(str,ids)),'folder_count':len(ids),'signal_min_DN':g.mean_signal_DN.min(),'signal_max_DN':g.mean_signal_DN.max(),'temporal_std_min_DN':g.temporal_std_DN.min(),'temporal_std_max_DN':g.temporal_std_DN.max(),'row_min_DN':g.row_energy_DN.min(),'row_max_DN':g.row_energy_DN.max(),'column_min_DN':g.column_energy_DN.min(),'column_max_DN':g.column_energy_DN.max(),'acf_min':g.radial_acf_lag1.min(),'acf_max':g.radial_acf_lag1.max(),'stable_min_DN':g.observed_stable_strength_DN.min(),'stable_max_DN':g.observed_stable_strength_DN.max(),'acquisition_order_min':min(cfg['folders'].index(x)+1 for x in ids),'acquisition_order_max':max(cfg['folders'].index(x)+1 for x in ids),'selection_basis':'signal/noise stratification only' if label=='A' else 'acquisition-order interleaving with state coverage'})
    split_compare=pd.DataFrame(split_rows); primary=cfg['primary_split']; pkey='split_b' if primary=='B' else 'split_a'; split_manifest=[]
    for order,folder in enumerate(cfg['folders'],1):
        role='calibration' if folder in cfg[pkey]['calibration'] else 'evaluation'; split_manifest.append({'folder':folder,'acquisition_order':order,'role':role,'split_id':f'primary_split_{primary}','selection_basis':'pre-registered signal/noise and acquisition order; no denoising outcomes','allowed_parameter_use':'estimate G/CG parameters' if role=='calibration' else 'holdout controlled real evaluation only','historical_candidate_a_used_all_folders':True})
    split_manifest=pd.DataFrame(split_manifest)
    cg_ready=bool(signal_ok and set(cfg[pkey]['calibration']).isdisjoint(cfg[pkey]['evaluation']))
    final='OBSERVED-STATE-SEPARATION-VERIFIED-WITH-LIMITATIONS' if cg_ready else ('OBSERVED-STATE-SEPARATION-PARTIAL' if state_ok or signal_ok else 'OBSERVED-STATE-SEPARATION-NO-GO')
    outputs={'folder_signal_noise_table.csv':adjusted,'signal_noise_model_fits.csv':fits,'leave_one_folder_out_results.csv':loo,'brightness_adjusted_residuals.csv':adjusted[['folder','mean_signal_DN','temporal_std_DN','predicted_temporal_std_DN','brightness_adjusted_temporal_residual_DN','standardized_adjusted_residual','residual_rank','residual_sign']], 'folder_subset_repeatability.csv':subsets_df,'roi_sensitivity_analysis.csv':rois,'feature_correlation_spearman.csv':sp.reset_index(),'feature_correlation_pearson.csv':pe.reset_index(),'feature_redundancy_analysis.csv':red,'condition_candidate_ranking.csv':candidates,'split_candidate_comparison.csv':split_compare,'primary_folder_blocked_split.csv':split_manifest}
    for n,d in outputs.items(): d.to_csv(output/n,index=False,encoding='utf-8-sig')
    manifest=repo/'manifests/e1_primary_calibration_evaluation_split_20260718.csv'
    if manifest.exists():
        existing=pd.read_csv(manifest)
        if list(existing.columns)!=list(split_manifest.columns) or not existing.astype(str).equals(split_manifest.astype(str)):
            raise RuntimeError(f'Existing split manifest drift: {manifest}')
    else:
        split_manifest.to_csv(manifest,index=False,encoding='utf-8-sig')
    dump_json(output/'cg_readiness.json',{'CG_READY':cg_ready,'route':'signal-conditioned' if signal_ok else ('observed-state-conditioned' if state_ok else 'none'),'selected_model':selected,'signal_condition_supported':signal_ok,'folder_level_condition_supported':state_ok,'optional_auxiliary_condition':'radial_acf_lag1' if acf_aux else None,'primary_calibration_folders':cfg[pkey]['calibration'],'primary_evaluation_folders':cfg[pkey]['evaluation'],'historical_candidate_a_used_all_folders':True,'G_calibration_rule':'median formal-E1 temporal_std_mean over primary calibration folders only; fixed center ROI; no evaluation folders','limitations':['n=10 folders','physical conditions constant/unknown','signal and folder/scene remain confounded','historical Candidate A used all folders']})
    dump_json(output/'cgs_gap_update.json',{'CGS_READINESS':'NOT-YET','missing':['radial ACF subset/ROI evidence must be calibration-only confirmed','row/column brightness-adjusted independence','stable-map scene leakage audit','component-energy de-duplication','calibration-only structural parameter estimation']})
    commit_files=[Path(__file__),cfg_path,repo/'scripts/json_serialization.py']; input_files=[formal/'verification_status.json',formal/'noise_summary/folder_noise_summary.csv',formal/'spatial/spatial_correlation_summary.csv',formal/'stable_component/stable_component_summary.csv']
    (output/'provenance/git_commit.txt').write_text(commit+'\n',encoding='utf-8'); (output/'provenance/git_status_before.txt').write_text(status_before,encoding='utf-8'); (output/'provenance/git_diff.patch').write_text(git(repo,'diff','--binary','HEAD'),encoding='utf-8'); (output/'provenance/command.txt').write_text(subprocess.list2cmdline(sys.argv)+'\n',encoding='utf-8'); (output/'provenance/environment.txt').write_text(f'python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\nscipy={__import__("scipy").__version__}\ntifffile={tifffile.__version__}\n',encoding='utf-8'); (output/'provenance/resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    pd.DataFrame([{'path':str(p.relative_to(repo)),'sha256':sha256(p)} for p in commit_files]).to_csv(output/'provenance/script_hashes.csv',index=False,encoding='utf-8-sig'); pd.DataFrame([{'path':str(p),'sha256':sha256(p)} for p in input_files]).to_csv(output/'provenance/input_hashes.csv',index=False,encoding='utf-8-sig')
    protection=[]
    for p in dict.fromkeys(protected):
        before_hash=protected_before[str(p)]['sha256']; before_mtime=protected_before[str(p)]['mtime_ns']
        after_hash=sha256(p); after_mtime=p.stat().st_mtime_ns
        protection.append({'path':str(p),'sha256_before':before_hash,'sha256_after':after_hash,'mtime_ns_before':before_mtime,'mtime_ns_after':after_mtime,'unchanged':before_hash==after_hash and before_mtime==after_mtime})
    protection_df=pd.DataFrame(protection); protection_df.to_csv(output/'provenance/source_protection.csv',index=False,encoding='utf-8-sig'); count_after=sum(len(fs) for _,_,fs in os.walk(root)); source_ok=bool(count_before==count_after and protection_df.unchanged.all())
    report=f'''# E1 Observed-State Reliability And Separation Audit\n\nStatus: `{final}`\n\nThe audit used ten 200-frame folders, the frozen 512x512 center ROI, seven pre-registered temporal subsets, and five fixed ROI positions. No denoising result, PMRID data, synthetic pair, or model training was used.\n\nThe best LOOCV model was `{selected}` with RMSE {fits.iloc[0].loocv_rmse_std_DN:.4f} DN. Signal-conditioned strength is {'supported with limitations' if signal_ok else 'not sufficiently stable'}. Brightness-adjusted folder-state conditioning is {'supported' if state_ok else 'not supported as a formal folder condition'}. Folder 5 and other high-signal folders are retained and their LOO influence is reported.\n\nPrimary split `{primary}` freezes calibration folders `{cfg[pkey]['calibration']}` and evaluation folders `{cfg[pkey]['evaluation']}`. Historical Candidate A remains an all-folder historical baseline; future `G-calibration` must estimate sigma only from primary calibration folders.\n\n`CG_READY={str(cg_ready).lower()}`. `CGS_READINESS=NOT-YET`. The result authorizes mathematical freezing of a signal-conditioned CG only; it does not authorize training, synthetic generation, CGS, or physical-condition claims.\n'''
    (output/'verification_report.md').write_text(report,encoding='utf-8')
    run={'experiment_id':cfg['experiment_id'],'started_at_utc':started,'ended_at_utc':now(),'git_commit':commit,'status':final,'folder_count':10,'source_file_count_before':count_before,'source_file_count_after':count_after,'source_data_protected':source_ok,'synthetic_pairs_generated':False,'model_training_performed':False}
    required=list(outputs)+['cg_readiness.json','cgs_gap_update.json','verification_report.md','provenance/git_commit.txt','provenance/git_status_before.txt','provenance/git_diff.patch','provenance/command.txt','provenance/environment.txt','provenance/resolved_config.yaml','provenance/script_hashes.csv','provenance/input_hashes.csv','provenance/source_protection.csv']
    allowed_empty={'provenance/git_status_before.txt','provenance/git_diff.patch'}
    provenance_complete=all((output/p).is_file() and ((output/p).stat().st_size>0 or p in allowed_empty) for p in required)
    verification={'experiment_id':cfg['experiment_id'],'status':final,'CG_READY':cg_ready,'CGS_READINESS':'NOT-YET','selected_model':selected,'primary_split':primary,'provenance_complete':provenance_complete,'source_data_protected':source_ok,'model_training_performed':False,'synthetic_pairs_generated':False}
    if not provenance_complete or not source_ok:
        verification['status']='OBSERVED-STATE-SEPARATION-NO-GO'; run['status']=verification['status']
    dump_json(output/'provenance/run_manifest.json',run); dump_json(output/'verification_status.json',verification); (output/'logs/run.log').write_text(json.dumps(verification,indent=2),encoding='utf-8'); (output/'provenance/git_status_after.txt').write_text(git(repo,'status','--porcelain=v1','--untracked-files=all'),encoding='utf-8')
    hashes=[]
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name!='output_hashes.csv': hashes.append({'relative_path':str(p.relative_to(output)),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    hashes.append({'relative_path':'../manifests/e1_primary_calibration_evaluation_split_20260718.csv','size_bytes':manifest.stat().st_size,'sha256':sha256(manifest)})
    pd.DataFrame(hashes).to_csv(output/'output_hashes.csv',index=False,encoding='utf-8-sig')
    for p in [output/'verification_status.json',output/'cg_readiness.json',output/'cgs_gap_update.json',output/'provenance/run_manifest.json']: json.loads(p.read_text(encoding='utf-8'))
    print(json.dumps({'status':verification['status'],'CG_READY':cg_ready,'selected_model':selected,'primary_split':primary,'source_protected':source_ok},indent=2)); return 0 if verification['status']!='OBSERVED-STATE-SEPARATION-NO-GO' else 2
if __name__=='__main__': raise SystemExit(main())
