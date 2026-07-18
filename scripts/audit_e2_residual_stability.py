"""Metrics and aggregation for the fixed E1-strength Gaussian stability audit."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage, stats


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64)
    return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else float("nan")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if np.ptp(a)==0 or np.ptp(b)==0:
        return float("nan")
    return float(stats.spearmanr(a,b).statistic)


def gradient_energy(image: np.ndarray) -> float:
    gy,gx=np.gradient(image.astype(np.float64)); return float(np.mean(gx*gx+gy*gy))


def high_frequency_energy(image: np.ndarray) -> float:
    hp=image.astype(np.float64)-ndimage.gaussian_filter(image.astype(np.float64),1.0)
    return float(np.mean(hp*hp))


def radial_acf_lag1(residual: np.ndarray) -> float:
    centered=residual.astype(np.float64)-float(residual.mean()); h,w=centered.shape
    ac=np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(centered))**2).real)
    ac/=ac[h//2,w//2]
    yy,xx=np.indices(ac.shape); radius=np.sqrt((yy-h//2)**2+(xx-w//2)**2)
    return float(ac[(radius>=0.5)&(radius<1.5)].mean())


def psd_flatness(residual: np.ndarray) -> float:
    centered=residual.astype(np.float64)-float(residual.mean())
    power=np.abs(np.fft.fft2(centered))**2
    values=power.ravel()[1:]
    return float(np.exp(np.mean(np.log(values+np.finfo(float).tiny)))/np.mean(values))


def tiff_round_trip(content: np.ndarray, noisy: np.ndarray, divisor: float) -> tuple[dict[str, Any],np.ndarray,np.ndarray]:
    content_scaled=(content*np.float32(divisor)).astype(np.float32); noisy_scaled=(noisy*np.float32(divisor)).astype(np.float32)
    content_u16=np.rint(content_scaled).astype(np.uint16); noisy_u16=np.rint(noisy_scaled).astype(np.uint16)
    content_buffer=io.BytesIO(); noisy_buffer=io.BytesIO()
    tifffile.imwrite(content_buffer,content_u16,compression=None); tifffile.imwrite(noisy_buffer,noisy_u16,compression=None)
    content_buffer.seek(0); noisy_buffer.seek(0)
    content_rt=tifffile.imread(content_buffer); noisy_rt=tifffile.imread(noisy_buffer)
    content_f=content_rt.astype(np.float32)/divisor; noisy_f=noisy_rt.astype(np.float32)/divisor
    clipped_residual_dn=noisy_scaled.astype(np.float64)-content_scaled.astype(np.float64)
    reconstructed_dn=noisy_rt.astype(np.float64)-content_rt.astype(np.float64)
    metrics={
      "content_round_trip_max_error_dn":float(np.max(np.abs(content_rt.astype(np.float64)-content_scaled.astype(np.float64)))),
      "noisy_round_trip_max_error_dn":float(np.max(np.abs(noisy_rt.astype(np.float64)-noisy_scaled.astype(np.float64)))),
      "residual_reconstruction_max_error_dn":float(np.max(np.abs(reconstructed_dn-clipped_residual_dn))),
      "residual_reconstruction_std_relative_error":abs(float(reconstructed_dn.std())-float(clipped_residual_dn.std()))/float(clipped_residual_dn.std()),
      "residual_reconstruction_mean_error_dn":float((reconstructed_dn-clipped_residual_dn).mean()),
      "round_trip_zero_ratio_change":float(np.mean(noisy_f==0)-np.mean(noisy==0)),
      "round_trip_one_ratio_change":float(np.mean(noisy_f==1)-np.mean(noisy==1)),
      "round_trip_dtype":str(noisy_rt.dtype),"round_trip_shape":"x".join(map(str,noisy_rt.shape)),"tiff_compression":"NONE","tiff_readable":True,
    }
    return metrics,content_u16,noisy_u16


def pair_metrics(content: np.ndarray, residual: np.ndarray, sigma_dn: float, divisor: float) -> tuple[dict[str,Any],np.ndarray,np.ndarray,np.ndarray]:
    residual_dn=residual.astype(np.float64)*divisor; unclipped=(content+residual).astype(np.float32); noisy=np.clip(unclipped,0,1).astype(np.float32)
    content_grad=gradient_energy(content); content_hf=high_frequency_energy(content)
    noisy_grad=gradient_energy(noisy); noisy_hf=high_frequency_energy(noisy)
    rt,content_u16,noisy_u16=tiff_round_trip(content,noisy,divisor)
    row_content=content.mean(axis=1); col_content=content.mean(axis=0); row_noisy=noisy.mean(axis=1); col_noisy=noisy.mean(axis=0)
    values={
      "content_mean_norm":float(content.mean()),"content_std_norm":float(content.std()),"content_p1_norm":float(np.percentile(content,1)),"content_p50_norm":float(np.percentile(content,50)),"content_p99_norm":float(np.percentile(content,99)),
      "content_robust_range_norm":float(np.percentile(content,99)-np.percentile(content,1)),"content_gradient_energy":content_grad,"content_high_frequency_energy":content_hf,"content_row_profile_std":float(row_content.std()),"content_column_profile_std":float(col_content.std()),
      "residual_mean_dn":float(residual_dn.mean()),"residual_std_dn":float(residual_dn.std()),"residual_std_relative_error":abs(float(residual_dn.std())-sigma_dn)/sigma_dn,
      "residual_skewness":float(stats.skew(residual.ravel())),"residual_excess_kurtosis":float(stats.kurtosis(residual.ravel())),"residual_min_dn":float(residual_dn.min()),"residual_max_dn":float(residual_dn.max()),
      "residual_p1_dn":float(np.percentile(residual_dn,1)),"residual_p50_dn":float(np.percentile(residual_dn,50)),"residual_p99_dn":float(np.percentile(residual_dn,99)),
      "residual_horizontal_lag1":correlation(residual[:,:-1],residual[:,1:]),"residual_vertical_lag1":correlation(residual[:-1,:],residual[1:,:]),"residual_radial_autocorrelation_lag1":radial_acf_lag1(residual),
      "residual_row_energy_dn":float(residual_dn.mean(axis=1).std()),"residual_column_energy_dn":float(residual_dn.mean(axis=0).std()),"residual_psd_flatness":psd_flatness(residual),
      "noisy_mean_norm":float(noisy.mean()),"noisy_std_norm":float(noisy.std()),"noisy_p1_norm":float(np.percentile(noisy,1)),"noisy_p50_norm":float(np.percentile(noisy,50)),"noisy_p99_norm":float(np.percentile(noisy,99)),
      "brightness_difference_dn":float((noisy.mean()-content.mean())*divisor),"brightness_difference_norm":float(noisy.mean()-content.mean()),"gradient_energy_ratio":noisy_grad/content_grad,"high_frequency_energy_ratio":noisy_hf/content_hf,
      "row_profile_std_change":float(row_noisy.std()-row_content.std()),"column_profile_std_change":float(col_noisy.std()-col_content.std()),
      "negative_before_clipping_ratio":float(np.mean(unclipped<0)),"above_one_before_clipping_ratio":float(np.mean(unclipped>1)),"added_zero_clipping_ratio":float(np.mean((noisy==0)&(content>0))),"added_one_clipping_ratio":float(np.mean((noisy==1)&(content<1))),
      "source_zero_ratio":float(np.mean(content==0)),"source_saturation_ratio":float(np.mean(content==1)),
    }
    values.update(rt)
    return values,noisy,content_u16,noisy_u16


def select_contents(manifest: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    ordered=manifest.sort_values(["roi_mean_dn","sha256"],kind="stable").reset_index(drop=True); used=set(); rows=[]
    for rank,q in enumerate(quantiles,1):
        target=float(ordered.roi_mean_dn.quantile(q,interpolation="linear"))
        candidates=ordered.assign(_distance=(ordered.roi_mean_dn-target).abs()).sort_values(["_distance","sha256"],kind="stable")
        selected=candidates[~candidates.content_id.isin(used)].iloc[0]; used.add(selected.content_id)
        row=selected.to_dict(); row.update({"selection_rank":rank,"target_quantile":q,"target_roi_mean_dn":target,"selection_distance_dn":float(selected._distance),"selection_reason":"nearest roi_mean_dn to preregistered quantile; SHA256 lexicographic tie-breaker"})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(pair_df: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str,Any]) -> dict[str,pd.DataFrame|dict[str,Any]]:
    content_rows=[]
    for content_id,group in pair_df.groupby("content_id",sort=False):
        content_rows.append({"content_id":content_id,"seed_count":len(group),"roi_mean_dn":float(group.content_mean_norm.iloc[0]*cfg["normalization_divisor"]),"roi_std_dn":float(group.content_std_norm.iloc[0]*cfg["normalization_divisor"]),
          "residual_std_mean_dn":group.residual_std_dn.mean(),"residual_std_sd_dn":group.residual_std_dn.std(ddof=1),"residual_std_min_dn":group.residual_std_dn.min(),"residual_std_max_dn":group.residual_std_dn.max(),"residual_std_mean_relative_error":abs(group.residual_std_dn.mean()-cfg["sigma_dn"])/cfg["sigma_dn"],
          "residual_mean_mean_dn":group.residual_mean_dn.mean(),"residual_mean_min_dn":group.residual_mean_dn.min(),"residual_mean_max_dn":group.residual_mean_dn.max(),"absolute_brightness_difference_mean_dn":group.brightness_difference_dn.abs().mean(),"brightness_difference_max_abs_dn":group.brightness_difference_dn.abs().max(),
          "negative_clipping_max":group.negative_before_clipping_ratio.max(),"above_one_clipping_max":group.above_one_before_clipping_ratio.max(),"added_zero_clipping_max":group.added_zero_clipping_ratio.max(),"added_one_clipping_max":group.added_one_clipping_ratio.max(),
          "gradient_ratio_mean":group.gradient_energy_ratio.mean(),"gradient_ratio_min":group.gradient_energy_ratio.min(),"gradient_ratio_max":group.gradient_energy_ratio.max(),"round_trip_max_error_dn":group.noisy_round_trip_max_error_dn.max(),"seed_anomaly":False})
    content=pd.DataFrame(content_rows)
    seed_rows=[]
    for seed,group in pair_df.groupby("seed"):
        seed_rows.append({"seed":seed,"content_count":len(group),"residual_std_mean_dn":group.residual_std_dn.mean(),"residual_std_relative_error":abs(group.residual_std_dn.mean()-cfg["sigma_dn"])/cfg["sigma_dn"],"residual_mean_mean_dn":group.residual_mean_dn.mean(),"brightness_difference_mean_dn":group.brightness_difference_dn.mean(),"brightness_difference_max_abs_dn":group.brightness_difference_dn.abs().max(),"max_clipping":group[["negative_before_clipping_ratio","above_one_before_clipping_ratio","added_zero_clipping_ratio","added_one_clipping_ratio"]].max().max(),"systematic_anomaly":False})
    seed_summary=pd.DataFrame(seed_rows)
    rho_mean=safe_spearman(content.residual_std_mean_dn,content.roi_mean_dn)
    rho_std=safe_spearman(content.residual_std_mean_dn,content.roi_std_dn)
    grouped=pair_df.groupby("content_id")
    clipping_by_content=grouped.negative_before_clipping_ratio.max().reindex(content.content_id).to_numpy()
    brightness_by_content=grouped.brightness_difference_dn.mean().reindex(content.content_id).to_numpy()
    selected_by_content=selected.set_index("content_id").reindex(content.content_id)
    correlation_rows=[
      {"analysis":"residual_std_vs_content_mean","spearman_rho":rho_mean,"n_content":len(content),"inference":"descriptive_only_single_unknown_source_group"},
      {"analysis":"residual_std_vs_content_std","spearman_rho":rho_std,"n_content":len(content),"inference":"descriptive_only_single_unknown_source_group"},
      {"analysis":"max_clipping_vs_content_p1","spearman_rho":safe_spearman(clipping_by_content,selected_by_content.roi_p1_dn),"n_content":len(content),"inference":"descriptive_only_single_unknown_source_group"},
      {"analysis":"brightness_vs_content_mean","spearman_rho":safe_spearman(brightness_by_content,content.roi_mean_dn),"n_content":len(content),"inference":"descriptive_only_single_unknown_source_group"},
      {"analysis":"gradient_ratio_vs_content_std","spearman_rho":safe_spearman(content.gradient_ratio_mean,content.roi_std_dn),"n_content":len(content),"inference":"descriptive_only_single_unknown_source_group"},]
    cv=float(content.residual_std_mean_dn.std(ddof=1)/content.residual_std_mean_dn.mean())
    cross={"content_count":len(content),"pair_count":len(pair_df),"residual_std_mean_dn":pair_df.residual_std_dn.mean(),"residual_std_min_dn":pair_df.residual_std_dn.min(),"residual_std_max_dn":pair_df.residual_std_dn.max(),"pair_residual_std_relative_error_min":pair_df.residual_std_relative_error.min(),"pair_residual_std_relative_error_max":pair_df.residual_std_relative_error.max(),"pair_residual_std_relative_error_median":pair_df.residual_std_relative_error.median(),"content_level_residual_std_mean_cv":cv,
      "worst_negative_clipping":pair_df.negative_before_clipping_ratio.max(),"worst_above_one_clipping":pair_df.above_one_before_clipping_ratio.max(),"worst_added_zero_clipping":pair_df.added_zero_clipping_ratio.max(),"worst_added_one_clipping":pair_df.added_one_clipping_ratio.max(),
      "brightness_difference_mean_dn":pair_df.brightness_difference_dn.mean(),"absolute_brightness_difference_mean_dn":pair_df.brightness_difference_dn.abs().mean(),"brightness_difference_max_abs_dn":pair_df.brightness_difference_dn.abs().max(),"gradient_ratio_min":pair_df.gradient_energy_ratio.min(),"gradient_ratio_max":pair_df.gradient_energy_ratio.max(),"noisy_round_trip_worst_dn":pair_df.noisy_round_trip_max_error_dn.max(),"residual_reconstruction_worst_dn":pair_df.residual_reconstruction_max_error_dn.max(),"residual_std_vs_content_mean_spearman":rho_mean,"residual_std_vs_content_std_spearman":rho_std}
    return {"content":content,"seed":seed_summary,"correlations":pd.DataFrame(correlation_rows),"cross":cross}


def save_representative(output: Path, role: str, content_id: str, seed: int, content: np.ndarray, residual: np.ndarray, noisy_unclipped: np.ndarray, noisy: np.ndarray, content_u16: np.ndarray, noisy_u16: np.ndarray, sigma_norm: float) -> None:
    stem=f"{role}_{content_id}_seed{seed}"; arrays=output/"float_arrays"/stem; tiffs=output/"tiff"/stem; previews=output/"previews"/stem
    arrays.mkdir(parents=True); tiffs.mkdir(parents=True); previews.mkdir(parents=True)
    np.save(arrays/"content_float.npy",content); np.save(arrays/"residual_float.npy",residual); np.save(arrays/"noisy_unclipped_float.npy",noisy_unclipped)
    tifffile.imwrite(tiffs/"content_uint16.tiff",content_u16,compression=None); tifffile.imwrite(tiffs/"noisy_uint16.tiff",noisy_u16,compression=None)
    Image.fromarray((content_u16/257).astype(np.uint8)).save(previews/"content_preview.png"); Image.fromarray((noisy_u16/257).astype(np.uint8)).save(previews/"noisy_preview.png")
    Image.fromarray(np.clip(127.5+residual/max(sigma_norm,1e-12)*24,0,255).astype(np.uint8)).save(previews/"residual_preview.png")
