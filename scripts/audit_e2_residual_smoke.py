"""Compute pre-save, round-trip, and E1 comparison metrics for one smoke pair."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, stats


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64)
    return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else float("nan")


def gradient_energy(image: np.ndarray) -> float:
    gy,gx=np.gradient(image.astype(np.float64)); return float(np.mean(gx*gx+gy*gy))


def audit(cfg: dict[str, Any], output: Path, built: dict[str, Any]) -> dict[str, Any]:
    content=built["content"]; residual=built["residual"]; unclipped=built["noisy_unclipped"]; noisy=built["noisy"]
    divisor=float(cfg["content_preprocessing"]["normalization_divisor"]); target=built["sigma_norm"]
    neg=float(np.mean(unclipped<0)); high=float(np.mean(unclipped>1))
    added_zero=float(np.mean((noisy==0)&(content>0))); added_one=float(np.mean((noisy==1)&(content<1)))
    residual_dn=residual.astype(np.float64)*divisor
    highpass=residual_dn-ndimage.gaussian_filter(residual_dn,1.0)
    pre={
      "content_mean_norm":float(content.mean()),"content_std_norm":float(content.std()),
      "content_p1_norm":float(np.percentile(content,1)),"content_p50_norm":float(np.percentile(content,50)),"content_p99_norm":float(np.percentile(content,99)),
      "residual_mean_norm":float(residual.mean()),"residual_mean_dn":float(residual_dn.mean()),"residual_std_norm":float(residual.std()),"residual_std_dn":float(residual_dn.std()),
      "residual_std_relative_error":abs(float(residual.std())-target)/target,"residual_skewness":float(stats.skew(residual.ravel())),"residual_excess_kurtosis":float(stats.kurtosis(residual.ravel())),
      "noisy_unclipped_min":float(unclipped.min()),"noisy_unclipped_max":float(unclipped.max()),
      "negative_before_clipping_ratio":neg,"above_one_before_clipping_ratio":high,"added_zero_clipping_ratio":added_zero,"added_one_clipping_ratio":added_one,
      "brightness_difference_norm":float(noisy.mean()-content.mean()),"brightness_difference_dn":float((noisy.mean()-content.mean())*divisor),
      "gradient_energy_ratio":gradient_energy(noisy)/gradient_energy(content),
      "residual_row_energy_dn":float(residual_dn.mean(axis=1).std()),"residual_column_energy_dn":float(residual_dn.mean(axis=0).std()),
      "residual_horizontal_lag1_correlation":corr(residual_dn[:,:-1],residual_dn[:,1:]),"residual_vertical_lag1_correlation":corr(residual_dn[:-1,:],residual_dn[1:,:]),
      "residual_high_frequency_std_dn":float(highpass.std()),
    }
    pd.DataFrame([pre]).to_csv(output/"metrics/pre_save_metrics.csv",index=False,encoding="utf-8-sig")
    content_rt=tifffile.imread(output/"tiff/content_uint16.tiff"); noisy_rt=tifffile.imread(output/"tiff/noisy_uint16.tiff")
    content_rt_f=content_rt.astype(np.float32)/divisor; noisy_rt_f=noisy_rt.astype(np.float32)/divisor
    clipped_residual=noisy-content; reconstructed=noisy_rt_f-content_rt_f
    rt={
      "content_dtype":str(content_rt.dtype),"noisy_dtype":str(noisy_rt.dtype),"content_shape":"x".join(map(str,content_rt.shape)),"noisy_shape":"x".join(map(str,noisy_rt.shape)),
      "content_max_abs_error_dn":float(np.max(np.abs(content_rt_f-content))*divisor),"noisy_max_abs_error_dn":float(np.max(np.abs(noisy_rt_f-noisy))*divisor),
      "residual_reconstruction_max_abs_error_dn":float(np.max(np.abs(reconstructed-clipped_residual))*divisor),
      "residual_reconstruction_mean_error_dn":float((reconstructed-clipped_residual).mean()*divisor),
      "residual_reconstruction_std_relative_error":abs(float(reconstructed.std())-float(clipped_residual.std()))/float(clipped_residual.std()),
      "zero_ratio_change":float(np.mean(noisy_rt_f==0)-np.mean(noisy==0)),"one_ratio_change":float(np.mean(noisy_rt_f==1)-np.mean(noisy==1)),
      "tiff_compression":"NONE","readable":True,
    }
    pd.DataFrame([rt]).to_csv(output/"metrics/round_trip_metrics.csv",index=False,encoding="utf-8-sig")
    comparison=[{"metric":"temporal_std_dn","e1_target":built["sigma_dn"],"synthetic_actual":pre["residual_std_dn"],"relative_error":pre["residual_std_relative_error"],"passed":pre["residual_std_relative_error"]<cfg["gates"]["residual_std_relative_error_max"]}]
    pd.DataFrame(comparison).to_csv(output/"metrics/e1_target_comparison.csv",index=False,encoding="utf-8-sig")
    unmatched=[
      ("row_energy","intentionally_unmatched"),("column_energy","intentionally_unmatched"),("radial_psd","intentionally_unmatched"),("spatial_autocorrelation","intentionally_unmatched"),
      ("repeatable_observed_stable_component","intentionally_unmatched"),("fano_like","intentionally_unmatched"),("signal_dependence","intentionally_unmatched")]
    pd.DataFrame([{"metric":m,"status":s,"reason":"Candidate A is the preregistered strength-only baseline"} for m,s in unmatched]).to_csv(output/"metrics/known_unmatched_statistics.csv",index=False,encoding="utf-8-sig")
    gates=cfg["gates"]
    checks={
      "residual_std":pre["residual_std_relative_error"]<gates["residual_std_relative_error_max"],"negative_clipping":neg<gates["negative_before_clipping_ratio_max"],"above_one_clipping":high<gates["above_one_before_clipping_ratio_max"],
      "added_zero_clipping":added_zero<gates["added_zero_clipping_ratio_max"],"added_one_clipping":added_one<gates["added_one_clipping_ratio_max"],
      "content_round_trip":rt["content_max_abs_error_dn"]<=gates["uint16_max_error_dn"],"noisy_round_trip":rt["noisy_max_abs_error_dn"]<=gates["uint16_max_error_dn"],
      "residual_round_trip":rt["residual_reconstruction_max_abs_error_dn"]<=gates["residual_reconstruction_max_error_dn"],"std_round_trip":rt["residual_reconstruction_std_relative_error"]<gates["round_trip_std_relative_error_max"],
      "uint16_shape":rt["content_dtype"]=="uint16" and rt["noisy_dtype"]=="uint16" and rt["content_shape"]=="512x512" and rt["noisy_shape"]=="512x512",
    }
    return {"pre":pre,"round_trip":rt,"checks":checks}

