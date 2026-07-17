# E1 Verification Report

- Status: **VERIFIED-RUN**

## Checks

- PASS `required_csv:input_audit/input_manifest.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\input_audit\input_manifest.csv
- PASS `required_csv:input_audit/data_integrity_report.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\input_audit\data_integrity_report.csv
- PASS `required_csv:input_audit/frame_level_statistics.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\input_audit\frame_level_statistics.csv
- PASS `required_csv:noise_summary/folder_noise_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\noise_summary\folder_noise_summary.csv
- PASS `required_csv:mean_variance/mean_variance_bins.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\mean_variance\mean_variance_bins.csv
- PASS `required_csv:mean_variance/fano_like_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\mean_variance\fano_like_summary.csv
- PASS `required_csv:robustness/robustness_by_crop_and_frames.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\robustness\robustness_by_crop_and_frames.csv
- PASS `required_csv:temporal_stability/temporal_drift_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\temporal_stability\temporal_drift_summary.csv
- PASS `required_csv:stable_component/stable_component_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\stable_component\stable_component_summary.csv
- PASS `required_csv:row_column/row_column_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\row_column\row_column_summary.csv
- PASS `required_csv:spatial/spatial_correlation_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\spatial\spatial_correlation_summary.csv
- PASS `required_csv:combined/folder_eligibility_summary.csv`: E:\PNGAN-main\iccd-photonics-denoising\reports\e1_formal_rerun_20260717\combined\folder_eligibility_summary.csv
- PASS `all_configured_folders_present`: expected=[1, 2, 4, 5, 7, 8, 9, 10, 11, 13] observed=[1, 2, 4, 5, 7, 8, 9, 10, 11, 13]
- PASS `integrity_all_pass`: statuses=['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']
- PASS `dtype_uint16`: rows=2000
- PASS `shape_expected`: 5120x5120
- PASS `robustness_full_factorial`: expected=120 observed=120
- PASS `principal_metrics_finite`: 
- PASS `cross_output_recompute_consistent`: {"1": 0.0, "2": 0.0, "4": 0.0, "5": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "13": 0.0}
- PASS `provenance_complete`: git_commit.txt,git_status.txt,git_diff.patch,environment.txt,pip_freeze.txt,gpu_info.txt,script_hashes.csv,run_manifest.json,config.resolved.yaml,command.txt
- PASS `worktree_unchanged_by_run`: before='' after=''
- PASS `committed_clean_code`: formal VERIFIED-RUN requires a clean worktree at start
- PASS `input_manifest_hashed`: e3c361f12aa94e9b116f632c0564aac3b7842553da04a6eb6061fa7a1e721604

## Recomputed Folder Values

| folder | temporal std (summary) | temporal std (robustness) | relative difference |
|---:|---:|---:|---:|
| 1 | 68.3813705 | 68.3813705 | 0 |
| 2 | 43.5904198 | 43.5904198 | 0 |
| 4 | 111.481941 | 111.481941 | 0 |
| 5 | 228.860138 | 228.860138 | 0 |
| 7 | 148.059998 | 148.059998 | 0 |
| 8 | 134.920212 | 134.920212 | 0 |
| 9 | 87.5539322 | 87.5539322 | 0 |
| 10 | 71.4045486 | 71.4045486 | 0 |
| 11 | 58.2051048 | 58.2051048 | 0 |
| 13 | 38.5847206 | 38.5847206 | 0 |

All values above are recomputed from the cited bottom-level CSV files.
