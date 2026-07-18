# E1 Scientific Completeness And Condition Readiness Audit

Status: `E1-PARTIAL-FOR-CONDITION-MODELING`

E1 contains ten complete 200-frame uint16 folders at 5120x5120. All formal statistics use the frozen center ROI `(top=2304, left=2304, height=512, width=512)`. Folder-level temporal, row/column, spatial, stable-map and drift outputs are present and reproducible.

Per-frame `PictureInfo.txt` records are identical across all ten folders: Exposure channel width 900 ms, Sync.A/Sync.B width 4 us and gain 60, with camera serial 20600555 in filenames. These records verify constant control values, not varying physical conditions. Mapping Sync width to the intensifier gate and gain 60 to calibrated MCP gain is not established; sensor exposure is conflicting with a non-snapshotted `Format.ini` value of 300.

Temporal standard deviation spans 38.585-228.860 DN (5.93x) and is stable across 16/32/64/128-frame estimates (folder CV 0.67-2.11%). However its Spearman correlation with folder mean signal is 1.000. Row, column and observed stable strength are also strongly correlated with brightness and each other. Radial ACF lag-1 is less brightness-confounded, but its frame-subset repeatability was not measured.

All ten folders were used in E1 and Candidate A strength estimation and also appear in real-surrogate evaluation. A future CG therefore requires a pre-registered folder-blocked calibration/evaluation boundary and must not choose roles or thresholds from denoising outcomes.

`CG_READY=false`: repeatable observed strength states exist, but physical conditions do not vary, brightness/scene confounding is unresolved, and calibration/evaluation separation is not frozen.

`CGS_READY=false`: folder-level structural statistics exist, but spatial subset repeatability, stable-component scene separation and non-overlapping component energy accounting are missing.

The sole next task is the priority-1 existing-data audit in `e1_gap_list.csv`: pre-register the folder boundary, then test brightness-adjusted frame-subset reliability and redundancy of the observed noise-state vector. This does not require reacquisition and does not authorize training or CG implementation.
