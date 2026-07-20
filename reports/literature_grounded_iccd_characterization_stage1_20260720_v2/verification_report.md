# Literature-Grounded Gated ICCD Characterization Stage 1

- Final status: **LITERATURE-GROUNDED-FRAMEWORK-READY-WITH-METADATA-GAPS**
- References: **23** (standards 3, detector references/papers 15, restoration papers 5)
- Methods assessed: **32**
- Eligibility: {"EXPLORATORY-ONLY": 5, "FORMALLY-ELIGIBLE": 9, "NOT-ELIGIBLE": 1, "PARTIALLY-ELIGIBLE": 6, "REQUIRES-NEW-ACQUISITION": 11}
- Recommended framework: **EMVA-inspired layered gated ICCD operational noise characterization**
- EMVA 1288 compliant: **no**

## Decision

The current project should use a layered, EMVA-inspired operational framework rather than a strict PTC claim or an unstructured list of E1 metrics. Formal results can be based on the frozen repeated ICCD folders: pixelwise temporal variability, difference-frame operational noise, frame-count convergence, split repeatability, drift, temporal-residual covariance/ACF/NPS, and directional residual energy. Physical labels that require flats, matched darks, calibrated irradiance, verified exposure, temperature, gain or photon-counting mode remain unavailable.

The existing E1 values remain valid within their frozen ROI and operational definitions. They require reorganization and terminology control, not replacement: `Fano-like statistic in DN`, `repeatable observed stable component`, `row/column profile energy of temporal residual`, and `temporal-residual spatial ACF/NPS`. The observed-signal-conditioned model retains an operational basis but is not a photon-transfer, conversion-gain or physical gate model.

## Data interpretation

- `D:/iccd/data/20260319`: the only primary gated ICCD repeated-frame asset. The recorded 900 ms value is an exposure-control width, not a verified gate width.
- `D:/PMRID4/data` and historical `F:/目标传感器噪声参数估计/data`: sCMOS/recovery assets with 1 ms through 1 s directory labels. They are not ICCD exposure-response data.
- `dark_Background`: an sCMOS dark candidate with unmatched settings. The derived dark offset is an excluded untraced artifact.
- `F:/ICCD_pir/...`: an auxiliary 8-bit background candidate from another acquisition context; not a matching dark for the main 16-bit ICCD batch.
- No verified matching ICCD 50 ms/1 s dark sequence was established in this stage.

## Paper readiness

`GENERAL_PAPER_DRAFT_SUPPORTED = true` for a reproducible operational characterization and controlled restoration study. `ACTA_PHOTONICA_SINICA_DRAFT_SUPPORTED = true_with_metadata_gaps`: a draft is supportable if it explicitly avoids EMVA compliance, strict PTC, physical gain, DSNU/PRNU and photon-counting claims. P0 terminology/metadata closure and selected P1 residual analyses block a strong final submission, while formal PTC/DSNU/PRNU and MCP/gate physics require new controlled acquisition and do not block an operational draft.

## Proposed section structure

1. Acquisition evidence, data roles, ROI and claim boundary.
2. Temporal noise and stability from repeated frames.
3. Spatial nonuniformity proxies, covariance and temporal-residual spectra.
4. Verified exposure response or, when unavailable, observed signal/noise relation.
5. Calibration-only parameters used by G/CG and controlled holdout validation.

Core tables: acquisition/metadata table; method-eligibility table; folder-level temporal/spatial statistics; calibration/evaluation role table. Core figures: temporal std maps/distributions and convergence; difference-frame noise versus observed signal; covariance/ACF and 2D/radial temporal-residual NPS; split-stable and drift diagnostics; G/CG controlled validation with explicit tradeoffs.

## Scope and safety

No image was opened for new pixel statistics, no stage-2 analysis was executed, no model was trained or inferred, and no source file was modified. F-volume direct access was unavailable in this session; its inventory entries are sourced from hashed prior formal audit artifacts and are marked accordingly.
