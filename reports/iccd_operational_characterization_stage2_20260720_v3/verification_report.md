# Gated ICCD Stage-2 Operational Characterization

Status: `OPERATIONAL-CHARACTERIZATION-PAPER-READY-WITH-LIMITATIONS`

This package reconstructs E1 as a DN-domain, repeated-frame, EMVA-inspired operational characterization. It is not an EMVA 1288 compliance test, a standard photon-transfer curve, DSNU/PRNU measurement, or physical ICCD noise decomposition.

## Frozen boundaries

- Folders: 1, 2, 4, 5, 7, 8, 9, 10, 11, 13; 200 frames each.
- ROI: top=2304, left=2304, height=512, width=512.
- Input: raw uint16 values converted directly to float64 DN.
- `EXPOSURE_CONTROL_WIDTH=900 ms`; physical meaning unresolved. Sync A/B=4 us are metadata only.
- Calibration/evaluation roles are preserved; this run does not refit CG or perform training/inference.

## Core folder table

| Folder | Role | Mean DN | Temporal std DN | Direct var DN2 | H/V/R ACF lag1 | NPS L/M/H | Row/column DN |
|---:|---|---:|---:|---:|---|---|---|
| 1 | calibration | 1157.400 | 70.780 | 5984.686 | 0.1750/0.1697/0.1646 | 0.114/0.113/0.772 | 12.087/5.890 |
| 2 | evaluation | 964.958 | 43.754 | 2028.643 | 0.0611/0.0476/0.0409 | 0.025/0.117/0.859 | 6.784/2.115 |
| 4 | calibration | 1782.133 | 112.622 | 16937.466 | 0.0544/0.0529/0.0509 | 0.055/0.103/0.842 | 9.838/12.346 |
| 5 | evaluation | 4690.910 | 236.518 | 78390.621 | 0.1497/0.1507/0.1488 | 0.160/0.093/0.747 | 23.841/47.632 |
| 7 | calibration | 2509.448 | 150.114 | 30508.240 | 0.0709/0.0705/0.0686 | 0.069/0.099/0.832 | 12.692/23.300 |
| 8 | calibration | 2240.655 | 136.026 | 24850.511 | 0.0460/0.0445/0.0428 | 0.043/0.102/0.854 | 11.329/15.471 |
| 9 | evaluation | 1399.160 | 87.798 | 9785.797 | 0.0294/0.0266/0.0243 | 0.024/0.105/0.871 | 8.101/6.319 |
| 10 | calibration | 1201.242 | 71.521 | 6199.766 | 0.0335/0.0290/0.0253 | 0.021/0.108/0.871 | 7.459/4.284 |
| 11 | evaluation | 1080.441 | 58.279 | 3825.083 | 0.0548/0.0477/0.0400 | 0.024/0.118/0.858 | 7.053/3.322 |
| 13 | calibration | 936.233 | 38.642 | 1590.974 | 0.0571/0.0391/0.0341 | 0.022/0.115/0.862 | 6.671/1.788 |

## Interpretation

The component-specific convergence rule recommends 200 frames for the complete temporal-map, ACF, NPS and directional package. The full 200-frame sequence is sufficient for this bounded operational description, but positional nonstationarity is retained and a stationary-population claim is not supported.

Difference-frame estimates are reported beside direct temporal variance. Departures are interpreted together with temporal correlation and drift, not silently forced to agree. Covariance and NPS use temporal residuals after per-frame DC removal; the mean image is never used as the formal noise spectrum.

The repeatable observed stable component retains the historical split/high-pass definition and remains scene-confounded. The observed-signal relation remains operational; observed DN is not exposure, irradiance, or photon count.

## Readiness

- data and claim boundary: `READY` - DN-domain repeated-frame characterization; no standard PTC/DSNU/PRNU claim
- temporal noise: `READY` - pixelwise ddof=1 statistics and historical comparison
- difference-frame noise: `READY-WITH-LIMITATIONS` - adjacent and non-overlapping pair estimates; temporal correlation noted
- frame convergence: `READY-WITH-LIMITATIONS` - component-specific first/middle/last/random convergence against internal N=200 reference
- drift: `READY-WITH-LIMITATIONS` - historical frame-level DC drift retained; physical time scale unavailable
- row/column structure: `READY-WITH-LIMITATIONS` - temporal-residual profile RMS; not DSNU
- 2D covariance: `READY` - frame-mean-centered temporal residual, exact non-circular lags
- ACF: `READY` - normalized temporal-residual covariance in pixel lags
- NPS: `READY-WITH-LIMITATIONS` - cycles/pixel, 2D Hann, no pixel-pitch conversion
- stable component: `READY-WITH-LIMITATIONS` - repeatable observed stable component with scene-leakage caveat
- observed-signal dependence: `READY-WITH-LIMITATIONS` - observed signal level is not exposure or irradiance
- noise-model parameter support: `READY-WITH-LIMITATIONS` - frozen calibration-only CG slope retained, not refit

## Claim boundary

Supported: folder-level temporal variability, difference-frame comparison, convergence, temporal-residual directional energy, covariance/ACF/NPS, split-repeatable observed stable structure, and observed-signal dependence at the frozen ROI.

Not supported: standard PTC, photon/conversion gain, dark current, DSNU, PRNU, pure FPN, physical gate-conditioned behavior, or unique physical noise-component separation.

Warnings: `none`.
