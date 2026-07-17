# E1 Formal Rerun Protocol

## Scope

This protocol characterizes operationally observed variability in ten repeated-frame
gated ICCD folders. It does not train a model, identify a unique physical noise
mechanism, or treat a temporal mean as clean ground truth.

## Legacy Script Audit

| Script | Input and defaults | Existing outputs | Formal limitation |
|---|---|---|---|
| `summarize_single_condition_noise.py` | Immediate TIFF folders; first 32 frames; 512 center crop; uint16 read then float32 raw DN | Folder summary CSV and Markdown | Uses `ddof=0`; labels spatial variation of the temporal mean as fixed; no provenance or anomaly gate |
| `fit_mean_variance_curve.py` | First 64 frames; 1024 center crop; 32 equal-width intensity bins | Bin CSV, summary CSV, plot, Markdown | Bins spatial pixels within one fixed scene; not a strict photon-transfer curve; no unified configuration |
| `evaluate_noise_robustness.py` | First 16/32/64/128 frames; 256/512/1024 center crops | Wide robustness CSV, plot, Markdown | Uses `ddof=1`, inconsistent with the legacy summary; overwrites named outputs; no provenance |
| `analyze_iccd_spatial_correlation.py` | First 64 frames; 512 center crop; radius 128 | Summary, radial PSD and radial autocorrelation CSVs | No directional PSD or 2D arrays; no temporal drift, repeatability, or row/column stability audit |

All legacy readers preserve uint16 values until conversion to float32 raw DN. None
normalizes to [0, 1] or quantizes to uint8. All use deterministic center crops.

## Formal Definitions

- Temporal variability: per-pixel sample variance across repeated frames (`ddof=1`).
- Fano-like statistic: mean temporal variance divided by mean raw DN. This is an
  operational statistic, not a photon-transfer measurement.
- Observed stable component: a repeatable high-pass component of independently
  averaged frame groups. No physical fixed-pattern attribution is made.
- Row/column structure: RMS row and column profiles of temporal-mean-subtracted
  residual frames.
- Spatial correlation: PSD and autocorrelation after per-pixel temporal-mean and
  per-frame residual-mean subtraction.

## Formal Entry Point

```powershell
python scripts\run_e1_formal_rerun.py --config configs\e1_formal_rerun_20260717.yaml
```

The runner refuses to overwrite an existing output directory and records source,
environment, input, command, log, and output hashes. A formal `VERIFIED-RUN`
requires a clean committed worktree at start and every automated verification gate
to pass.
