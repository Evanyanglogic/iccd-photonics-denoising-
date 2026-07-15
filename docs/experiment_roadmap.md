# ICCD Experiment Roadmap

## Material Passport

- Origin Skills: `iccd-denoising-optimizer`, `academic-research-suite`
- Origin Mode: experiment planning
- Origin Date: 2026-07-15
- Verification Status: PARTIALLY VERIFIED
- Version Label: iccd_experiment_roadmap_v1

This roadmap turns the high-level workflow into concrete experiments for an
ICCD low-light denoising paper. It is written for the current local repository
state and current local data under `D:/iccd/data/20260319`.

## Paper-Level Objective

Build a Photonics Journal-oriented paper around real ICCD device evidence:

1. Real gated ICCD data have measurable signal-dependent and fixed-pattern
   noise behavior.
2. Generic Poisson-Gaussian or sCMOS-only assumptions are insufficient for this
   device chain.
3. An ICCD-aware noise prior or generated-noise pipeline should better match
   real device statistics.
4. A denoiser trained or calibrated with ICCD-aware evidence should improve
   held-out real ICCD data without only performing brightness correction or
   oversmoothing.

## Current Evidence Boundary

Already supported:

- Ten complete repeated-frame gated ICCD folders are available:
  `1,2,4,5,7,8,9,10,11,13`.
- Each complete folder has 200 TIFF frames and 200 metadata rows.
- Current metadata show exposure width 900 ms, Sync A/B width 4 us, and gain 60.
- Mean signal spans about 936 to 4717 DN on 512x512 center crops.
- Temporal standard deviation spans about 37.7 to 217.9 DN.
- Approximate Fano factor spans about 1.64 to 14.0.
- Fixed-pattern variation grows strongly with brightness.

Not yet supported:

- Supervised ICCD clean/noisy training claims from the gated data alone.
- Exposure-normalized denoising claims across short/long ICCD exposure pairs.
- Dark-current or dark-count correction claims without identified dark frames.
- Flat-field correction claims beyond repeated-scene empirical fixed-pattern
  estimates.
- Device-general claims beyond this acquisition setting.

## Decision Gates

| Gate | Required before moving on | Current status |
|---|---|---|
| G1 data inventory | Complete folders, metadata, dtype/range, saturation status | mostly passed |
| G2 noise statistics | Mean-variance, fixed pattern, residual distribution, spatial correlation | in progress |
| G3 calibration evidence | Dark or flat frames, or explicit surrogate calibration limits | not passed |
| G4 training correctness | Manifest, tensor range, split, PSNR/SSIM, no-model baseline | passed for PMRID/smoke, not for gated ICCD pairs |
| G5 model claim | Real held-out improvement over stable baselines | not passed |
| G6 paper claim | Every figure/table maps to a bounded claim | not passed |

## Experiment Set A: Device Noise Characterization

### E1.1 Multi-Folder Temporal Noise Summary

- Status: done once, should be repeated with robustness checks.
- Purpose: establish signal-dependent temporal noise across complete folders.
- Current entry command:

```powershell
python scripts\summarize_single_condition_noise.py `
  --root D:\iccd\data\20260319 `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --output-dir reports\gated_iccd_20260319_noise_summary `
  --max-frames 32 `
  --crop-size 512
```

- Primary outputs:
  - `single_condition_noise_summary.csv`
  - `single_condition_noise_summary.md`
- Success criteria:
  - All ten complete folders produce rows.
  - Mean signal and temporal noise increase consistently enough to justify
    signal-dependent modeling.
  - No folder is silently treated as paired clean/noisy data.
- Paper use:
  - First noise-characterization table.
  - Evidence for ICCD-specific signal-dependent noise.

### E1.2 Crop and Frame-Count Robustness

- Status: needs script option or wrapper.
- Purpose: check whether E1.1 is an artifact of the 512x512 center crop or first
  32 frames.
- Inputs:
  - Same ten complete folders.
- Planned runs:
  - Crop sizes: 256, 512, 1024.
  - Frame counts: 16, 32, 64, 128.
  - Optional crop positions: center and four quadrants.
- Primary metrics:
  - Mean signal.
  - Temporal standard deviation.
  - Fano approximation.
  - Fixed-to-temporal ratio.
- Success criteria:
  - Main trends remain after changing crop size or frame count.
  - If trends change by region, report this as spatial nonuniformity rather than
    hide it.
- Paper use:
  - Supplementary robustness table.
  - Reviewer defense against "single crop cherry-picking."

### E1.3 Brightness-Bin Mean-Variance Curve

- Status: next recommended implementation.
- Purpose: fit real ICCD mean-variance behavior from repeated frames.
- Inputs:
  - Repeated frames from complete folders.
- Planned output:
  - Per-folder brightness-bin table.
  - Combined mean-variance CSV.
  - Mean-variance plot.
  - Simple fitted parameters for comparison with Poisson and over-dispersed
    models.
- Candidate command:

```powershell
python scripts\fit_mean_variance_curve.py `
  --root D:\iccd\data\20260319 `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --output-dir reports\gated_iccd_20260319_mean_variance `
  --max-frames 64 `
  --crop-size 1024 `
  --bins 32
```

- Success criteria:
  - Variance is not adequately explained by unit-slope Poisson behavior in raw
    DN.
  - A calibrated over-dispersed or signal-dependent model reduces bin-wise
    variance error relative to a simple Poisson-Gaussian baseline.
  - If a simple model fits poorly, the failure mode is reported, not discarded.
- Paper use:
  - Main figure for real ICCD noise statistics.
  - Parameter source for the ICCD physical prior.

### E1.4 Fixed-Pattern Correction Baseline

- Status: needs implementation.
- Purpose: estimate how much repeated-frame fixed structure can be removed
  before learning-based denoising.
- Inputs:
  - Repeated frames from each complete folder.
- Planned method:
  - Estimate per-pixel temporal mean map.
  - Decompose each frame into global brightness, fixed-pattern component, and
    temporal residual.
  - Test subtraction or flat-field style normalization on held-out frames from
    the same folder.
- Candidate command:

```powershell
python scripts\evaluate_fixed_pattern_correction.py `
  --root D:\iccd\data\20260319 `
  --folders 1 2 4 5 7 8 9 10 11 13 `
  --output-dir reports\gated_iccd_20260319_fixed_pattern `
  --train-frames 100 `
  --test-frames 100 `
  --crop-size 1024
```

- Primary metrics:
  - Spatial fixed-pattern standard deviation before/after correction.
  - Temporal residual standard deviation before/after correction.
  - Residual mean bias.
  - Visual map of fixed-pattern component.
- Working success threshold:
  - Reduce spatial fixed-pattern standard deviation by at least 50% on held-out
    frames.
  - Do not increase temporal residual standard deviation by more than 10%.
  - Residual mean remains close to zero after correction.
- Paper use:
  - Calibration baseline.
  - Evidence that fixed-pattern correction is necessary before final denoising
    claims.

### E1.5 Spatial Correlation and PSD

- Status: needs implementation.
- Purpose: test whether residual noise has spatial structure from phosphor
  diffusion, optics, readout, or processing.
- Inputs:
  - Residuals after subtracting per-pixel temporal mean.
- Planned outputs:
  - Radially averaged PSD.
  - Autocorrelation map.
  - Row/column correlation summary.
- Success criteria:
  - Report whether residuals are spatially white or correlated.
  - If spatial correlation is present, use it to justify a phosphor/spatial blur
    term in the ICCD prior.
- Paper use:
  - Optional figure if spatial structure is clear.
  - Ablation motivation for the ICCD prior.

### E1.6 Dark and Flat Data Gate

- Status: blocked until data are identified or acquired.
- Purpose: separate true dark/read components from scene or illumination
  structure.
- Data needed:
  - Dark frames at gain 60, exposure width 900 ms, Sync A/B width 4 us.
  - Flat-field frames at several illumination levels under the same device
    condition.
  - Optional matched sCMOS dark/flat data.
- Success criteria:
  - Dark and flat folders have metadata matching the target ICCD condition.
  - At least 50 frames per condition, 100+ preferred.
  - No saturation in useful flat-field levels.
- Paper use:
  - Stronger calibration section.
  - Cleaner separation between device noise and scene texture.

## Experiment Set B: Noise Model and Synthetic Fidelity

### E2.1 ICCD Prior Parameterization

- Status: partial code exists; needs calibration from E1.3/E1.5.
- Purpose: turn real ICCD statistics into prior parameters.
- Inputs:
  - Mean-variance fits.
  - Fixed-pattern maps or fixed-pattern statistics.
  - Spatial correlation estimates if available.
- Model components:
  - Signal-dependent over-dispersion.
  - Additive read/dark term.
  - Fixed-pattern component.
  - Optional spatial diffusion/correlation component.
- Success criteria:
  - Parameters are derived from reports, not hand-tuned only by visuals.
  - The prior reproduces brightness-bin variance better than generic
    Poisson-Gaussian.
- Paper use:
  - Method section for ICCD-aware physical prior.

### E2.2 Synthetic Noise Fidelity Baseline

- Status: runnable for paired manifests, but gated ICCD pairing is not available
  yet.
- Existing entry point:

```powershell
python scripts\compare_noise_priors.py `
  --pairs-csv reports\pmrid_500ms_15ms\pairs.csv `
  --config configs\noise_prior_baselines.yaml `
  --output-dir reports\pmrid_500ms_15ms\noise_priors
```

- For gated ICCD:
  - Use only after clean/reference pairing is defined, or define a repeated-frame
    surrogate carefully.
- Primary metrics:
  - Mean error.
  - Variance error.
  - Histogram distance.
  - Brightness-bin residual error.
  - PSD/autocorrelation distance after E1.5 exists.
- Working success threshold:
  - ICCD prior reduces variance or brightness-bin residual error by at least 20%
    versus Poisson-Gaussian in most valid bins.
  - It must not improve one statistic while severely degrading mean or histogram
    fidelity.
- Paper use:
  - Main comparison table for noise fidelity.

### E2.3 sCMOS Comparison

- Status: data-dependent.
- Purpose: show why ICCD cannot be reduced to a normal sCMOS noise model.
- Inputs:
  - Matched or comparable sCMOS repeated data.
  - ICCD repeated data from the same scene or controlled target if available.
- Primary comparisons:
  - Mean-variance curve.
  - Fano approximation.
  - Fixed-pattern ratio.
  - Spatial PSD/autocorrelation.
- Success criteria:
  - Either show a measurable ICCD/sCMOS difference, or state the conditions
    where the current data cannot support that comparison.
- Paper use:
  - Device comparison figure if data are comparable.

## Experiment Set C: Training Pipeline and Baselines

### E3.1 Paired Dataset Audit

- Status: done for PMRID lists and smoke data; gated ICCD not yet paired.
- Purpose: ensure no training starts from broken pairing or wrong tensor range.
- Existing commands:

```powershell
python scripts\audit_iccd_dataset.py `
  --clean-dir E:\PMRID\PMRID7\data\500ms `
  --noisy-dir E:\PMRID\PMRID7\data\15ms `
  --dark-dir E:\PMRID\PMRID7\data\dark_Background `
  --output-dir reports\pmrid_500ms_15ms `
  --pairs-out reports\pmrid_500ms_15ms\pairs.csv `
  --splits-out reports\pmrid_500ms_15ms\splits.yaml
```

- Gate:
  - TIFFs remain 16-bit.
  - Clean/noisy orientation is correct.
  - Split is scene/condition safe.
  - `data_range` used by PSNR/SSIM matches tensor scale.
- Paper use:
  - Reproducibility appendix and baseline validity.

### E3.2 No-Model Baseline

- Status: runnable.
- Purpose: establish how hard the pair mapping is before training a network.
- Existing command:

```powershell
python scripts\evaluate_pair_baseline.py `
  --pairs-csv reports\pmrid_500ms_15ms\pairs.csv `
  --output-dir reports\pmrid_500ms_15ms\b0 `
  --range-max 65535 `
  --bins 8
```

- Success criteria:
  - Baseline PSNR/SSIM and brightness-bin metrics are recorded before any model.
  - Any later model gain is compared against this baseline.
- Paper use:
  - Baseline row and sanity check.

### E3.3 First Supervised Denoiser Baseline

- Status: planned after PyTorch audit.
- Purpose: create a stable reference model before proposing ICCD-specific
  improvements.
- Candidate models:
  - Existing model in this repo after audit.
  - Lightweight U-Net or RLFN-style baseline.
- Required controls:
  - Same train/val/test split.
  - Same crop size.
  - Same normalization.
  - Same random seed list, ideally 3 seeds.
  - Same metric script.
- Working success threshold:
  - Improve no-model baseline by at least 0.3 dB PSNR or 0.005 SSIM on held-out
    real data, while visual panels do not show obvious oversmoothing.
  - Effect should be larger than run-to-run seed noise.
- Paper use:
  - Baseline denoising performance table.

### E3.4 Real-Only vs Synthetic-Only vs Mixed Training

- Status: blocked until E2 synthetic noise and a supervised test set are stable.
- Purpose: test whether ICCD-aware synthetic data actually helps denoising.
- Arms:
  - Real paired data only.
  - Poisson-Gaussian synthetic data.
  - sCMOS-like synthetic data.
  - ICCD-prior synthetic data.
  - Real + ICCD-prior mixed.
  - ICCD-aware PNGAN generated data if later implemented.
- Primary metrics:
  - PSNR/SSIM on held-out real ICCD or validated proxy data.
  - Brightness-bin PSNR.
  - Residual noise statistics.
  - Visual best/median/worst panels.
- Success criteria:
  - ICCD-aware data improves held-out real performance over generic synthetic
    data.
  - Improvement is not only global brightness correction.
  - No severe loss of fine structure in visual panels.
- Paper use:
  - Main denoising validation table.

## Experiment Set D: Ablation

### E4.1 Physical Prior Ablation

- Status: planned after E2.1.
- Variants:
  - Full ICCD prior.
  - No over-dispersion/MCP-like gain term.
  - No fixed-pattern component.
  - No spatial diffusion/correlation component.
  - Poisson-Gaussian only.
  - sCMOS-like only.
- Metrics:
  - Mean-variance error.
  - Histogram distance.
  - PSD/autocorrelation distance if available.
  - Downstream denoising if training is stable.
- Success criteria:
  - Each retained component has a measurable benefit on at least one targeted
    statistic without unacceptable degradation elsewhere.
- Paper use:
  - Ablation table tied to ICCD imaging chain.

### E4.2 Training Objective Ablation

- Status: later-stage only.
- Variants:
  - L1 only.
  - L1 + SSIM or MS-SSIM.
  - L1 + residual/statistical consistency.
  - Optional adversarial noise-domain term if PNGAN is reintroduced.
- Rule:
  - Do not run this before data and noise-prior gates are stable.
- Success criteria:
  - A loss term must improve a targeted failure mode, not only one average
    metric.
- Paper use:
  - Secondary ablation if model contribution becomes part of the paper.

## Experiment Set E: Literature and Paper Evidence

### E5.1 Literature Matrix

- Tool route:
  - arXiv MCP for paper discovery.
  - Brave Search MCP or web browsing for non-arXiv sources, journal scope, and
    device documents.
  - `academic-research-suite` for literature synthesis and claim boundary.
- Rows to collect:
  - Low-light real noise modeling.
  - Poisson-Gaussian and camera noise calibration.
  - sCMOS noise modeling.
  - ICCD or image intensifier noise characteristics.
  - Synthetic noise for denoising.
  - Real low-light denoising datasets.
- Output:
  - `docs/literature_matrix.md` or a future `paper_rewriting_output/`
    evidence bank.
- Gate:
  - No fabricated references.
  - Every cited method has source URL/DOI/arXiv ID.

### E5.2 Figure and Table Plan

Minimum paper figures:

| Figure/Table | Source experiment | Claim supported |
|---|---|---|
| Data/acquisition overview | G1 + E1.1 | Real ICCD data and metadata are controlled enough for analysis |
| Mean-variance curve | E1.3 | ICCD noise is signal-dependent and over-dispersed |
| Fixed-pattern map/correction | E1.4 | Fixed-pattern correction is necessary and measurable |
| PSD/autocorrelation | E1.5 | Residual noise has or lacks spatial correlation |
| Noise model fidelity table | E2.2/E4.1 | ICCD-aware prior matches statistics better |
| Denoising baseline table | E3.2/E3.3/E3.4 | Real held-out denoising improves under controlled comparison |
| Ablation table | E4.1/E4.2 | Each proposed component has evidence |

## Next Sprint Plan

The next sprint should implement and run the first three Stage 2 experiments:

1. Add `scripts/fit_mean_variance_curve.py`.
2. Add `scripts/evaluate_fixed_pattern_correction.py`.
3. Add a crop/frame-count robustness mode or wrapper around
   `summarize_single_condition_noise.py`.
4. Run all three on `D:/iccd/data/20260319`.
5. Promote only stable summaries into `docs/gated_iccd_data_inventory.md`.

Do not start major network changes until E1.3 and E1.4 have been run and their
outputs show which noise component is the dominant bottleneck.

## User Data Needed Later

Priority order:

1. Dark frames matching gain 60, exposure width 900 ms, Sync A/B width 4 us.
2. Flat-field frames at multiple brightness levels under the same condition.
3. Short/long ICCD paired exposure data for the same scene.
4. Matched sCMOS repeated data if the paper will include device comparison.
5. Any acquisition notes explaining why folders `1` to `13` differ in
   brightness despite identical recorded exposure/gate/gain metadata.

