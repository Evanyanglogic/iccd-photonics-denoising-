# ICCD Photonics Denoising

This repository is the working scaffold for a Photonics Journal oriented paper on
ICCD low-light denoising.

## Working Title

面向 ICCD 弱光成像的物理先验噪声建模与生成式去噪数据增强方法

## Positioning

The paper should not be framed as a generic deep denoising network paper. The
core angle is an optical imaging and device-noise problem:

- ICCD and sCMOS real-device data are available.
- Calibration sequences include dark, flat, gain/exposure/gating variations.
- The current PNGAN work already provides a transferable pipeline:
  physical noise prior -> adversarial noise refinement -> downstream denoising.
- The new contribution is to make the noise prior ICCD-aware and validate it
  with device-level statistics and real low-light denoising performance.

## Core Contribution Draft

We propose an ICCD-aware physical-prior generative noise modeling framework for
low-light imaging. The method models photon statistics, photocathode/MCP gain
fluctuation, phosphor-screen spatial diffusion, and sensor readout noise, then
uses adversarial refinement to generate realistic noisy-clean training pairs
that improve downstream denoising on real ICCD data.

## Repository Layout

```text
configs/          Experiment and model configuration drafts
data_manifest/    Dataset organization and capture protocol notes
docs/             Research route, contribution lock, experiment matrix
experiments/      Runnable experiment entry points and logs later
paper/            Manuscript outline and figure/table planning
scripts/          Dataset audit and experiment utility scripts
src/iccd_eval/    Float-domain metrics and residual analysis helpers
src/iccd_noise/   ICCD noise model and analysis code
```

## Immediate Milestones

1. Run the dataset audit gate before training:

   ```powershell
   python scripts\audit_iccd_dataset.py `
     --config configs\dataset_iccd.yaml `
     --output-dir reports `
     --pairs-out data_manifest\pairs.csv `
     --splits-out data_manifest\splits.yaml
   ```

2. Confirm strict clean/noisy pairing, 16-bit range handling, metadata coverage,
   dark/flat calibration coverage, and held-out scene/condition splits. See
   `docs/data_audit_gate.md`; start metadata from
   `data_manifest/metadata_template.csv`.
3. Use `src/iccd_eval/metrics.py` for float-domain PSNR/SSIM and residual
   statistics; do not quantize normalized scientific images to uint8.
4. Measure the no-model B0 baseline from the generated pair manifest:

   ```powershell
   python scripts\evaluate_pair_baseline.py `
     --pairs-csv data_manifest\pairs.csv `
     --output-dir reports\b0_noisy_baseline
   ```

5. Compare sCMOS prior, Poisson-Gaussian prior, and ICCD prior on noise
   statistics after the audit gate passes.
6. Connect the ICCD prior to the existing PNGAN training loop only after the
   data and metric gates are stable.
