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
src/iccd_noise/   ICCD noise model and analysis code
```

## Immediate Milestones

1. Build an ICCD calibration dataset manifest.
2. Implement a first ICCD physical noise simulator.
3. Compare sCMOS prior, Poisson-Gaussian prior, and ICCD prior on noise statistics.
4. Connect the ICCD prior to the existing PNGAN training loop.
5. Run downstream denoising validation on real ICCD test data.

