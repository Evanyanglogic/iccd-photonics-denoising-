# Manuscript Outline

## Title

面向 ICCD 弱光成像的物理先验噪声建模与生成式去噪数据增强方法

## Abstract Contract

1. Problem: ICCD weak-light imaging suffers from device-specific noise, and real
   paired data are hard to acquire.
2. Gap: generic synthetic noise and sCMOS-oriented models do not capture the
   ICCD image-intensifier chain.
3. Contribution: an ICCD-aware physical-prior generative noise model.
4. Evidence: calibration statistics and real-data denoising validation.
5. Payoff: lower-cost training data construction for ICCD weak-light denoising.

## Figure Plan

1. ICCD imaging chain and noise components.
2. Overall method: physical prior -> PNGAN refinement -> denoiser training.
3. Calibration statistics: mean-variance, PSD, dark distribution.
4. Noise visual comparison: real vs baselines vs generated.
5. Denoising visual comparison.

## Table Plan

1. Device acquisition settings.
2. Noise-statistical fidelity comparison.
3. Downstream denoising results on real ICCD.
4. Ablation study.

