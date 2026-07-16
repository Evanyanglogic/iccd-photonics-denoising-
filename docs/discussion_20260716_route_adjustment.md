# Discussion: Denoising vs Low-Light Detail Restoration

Date: 2026-07-16

## Trigger

The user challenged the current route on three points:

- low-exposure data may lack recoverable detail, so "detail restoration" may be
  an ill-posed target;
- the current model and metrics may not match a denoising-centered paper;
- more literature and tool-assisted challenge are needed before continuing.

## Current Position

The project should not be framed as ultra-low-light detail restoration.

For the current data, a safer and more defensible paper direction is:

```text
Gated ICCD weak-light noise characterization, condition-aware noise modeling,
and fidelity-controlled denoising validation.
```

The route should avoid claims that the model recovers details that were not
captured by photons. Such claims would shift the paper into enhancement or
hallucinated restoration and would be difficult to defend without true clean
references or task-level validation.

## Revised Claim Boundary

Allowed claims:

- real gated ICCD repeated frames show measurable signal-dependent noise;
- fixed-pattern structure is important under the current acquisition condition;
- synthetic noise training can be evaluated as a device-statistical proxy;
- real surrogate validation can test whether a denoiser reduces residual noise
  without large condition-specific failures.

Avoid for now:

- "recovering missing details";
- "true clean ICCD reconstruction";
- "super-resolution-like weak-light restoration";
- "real paired ICCD denoising performance" unless true paired data exist.

## Metric Implication

PSNR and SSIM should remain secondary sanity metrics. The paper needs
device-facing metrics:

- residual mean and standard deviation;
- residual histogram distance;
- PSD / autocorrelation;
- fixed-pattern ratio;
- brightness-bin PSNR;
- edge/gradient preservation;
- folder-level and condition-level gain, not only average gain.

## Model Implication

The current small CNN is a training-pipeline sanity baseline only.

The E3 real surrogate result suggests condition dependence:

- p99 synthetic model: stable but tiny real-surrogate gain;
- physical-scale synthetic model: stronger average gain but many negative-gain
  pairs and strong folder dependence.

Therefore the next technical step should not be a larger MIRNet/SMNet/PNGAN
run. It should be condition-stratified analysis and possibly condition-aware
noise modeling.

## Literature Direction

The most relevant adjacent literature is not generic low-light enhancement but:

- physics-based low-light RAW noise formation;
- noise synthesis from real sensor statistics;
- dark-frame or residual-bank noise modeling;
- self-supervised/repeated-frame denoising;
- image-intensifier / ICCD noise and fixed-pattern characterization.

Working hypothesis:

```text
The strongest contribution is the ICCD device evidence chain, not the neural
network architecture.
```

## Next Research Gate

Before new model training:

1. Diagnose Brave Search MCP failure and restore reliable web search.
2. Expand and verify the literature matrix with stronger sources.
3. Run condition-stratified analysis linking real-surrogate PSNR gain to E1
   folder statistics: mean signal, temporal std, Fano, and fixed-pattern ratio.
4. Decide whether to implement condition-aware denoising or residual-bank noise
   synthesis.
