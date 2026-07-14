# Research Route

## Target Journal

Target: 光子学报.

Preferred framing: optical/electronic imaging system, low-light acquisition,
device noise, calibration, and denoising validation.

Less suitable framing: only proposing a new neural network block and comparing
against natural-image denoising benchmarks.

## Main Research Question

Can an ICCD-aware physical-prior generative noise model create training pairs
that match real ICCD low-light noise statistics better than generic synthetic
noise, and improve denoising performance on real ICCD measurements?

## Paper Spine

1. Real ICCD weak-light paired data are difficult and expensive to collect.
2. Generic Poisson-Gaussian or sCMOS-only noise priors miss ICCD chain effects.
3. ICCD noise should be modeled through the imaging chain:
   photon arrival -> photocathode -> MCP gain -> phosphor diffusion -> CCD/sCMOS
   readout.
4. A physical prior gives a reasonable noisy starting point.
5. PNGAN-style adversarial refinement aligns generated noise with real device
   statistics while keeping content stable.
6. The generated data should be judged by both device-statistical fidelity and
   downstream real-data denoising gain.

## Recommended Claim Boundary

Strong claims allowed:

- The method models a more ICCD-specific noise path than plain Poisson-Gaussian
  or sCMOS-only priors.
- The method can be calibrated from real device sequences.
- The generated noise can be evaluated against real ICCD statistical signatures.
- Downstream denoising can be improved if the generated data closes the
  synthetic-to-real gap.

Claims to avoid until proven:

- The method fully reproduces all ICCD noise components.
- The generated noise is physically exact.
- The denoiser generalizes to all ICCD devices or all low-light conditions.
- The method is better than all recent low-light denoising methods without a
  broad benchmark.

## Suggested Paper Sections

1. Introduction
2. ICCD imaging chain and noise characteristics
3. ICCD-aware physical-prior noise model
4. PNGAN-based generative refinement
5. Calibration and experimental setup
6. Noise-statistical validation
7. Downstream denoising validation
8. Discussion and limitations
9. Conclusion

