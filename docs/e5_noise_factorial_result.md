# E5 Noise Structure-by-Strength Factorial Result

## Construction Finding

The original datasets could not be used as an absolute low/high pair. The p99
dataset has larger absolute residual standard deviation because its clean
content was rescaled to p99 = 0.25, while the physical dataset kept the darker
source scale. E5 therefore transferred each source residual-to-clean standard
deviation ratio to one shared clean domain.

The first smoke construction failed with about 91% lower-bound clipping because
the dark-corrected shared clean contains many exact zeros. A common 1024-DN
pedestal was then applied identically to all four variants. This reduced mean
clipping to about 0.15-0.17% without introducing a factor-dependent offset.

## Decoupling Gate

The full 100-pair construction passed all preregistered checks:

| Variant | Residual std | Mean | Skewness | Excess kurtosis | Signal-residual correlation |
|---|---:|---:|---:|---:|---:|
| P-L | 0.00168231 | -1.53e-10 | -4.448 | 266.33 | -0.180 |
| P-H | 0.00470248 | 1.35e-09 | -5.386 | 264.95 | -0.197 |
| H-L | 0.00168231 | -2.88e-10 | -1.590 | 76.59 | -0.130 |
| H-H | 0.00470248 | 8.46e-10 | -1.769 | 75.12 | -0.137 |

Matched-strength construction error is approximately 0.000003 dB. Normalized
PSD and autocorrelation remain stable when only scale changes. P/H structure
remains distinguishable mainly through tail distribution, kurtosis,
signal-residual dependence, and column-pattern energy rather than correlation
length.

## Three-Seed Real Transfer

All cells used the same 2,625-parameter residual small CNN, 100 epochs, 128
patches, AdamW, L1 loss, synthetic-validation checkpoint selection, and seeds
20260716/17/18. Real evaluation used the same 80 held-out surrogate pairs and both
odd/even temporal-mean references.

| Variant | Folder PSNR gain | Seed SD | Seed 95% t-CI | Positive folders | Grad/noisy |
|---|---:|---:|---:|---:|---:|
| P-L | -0.4955 | 0.3514 | [-1.3684, 0.3774] | 5/10 | 0.5697 |
| P-H | -4.3149 | 0.2215 | [-4.8653, -3.7646] | 0/10 | 0.3515 |
| H-L | -1.8917 | 0.4700 | [-3.0592, -0.7242] | 0/10 | 0.5305 |
| H-H | -4.8911 | 0.6680 | [-6.5505, -3.2317] | 0/10 | 0.4009 |

Both references give identical folder-gain signs. The high-strength cells show
severe oversmoothing, and even P-L removes visible scene structure in some
folders. These are negative-transfer results, not restoration gains.

## Factor Decision

| Effect | PSNR effect | Folder bootstrap 95% CI | Seed SD |
|---|---:|---:|---:|
| Strength main effect | -3.4094 dB | [-4.4662, -2.3572] | 0.4550 |
| Structure main effect | -0.9861 dB | [-1.5633, -0.4562] | 0.6908 |
| Structure-strength interaction | +0.8200 dB | [0.2601, 1.4452] | 0.1533 |

The preregistered case is `C_INTERACTION`: strength dominates, structure also
changes transfer, and the strength penalty depends on structure. However, the
maximum cell seed variation is 0.6680 dB, far above the previous 0.036 dB
condition-strategy advantage. That earlier advantage must remain exploratory
and cannot support a deployable gate or condition-aware network claim.

The next experiment should repair the synthetic-real gap in the conservative
P-L cell using real ICCD residual statistics while keeping the probe and
strength fixed. No stronger backbone is justified by E5.
