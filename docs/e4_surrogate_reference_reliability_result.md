# E4 Surrogate Reference Reliability Result

## Decision

The preregistered decision is `GO`, but only for the narrow claim that the
existing condition-aware selection result is not an artifact of one particular
temporal-mean surrogate realization.

Odd frames 1-99 and even frames 2-100 formed two disjoint 50-frame references.
The held-out noisy inputs remained frames 101, 111, 121, 131, 141, 151, 161,
and 171. No model, threshold, or checkpoint was selected on either reference.

## Main Evidence

| Result | Reference A | Reference B |
|---|---:|---:|
| LOFO-linear folder PSNR gain | +0.376539 dB | +0.376989 dB |
| Positive folders | 10/10 | 9/10 |
| Worst folder gain | +0.006406 dB | -0.003853 dB |
| Linear minus physical | +0.038255 dB | +0.035602 dB |
| Paired 95% CI | [0.007836, 0.078854] | [0.003657, 0.080092] |
| Gradient ratio to noisy | 0.9650 | 0.9650 |

Folder-gain sign agreement between references is 0.90 and the folder-gain
correlation is 0.99996. The split-reference PSNR ranges from 62.01 to 78.28 dB.

## Interpretation

The result does not prove that condition-aware denoising is a final method.
Relative to physical, LOFO-linear is strictly better in five folders, equal in
four, and slightly worse in one. Its advantage therefore comes mainly from
avoiding physical-model regressions in low-condition folders, not from a
universal restoration gain. The temporal means also retain any fixed pattern
shared across frames, so they remain surrogate references rather than clean
ground truth.

The next experiment should strengthen the statistical representation by
scale-matching p99 and physical synthetic residuals and testing distributional
differences separately from residual strength. Stronger backbones remain
blocked until that confound is resolved.
