# Surrogate Reference Reliability Audit

## Preregistered Hypothesis

When the temporal-mean surrogate is rebuilt from two disjoint, interleaved frame sets, the preregistered LOFO linear condition strategy outperforms both fixed p99 and fixed physical strategies on each reference replicate.

Decision: **GO**

## Strategy Results

| Reference | Strategy | Folder gain | 95% CI | Positive folders | Worst folder | SSIM gain | Grad/noisy | Residual std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reference_a_odd | lofo_best_linear | 0.376539 | [0.107083, 0.698762] | 10/10 | 0.006406 | 0.000897358 | 0.9650 | 0.00169728 |
| reference_a_odd | lofo_best_hard | 0.371397 | [0.099974, 0.690723] | 10/10 | 0.005475 | 0.000894935 | 0.9654 | 0.00169795 |
| reference_a_odd | always_physical | 0.338284 | [0.048407, 0.689793] | 6/10 | -0.129112 | 0.000886996 | 0.9674 | 0.00169875 |
| reference_a_odd | always_p99 | 0.039268 | [0.028507, 0.049564] | 10/10 | 0.006406 | 4.81422e-05 | 0.9949 | 0.0018209 |
| reference_a_odd | always_noisy | 0.000000 | [0.000000, 0.000000] | 0/10 | 0.000000 | 0 | 1.0000 | 0.00183073 |
| reference_b_even | lofo_best_linear | 0.376989 | [0.104651, 0.695688] | 9/10 | -0.003853 | 0.000897049 | 0.9650 | 0.0016946 |
| reference_b_even | lofo_best_hard | 0.372624 | [0.106062, 0.705750] | 9/10 | -0.003853 | 0.000894948 | 0.9654 | 0.00169527 |
| reference_b_even | always_physical | 0.341387 | [0.053795, 0.684107] | 6/10 | -0.147342 | 0.000887425 | 0.9674 | 0.00169606 |
| reference_b_even | always_p99 | 0.038501 | [0.025319, 0.050341] | 9/10 | -0.003853 | 4.7924e-05 | 0.9949 | 0.00181849 |
| reference_b_even | always_noisy | 0.000000 | [0.000000, 0.000000] | 0/10 | 0.000000 | 0 | 1.0000 | 0.00182832 |

## Reference Agreement

| Strategy | Folder sign agreement | Mean abs gain delta | Max abs gain delta | Correlation |
|---|---:|---:|---:|---:|
| always_noisy | 1.000 | 0.000000 | 0.000000 | nan |
| always_p99 | 0.900 | 0.002952 | 0.010259 | 0.9866 |
| always_physical | 1.000 | 0.007043 | 0.025360 | 0.9998 |
| lofo_best_hard | 0.900 | 0.004180 | 0.010259 | 1.0000 |
| lofo_best_linear | 0.900 | 0.003791 | 0.010259 | 1.0000 |

## Paired Strategy Contrasts

| Reference | Comparator | Linear minus comparator | 95% paired CI | Positive/equal/negative folders |
|---|---|---:|---:|---:|
| reference_a_odd | always_p99 | 0.337271 | [0.072855, 0.645298] | 6/4/0 |
| reference_a_odd | always_physical | 0.038255 | [0.007836, 0.078854] | 5/4/1 |
| reference_b_even | always_p99 | 0.338488 | [0.076857, 0.666733] | 6/4/0 |
| reference_b_even | always_physical | 0.035602 | [0.003657, 0.080092] | 5/4/1 |

## Automatic Checks

- PASS: `reference_a_odd_advantage`
- PASS: `reference_a_odd_positive_folders`
- PASS: `reference_a_odd_worst_folder`
- PASS: `reference_a_odd_gradient`
- PASS: `reference_b_even_advantage`
- PASS: `reference_b_even_positive_folders`
- PASS: `reference_b_even_worst_folder`
- PASS: `reference_b_even_gradient`
- PASS: `folder_sign_agreement`

## Claim Boundary

- Both references are temporal means from repeated frames, not clean ground truth.
- This audit tests evaluation stability; it does not establish recovery of unobserved scene detail.
- Fixed-pattern content shared by both split references may remain and is not removed by split-half agreement alone.