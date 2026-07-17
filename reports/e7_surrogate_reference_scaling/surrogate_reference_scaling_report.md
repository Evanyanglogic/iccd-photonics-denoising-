# E7 Surrogate Reference Scaling Audit

References use disjoint 25-frame quarters, disjoint 50-frame halves, and the first 100 frames. Test inputs remain frames 101-200 from the existing held-out manifest.

## Strategy Summary

| reference | frames | strategy | folder gain dB | worst folder dB | positive folders | rank |
|---|---:|---|---:|---:|---:|---:|
| ref25_q1 | 25 | lofo_best_linear | 0.3164 | 0.0127 | 10 | 1 |
| ref25_q1 | 25 | lofo_best_hard | 0.3083 | -0.0034 | 9 | 2 |
| ref25_q1 | 25 | always_physical | 0.2725 | -0.1424 | 5 | 3 |
| ref25_q1 | 25 | always_p99 | 0.0404 | 0.0127 | 10 | 4 |
| ref25_q1 | 25 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref25_q2 | 25 | lofo_best_linear | 0.3496 | -0.0056 | 9 | 1 |
| ref25_q2 | 25 | lofo_best_hard | 0.3460 | -0.0056 | 9 | 2 |
| ref25_q2 | 25 | always_physical | 0.3159 | -0.1148 | 6 | 3 |
| ref25_q2 | 25 | always_p99 | 0.0388 | -0.0056 | 9 | 4 |
| ref25_q2 | 25 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref25_q3 | 25 | lofo_best_linear | 0.3895 | 0.0030 | 10 | 1 |
| ref25_q3 | 25 | lofo_best_hard | 0.3848 | 0.0030 | 10 | 2 |
| ref25_q3 | 25 | always_physical | 0.3528 | -0.1428 | 6 | 3 |
| ref25_q3 | 25 | always_p99 | 0.0368 | 0.0030 | 10 | 4 |
| ref25_q3 | 25 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref25_q4 | 25 | lofo_best_linear | 0.4197 | -0.0059 | 9 | 1 |
| ref25_q4 | 25 | lofo_best_hard | 0.4176 | -0.0059 | 9 | 2 |
| ref25_q4 | 25 | always_physical | 0.3894 | -0.1424 | 6 | 3 |
| ref25_q4 | 25 | always_p99 | 0.0350 | -0.0059 | 9 | 4 |
| ref25_q4 | 25 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref50_h1 | 50 | lofo_best_linear | 0.3383 | 0.0040 | 10 | 1 |
| ref50_h1 | 50 | lofo_best_hard | 0.3323 | 0.0028 | 10 | 2 |
| ref50_h1 | 50 | always_physical | 0.2988 | -0.1311 | 6 | 3 |
| ref50_h1 | 50 | always_p99 | 0.0403 | 0.0040 | 10 | 4 |
| ref50_h1 | 50 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref50_h2 | 50 | lofo_best_linear | 0.4139 | -0.0015 | 9 | 1 |
| ref50_h2 | 50 | lofo_best_hard | 0.4104 | -0.0015 | 9 | 2 |
| ref50_h2 | 50 | always_physical | 0.3798 | -0.1452 | 6 | 3 |
| ref50_h2 | 50 | always_p99 | 0.0367 | -0.0015 | 9 | 4 |
| ref50_h2 | 50 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |
| ref100_all | 100 | lofo_best_linear | 0.3804 | 0.0013 | 10 | 1 |
| ref100_all | 100 | lofo_best_hard | 0.3756 | 0.0013 | 10 | 2 |
| ref100_all | 100 | always_physical | 0.3431 | -0.1395 | 6 | 3 |
| ref100_all | 100 | always_p99 | 0.0392 | 0.0013 | 10 | 4 |
| ref100_all | 100 | always_noisy | 0.0000 | 0.0000 | 0 | 5 |

## Decision

- Fixed-model folder sign stability: 0.971
- LOFO-linear folder sign stability: 0.957
- Maximum LOFO-linear gain range across references: 0.7153 dB
- Top-ranked strategy stable: True
- These references can support cautious relative comparisons. They cannot support absolute clean-image recovery claims because all means retain the repeatable scene-plus-stable-pattern component.
