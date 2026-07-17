# E7 Data-to-Route Eligibility Audit

## Module Decision

- Gated ICCD noise characterization: **SUPPORTED**, limited to operationally observed components.
- Condition-aware noise modeling: **NOT SUPPORTED** as a deployable generator or selector.
- Controlled denoising validation: **LIMITED SUPPORT**, for surrogate-based applicability and failure boundaries.

## Folder Gates

| folder | integrity | stability | characterization | surrogate | condition | denoising | repeated supervision | reason |
|---:|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | frame residuals are correlated; row/column residuals are correlated |
| 2 | PASS | PASS | PASS | WARN | WARN | WARN | PASS | dual-reference folder result is sensitive |
| 4 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | row/column residuals are correlated |
| 5 | PASS | WARN | PASS | WARN | WARN | WARN | FAIL | local drift exceeds E6 gate; frame residuals are correlated; row/column residuals are correlated |
| 7 | PASS | WARN | PASS | WARN | WARN | WARN | FAIL | local drift exceeds E6 gate; row/column residuals are correlated |
| 8 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | row/column residuals are correlated |
| 9 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | row/column residuals are correlated |
| 10 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | row/column residuals are correlated |
| 11 | PASS | PASS | PASS | PASS | WARN | PASS | PASS | usable for operational characterization and relative surrogate comparison |
| 13 | PASS | PASS | WARN | WARN | WARN | WARN | FAIL | split-half stable component is not stable; dual-reference folder result is sensitive |

## Condition Audit

- Maximum feature correlation: 1.0000
- PC1 explained fraction: 0.9691
- Maximum VIF: 54164.69
- LOFO ridge RMSE: 0.2999 dB
- Null RMSE: 0.5581 dB
- Maximum E5 seed SD: 0.6680 dB
- All complete folders share the recorded gate/exposure/sync/gain settings. The score is therefore an image-statistical state descriptor confounded with folder and scene, not a verified acquisition-condition variable.

## Route

Select route 2: characterization + conditional mismatch analysis + controlled denoising applicability validation.
Do not claim a validated condition-aware generator, deployable selector, clean ground truth, or true-detail recovery.
