# E5 Noise Structure-by-Strength Factorial Result

Decision: **C_INTERACTION**

## Three-Seed Cell Results

| Variant | Folder gain | Seed SD | Seed 95% t-CI | Positive folders | Positive pairs | Grad/noisy | Worst folder | Reference sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-L | -0.495504 | 0.351404 | [-1.368439, 0.377432] | 5/10 | 0.529 | 0.5697 | -4.812298 | 1.000 |
| P-H | -4.314942 | 0.221548 | [-4.865297, -3.764587] | 0/10 | 0.000 | 0.3515 | -6.200089 | 1.000 |
| H-L | -1.891664 | 0.469982 | [-3.059163, -0.724164] | 0/10 | 0.000 | 0.5305 | -5.443265 | 1.000 |
| H-H | -4.891073 | 0.667996 | [-6.550468, -3.231679] | 0/10 | 0.000 | 0.4009 | -8.261544 | 1.000 |

## Factor Effects

| Effect | PSNR effect | Folder 95% CI | Seed SD | Reference half-diff | Stable |
|---|---:|---:|---:|---:|---:|
| H-H_minus_H-L | -2.999410 | [-4.073059, -1.957461] | 0.425968 | 0.001860 | True |
| H-H_minus_P-H | -0.576131 | [-1.322012, 0.152737] | 0.741765 | 0.000421 | False |
| H-L_minus_P-L | -1.396160 | [-1.925031, -0.936231] | 0.645033 | 0.000122 | True |
| P-H_minus_P-L | -3.819438 | [-4.911539, -2.703401] | 0.494281 | 0.001318 | True |
| interaction | 0.820028 | [0.260145, 1.445168] | 0.153270 | 0.000542 | True |
| strength_main | -3.409424 | [-4.466181, -2.357237] | 0.454982 | 0.001589 | True |
| structure_main | -0.986146 | [-1.563296, -0.456186] | 0.690846 | 0.000150 | True |

## Uncertainty Budget

- Maximum seed SD: 0.667996 dB
- Maximum reference half-difference: 0.002028 dB
- Matched-strength construction error: 0.000003 dB
- Prior 0.036 dB condition gain exceeds uncertainty: False

## Claim Boundary

- Results use temporal-mean surrogate references, not clean ground truth.
- A factor is called stable only when its folder bootstrap CI excludes zero and its mean exceeds seed/reference variability.