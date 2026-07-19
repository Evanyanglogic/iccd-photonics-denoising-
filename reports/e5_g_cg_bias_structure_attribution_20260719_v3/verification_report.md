# E5 G/CG-NC Bias and Structure Attribution

Status: `BIAS-STRUCTURE-ATTRIBUTION-VERIFIED-WITH-LIMITATIONS`

Six frozen best checkpoints were evaluated on folders 2, 5, 9, and 11 (200 frames per folder). No training, checkpoint modification, or input adaptation was performed.

## Brightness bias

| Seed | G mean / mean-absolute shift (DN) | CG-NC mean / mean-absolute shift (DN) |
|---:|---:|---:|
| 20260719 | -0.981727 / 0.981727 | -3.685046 / 3.685046 |
| 20260720 | -7.796048 / 7.796048 | -2.890306 / 2.890464 |
| 20260721 | -3.344191 / 12.656963 | 1.351022 / 10.797342 |

Pooled input-signal Pearson/Spearman correlations were G `-0.684447/-0.570154` and CG-NC `-0.505499/-0.268222`.
CG predicted sigma is exactly proportional to input mean, so signal-dependent and sigma-dependent bias cannot be separately identified.
Folder 5 had the dominant negative shift: G `-16.239651` DN and CG-NC `-8.074270` DN.
Selected attribution: A. seed-dependent network DC bias; B. input-signal-dependent bias; F. unresolved checkpoint-specific contribution. Overfitting causality is not established because non-best epochs were not evaluated on the real holdout.

## Temporal attribution

| Metric | G | CG-NC | CG-NC - G |
|---|---:|---:|---:|
| Raw temporal reduction | 0.030747 | 0.040408 | 0.009660 |
| Mean-centered temporal reduction | 0.030838 | 0.040538 | 0.009700 |
| DC-restored temporal reduction | 0.030738 | 0.040409 | 0.009672 |

The CG-NC advantage remains positive for all three seeds and all four folders after mean centering. It is not primarily a frame-level DC effect.

## Frequency and structure

| Band | CG-NC - G output/input energy ratio |
|---|---:|
| dc | 0.003562 |
| very_low | 0.003538 |
| low | 0.003105 |
| mid | -0.002612 |
| high | -0.012456 |

Flat-region temporal-reduction advantage was `0.010397`; high-gradient retention difference was `-0.002739`.
The conditional advantage is concentrated in local/high-frequency suppression, with a small but consistent high-gradient retention cost. CG-NC does not obtain its advantage by stronger DC suppression than G.
Selected structure attribution: A. flat-region noise suppression; B. edge attenuation; D. broad smoothing.

### Folder 5

Flat reduction G/CG-NC: `0.030014/0.043613`; high-gradient reduction: `0.020303/0.021911`.
High-gradient retention G/CG-NC: `0.993501/0.994662`. The large brightness shift is folder-level and signal-associated, while the temporal benefit remains after mean centering.

## Frame-wise DC mean restoration

| Model | Pre/post mean-absolute shift (DN) | Raw/DC-restored temporal reduction | Gradient pre/post | |structure corr| pre/post |
|---|---:|---:|---:|---:|
| G | 7.144912 / 0.000027 | 0.030747 / 0.030738 | 0.993008 / 0.993008 | 0.222157 / 0.222158 |
| CG_NC | 5.790951 / 0.000026 | 0.040408 / 0.040409 | 0.988660 / 0.988660 | 0.178469 / 0.178469 |

Decision: `DC-CORRECTION-BENEFICIAL`. The correction restores only each frame's global DC mean and does not modify checkpoint weights or spatial gradients.

## Overfitting limitation

All six runs show best-to-final PMRID validation PSNR drops while train loss decreases. The drop range is `0.750377-7.139937 dB`. Its causal relation to real-domain DC bias remains unresolved because epochwise holdout inference was not preregistered or performed.

## Decision

The limited conditional temporal benefit remains after mean centering and DC restoration, but seed-dependent DC bias, rapid overfitting, and the high-gradient retention tradeoff remain unresolved.
`CGS_ENTRY_ALLOWED = false`.

This audit supports operational attribution only. It does not establish clean-image recovery, physical causality, acceptable final image quality, or permission to implement CGS.
