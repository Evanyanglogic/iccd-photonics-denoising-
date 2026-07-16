# E3.5-A Condition Gate

## Purpose

Test whether E1 folder-level ICCD condition statistics can decide when to apply
the physical-scale denoiser on real gated ICCD surrogate pairs.

This is a minimal condition-aware validation step. It does not train a new
network and does not use per-image ground-truth gain to decide the gate.

## Script

```powershell
python scripts\evaluate_condition_gate.py `
  --eval-csv reports\e3_real_surrogate_eval_physical_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --p99-eval-csv reports\e3_real_surrogate_eval_p99_smallcnn_100ep\checkpoint_eval_metrics.csv `
  --condition-summary-csv reports\e3_condition_gain_analysis\condition_gain_summary.csv `
  --model-label physical `
  --output-dir reports\e3_5_condition_gate
```

Outputs:

- `reports\e3_5_condition_gate\condition_gate_pair_metrics.csv`
- `reports\e3_5_condition_gate\condition_gate_summary.csv`
- `reports\e3_5_condition_gate\condition_gate_report.md`

## Compared Strategies

| Strategy | Meaning |
|---|---|
| `always_noisy` | Never apply denoiser |
| `always_model` | Always apply physical-scale checkpoint |
| `always_p99` | Always apply p99 checkpoint |
| `gate_*_q40/q50/q60` | Apply physical checkpoint only when the E1 condition metric is above the given folder quantile |
| `oracle_folder_positive` | Diagnostic upper bound: apply physical checkpoint only to folders with positive observed mean gain |

The deployable candidates are the `gate_*` rows. The oracle row is not a valid
paper method, only an upper bound.

## Key Result

Best non-oracle gate:

```text
gate_mean_signal_q40
```

Equivalent q40 gates were obtained for temporal std, Fano, fixed-map std, and
fixed/temporal std ratio because the ten folders are similarly ordered by these
condition statistics.

| Strategy | Mean folder PSNR gain | Positive folders | Negative folders | Selected source count |
|---|---:|---:|---:|---|
| always_model | 0.3431 dB | 6/10 | 4 | physical:80 |
| always_p99 | 0.0392 dB | 10/10 | 0 | p99:80 |
| best q40 condition gate | 0.3669 dB | 6/10 | 0 | noisy:32; physical:48 |
| oracle_folder_positive | 0.3669 dB | 6/10 | 0 | noisy:32; physical:48 |

## Interpretation

The condition gate removes the four negative-gain folders from `always_model`
while preserving the high-gain folders. This supports the claim that the current
denoising benefit is condition-dependent and tied to real ICCD noise statistics.

It also reinforces the route change:

```text
condition-aware denoising validation > generic low-light detail restoration
```

## Guardrails

- The result uses only ten folders, so it is a diagnostic validation, not a final
  deployable model.
- The q40 threshold should not be over-interpreted; it is a simple condition
  split, not a learned classifier.
- Final manuscript wording should say "condition-aware gating reduces
  condition-specific degradation in this repeated-frame surrogate evaluation,"
  not "the method universally improves ICCD denoising."

## Next Step

Run E3.5-B:

```text
low-noise vs high-noise subset validation
```

This should report separate performance on low-condition and high-condition
folders, and then inspect visual samples from the gate boundary folders.
