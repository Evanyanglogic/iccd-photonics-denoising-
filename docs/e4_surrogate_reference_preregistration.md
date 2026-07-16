# E4 Surrogate Reference Reliability Preregistration

## Decision Question

Can the existing condition-aware conclusion survive when the temporal-mean
surrogate is replaced by two disjoint reference estimates?

## Primary Hypothesis

When the temporal-mean surrogate is rebuilt from two disjoint, interleaved
frame sets, the preregistered LOFO linear condition strategy outperforms both
fixed p99 and fixed physical strategies on each reference replicate.

The null is accepted if the advantage over the best fixed strategy is absent
on either reference, or the folder-level conclusion changes materially with
the reference replicate.

## Single Primary Independent Variable

`surrogate_reference_replicate`: odd frames 1-99 versus even frames 2-100.
Each reference contains 50 frames. Held-out inputs remain frames 101, 111,
121, 131, 141, 151, 161, and 171.

## Frozen Controls

- Data root: `D:/iccd/data/20260319`
- Pair manifest: `reports/gated_iccd_20260319_surrogate_pairs/pairs.csv`
- Crop/data range: center 512 x 512, linear float `[0,1]`, `data_range=1`
- Model outputs: existing p99 and physical small-CNN best checkpoints
- Training controls: 128 patch, 100 epochs, Adam, L1, seed 20260716
- Checkpoint rule: synthetic-validation PSNR only
- Condition rule: existing E3.8 LOFO selections, with each folder excluded from
  its threshold/interval selection
- Metrics: shared `src/iccd_eval/metrics.py` implementation

No model is retrained and no threshold is selected on either reference in this
experiment.

## Preregistered Decision

GO requires both reference replicates to show a positive LOFO-linear advantage
over the best fixed model, at least 9/10 positive folders, worst-folder gain no
lower than -0.05 dB, at least 0.8 folder-sign agreement, and no additional
gradient-ratio drop larger than 0.01 relative to physical.

PARTIAL means the average advantage survives both references but one of the
folder robustness or smoothing checks fails. Any loss of average advantage or
reference sign stability is NO-GO for the gate/blend main claim.

## Command

```powershell
python scripts\audit_surrogate_reference_reliability.py `
  --config configs\e4_surrogate_reference_reliability.yaml
```

The experiment writes pair-, folder-, and strategy-level CSV files, reference
agreement statistics, best/median/worst diagnostic panels, a machine-readable
decision, and a Markdown report under
`reports/e4_surrogate_reference_reliability`.
