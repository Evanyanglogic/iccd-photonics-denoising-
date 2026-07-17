# E2 Formal Input and Provenance Audit Protocol

## Scope

This audit reads the existing sCMOS content source, historical E2 manifests,
formal E1 outputs, and generator code. It does not generate a synthetic dataset
or train a model. Two fixed single-pair replays are allowed solely for numeric
round-trip and historical reproducibility checks.

## Operational Terminology

- `clean-content source`: one 500 ms sCMOS frame after center cropping, optional
  dark-offset subtraction, masking, normalization, and invalid-pixel filling. It
  is not ICCD clean ground truth and still contains unresolved sCMOS noise.
- `legacy_unscaled_content`: the historical output previously called
  `physical-scale`; it preserves the offset-corrected sCMOS normalized scale.
  It is not a calibrated physical ICCD model.
- `legacy_per_image_p99_0p25`: the historical output previously called `p99`;
  each corrected clean crop is independently multiplied so its valid-pixel p99
  equals 0.25 before noise injection.
- `formal E1 target`: an operational statistic measured at the frozen 512x512
  ICCD ROI. It is not automatically a parameter of a synthetic generator.

## Required Gates

`VERIFIED-INPUT` requires complete file hashes, valid dtype/shape, explicit
scene/source-group metadata, scene-isolated splits, reproducible numeric
round-trip, supported parameter units, and complete provenance. Any large
clipping, absent scene grouping, unsupported physical attribution, or
result-driven parameter selection blocks batch generation.

## Entry Point

```powershell
python scripts\run_e2_formal_input_audit.py --config configs\e2_formal_input_audit_20260717.yaml
```
