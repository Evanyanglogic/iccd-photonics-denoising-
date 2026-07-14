# Dataset Protocol

## Proposed Directory Layout

```text
data/
  iccd/
    dark/
      gain_<id>_gate_<id>_exp_<id>/
    flat/
      level_<id>_gain_<id>_gate_<id>_exp_<id>/
    paired/
      clean/
      noisy/
    test_real/
      clean/
      noisy/
  scmos/
    dark/
    flat/
    paired/
    test_real/
```

Do not commit raw data to git. Commit only manifests, scripts, and derived small
figures where needed.

## Metadata Fields

Each capture group should record:

- device: ICCD or sCMOS
- sensor/camera model
- gain
- gate width
- exposure time
- illumination level
- frame count
- bit depth
- temperature if available
- optical setup notes
- file naming convention

## Split Rule

Use device-condition splits, not only random image splits:

- Train: a subset of scenes and parameter settings.
- Validation: held-out frames from seen settings.
- Test: held-out scenes and at least one held-out device setting.

This prevents reporting only memorization of one capture condition.

