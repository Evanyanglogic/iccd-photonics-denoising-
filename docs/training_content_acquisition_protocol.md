# Training Content Acquisition Protocol

This protocol defines a future `training_content_only` operational source. It does not authorize acquisition, synthetic generation, splitting, or model training.

## Scope

- Use a controllable high-bit-depth grayscale-capable camera, preferably the available sCMOS after its export behavior is documented.
- Name the source `newly acquired operational training content`; do not call it clean ground truth or ICCD reference data.
- Keep it dataset-level isolated from PMRID validation and from every real ICCD evaluation scene.

## Minimum And Recommended Scale

- Minimum: 20 independently composed scenes, 3 files per scene, 60 files total.
- Recommended: 40 independently composed scenes, 5 files per scene, 200 files total.
- Repeated captures from one scene remain in one `acquisition_group` and one `isolation_block_id`.
- Consecutive frames do not increase the independent scene count.

## Scene Design

Cover low, medium and high brightness; flat and textured areas; fine lines and hard edges; natural and industrial objects; low and high spatial frequencies; and varied but non-saturated dynamic ranges. Do not reproduce the composition of a real ICCD test scene.

## Directory Layout

```text
training_content_acquisition/
  acquisition_<date>_<device>/
    acquisition_manifest.csv
    scene_<scene_id>/
      group_<acquisition_group>/
        <camera-native files>
```

## Capture Rules

Preserve camera-native source files. Record device, serial when available, date/time and timezone, exposure, gain/ISO, temperature, readout mode, binning, ROI, black-level setting, lens, capture software, and internal correction settings. Avoid clipping and use low gain where practical. Do not apply dark subtraction, scalar pedestal subtraction, per-image p99 scaling, gamma, histogram adjustment, denoising, bad-pixel interpolation, or flat-field correction before a separate input audit freezes preprocessing.

## Post-Capture Gate

Before assigning `training_content_only`, verify file count, SHA256, dtype, shape, bit depth, metadata, processing status, zero/saturation ratios, scene/group integrity, duplicate and near-duplicate rates, PMRID independence, and real-ICCD-scene non-overlap. Until that audit passes, the materialized source remains a candidate and formal synthetic generation remains prohibited.
