# Tooling Install Plan

Saved on 2026-07-15. This plan records which image, experiment, and research
tools are worth adding to the ICCD denoising workflow, and which ones should be
deferred.

## Current Environment

Already available:

- MCP: `arxiv`, `brave-search`, `github`.
- Skills: `iccd-denoising-optimizer`, `pytorch-patterns`,
  `academic-research-suite`, `paper-spine`, `research-paper-writing`.
- Local project scripts for TIFF inventory, paired-data audit, no-model
  baseline, noise-prior comparison, manifest dataloader checks, and
  single-condition ICCD noise summaries.

Current limitation:

- The repo does not yet have `requirements.txt`, `pyproject.toml`, or
  `environment.yml`.
- Any Python package installation should therefore start by creating an explicit
  dependency file and, preferably, using an isolated environment.

## Install First

### 1. FiftyOne

Priority: high.

Reason:

- Best fit for visual inspection of image datasets, noisy/clean pairs, model
  outputs, metric-based filtering, and worst-case sample review.
- Directly supports the project need to connect image path, metadata, PSNR/SSIM,
  model output, and experiment ID.

Initial use:

- Import `pairs.csv` plus model outputs into a FiftyOne dataset.
- Store fields such as `noisy_path`, `clean_path`, `output_path`, `exposure`,
  `scene_id`, `frame_id`, `psnr`, `ssim`, and `experiment_id`.
- Use the UI to inspect low-PSNR samples, suspected mispairing, brightness drift,
  oversmoothing, and fixed-pattern artifacts.

Install timing:

- Install after adding a dependency file.
- Does not need to block the next mean-variance and fixed-pattern scripts.

### 2. MLflow

Priority: high.

Reason:

- Needed once model training starts.
- Tracks configs, git commit, seeds, metrics, artifacts, checkpoints, and model
  comparisons.

Initial use:

- Record `experiment_id`, model name, dataset manifest, git commit, patch size,
  loss weights, learning rate, seed, best PSNR/SSIM, parameter count, latency,
  and visual artifacts.

Install timing:

- Install before the first serious supervised denoising baseline.
- Not required for current Stage 2 noise-statistics scripts.

### 3. Kornia

Priority: medium-high.

Reason:

- Useful inside PyTorch training and validation for differentiable image
  processing.
- Better than OpenCV for tensor-native losses and augmentations.

Initial use:

- Gradient/edge loss.
- Differentiable filtering.
- Possible registration or alignment experiments if paired ICCD data require it.

Install timing:

- Install when editing training code or adding loss functions.
- Do not use it for the current offline raw TIFF audits unless tensor-native
  processing is needed.

### 4. PIQ / TorchMetrics / LPIPS

Priority: medium.

Reason:

- Adds complementary image-quality metrics beyond PSNR/SSIM.
- Useful for validation logging and perceptual checks, but should not replace
  PSNR/SSIM as the main paper metrics.

Initial use:

- PIQ or TorchMetrics for validation-loop metrics.
- LPIPS for supplemental perceptual inspection when denoising outputs exist.

Install timing:

- Install with the training stack, not before the current device-noise scripts.

## Install Later or Conditionally

### 5. DVC

Priority: medium, but storage-dependent.

Reason:

- The project has large TIFF datasets and checkpoints that should not be stored
  in Git.
- DVC is appropriate for dataset versions, split manifests, model checkpoints,
  and reproducible pipelines.

Condition before installing:

- Decide the DVC remote target: local external disk, NAS, S3-compatible object
  storage, OneDrive-synced folder, or another remote.
- Define what is tracked by DVC versus Git.

Recommended policy:

- Git tracks code, configs, manifests, and compact reports.
- DVC tracks large TIFF datasets, generated model outputs, checkpoints, and
  large visual artifact sets.
- MLflow tracks run metadata and selected artifacts.

### 6. FiftyOne Skills

Priority: conditional.

Reason:

- Potentially useful for dataset curation workflows.
- We should first confirm the repository structure and install path, because
  the current Codex skill installer list request returned HTTP 403.

Condition before installing:

- Inspect the GitHub skill contents.
- Install only the specific dataset-curation skill if it maps cleanly to Codex.

### 7. OpenCV MCP Server

Priority: low-medium.

Reason:

- OpenCV is useful, but an MCP wrapper is not necessary for most local analysis
  because the repo can call `opencv-python`, `tifffile`, `numpy`, and
  `scikit-image` directly in auditable scripts.

Condition before installing:

- Inspect the community MCP server source and permissions.
- Confirm it will not modify or overwrite raw data.
- Prefer read-only usage and output to `reports/`.

Recommended alternative now:

- Add `opencv-python` only if a script needs registration, morphology, feature
  matching, or classical filtering.

### 8. AI-Research-SKILLs / scientific-agent-skills

Priority: low for now.

Reason:

- The project already has `academic-research-suite`, `paper-spine`,
  `pytorch-patterns`, and the custom `iccd-denoising-optimizer`.
- More broad research skills may add overlap and routing noise before the
  baseline experiments are stable.

Condition before installing:

- Install only if a specific gap appears:
  - statistics/reporting templates not covered by current docs;
  - experiment observability not covered by MLflow;
  - model architecture guidance not covered by `pytorch-patterns`.

## Do Not Install Yet

- Full AI-Research-SKILLs collection.
- Full scientific-agent-skills collection.
- Any image-processing MCP that requests broad filesystem write access.
- Any tool that automatically mutates raw TIFF datasets.

## Recommended Immediate Dependency File

Add a small dependency file before installing packages:

```text
tifffile
numpy
pandas
matplotlib
scikit-image
opencv-python
fiftyone
mlflow
kornia
piq
torchmetrics
lpips
```

The first dependency pass can exclude training-only packages if we only run
Stage 2 scripts:

```text
tifffile
numpy
pandas
matplotlib
scikit-image
opencv-python
```

## Recommended Install Order

1. Add `requirements-analysis.txt` for offline data/noise scripts.
2. Install analysis dependencies in an isolated environment.
3. Implement and run mean-variance and fixed-pattern scripts.
4. Add `requirements-training.txt` before supervised training starts.
5. Install FiftyOne and MLflow when outputs and metrics need visual and
   experiment tracking.
6. Add DVC after the storage remote is decided.
7. Reconsider MCP/extra skills only after the first stable baseline.

