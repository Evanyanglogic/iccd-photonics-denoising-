# E3 PyTorch Training-Code Audit

## Scope

Reviewed the current parent-repository training code under `E:/PNGAN-main`:

- `train_pngan.py`
- `train_denoiser_final.py`
- `train_denoiser_scmos.py`
- `dataloaders/dataset_grayscale_tif.py`
- `scmos_noise_model.py`
- `utils/metrics.py`

Audit rules came from `iccd-denoising-optimizer` and `pytorch-patterns`: data
and metrics must be fixed before changing model architecture.

## Current Decision

Do not use the legacy training scripts directly for the ICCD paper experiments.

Reuse these parts only after wrapping them in a manifest-driven training entry:

- MIRNet/SMNet grayscale model definitions;
- patch discriminator if PNGAN is later needed;
- existing grayscale loss components after metric/range checks;
- visualization/checkpoint patterns after reproducibility fixes.

## Findings

### 1. Pairing Is Directory-Sorted, Not Manifest-Safe

`dataloaders/dataset_grayscale_tif.py` pairs images by sorted filenames under
`clean/` and `noisy/`.

Risk:

- It ignores `pairs.csv` and `splits.yaml`.
- It cannot enforce train/val/test split consistency.
- It cannot carry source metadata such as device type, exposure, prior config,
  dark-offset artifact, or claim boundary.

Required fix:

- Use `src/iccd_data.ICCDPairDataset` from this subproject for ICCD experiments.
- Every training run must record the exact manifest and split file.

### 2. Paths and Experimental Roles Are Hardcoded

Examples:

- `train_pngan.py` uses `E:/PMRID/PMRID7/data`.
- `train_denoiser_final.py` uses `./synthetic_data`.
- Validation often reuses the same PMRID root as training.

Risk:

- Easy to train on the wrong data.
- Hard to separate real-only, synthetic-only, and mixed-training arms.
- Not acceptable for a paper evidence chain.

Required fix:

- Training entry should accept CLI/config paths:
  - `--train-pairs`
  - `--train-splits`
  - `--val-pairs`
  - `--val-splits`
  - `--experiment-id`
  - `--output-dir`

### 3. Reproducibility Is Incomplete

Observed:

- No global seed setup for Python `random`, NumPy, Torch CPU, Torch CUDA.
- Random crops and augmentations use unseeded `random`/`np.random`.
- Checkpoints do not consistently save random states.

Required fix:

- Add one `set_seed(seed)` utility.
- Save `config`, git commit, seed, manifest paths, and best/last checkpoints.

### 4. Legacy Noise Model Is Not Suitable for ICCD Training

`scmos_noise_model.py`:

- models sCMOS-like noise, not ICCD;
- converts Torch tensors to CPU NumPy inside `batch_add_noise`;
- uses global NumPy randomness;
- is not tied to E1-derived ICCD statistics.

Required fix:

- Do not use this model for ICCD claims.
- For synthetic ICCD data, use the already generated manifest-backed TIFF pairs
  from `scripts/generate_iccd_like_synthetic_pairs.py`.
- If online noise generation is later needed, implement a Torch-native
  `ICCDNoiseModel` equivalent with explicit seed control.

### 5. Metric Implementations Are Not Yet Consistent

`utils/metrics.py`:

- PSNR assumes normalized `[0, 1]`, which is acceptable if loaders are correct.
- SSIM converts images to uint8 before calling `skimage.metrics.ssim`.

Risk:

- Quantizing weak 16-bit low-light images to uint8 can hide small residual
  structure.
- Results may differ from `src/iccd_eval.metrics`, which is used by the data
  audit scripts.

Required fix:

- Use one metric implementation for all E3/E4 tables.
- Prefer `src/iccd_eval.metrics` for manifest-based evaluation reports.
- Record `data_range=1.0` for tensors and `range_max=65535` for TIFF tools.

### 6. Training Loops Need Paper-Experiment Controls

Positive patterns:

- Device selection is mostly device-agnostic.
- `model.train()` / `model.eval()` are used.
- Checkpoints save model and optimizer state.
- Gradient clipping is present.

Gaps:

- `optimizer.zero_grad()` should use `set_to_none=True`.
- AMP handling is incomplete or unused in denoiser scripts.
- `torch.cuda.empty_cache()` is called inside loops as a workaround rather than
  a measured memory strategy.
- No per-epoch metrics CSV is written.
- No best/median/worst validation sample selection.

Required fix:

- Build a small, manifest-based baseline trainer first.
- Start with L1-only or L1+SSIM denoising, not PNGAN.
- Only introduce PNGAN/adversarial components after the supervised baseline is
  reproducible.

## Recommended E3 Path

1. Implement `scripts/train_manifest_denoiser_baseline.py`.
2. Use `reports/target_scmos_iccd_like_synthetic_512_p99_0p25/pairs.csv` as the
   first synthetic training-source candidate.
3. Use the same split manifest for train/val/test initially, then later add
   real ICCD surrogate validation when the target claim requires it.
4. Train a small baseline first:
   - input: noisy synthetic ICCD-like TIFF;
   - target: clean sCMOS content/reference TIFF;
   - loss: L1 only for the first smoke test;
   - metrics: PSNR/SSIM from `src/iccd_eval.metrics`;
   - outputs: `metrics.csv`, `config.yaml`, `best.pth`, `last.pth`, visual
     samples.
5. After the baseline runs end-to-end, compare:
   - no-model B0;
   - strict physical-scale synthetic training;
   - p99-normalized synthetic training;
   - later real ICCD surrogate or true paired ICCD test data.

## Current Gate

E3 should not start by modifying SMNet, MIRNet, loss weights, or PNGAN. The next
engineering task is a manifest-driven supervised denoiser baseline with a smoke
training mode.
