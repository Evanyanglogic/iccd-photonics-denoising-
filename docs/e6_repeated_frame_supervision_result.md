# E6 Repeated-Frame Supervision Feasibility

## Decision

The preregistered audit selected `E_REACQUIRE`. Formal real-domain training was
not started. This is a data-assumption failure, not a training failure.

Only 2/10 folders passed all preregistered stability and independence checks.
The 8-frame target reduced median random target noise by 2.627x, but this did
not make the target conditionally independent of the input across folders.

## Evidence Against Unqualified Noise2Noise

- Eight-frame group registration returned zero shift for every folder. The
  scene is mechanically stable at the measured scale, although the estimate
  can be anchored by sensor-fixed structure.
- Global first-to-last brightness shifts were below 0.17 temporal standard
  deviations, but local drift reached 0.684 in folder 5 and 0.420 in folder 7.
- High-frequency residual lag-1 correlations were small in every folder
  (absolute values below 0.007), supporting approximate independence of the
  highest-frequency temporal component.
- Pixel residual lag-1 correlation reached 0.132 in folder 1 and 0.129 in
  folder 5. Brightness-stratified residual correlation reached 0.260 and 0.308
  in the same folders.
- Persistent row/column correlations were large in several folders. Examples
  are folder 4 column 0.626, folder 5 column 0.788, folder 7 column 0.716, and
  folder 1 row/column 0.502/0.561. Correlation decayed over tens of frames.
- Split-half fixed maps were highly correlated in 9/10 folders. Existing E1
  fixed-to-temporal ratios range from 0.38 to 18.50. A repeated-frame target
  therefore retains sensor-fixed content that a model can learn as signal.
- Folder 13 has weak fixed structure and a split-half map correlation of
  0.916; its split-map RMSE is only 0.146 temporal standard deviations. This
  illustrates that correlation alone is unstable when fixed-map energy is
  low, but it does not rescue the multi-folder supervision assumption.
- The odd/even 50-frame references agree closely (62.0-78.3 dB), yet both
  retain shared fixed-pattern bias and are not clean ground truth.

Noise2Noise-style supervision could suppress approximately independent
high-frequency temporal noise. It cannot identify static fixed-pattern bias,
slow row/column fluctuations, gate/illumination drift, or scene detail shared
by input and target. The current 200-frame sequences can separate temporal
variation from the combined stable image, but cannot separate the true scene
from fixed-pattern noise without dark/flat calibration or an independent
high-SNR reference.

## Candidate Target Audit

| target | nonreused pairs/folder | mean target residual std (DN) | use boundary |
|---|---:|---:|---|
| independent single frame | 100 | 114.56 | maximum sample count; noisy and correlated target |
| disjoint 8-frame mean | 22 | 48.35 | best single-frame inference match, but retains fixed/slow noise |
| disjoint 16-frame mean | 11 | 38.94 | fewer samples and stronger temporal-mean bias |
| independent 8-frame means | 12 | 48.40 | input domain does not match single-frame inference |
| odd/even 100-frame means | 1 | 11.07 | evaluation replicate only, not a useful training set |
| synthetic to real temporal mean | 0 | not pairable | content domains do not correspond |

Protocol B would have been the only justified candidate if the audit had
passed: one real frame to a disjoint 8-frame temporal-mean surrogate. The audit
instead selected protocol E, so this candidate remains blocked.

## LOFO Leakage Design

Ten blocked fold manifests were generated to preserve the intended protocol.
Each fold has eight training folders (176 candidate pairs), one validation
folder (22 pairs), and one fully held-out test folder. Test references use
frames 1-100; test inputs use frames 101, 111, ..., 171. Every source frame has
one role per fold, and all ten folds pass the leakage checker.

The representative-fold rule selected folders 13, 10, and 5 from the minimum,
median, and maximum full-sequence temporal standard deviation. No feasibility
training was run because the audit gate failed before this selection could be
used.

## Preregistered Hypothesis and Unchanged Go Criteria

Core hypothesis: with a completely independent test folder, a small CNN
trained using real repeated-frame supervision outperforms the P-L synthetic
model on both temporal-mean references without greater gradient loss.

The null holds if real supervision does not stably beat P-L, its gain is below
seed variability, or the gain is produced mainly by smoothing.

Future training remains gated on both-reference improvement, positive mean
folder gain, at least 8/10 positive folders, worst-folder gain at least -0.05
dB, improvement larger than seed SD, gradient/noisy at least 0.95, negligible
brightness bias, and no systematic structure removal in error panels.

## Minimum Reacquisition

- Acquire at least 18 independent static scenes, six scenes at each of low,
  medium, and high signal conditions. Keep a final scene-level test split.
- Record at least 128 light frames per scene. Randomize or interleave frame
  roles rather than acquiring all target frames in one contiguous block.
- Record at least 64 dark frames before and after each gain/gate configuration.
- Record at least 64 flat-field frames at five illumination levels for every
  gain/gate configuration. This supports offset, fixed-pattern, photon-transfer,
  and signal-dependent variance separation.
- Use at least three preregistered gate or illumination conditions. Hold gain,
  exposure, lens, focus, temperature, and processing constant within a sequence
  and preserve all metadata.
- Use a static resolution/contrast target plus textured scenes. Monitor
  illumination with a reference detector when possible.
- Acquire an aligned high-SNR reference for every scene, preferably multiple
  low-gain/long-integration frames under unchanged geometry. Treat it as a
  measured reference with its own uncertainty, not perfect ground truth.
- Dark frames estimate offset/read and dark fixed pattern; flat-field
  mean-variance data estimate photon and gain components; repeated illuminated
  frames estimate temporal and row/column components; before/after calibration
  measures drift.

This acquisition is the minimum route to test real repeated-frame supervision
without asking the network to learn an inseparable scene-plus-fixed-pattern
target.

## Execution Record

- Completed: repeated-frame independence audit, target comparison, ten-fold
  role manifest generation, LOFO split generation, and leakage checking.
- Reused and reviewed: 2,625-parameter trainer, odd/even reference builder,
  float-domain PSNR/SSIM, gradient, brightness, and residual metrics.
- Deliberately skipped: target materialization, small-CNN training, checkpoint
  evaluation, seed aggregation, and visual prediction panels. The
  preregistered data gate prohibits these steps under `STOP_REACQUIRE`.
