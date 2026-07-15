# ICCD Research Workflow

This workflow records when to use each installed MCP server and Codex skill for
the ICCD low-light denoising project. The operating rule is:

1. Local data evidence first.
2. Training and metric correctness second.
3. Literature and paper framing third.
4. Model changes last, unless a prior audit identifies a specific bottleneck.

## Current Saved State

Saved as of 2026-07-15:

- Latest commit before this workflow: `6d7cfb3 Add single-condition ICCD noise summary`.
- Complete gated ICCD folders under `D:/iccd/data/20260319`:
  `1,2,4,5,7,8,9,10,11,13`.
- Incomplete folders: `3`, `6`, `12`, and helper folder
  `1_20260715_143749`.
- All complete gated ICCD folders currently share exposure width 900 ms,
  Sync A/B width 4 us, and gain 60.
- Single-condition center-crop summary shows mean signal about 936 to 4717 DN,
  temporal standard deviation about 37.7 to 217.9 DN, and approximate Fano
  factor about 1.64 to 14.0.

## Tool Routing

| Trigger | Primary tool | Secondary tool | Output |
|---|---|---|---|
| New ICCD/sCMOS data appears locally | `iccd-denoising-optimizer` + local audit scripts | None | Inventory, metadata table, range stats, split status |
| Need real device noise evidence | `iccd-denoising-optimizer` | Local scripts under `scripts/` | Mean-variance, dark/flat, fixed-pattern, temporal residual reports |
| Need to inspect PyTorch data/training code | `pytorch-patterns` | Local tests/smoke scripts | Risk list, minimal fixes, reproducibility checklist |
| Need recent low-light/denoising papers | arXiv MCP | Brave Search MCP | Candidate papers, abstracts, downloaded full text when needed |
| Need non-arXiv sources, journal policy, venue scope, official pages | Brave Search MCP | Codex web browsing fallback | Source links and verification notes |
| Need structured literature review or experiment plan | `academic-research-suite` | arXiv MCP + Brave Search MCP | Research question, source matrix, experiment plan, claim boundary |
| Need manuscript/report construction | `paper-spine` | `academic-research-suite` | Confirmed contribution, section blueprint, draft, audit reports |
| Need repository/remote coordination | GitHub MCP | local `git` | Remote issue/PR/release context, if a remote repo is used |

## MCP Use Rules

### arXiv MCP

Use for peer-reviewed or preprint discovery in computer vision, signal
processing, image restoration, and low-light noise modeling.

Preferred sequence:

1. `search_papers`: broad candidate discovery.
2. `get_abstract`: relevance check before full download.
3. `download_paper`: only for papers likely to support the paper route.
4. `read_paper`: extract method, dataset, metric, and limitation evidence.
5. `watch_topic`: optional, after the final keywords stabilize.

Initial smoke result on 2026-07-15: arXiv MCP search worked for
`"low light image denoising"` and returned relevant low-light denoising/noise
synthesis papers.

### Brave Search MCP

Use for source types that arXiv does not cover well:

- Journal scope and formatting requirements.
- Official venue pages and publisher policies.
- Device or camera documentation.
- Non-arXiv papers, project pages, datasets, and code repositories.
- Recent web-only updates.

Current note: a Brave web search smoke query returned `fetch failed` on
2026-07-15. Treat Brave as the first web MCP to try, but fall back to normal
Codex web browsing or arXiv when it fails.

### GitHub MCP

Use only when repository-hosted context is needed:

- Inspecting remote issues, pull requests, branches, or templates.
- Creating issues or pull requests after local work is ready.
- Searching upstream code if a dependency or baseline implementation matters.

Do not use GitHub MCP for normal local edits; local `git` is enough.

## Skill Use Rules

### `iccd-denoising-optimizer`

Use first for any ICCD/sCMOS data, metric, experiment, or denoising-route
question.

It blocks large model changes until these are checked:

- 16-bit TIFF preservation.
- Short/long or noisy/clean pairing.
- Exposure, gain, gate, and device metadata.
- Dark/flat calibration status.
- Train/validation/test condition separation.
- PSNR/SSIM `data_range`.
- Whether improvement is just brightness correction.

### `pytorch-patterns`

Use after data manifests and metrics are stable, or whenever touching PyTorch
code.

Checklist:

- Device-agnostic CPU/GPU code.
- Random seed and deterministic settings.
- Tensor range and shape checks.
- Dataset and DataLoader performance.
- Correct `train()` and `eval()` behavior.
- Checkpoint contents sufficient for resume.
- AMP, gradient clipping, and memory use only after correctness checks.

### `academic-research-suite`

Use for research planning, literature review, experiment validation, and claim
discipline.

Project-specific routing:

- Use deep-research mode for literature map and gap analysis.
- Use experiment-agent mode for experiment design and statistical
  interpretation.
- Use academic-pipeline mode when moving from evidence to paper structure.

Do not use it to invent citations or results. Every claim must map back to local
reports, downloaded papers, or verified sources.

### `paper-spine`

Use only after the contribution and evidence boundary are mature enough for a
manuscript workflow.

Entry criteria:

- Main research question is fixed.
- Noise statistics figures exist.
- At least one reproducible denoising baseline exists.
- Claim boundary is written.
- Target journal requirements are known.

Expected artifacts:

- Confirmed contribution.
- Section blueprints.
- Writing rationale matrix.
- Evidence bank and claim register.
- Final LaTeX/PDF/Word only after audit gates pass.

## Stage Plan

### Stage 0: Progress Save

Run before any new branch of work:

```powershell
git status --short
git log -1 --oneline
```

If useful work was produced, update `docs/progress_memory.md` and commit it.

### Stage 1: Data Inventory

Use `iccd-denoising-optimizer`.

Inputs:

- Raw ICCD/sCMOS folders.
- Metadata files.
- Known exposure/gain/gate mapping.

Outputs:

- Folder inventory.
- Metadata coverage.
- Complete/incomplete folder list.
- TIFF dtype/shape/range summary.

Gate to Stage 2: complete folder list and metadata interpretation are written.

### Stage 2: Device Noise Characterization

Use `iccd-denoising-optimizer`.

Outputs:

- Dark-frame distribution.
- Flat-field mean-variance curve.
- Temporal residual statistics.
- Fixed-pattern map.
- Brightness-bin residual statistics.
- PSD/autocorrelation if spatial structure is visible.

Gate to Stage 3: at least one real-device statistical signature is repeatable
across folders or conditions.

### Stage 3: Literature and Claim Boundary

Use arXiv MCP, Brave Search MCP, and `academic-research-suite`.

Outputs:

- Literature matrix.
- SOTA gap map.
- Device-noise comparison table.
- Claim boundary for Photonics Journal style.

Gate to Stage 4: every planned claim has either local evidence or literature
support.

### Stage 4: Training Pipeline Audit

Use `pytorch-patterns` plus `iccd-denoising-optimizer`.

Outputs:

- Dataset manifest validation.
- Tensor range validation.
- Baseline metric reproduction.
- Train/validation/test split check.
- One no-model baseline and one simple supervised baseline.

Gate to Stage 5: no unresolved high-risk data or metric bugs.

### Stage 5: Minimal Model Experiments

Use `iccd-denoising-optimizer` and `pytorch-patterns`.

Allowed first experiments:

- Dark/flat corrected baseline.
- ICCD-prior synthetic noise baseline.
- Real-only vs synthetic-only vs mixed training.
- One controlled loss or architecture change at a time.

Outputs:

- Config, commit, seed, split, metrics, visual panels, checkpoint, and latency.

Gate to Stage 6: at least one result improves real held-out data without only
performing brightness correction or oversmoothing.

### Stage 6: Paper Construction

Use `academic-research-suite` first, then `paper-spine`.

Outputs:

- Confirmed contribution.
- Section blueprint.
- Evidence bank.
- Claim register.
- Manuscript draft and audit report.

Gate: no unsupported citation, metric, figure, or contribution claim.

## Default Next Action

The next technical step should be Stage 2:

1. Fit brightness-bin mean-variance curves from the repeated gated ICCD folders.
2. Add a fixed-pattern correction baseline.
3. Compare residual statistics before and after correction.
4. Save the resulting tables and plots under `reports/`.
5. Promote stable summaries into `docs/gated_iccd_data_inventory.md`.

