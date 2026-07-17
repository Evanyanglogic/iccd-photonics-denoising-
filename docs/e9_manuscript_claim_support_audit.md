# E9 Manuscript Claim-Support Audit

Audit date: 2026-07-17

Frozen manuscript route:

> Gated ICCD noise characterization, conditional synthetic-real mismatch analysis, and denoising applicability validation.

This audit does not add model training, synthetic data, feature fitting, or result-driven statistical tests. It traces candidate manuscript claims to the frozen E1-E8 evidence chain and verified primary literature.

## 1. Scientific question

Under the current gated-ICCD repeated-frame acquisition and surrogate-based evaluation protocol, how are observed noise statistics and controlled synthetic-noise strength/structure mismatch associated with real-domain denoising transfer and oversmoothing boundaries?

This question is answerable without claiming physical-source identification, clean-ground-truth recovery, a deployable predictor, or cross-camera generalization.

## 2. Maximum defensible contributions

1. A multidimensional operational characterization of ten repeated-frame gated-ICCD folders, covering signal level, temporal variability, Fano-like dispersion, repeatable stable components, directional/spatial correlation, and drift, with crop/frame-count robustness checks.
2. A controlled 2x2 probe showing that the four tested synthetic priors all transfer negatively to the present real surrogate task, while residual strength, residual structure, and their interaction alter the magnitude of negative transfer; high-strength training is also accompanied by stronger gradient loss in the fixed small CNN.
3. A narrow folder-level association result: within the frozen E5 variants, larger residual-strength mismatch is associated with poorer surrogate-based transfer across the ten folders, with exact folder-profile permutation, leave-one-folder, seed/reference, gradient-partial, bootstrap, and negative-control results reported together.

The surrogate audit and repeated-frame supervision audit are methodological support and limitations, not separate headline contributions.

## 3. Claim grades

The machine-readable matrix is `reports/e9_manuscript_claim_support_audit/claim_evidence_matrix.csv`.

### A. Between-folder observed noise differences: C1

The differences span all ten folders and are much larger than the median crop/frame-count sensitivity for the principal E1 statistics. At the 512-pixel/32-frame baseline, mean signal spans about 936-4717 DN, temporal standard deviation 38.3-221.3 DN, Fano-like dispersion 1.70-14.46, and stable-to-temporal ratio 0.38-18.21. Median robustness changes are about 5.8%, 9.8%, 6.4%, and 16.3%, respectively. These measurements support `observed temporal variability`, `row/column-associated residual energy`, `spatial correlation`, and `repeatable stable component`. They do not uniquely identify dark-signal nonuniformity, PRNU, intensifier nonuniformity, or any physical source because scene and folder identity are confounded and matching dark/flat data are absent.

This claim may appear in the abstract and contributions only in operational language. It must not be phrased as verified acquisition-condition dependence because the recorded gate/gain/exposure metadata do not vary across the ten complete folders.

### B. Tested unified synthetic priors are not reliably transferable: C1

The claim is restricted to the priors and protocol tested here. E5 gave negative mean real-surrogate transfer for P-L (-0.496 dB), P-H (-4.315 dB), H-L (-1.892 dB), and H-H (-4.891 dB) across three seeds and two references. Earlier p99/physical/gate observations are retained as exploratory because their small positive differences are below the later seed variability. This supports `the tested unified synthetic priors did not transfer reliably`. It does not support `all synthetic noise modeling is ineffective`.

### C. Strength mismatch association: C1, narrow

The correct primary direction is negative: mean variant-specific Spearman rho is -0.6061, not +0.6061. Exact folder-profile permutation gives p=0.00293; all 10 leave-one-folder statistics and all 6 seed/reference summaries remain negative; the gradient-controlled partial rank correlation is -0.5077; and the negative control is near zero. The four variant results are heterogeneous: P-L -0.8424, P-H -0.6000, H-L -0.2485, H-H -0.7333. The folder percentile-bootstrap interval is [-0.500, 0.000], so uncertainty reaches the null and the H-L replication is weak.

This is eligible for a cautious abstract sentence and contribution item only when tied to the current camera batch, E5 variants, folder-level repeated-measure design, and temporal-mean surrogate evaluation. The permitted verb is `was associated with`, not `caused` or `predicted`.

### D. Spatial mismatch result: C2 negative result

The frozen radial-PSD distance did not support the preregistered negative association: mean rho +0.3455, exact p=0.1890, and three variants had positive signs. This only means that the current radial-PSD distance did not form a stable association in the expected direction. It does not establish that spatial structure is unimportant, especially because E5's controlled structure main effect was nonzero and prior real-noise synthesis work treats local/spatial correlation as consequential.

### E. High-strength training and oversmoothing: C2

The high-strength cells have much lower gradient/noisy ratios (P-H 0.3515 and H-H 0.4009) than their low-strength counterparts (P-L 0.5697 and H-L 0.5305), alongside strongly negative transfer. Gradient loss is an evaluation axis independent of surrogate PSNR, and visual/error panels are available. This supports `increased oversmoothing risk in the fixed small-CNN probe`. It does not support a statement about all denoisers or all backbones.

### F. Repeated-frame supervision limits: C2

Only 2/10 folders passed all frozen supervision gates. High-frequency temporal residual correlation was low and eight-frame averaging reduced random target noise, but pixel, brightness-bin, row/column correlations, stable components, and local drift violate naive independence or unbiased-target assumptions in most folders. The correct scope is `the present ten-folder dataset does not support a universal naive Noise2Noise protocol`. It is not evidence that all ICCD repeat sequences are unusable; local high-frequency self-supervision remains an untested possibility.

### C3 and C4 boundaries

- C3: prior gate/blend gains near 0.036 dB, single-model slight positive gains, folder-specific trends, and unregistered sensitivity observations.
- C4: precise physical-source attribution; a validated condition-aware generator; strength mismatch causes negative transfer; deployable gate or performance predictor; clean/true-detail recovery; generalization to other ICCD devices; all synthetic modeling is ineffective; spatial structure is irrelevant.

## 4. E8 wording by manuscript location

### Abstract

For this gated-ICCD batch and the controlled E5 small-CNN protocol, residual-strength mismatch showed a consistent folder-level association with poorer temporal-mean-surrogate transfer across seeds and two reference constructions; the association remained uncertain at the ten-folder sample size and is not interpreted causally.

### Results

Across four frozen synthetic variants, the unweighted mean of variant-specific folder-level Spearman correlations was -0.6061 (P-L -0.8424, P-H -0.6000, H-L -0.2485, H-H -0.7333). Exact folder-profile permutation yielded p=0.00293; all 10 leave-one-folder statistics and all 6 seed-reference summaries remained negative, and gradient-controlled partial rank correlation was -0.5077. However, the percentile folder-bootstrap interval was [-0.500, 0.000], and the H-L association was weak, limiting the result to a narrow association within this device and protocol.

### Discussion

One possible explanation is that residual scale mismatch changes the effective denoising strength learned by the probe, thereby changing both transfer and smoothing. This remains a hypothesis: folder/scene identity, imperfect surrogate references, variant dependence, and unmeasured camera state can produce or modify the association.

## 5. Bootstrap and permutation audit

The two results answer different questions and are not logically contradictory.

- Exact permutation tests the null that folder-level mismatch profiles are exchangeable relative to transfer profiles while preserving each folder's full repeated-measure block. It is the primary preregistered inferential result.
- Folder bootstrap describes sampling instability when only ten folders are available. The frozen 10,000-resample run produced 9,834 finite statistics; 8,769 were negative, 1,065 were exactly zero, none were positive, and 166 were undefined because duplicate-heavy resamples produced degenerate ranks. The 97.5th percentile is exactly 0.0, not rounded from a positive value.
- The bootstrap distribution is discrete and biased enough that its interval does not contain the observed -0.6061 statistic. This is itself a warning against presenting the point estimate as precise.

The manuscript must report both the exact permutation p-value and the zero-touching percentile interval. BCa bootstrap is not added after seeing the result: at n=10 with ties, degenerate resamples, and a mean-of-rank-correlations statistic, its jackknife acceleration can be unstable, and post hoc interval substitution would violate the frozen audit. A future larger, preregistered dataset can use a prespecified interval procedure.

## 6. Variant aggregation audit

The four variants have equal status in the frozen 2x2 design and share clean content, generator components, architecture, and evaluation folders. Their correlations are therefore related repeated views, not four independent replications. The unweighted mean has a clear but limited interpretation: average direction across the four prespecified E5 cells. It is not a pooled n=40 effect and it must not hide cell heterogeneity.

The manuscript must report the aggregate statistic, all four variant-specific correlations, and H-L as the weakest/worst replication. P-L cannot be promoted post hoc to the primary variant. P-L may be discussed as the most conservative cell, but changing the confirmatory hierarchy after observing results would invalidate the E8 preregistration.

## 7. Minimal figure and table set

The machine-readable map is `reports/e9_manuscript_claim_support_audit/figure_claim_map.csv`.

- Figure 1: acquisition, repeated-frame partitions, temporal-mean surrogate, E5/E8 workflow. New assembly only; no new result.
- Figure 2: folder-level operational characterization and robustness. Assemble existing mean-variance/Fano, stable-component, directional/spatial-correlation, and drift outputs.
- Figure 3: controlled synthetic construction and E5 strength/structure/interaction effects. Combine existing construction checks and factor-effect plot; do not duplicate every PSD panel.
- Figure 4: E8 strength-mismatch association as the primary panel, with the unsupported radial-PSD result as a clearly labeled sensitivity/null panel.
- Figure 5: predetermined representative E5 output/residual/error panels with gradient ratios, including adverse cases. Selection rule must be stated before assembly.
- Table 1: ten-folder data integrity, metadata, characterization/surrogate/supervision eligibility.
- Table 2: four E5 cells, three-seed uncertainty, two-reference agreement, gradients, worst folder, and factor effects.
- Table 3: E8 aggregate and four variant correlations, exact permutation, bootstrap, LOO, seed/reference, partial-gradient result, and negative control.

## 8. Literature positioning and nearest neighbors

Direct ICCD detector studies already establish multistage stochastic gain, dark/shot components, nonuniformity, and the need for controlled calibration. Sandel and Broadfoot (1986), Williams and Shaddix (2007), Jin et al. (2012), and Pelaez et al. (2012) prevent claims that this paper first characterizes ICCD noise. Wang et al. (2026) further shows that a modern intensifier-detector characterization uses controlled illumination, dark frames, excess-noise-factor measurements, nonlinearity, uniformity, MTF, and SNR modeling; the current data do not meet that calibration standard.

Direct ICCD denoising also predates this work: Wang et al. and Yang et al. (Sensors, 2017) address randomly clustered ICCD noise. Han et al. (Sensors, 2021) reports that Poisson-Gaussian training mismatch degrades intensified-CMOS denoisers. Luo et al. (Remote Sensing, 2025) is the closest intensified-sensor noise-generator paper, using 2,000 ICMOS contents, 40,000 noisy instances, frame-integrated references, a learned local-correlation generator, and downstream denoising. It is ICMOS, not ICCD, but substantially narrows broad novelty claims.

General real-noise work by Plötz and Roth (2017), Zhang et al. (ICCV 2021), and Fu et al. (CVPR 2023) already establishes that marginal scale matching is insufficient and that downstream real denoising and spatial/signal dependence matter. In the verified search, no direct gated-ICCD paper was found that combines a prespecified 2x2 strength/structure probe with folder-blocked mismatch-to-transfer association and seed/reference/gradient uncertainty. The defensible novelty is this linked, failure-aware evidence chain, not any individual characterization metric, denoising network, or noise-generation concept.

With ten scene-confounded folders and no matched dark/flat or controlled multi-setting acquisition, the work is best positioned for *Acta Photonica Sinica* if written as detector-noise observation and algorithm-applicability analysis. *Applied Optics* would require a stronger calibration and independent acquisition evidence chain. *Sensors* is possible but has close intensified-camera predecessors with substantially richer paired data. *Optics Express*, IEEE TIM, and Measurement should not be primary targets without additional calibrated acquisition.

## 9. Decision

**GO-WRITE**

The evidence is sufficient to write the data, methods, characterization, controlled mismatch, applicability-results, and limitations sections under the frozen claims above. No further decisive experiment is required before drafting. Remaining work is figure/table assembly, exact traceability in captions, and claim-constrained writing; it must not alter the frozen findings.

## 10. Recommended titles

- Conservative: `门控 ICCD 重复帧噪声表征与去噪适用性分析`
- Balanced: `门控 ICCD 噪声表征、合成噪声失配与受控去噪验证`
- Assertive, only with the qualified E8 sentence retained: `门控 ICCD 合成噪声强度失配与真实域去噪迁移的文件夹级关联分析`

## 11. Final conclusion boundary

The paper may conclude that the present gated-ICCD repeated-frame batch exhibits substantial between-folder differences in operational noise statistics; the tested synthetic priors transfer unreliably under the fixed small-CNN surrogate protocol; controlled strength, structure, and interaction changes alter negative transfer; and residual-strength mismatch has a narrow, noncausal folder-level association with transfer. It may also document oversmoothing and naive repeated-frame-supervision boundaries.

It may not conclude physical-source identification, calibrated condition dependence, clean-image recovery, a universal synthetic generator, a deployable gate/predictor, backbone-independent behavior, or validity on other ICCD systems.
