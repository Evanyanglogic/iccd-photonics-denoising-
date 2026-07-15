# Literature Matrix: Gated ICCD Noise Characterization and Synthetic-Data Denoising

Date started: 2026-07-15

Working paper title:

```text
基于重复帧统计的门控 ICCD 弱光噪声表征与合成数据驱动去噪方法
```

## Claim Boundary

This matrix is for positioning, not final citation verification. Each source
listed here needs DOI/publisher metadata verification before manuscript
submission. The current contribution should not be framed as a generic deep
denoising network. The stronger framing is:

```text
real gated-ICCD repeated-frame statistics
-> device-aware noise prior
-> synthetic paired data for denoising
-> real-device statistical validation
```

## Positioning Summary

| Literature area | What prior work already covers | Gap for this project | How our work should differ |
|---|---|---|---|
| Low-light RAW denoising | Paired short/long exposure training for consumer sensors | Not focused on gated ICCD or intensifier-chain noise | Use as denoising baseline context, not as the novelty claim |
| Physics-based raw noise synthesis | Poisson-Gaussian and learned raw-noise synthesis for CMOS-like cameras | Does not directly calibrate from gated ICCD repeated-frame statistics | Replace generic raw noise assumptions with ICCD-specific Fano/fixed-pattern evidence |
| Real-camera noise datasets | Real noisy/clean pairs and smartphone sensor statistics | Mostly Bayer/CMOS/smartphone sensors, not ICCD | Use sCMOS/CMOS literature as contrast class |
| ICCD/MCP/image-intensifier physics | MCP gain, phosphor, photon-counting and intensified imaging behavior | Often not connected to deep denoising data generation | Use this to justify ICCD-specific statistical terms |
| Self-supervised denoising | Training without clean targets | Does not by itself solve device-specific ICCD noise fidelity | Keep as fallback/auxiliary route if paired training data remain unavailable |
| Camera characterization standards | Mean-variance and photon transfer style calibration | Requires dark/flat or controlled flat-field data for strong calibration | Use current repeated-frame results conservatively; add matching dark/flat when available |

## Source Matrix

| Source | Area | What it did | What it does not solve for us | Use in our paper |
|---|---|---|---|---|
| Chen et al., "Learning to See in the Dark" ([arXiv:1805.01934](https://arxiv.org/abs/1805.01934)) | Low-light RAW denoising | Uses paired short-exposure noisy and long-exposure reference RAW data for extreme low-light enhancement/denoising | Consumer camera RAW, not gated ICCD; relies on paired exposure data | Cite as baseline low-light paired-denoising paradigm; distinguish our ICCD data limitation and device prior |
| Wei et al., "A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising" ([arXiv:2108.02158](https://arxiv.org/abs/2108.02158)) | Physics-based noise synthesis | Models extreme low-light raw noise and improves synthetic-to-real denoising fidelity | Generic raw sensor setting; not calibrated from gated ICCD repeated-frame Fano/fixed-pattern evidence | Cite as closest synthetic-noise training precedent; our novelty is ICCD-specific calibration |
| Cao et al., "Towards General Low-Light Raw Noise Synthesis and Modeling" ([arXiv:2307.16508](https://arxiv.org/abs/2307.16508)) | General raw noise synthesis | Targets generalized low-light raw-noise synthesis across cameras | Not focused on intensifier-chain ICCD noise or gated acquisition | Cite to show synthetic raw noise is active but sensor-general |
| Abdelhamed et al., "Noise Flow: Noise Modeling with Conditional Normalizing Flows" ([arXiv:1908.00129](https://arxiv.org/abs/1908.00129)) | Learned real-camera noise model | Learns camera noise distribution with normalizing flows | Needs suitable real sensor noise data; not ICCD-specific | Cite as learned noise-model alternative; we choose interpretable ICCD prior first |
| Abdelhamed et al., SIDD / smartphone denoising dataset ([project](https://www.eecs.yorku.ca/~kamel/sidd/), [arXiv search](https://arxiv.org/search/?query=Smartphone+Image+Denoising+Dataset&searchtype=all)) | Real image denoising dataset | Provides real noisy smartphone image pairs and benchmark practice | Smartphone Bayer/ISP domain, not ICCD | Cite as real-noise dataset precedent and contrast with lack of public ICCD pairs |
| Guo et al., CBDNet ([arXiv:1807.04686](https://arxiv.org/abs/1807.04686)) | Blind denoising with real/synthetic noise | Combines synthetic and real noise handling for blind denoising | Not device-calibrated for ICCD; RGB/sRGB-oriented | Cite as synthetic-real mixed-denoising precedent, not as core novelty |
| Lehtinen et al., "Noise2Noise" ([arXiv:1803.04189](https://arxiv.org/abs/1803.04189)) | Self-supervised/no-clean denoising | Shows restoration can be learned from pairs of independently noisy observations | Assumes independent noise and suitable repeated noisy pairs; may not remove fixed-pattern bias without care | Cite as optional training route for repeated ICCD frames |
| Krull et al., "Noise2Void" ([arXiv:1811.10980](https://arxiv.org/abs/1811.10980)) | Single-image blind-spot denoising | Learns denoising from single noisy images using blind-spot masking | Can struggle with structured fixed-pattern noise; not a physical ICCD model | Cite as no-clean-target fallback; contrast with our device-statistical prior |
| Daigle et al., MCP/phosphor gain measurement ([arXiv:1906.05481](https://arxiv.org/abs/1906.05481)) | Image intensifier / MCP characterization | Measures gain behavior in MCP/phosphor detector chain | Detector characterization, not denoising data synthesis | Cite for intensifier-chain gain/noise motivation |
| "High Speed Time Gated Single Photon Imaging" ([arXiv:1408.6381](https://arxiv.org/abs/1408.6381)) | Time-gated ICCD imaging | Demonstrates time-gated ICCD/single-photon imaging context | Imaging demonstration, not denoising training pipeline | Cite for gated ICCD acquisition context |
| "Reducing MCP cross-talk..." ([arXiv:1805.04106](https://arxiv.org/abs/1805.04106)) | MCP artifacts / cross-talk | Studies mitigation of microchannel-plate cross-talk | Artifact mitigation, not weak-light denoising with synthetic data | Cite as evidence that MCP/intensifier artifacts are device-specific |
| EMVA 1288 standard ([official site](http://www.standard1288.org/)) | Camera characterization | Defines standardized camera characterization concepts such as sensitivity/noise-related measurement practice | Formal calibration needs controlled dark/flat data; not a denoising method | Cite for why mean-variance and calibration discipline matter |

## Our Differentiation

| Manuscript claim candidate | Evidence already available | Evidence still needed | Risk if overstated |
|---|---|---|---|
| Gated ICCD noise is not well represented by a simple raw Poisson-Gaussian assumption | E1.3 shows temporal Fano about 1.70 to 14.46 across brightness folders | Fit explicit Poisson-Gaussian vs ICCD prior error in E2.2 | Reviewer may say Fano alone is descriptive, not a model comparison |
| Fixed-pattern structure is a dominant component in the current gated ICCD data | E1.4 shows median held-out spatial fixed-pattern reduction about 95.1% | Add visual maps and confidence intervals; verify larger crop/full-field behavior | Without calibration data, it is empirical fixed-pattern, not true flat-field correction |
| Residual temporal noise is only weakly spatially correlated after fixed-pattern removal | E1.5 median row/column lag-1 correlations about 0.032/0.030; autocorr below 0.1 by 1 px | Check larger crops and other device conditions | Do not claim strong phosphor-blur residual unless later evidence supports it |
| ICCD-aware synthetic noise can improve denoising compared with generic priors | Not yet complete | E2.2 fidelity comparison and E3 denoising baseline | This is the core future claim; cannot be written until experiments run |
| Auxiliary `ICCD_pir` data provide calibration candidates | `F:\ICCD_pir\dark`, `F:\ICCD_pir\mid`, and `2025.07.09` path are found | Need metadata or acquisition notes; audit 8-bit sequence separately | Cannot use 8-bit 2048x2048 data as matching dark/flat for 16-bit 5120x5120 batch |

## Recommended Citation Roles

| Paper section | Citation role |
|---|---|
| Introduction | SID, physics-based low-light raw denoising, general raw noise synthesis |
| Related Work: low-light denoising | SID, CBDNet, SIDD, Noise2Noise/Noise2Void |
| Related Work: noise synthesis | Physics-based model, Noise Flow, general raw noise synthesis |
| Related Work: ICCD/intensifier imaging | Time-gated ICCD imaging, MCP/phosphor gain, MCP cross-talk |
| Method: device characterization | EMVA 1288 / photon-transfer-style characterization references |
| Experiments | Prior comparisons should include Poisson-Gaussian, sCMOS-like, ICCD-aware prior, and fixed-pattern correction baseline |

## Search Log

Initial queries used:

- `ICCD denoising noise model`
- `gated ICCD denoising`
- `intensified CCD noise model denoising`
- `ICCD fixed pattern noise`
- `range-gated ICCD image enhancement`
- `low-light raw noise synthesis modeling`
- `Physics-based Noise Modeling for Extreme Low-light Photography`
- `Learning to See in the Dark`

Preliminary result:

No exact duplicate of the planned route was identified in the initial scan:

```text
real gated ICCD repeated-frame statistics
-> ICCD-aware noise prior
-> synthetic paired data for denoising
-> validation on real gated ICCD statistics
```

However, each component has precedent in adjacent literature. The manuscript
must therefore make the device-specific ICCD evidence chain explicit.
