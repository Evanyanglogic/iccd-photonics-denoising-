# Literature Matrix: Gated ICCD Noise Characterization and Synthetic-Data Denoising

Date started: 2026-07-15
Last updated: 2026-07-17 (E7 source verification and route audit)

Working paper title:

```text
门控 ICCD 噪声表征、条件噪声失配与去噪适用性分析
```

## Claim Boundary

The E7 core matrix below has been checked against official publisher, proceedings,
or DOI pages. Older legacy entries later in this file remain positioning notes and
must not be copied into the manuscript without the same source check.

The manuscript must not be framed as a generic deep denoising network, a
super-resolution/detail-restoration method, a validated condition-aware noise
generator, or the first ICCD denoising paper. E5 showed negative real transfer
for all four controlled synthetic cells, and E6 admitted only 2/10 folders for
repeated-frame supervision. The E7-selected route is:

```text
operational gated-ICCD repeated-frame characterization
-> conditional synthetic-real mismatch analysis
-> surrogate-based denoising applicability and failure-boundary validation
```

## E7 Verified Core Literature Matrix

Evidence grades follow the project protocol: A = direct gated ICCD; B = ICCD
or image-intensifier camera; C = adjacent sensor analogy; D = general denoising
methodology. E denotes internal project evidence and is recorded in the E7 route
report rather than treated as external literature.

| Complete citation / official source | Object and data | Method / key finding | Relation to this project | Grade | Directly citable? |
|---|---|---|---|---|---|
| B. R. Sandel and A. L. Broadfoot, “Statistical performance of the intensified charged coupled device,” *Applied Optics* 25, 4135–4140 (1986), [doi:10.1364/AO.25.004135](https://doi.org/10.1364/AO.25.004135) | ICCD; analytical detector statistics | Models photon Poisson statistics, exponential MCP pulse-height distribution, gain regime, and CCD digitization | Direct support that an ICCD is not adequately described by a single AWGN scale | A | Yes |
| T. C. Williams and C. R. Shaddix, “Simultaneous correction of flat field and nonlinearity response of intensified charge-coupled devices,” *Review of Scientific Instruments* 78, 123702 (2007), [doi:10.1063/1.2821616](https://doi.org/10.1063/1.2821616) | ICCD flat-field calibration over dynamic range | Pixelwise response characterization corrects nonuniform gain and nonlinearity | Shows why current scene repeats without flat fields cannot uniquely identify PRNU/FPN | A | Yes |
| W. Jin et al., “Three-step nonuniformity correction for a highly dynamic intensified charge-coupled device star sensor,” *Optics Communications* 285, 1753–1758 (2012), [doi:10.1016/j.optcom.2011.12.043](https://doi.org/10.1016/j.optcom.2011.12.043) | ICCD star sensor; controlled calibration images | Gain- and integration-dependent photoelectric response and nonlinear correction | Supports condition dependence, but also requires controlled radiance/calibration absent here | A | Yes |
| R. J. Peláez et al., “Integration of an Intensified Charge-Coupled Device (ICCD) Camera for Accurate Spectroscopic Measurements,” *Applied Spectroscopy* 66, 970–978 (2012), [doi:10.1366/12-06612](https://doi.org/10.1366/12-06612) | ICCD spectroscopy; dark/shot/spatial response | Reports dark and shot noise, spatial inhomogeneity, and smooth spatial dark-current dependence | Directly supports separating temporal and spatial terms and warns against calling all stable structure scene-free FPN | A | Yes |
| J. J. Selb, D. K. Joseph, and D. A. Boas, “Time-gated optical system for depth-resolved functional brain imaging,” *Journal of Biomedical Optics* 11, 044008 (2006), [doi:10.1117/1.2337320](https://doi.org/10.1117/1.2337320) | Time-gated ICCD; repeated phantom measurements | SNR depends on MCP gain; enumerates intensifier shot noise, MCP gain noise, phosphor shot noise, CCD dark/read noise | Direct physical motivation for multi-stage and gain-dependent noise | A | Yes |
| S. Liu et al., “Low noise and high resolution microchannel plate,” *Proc. SPIE* 6621, 662105 (2008), [doi:10.1117/12.790585](https://doi.org/10.1117/12.790585) | MCP/image intensifier | Relates MCP design to equivalent background input, FPN, and scintillation noise | Supports device-origin pattern/noise terminology, not a full ICCD image model | B | Yes, with scope caveat |
| M. Yang, F. Wang, Y. Wang, and N. Zheng, “A Denoising Method for Randomly Clustered Noise in ICCD Sensing Images Based on Hypergraph Cut and Down Sampling,” *Sensors* 17, 2778 (2017), [doi:10.3390/s17122778](https://doi.org/10.3390/s17122778) | Single ICCD images with clustered noise | Hypergraph/downsampling plus conventional restoration | Required direct predecessor; prevents “first ICCD denoising” claims | B | Yes |
| A. Foi et al., “Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data,” *IEEE TIP* 17, 1737–1754 (2008), [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399) | Ordinary raw image sensors | Signal-dependent Poisson-Gaussian model with clipping | Valid baseline, but ICCD multiplication/spatial terms exceed its assumptions | C | Yes, as analogy/baseline |
| T. Plötz and S. Roth, “Benchmarking Denoising Algorithms with Real Photographs,” *CVPR* (2017), [doi:10.1109/CVPR.2017.294](https://doi.org/10.1109/CVPR.2017.294) | Four consumer cameras; carefully aligned low-ISO references | Synthetic rankings can fail on real photographs; reference requires alignment, scaling, and low-frequency bias correction | Strong support for E5 negative transfer and reference-bias audit | C/D | Yes |
| A. Abdelhamed, S. Lin, and M. S. Brown, “A High-Quality Denoising Dataset for Smartphone Cameras,” *CVPR* (2018), [doi:10.1109/CVPR.2018.00182](https://doi.org/10.1109/CVPR.2018.00182) | Five smartphones, 10 scenes, about 30k noisy captures | Systematic reference estimation; real-device training outperforms low-ISO proxy strategies | Shows the evidence level expected for repeated-frame references and diverse real data | C/D | Yes |
| T. Brooks et al., “Unprocessing Images for Learned Raw Denoising,” *CVPR* (2019), [official page](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html) | Consumer RAW/sRGB | Noise realism also depends on the processing pipeline, not only noise variance | Supports treating sCMOS content and ICCD output-domain mismatch explicitly | D | Yes |
| K. Wei et al., “A Physics-Based Noise Formation Model for Extreme Low-Light Raw Denoising,” *CVPR* (2020), [official page](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.html) | Calibrated CMOS RAW cameras | Models shot/read/banding/quantization effects and validates through real denoising transfer | Supports richer calibration, but cannot be transferred to ICCD without device data | C/D | Yes |
| Y. Zhang et al., “Rethinking Noise Synthesis and Modeling in Raw Denoising,” *ICCV* (2021), [official page](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html) | SIDD/ELD real residuals | Real-noise sampling, pattern alignment, and high-bit reconstruction preserve spatial correlation; downstream denoising is decisive | Directly challenges the sufficiency of histogram/std/PSD similarity alone | D | Yes |
| A. Abdelhamed, M. A. Brubaker, and M. S. Brown, “Noise Flow: Noise Modeling With Conditional Normalizing Flows,” *ICCV* (2019), [official page](https://openaccess.thecvf.com/content_ICCV_2019/html/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.html) | Multiple cameras/gains from SIDD | Conditional likelihood model outperforms coarse parametric noise models | Shows what a validated condition model requires: actual condition labels and real noise data | D | Yes |
| J. Lehtinen et al., “Noise2Noise: Learning Image Restoration without Clean Data,” *ICML/PMLR* 80, 2965–2974 (2018), [official page](https://proceedings.mlr.press/v80/lehtinen18a) | Independent corruptions of a shared latent target | Noisy targets work when the loss expectation has the correct target statistic | E6 correlated/stable components violate the direct-use assumptions in most folders | D | Yes |
| J. Batson and L. Royer, “Noise2Self: Blind Denoising by Self-Supervision,” *ICML/PMLR* 97, 524–533 (2019), [official page](https://proceedings.mlr.press/v97/batson19a.html) | Natural and microscopy data | J-invariant risk estimate requires independence across measurement dimensions | Row/column and pixel correlation make naive blind spots unsafe here | D | Yes |
| W. Lee, S. Son, and K. M. Lee, “AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network,” *CVPR* (2022), [official PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.pdf) | Real sRGB noise | Pixel-downsampling is used specifically to weaken spatial correlation before blind-spot learning | A possible future method only after ICCD correlation length and scene aliasing are validated | D | Yes, not yet executable evidence |
| H. Jang et al., “Self-supervised Image Denoising with Downsampled Invariance Loss and Conditional Blind-Spot Network,” *ICCV* (2023), [official PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Jang_Self-supervised_Image_Denoising_with_Downsampled_Invariance_Loss_and_Conditional_Blind-Spot_ICCV_2023_paper.pdf) | Spatially correlated real sRGB noise | Explicitly notes standard blind spots fail with pixel correlation; random subsampling decorrelates noise | Supports E6 No-Go for naive N2N/N2S and suggests a separately testable future protocol | D | Yes |
| Y. Luo, T. Zhang, R. Li, B. Zhang, N. Jia, and L. Fu, “A Novel Framework for Real ICMOS Image Denoising: LD-NGN Noise Modeling and a MAST-Net Denoising Network,” *Remote Sensing* 17, 1219 (2025), [doi:10.3390/rs17071219](https://doi.org/10.3390/rs17071219) | ICMOS, not ICCD; real platform and multi-scene pairs | Models sparse clustered intensified-sensor noise and validates a generator via denoising | Closest intensified-CMOS comparison; must be labeled adjacent and prevents broad novelty claims about intensified-sensor noise generation | C | Yes, with device distinction |

### Verified Counter-Evidence

- Direct ICCD calibration studies require controlled dark/flat or radiance-response data; repeated scene frames alone cannot identify DSNU, PRNU, scene texture, and intensifier nonuniformity separately.
- DND, SIDD, Brooks, Wei, Zhang, and Noise Flow all show that matching a marginal noise scale is insufficient. Pipeline, conditional distribution, spatial structure, high-bit behavior, and downstream real transfer matter.
- Noise2Self, AP-BSN, and Jang et al. explicitly make independence/correlation handling part of the method. E6 therefore rules out naive repeated-frame or blind-spot training across all 10 folders.
- The 2024 MIDD study reports strong blur when a fixed architecture is trained on a mismatched real dataset, reinforcing that higher PSNR or stronger backbones do not resolve target-domain mismatch by themselves.

## Reference Strategy

Do not let low-level directly related papers define the paper's intellectual
standard. Use three layers:

| Layer | Function | How to use it |
|---|---|---|
| A. High-level method anchors | Set the technical standard for low-light noise modeling, synthetic data, and no-clean denoising | Emulate their problem framing, validation discipline, and ablation structure |
| B. Optics / detector characterization anchors | Set the measurement language for camera noise, photon transfer, fixed pattern, and intensifier-chain behavior | Use to justify E1 device statistics and calibration limitations |
| C. Direct ICCD denoising predecessors | Prevent novelty overclaim and define the nearest same-device baseline | Cite clearly, but do not imitate their writing level or claim structure |
| D. Background / gray literature | Explain device details or practical context | Use sparingly; never carry core claims |

## A. High-Level Method Anchors

| Source | Venue/status | What it contributes | What it does not solve for us | Role in our paper |
|---|---|---|---|---|
| Chen et al., "Learning to See in the Dark" ([arXiv:1805.01934](https://arxiv.org/abs/1805.01934), CVPR 2018) | Top CV conference | Paired short/long-exposure RAW low-light denoising/enhancement paradigm | Consumer RAW cameras; assumes paired long-exposure references | Use as the canonical paired low-light benchmark paradigm; distinguish our lack of true ICCD clean pairs |
| Wei et al., "A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising" ([arXiv:2003.12751](https://arxiv.org/abs/2003.12751), CVPR 2020) | Top CV conference | Physics-based synthetic raw noise calibrated from camera characteristics | CMOS/raw-camera pipeline, not ICCD intensifier chain | Primary method anchor for physically motivated synthetic noise |
| Cao et al., "Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography" ([CVPR 2023 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Physics-Guided_ISO-Dependent_Sensor_Noise_Modeling_for_Extreme_Low-Light_Photography_CVPR_2023_paper.pdf)) | Top CV conference | ISO/condition-dependent low-light sensor noise modeling | Still camera-RAW oriented, not gated ICCD | Strong support for our "condition-aware" framing |
| Feng et al., "Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising" ([arXiv:2310.09126](https://arxiv.org/abs/2310.09126), IEEE TPAMI 2026 per project/IEEE listing) | Top journal / high-level method | Learns physics-informed noise proxy from dark frames, reducing paired-data dependency | Requires dark frames; targets raw camera sensors, not ICCD | High-level anchor for dark-frame/detector-statistics-driven noise modeling |
| Abdelhamed et al., "Noise Flow" ([arXiv:1908.08453](https://arxiv.org/abs/1908.08453), ICCV 2019) | Top CV conference | Learns real camera noise distribution with conditional normalizing flows | Needs suitable real sensor noise data; not ICCD-specific | Alternative learned noise model to contrast with interpretable ICCD prior |
| Lehtinen et al., "Noise2Noise" ([arXiv:1803.04189](https://arxiv.org/abs/1803.04189), ICML 2018) | Top ML conference | Shows denoising from independently noisy observations without clean targets | Independence assumptions can fail under fixed-pattern or structured noise | Use to motivate repeated-frame validation, with caveats |
| Krull et al., "Noise2Void" ([arXiv:1811.10980](https://arxiv.org/abs/1811.10980), CVPR 2019) | Top CV conference | Blind-spot denoising from noisy data only | Structured fixed-pattern noise can violate blind-spot assumptions | Optional no-clean-target comparison route |
| Abdelhamed et al., SIDD / Smartphone Image Denoising Dataset ([project](https://www.eecs.yorku.ca/~kamel/sidd/)) | Real-noise benchmark | Real noisy/clean smartphone-image denoising benchmark | Smartphone Bayer/ISP domain, not ICCD | Contrast class showing why real-device data matters |

## B. Optics / Detector Characterization Anchors

| Source | Venue/status | What it contributes | What it does not solve for us | Role in our paper |
|---|---|---|---|---|
| EMVA Standard 1288 Release 4.0 ([official PDF](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)) | Authoritative machine-vision camera standard | Formal language for temporal dark noise, spatial variance, photon-transfer style characterization | Full compliance needs controlled dark/flat acquisition | Anchor E1 measurement terminology and clearly state what is not full EMVA calibration |
| Photon-transfer / camera characterization practice | Detector measurement tradition | Mean-variance and temporal/spatial noise separation | Needs controlled acquisition and calibration metadata | Justifies our repeated-frame mean-variance/Fano analysis as partial characterization |
| Daigle et al., MCP/phosphor gain measurement ([arXiv:1906.05481](https://arxiv.org/abs/1906.05481)) | Detector characterization | Intensifier-chain gain behavior | Not a denoising method | Supports MCP/phosphor gain-noise motivation |
| "High Speed Time Gated Single Photon Imaging" ([arXiv:1408.6381](https://arxiv.org/abs/1408.6381)) | Gated imaging context | Time-gated intensified imaging context | Not a denoising or noise-synthesis paper | Background for gated ICCD acquisition context |
| "Reducing MCP cross-talk..." ([arXiv:1805.04106](https://arxiv.org/abs/1805.04106)) | MCP artifact study | Shows MCP/intensifier artifacts are device-specific | Not a weak-light denoising pipeline | Supports device-specific artifact motivation |

## C. Direct ICCD Denoising Predecessors

| Source | What it did | Limitation relative to our route | How to cite |
|---|---|---|---|
| Yang et al., "A Denoising Method for Randomly Clustered Noise in ICCD Sensing Images Based on Hypergraph Cut and Down Sampling" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751643/), Sensors 2017, doi:10.3390/s17122778) | Removes randomly clustered ICCD noise using patch segmentation, hypergraph cut, BM3D, and RPCA | Classical single-image restoration; no repeated-frame noise characterization, condition-aware synthesis, or deep synthetic-data validation | Required direct predecessor; use it to avoid claiming first ICCD denoising |
| "A Denoising Scheme for Randomly Clustered Noise Removal in ICCD Sensing Image" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5336087/)) | Earlier clustered-noise ICCD removal route | Same limitation: algorithmic single-image denoising rather than device-statistics-driven modeling | Cite as direct same-device background if needed |

Direct predecessor positioning:

```text
Prior ICCD denoising work mainly targets randomly clustered noise removal from
single ICCD images. In contrast, our work starts from repeated gated-ICCD frames
to quantify condition-dependent device noise, then uses those statistics for
synthetic-noise construction and controlled denoising validation.
```

## D. Background / Gray Literature

| Source | Use | Caveat |
|---|---|---|
| Andor / Oxford Instruments, "An Introduction to Gated Intensified Cameras (ICCDs)" ([technical article](https://andor.oxinst.com/learning/view/article/intensified-ccd-cameras)) | Device background: gain, gate timing, MCP/phosphor concepts | Vendor page; do not use for core claims |
| Stanford Computer Optics ICCD/EMCCD noise comparison ([PDF](https://stanfordcomputeroptics.com/download/Noise-comparison-ICCD-EMCCD-SPIE.pdf)) | Practical ICCD/EMCCD noise-factor context | Needs venue verification before manuscript use |
| Laser Focus World ICCD/EMCCD low-light article | Practical explanation of gain/noise factor | Trade/technical article; background only |

## Our Differentiation

| Manuscript claim candidate | Evidence already available | Evidence still needed | Risk if overstated |
|---|---|---|---|
| Gated ICCD noise is not well represented by a simple raw Poisson-Gaussian assumption | E1.3 shows temporal Fano about 1.70 to 14.46 across brightness folders | Explicit Poisson-Gaussian vs ICCD-prior fidelity comparison with uncertainty intervals | Fano alone is descriptive, not sufficient as a model-comparison proof |
| Fixed-pattern structure is a dominant component in current gated ICCD data | E1.4 shows median held-out spatial fixed-pattern reduction about 95.1% | Visual maps, crop/frame robustness, and ideally matching dark/flat frames | Without dark/flat, it is empirical fixed-pattern correction, not formal flat-field calibration |
| Denoising gain appears folder-state dependent | E3.5 correlation and E3.8 LOFO strategy are positive | Independent acquisition conditions and gains larger than seed/reference uncertainty | E7 shows the score is highly collinear and confounded with folder/scene identity; exploratory only |
| Current ICCD-like synthetic priors transfer reliably | E5 directly tests strength and structure | None under the current data protocol | Rejected: all four E5 cells have negative real-domain transfer; retain as mismatch/failure-boundary evidence |
| Auxiliary `ICCD_pir` data provide calibration candidates | `F:\ICCD_pir\dark`, `F:\ICCD_pir\mid`, and `2025.07.09` path are found | Need metadata or acquisition notes; audit separately | Cannot use 8-bit 2048x2048 data as matching dark/flat for 16-bit 5120x5120 batch |

## Recommended Citation Roles

| Paper section | Citation role |
|---|---|
| Introduction | SID, CVPR 2020 physics-based raw noise, CVPR 2023 condition-dependent sensor noise, PNNP dark-frame noise model |
| Related Work: low-light denoising | SID, SIDD, Noise2Noise, Noise2Void, Noise Flow |
| Related Work: sensor noise synthesis | CVPR 2020, CVPR 2023, PNNP, general raw noise synthesis |
| Related Work: ICCD/intensifier imaging | Yang et al. 2017 direct ICCD denoising, MCP/phosphor gain, time-gated ICCD imaging, MCP artifacts |
| Method: device characterization | EMVA 1288 and photon-transfer-style measurement language |
| Experiments | Noisy input, fixed-pattern correction, Poisson-Gaussian, sCMOS-like, ICCD-aware prior, condition gate |

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

Brave fallback queries added on 2026-07-16:

- `ICCD image denoising gated intensified CCD noise model`
- `ICCD denoising clustered noise hypergraph cut`
- `image intensifier MCP phosphor noise fixed pattern ICCD`
- `Learning physics informed noise models from dark frames low light raw denoising`
- `Photon transfer curve camera noise characterization EMVA 1288 sensor noise`
- `Optica Photonics Research low light imaging denoising noise model image sensor`

Preliminary conclusion:

No exact duplicate of the **reduced E7 route** was identified in the verified scan:

```text
operational gated ICCD repeated-frame statistics
-> condition and synthetic-real mismatch analysis
-> surrogate-based denoising applicability and failure boundaries
```

Every component has strong precedent, and direct ICCD calibration/denoising work
already exists. Novelty must therefore come from the linked evidence chain across
repeated-frame characterization, controlled 2x2 mismatch causality, seed/reference
uncertainty, and applicability boundaries, not from claiming a new backbone or an
already validated condition-aware generator.

## E9 Nearest-Neighbor and Claim Audit Addendum (2026-07-17)

The word `causality` in the preceding E7 positioning sentence is superseded by
`controlled factor analysis and observational mismatch-to-transfer association`.
E8 does not identify a causal mechanism.

| Complete citation / official source | Why it is a nearest neighbor | What it prevents us from claiming | Remaining distinction |
|---|---|---|---|
| Y. Wang et al., "Characteristic Evaluation of an Intensifier Detector for SMILE UVI," *Sensors* 26, 483 (2026), [doi:10.3390/s26020483](https://doi.org/10.3390/s26020483) | Recent full intensifier/ICCD detector calibration covering radiant gain, background, excess noise factor, nonlinearity, uniformity, MTF, and SNR under controlled illumination and dark acquisition | The present scene repeats are not a complete physical detector calibration and cannot identify ENF, DSNU, PRNU, or component-level noise sources | Our study links operational repeated-frame statistics to controlled denoising-transfer failure boundaries rather than claiming full calibration |
| Y. Luo et al., "A Novel Framework for Real ICMOS Image Denoising: LD-NGN Noise Modeling and a MAST-Net Denoising Network," *Remote Sensing* 17, 1219 (2025), [doi:10.3390/rs17071219](https://doi.org/10.3390/rs17071219) | Closest intensified-sensor learned noise generator and downstream denoising study, with multi-scene paired/frame-integrated data | No claim that intensified-sensor noise generation or downstream generator validation is new; ICMOS evidence cannot be relabeled ICCD evidence | Our device is ICCD and our contribution is a failure-aware 2x2 mismatch and uncertainty audit, not a successful generator |
| S. Han et al., "Denoising and Motion Artifact Removal Using Deformable Kernel Prediction Neural Network for Color-Intensified CMOS," *Sensors* 21, 3891 (2021), [doi:10.3390/s21113891](https://doi.org/10.3390/s21113891) | Explicitly reports degradation when Poisson-Gaussian training noise and real intensified-CMOS noise/image characteristics differ | Synthetic-real mismatch in intensified cameras is not a new observation | E5 controls strength and structure and E8 tests folder-level association under repeated seed/reference evaluation |
| Y. Zhang et al., "Rethinking Noise Synthesis and Modeling in Raw Denoising," *ICCV* (2021), [official page](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html) | Uses real-noise sampling, pattern alignment, high-bit reconstruction, and downstream denoising to validate synthetic noise | Marginal histogram or standard-deviation matching is not sufficient evidence of simulator realism | Our result is a gated-ICCD, small-sample applicability audit with explicit negative transfer and uncertainty boundaries |
| Z. Fu, L. Guo, and B. Wen, "sRGB Real Noise Synthesizing With Neighboring Correlation-Aware Noise Model," *CVPR* (2023), [official page](https://openaccess.thecvf.com/content/CVPR2023/html/Fu_sRGB_Real_Noise_Synthesizing_With_Neighboring_Correlation-Aware_Noise_Model_CVPR_2023_paper.html) | Demonstrates signal dependency and neighboring correlation as relevant real-noise dimensions and validates them downstream | E8's radial-PSD null cannot be interpreted as spatial structure being irrelevant | Our current radial-PSD metric is retained as an unsupported preregistered direction, not a universal negative finding |

Verified-search conclusion for E9: no direct gated-ICCD paper was identified that
combines (i) repeated-frame operational characterization, (ii) a prespecified
2x2 synthetic strength/structure probe, and (iii) a folder-blocked
mismatch-to-transfer association with seed, surrogate-reference, gradient, LOO,
permutation, bootstrap, and negative-control auditing. This is a bounded search
finding, not a priority or "first" claim.
