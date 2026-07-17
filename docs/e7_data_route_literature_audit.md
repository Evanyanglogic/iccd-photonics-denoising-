# E7 Data-to-Route and Literature Audit

Date: 2026-07-17

## 1. 数据适配性总评

当前数据**不适合**完整的“门控 ICCD 噪声表征 + 条件感知噪声建模 + 受控去噪验证”，但适合缩小为“门控 ICCD 噪声表征 + 条件噪声失配分析 + surrogate-based 去噪适用性验证”。

## 2. 三个模块分别评级

| 模块 | 评级 | 证据边界 |
|---|---|---|
| 门控 ICCD 噪声表征 | 支持 | 仅支持 temporal、repeatable stable component、row/column、spatial correlation 和 drift 的操作性区分；无 dark/flat 时不能唯一物理归因 |
| 条件感知噪声建模 | 不支持 | 记录的 gate/exposure/sync/gain 在 10 个文件夹中不变；score 高度共线且与 scene/folder 混杂；E5 全部负迁移 |
| 受控去噪验证 | 有限支持 | 可做 folder-isolated、multi-reference、seed-aware 的适用性和失败边界分析；不能做绝对 clean recovery 或真实细节恢复结论 |

## 3. Folder-level 数据门禁表

| folder | 完整性 | 稳定性 | 表征 | surrogate | condition | 去噪验证 | 重复帧监督 | 主要原因 |
|---:|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | pixel 与 row/column residual 相关 |
| 2 | PASS | PASS | PASS | WARN | WARN | WARN | PASS | 部分策略的 dual-reference folder gain 差超过 0.01 dB |
| 4 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | column residual 相关 |
| 5 | PASS | WARN | PASS | WARN | WARN | WARN | FAIL | local drift 0.684 temporal-std；pixel 与 row/column residual 相关；reference range 最大 |
| 7 | PASS | WARN | PASS | WARN | WARN | WARN | FAIL | local drift 0.420 temporal-std；column residual 相关 |
| 8 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | row/column residual 相关 |
| 9 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | column residual 相关 |
| 10 | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | column residual 相关 |
| 11 | PASS | PASS | PASS | PASS | WARN | PASS | PASS | 通过 E6 全部重复帧门禁，但仍无 scene/FPN 分离标定 |
| 13 | PASS | PASS | WARN | WARN | WARN | WARN | FAIL | split-half stable-map correlation 0.916；reference gain 符号敏感 |

完整性检查读取了全部 2,000 个 TIFF 的中心块：每个文件夹 200 帧和 200 条 metadata，编号 1--200 连续，尺寸 5120x5120，`uint16`，无不可读文件、dtype/shape 异常、重复中心块、零值或饱和中心像素。所有完整文件夹记录的 exposure width 900 ms、Sync A/B 4 us、gain 60 相同。

## 4. 当前数据能够证明什么

- **已验证事实 E**：10 个文件夹的 mean signal、temporal std、Fano-like、stable-map std 和 stable/temporal ratio 存在明显差异。
- **已验证事实 E**：9/10 split-half stable maps 可重复；这只能称为 repeatable stable component，不能自动称为纯 FPN。
- **已验证事实 E**：高频 temporal residual 的 lag correlation 很低，但多个文件夹仍有 pixel、brightness-bin、row/column correlation。
- **已验证事实 E**：E5 四个 synthetic cells 均为真实域负迁移；strength、structure 和 interaction 都影响迁移，高强度伴随明显 gradient loss。
- **已验证事实 E**：25/50/100-frame references 上五种已有策略的总体排名一致；相对比较具有有限稳定性。
- **支持性推论 A/B/E**：多级 intensifier/CCD 链路与非均匀响应使统一 AWGN 或仅按 residual std 拟合的 prior 不充分。

## 5. 当前数据不能证明什么

- 不能把 stable component 唯一分解成 DSNU、PRNU、MCP FPN、phosphor nonuniformity 和 scene texture。
- 不能证明 gate width、delay、gain 或 illumination 与 noise statistics 的映射，因为设备设置没有跨文件夹变化且 illumination 无标定。
- 不能证明当前 condition score 是可部署分类器；PC1 解释 96.9%，最大特征相关 0.99996，最大 VIF 54,165。
- 不能证明 p99、physical 或 condition-scaled synthetic generator 真实有效；E5 的下游迁移证据否定了该表述。
- 不能把 temporal mean 称为 clean ground truth，也不能给出绝对恢复 PSNR 或真实细节恢复结论。
- 不能用更强 backbone、单次 seed、平均 PSNR 或视觉平滑掩盖 synthetic-real gap。

## 6. 文献检索方法

- 工具：先实际调用 Brave Search MCP（4/4 `fetch failed`）和 arXiv MCP（`Transport closed`）；随后使用出版商/官方论文页检索，包括 Optica、SPIE、AIP、SAGE、Elsevier、CVF、PMLR、NeurIPS、PubMed 和期刊 DOI 页面。
- 时间：覆盖 1986--2026 的 ICCD 基础研究、2017--2026 的真实去噪与 synthetic-real 研究、2018--2025 的自监督方法。
- 关键词：`gated ICCD noise`, `MCP gain noise`, `ICCD flat field nonlinearity`, `real camera noise synthesis`, `spatially correlated noise`, `Noise2Noise correlated noise`, `surrogate reference denoising`, `ICCD nonuniformity correction`，并执行中文期刊站点检索。
- 纳入：原始研究、官方期刊/会议页、对象和假设可核验、与当前证据链直接相关。
- 排除：博客、营销材料、只看标题无法核验正文者、把 sCMOS/EMCCD/SPAD 直接当 ICCD 证据者。
- 最终核心矩阵：19 篇，其中直接 ICCD/增强器证据 7 篇，邻近相机/数据集证据 5 篇，通用噪声建模或自监督方法 7 篇。完整引用见 `docs/literature_matrix.md`。

## 7. 核心文献矩阵

| 文献 | 对象 | 核心结果 | 当前作用 | 等级 |
|---|---|---|---|---|
| Sandel & Broadfoot, *Applied Optics* 1986, doi:10.1364/AO.25.004135 | ICCD | Poisson arrivals + exponential MCP pulse height + CCD digitization | 支持多级、非单尺度噪声 | A |
| Williams & Shaddix, *RSI* 2007, doi:10.1063/1.2821616 | ICCD | 逐像素 flat-field/nonlinearity calibration | 反证无 flat 数据时的 FPN 物理归因 | A |
| Jin et al., *Optics Communications* 2012, doi:10.1016/j.optcom.2011.12.043 | ICCD star sensor | gain/integration response-dependent NUC | 支持条件依赖，但要求受控标定 | A |
| Peláez et al., *Applied Spectroscopy* 2012, doi:10.1366/12-06612 | ICCD | dark、shot、spatial inhomogeneity | 支持 temporal/spatial 分开报告 | A |
| Selb, Joseph & Boas, *JBO* 2006, doi:10.1117/1.2337320 | gated ICCD | SNR 随 MCP gain，列出 intensifier/MCP/phosphor/CCD 噪声 | 物理动机 | A |
| Yang et al., *Sensors* 2017, doi:10.3390/s17122778 | ICCD image | clustered-noise single-image denoising | 最近直接前作；禁止“首个 ICCD 去噪” | B |
| Foi et al., *IEEE TIP* 2008, doi:10.1109/TIP.2008.2001399 | ordinary raw sensor | clipped Poisson-Gaussian model | 合理基线，不足以覆盖 ICCD | C |
| Plötz & Roth, CVPR 2017, doi:10.1109/CVPR.2017.294 | real cameras | synthetic ranking 可在 real data 上失效；reference 要配准/缩放/去低频偏差 | 支持 E5 和 surrogate audit | C/D |
| Abdelhamed et al., CVPR 2018, doi:10.1109/CVPR.2018.00182 | SIDD | 5 cameras/10 scenes 的系统 reference 构造 | 显示高质量真实基准证据要求 | C/D |
| Brooks et al., CVPR 2019 | RAW pipeline | pipeline mismatch 同样影响迁移 | 支持 sCMOS-content/ICCD-domain 边界 | D |
| Wei et al., CVPR 2020 | calibrated CMOS RAW | shot/read/banding/quantization + real transfer | 说明 calibration 和 downstream validation 必要 | C/D |
| Zhang et al., ICCV 2021 | SIDD/ELD residual | pattern-aligned real residual sampling、高位重建 | 反证只匹配 std/hist/PSD | D |
| Abdelhamed et al., ICCV 2019 | Noise Flow | camera/gain conditional likelihood | 说明 condition 必须有真实标签和数据 | D |
| Lehtinen et al., ICML 2018 | Noise2Noise | noisy targets 依赖正确条件期望 | E6 多数文件夹不满足 | D |
| Batson & Royer, ICML 2019 | Noise2Self | J-invariance 依赖维度间 noise independence | row/column correlation 构成冲突 | D |
| Lee et al., CVPR 2022 | AP-BSN | 用 PD 缓解 real noise correlation | 仅是补采/新协议后的候选，不是当前证据 | D |
| Jang et al., ICCV 2023 | correlated real noise | 标准 blind spot 对相关噪声失败 | 支持 E6 No-Go | D |
| Zhang et al., *Remote Sensing* 2025, doi:10.3390/rs17071219 | ICMOS | real platform + learned generator + downstream denoising | 最近邻 intensified-sensor 工作，但设备不同 | C |
| Flepp et al., CVPR 2024 | mobile real data | 跨数据训练可产生强 blur | 支持 backbone 不解决 domain mismatch | C/D |

## 8. 文献与当前结果对照

- **支持**：多级 ICCD 噪声、gain/response dependence、spatial nonuniformity、synthetic-real gap、自监督 independence 限制，均与 E1/E5/E6 一致。
- **不支持**：文献不支持把 scene repeats 中的 stable map 直接称为 calibrated FPN，也不支持把图像统计 score 当设备 condition。
- **冲突**：邻近 CMOS 文献中经标定的 physical generator 可实现正迁移；本项目 E5 全负，说明当前 prior 或 content/domain protocol 未达到同等校准水平，而不是物理建模概念本身无效。
- **可能新意**：在同一 gated-ICCD batch 上把 repeated-frame operational statistics、2x2 strength/structure causal audit、seed/reference uncertainty 和 negative-transfer/oversmoothing boundary 连成一条证据链。
- **待验证**：synthetic-real mismatch 的哪一组预注册统计距离能跨 folder 预测 negative transfer；当前“同时观察到 mismatch 和失败”还不是因果链。

## 9. Skill 使用记录

| Skill/工具 | 状态 | 实际用途 |
|---|---|---|
| `academic-research-suite` | 已调用 | 采用 deep-research 的检索、source verification、counter-evidence 和 synthesis 规则 |
| `iccd-denoising-optimizer` | 已调用 | 数据优先、16-bit/domain/配对/指标/过平滑和实验门禁审查 |
| `pytorch-patterns` | 已调用 | 检查 float data domain、checkpoint selection、seed 和训练/评估隔离 |
| `research-paper-writing` / `paper-spine` | 可用但本轮未调用写作 | 遵守“路线审查完成前不预写结论”；后续仅用于正式稿 |
| PaperJury | 不可用 | 以 source verification + reviewer-style counter-evidence 替代 |
| camera-noise / stats 独立 Skill | 不可用 | 以 NumPy/SciPy、既有 E1-E6 scripts 和 E7 审查脚本替代 |
| Brave Search MCP | 已调用但失败 | 4 个查询均返回 `fetch failed` |
| arXiv MCP | 已调用但失败 | 3 个查询均返回 `Transport closed` |
| Web/出版商官方页 | 已使用替代 | 核验标题、作者、年份、venue、DOI、器件和假设 |

新增产物：E7 integrity/folder gate、condition correlation/VIF/PCA/LOFO、25/50/100 reference scaling、route decision、更新后的 verified literature matrix。

## 10. 唯一推荐论文主线

选择**路线 2：门控 ICCD 噪声表征 + 条件噪声失配分析 + 去噪适用性验证**。

- 数据依据：完整性强，operational characterization 大部分稳定，但 acquisition condition 不变化、无 dark/flat。
- 实验依据：E5 证明 strength/structure/interaction 会改变真实迁移，但所有 synthetic cells 失败；E6 禁止直接 real repeated-frame training。
- 文献依据：直接 ICCD 文献支持多级噪声和校准必要性；真实去噪文献支持 mismatch/downstream validation；自监督文献支持 correlation 门禁。
- 新意：不是新网络，而是 gated-ICCD 的“表征 -> 受控 mismatch -> negative transfer/oversmoothing boundary”闭环。
- 边界：只能称 operational statistics、conditional mismatch、surrogate-based applicability；不能称 precise physical model、validated generator 或 true detail recovery。

## 11. 需要保留、降级、删除的实验

**保留**：E1 robustness/spatial correlation；E1.4 empirical stable-pattern correction baseline；E4 multi-reference reliability；E5 2x2 + 3 seeds；E6 independence audit；E7 integrity/condition/reference audits。

**降级**：p99/physical 的正增益仅作早期 probe；gate/blend 仅作 exploratory observation；condition score 改称 image-statistical state score；temporal mean 改称 surrogate reference；FPC 改称 empirical repeatable-component subtraction。

**从主证据删除**：condition-scaled、constant condition channel 的“方法贡献”表述；任何 SMNet/MIRNet/DnCNN 强 backbone 路线；任何 synthetic generator 有效、可部署 gate、真实细节恢复或 universal ICCD denoising 表述。失败结果本身保留在 appendix/failure analysis，不删除数据。

## 12. 文献驱动的路线修正

1. 共同支持：operational temporal/stable/row-column/spatial/drift characterization 和 synthetic-real mismatch 风险。
2. 探索性支持：score 与 physical-vs-p99 gain 的 LOFO 相关；其 RMSE 0.300 dB，小于 null 0.558 dB，但仍接近且未压过 0.668 dB 最大 seed SD。
3. 被否定：有效 condition-aware generator、纯 synthetic 可稳定迁移、naive repeated-frame supervision。
4. 需补数据：dark、flat、多 gain/gate/delay/illumination、同步高 SNR reference、多独立 scene/condition replicates。
5. 不补采主线：route 2 的 mismatch/applicability/failure-boundary 论文。
6. 新意：以同一真实设备批次连接统计表征、受控因子实验和不确定性，不以新 backbone 为贡献。
7. 最近工作：Sandel/Broadfoot；Williams/Shaddix；Jin et al.；Yang et al. ICCD denoising；2025 ICMOS LD-NGN/MAST-Net。
8. 避免重复：不能重复 clustered-noise removal 或只提出另一个网络；不能复述 CMOS physical noise model 而无 ICCD calibration；不能把 ICMOS 结果当 ICCD 证据。

## 13. 论文题目建议

- 最保守：**门控 ICCD 重复帧噪声统计与合成去噪失配分析**
- 平衡型：**门控 ICCD 噪声表征、合成噪声失配及去噪适用性研究**
- 证据充分后才可用：**面向门控 ICCD 的条件噪声表征与受控去噪适用性分析**

## 14. 论文最小证据链

1. 设备、metadata 和 10-folder/2,000-frame 完整性表。
2. frame/crop robustness；mean-variance/Fano；temporal vs repeatable stable component 图。
3. row/column profile、PSD、autocorrelation、drift 与 split-half uncertainty 图。
4. 25/50/100 surrogate convergence 和 ranking/sign/range 表。
5. real-vs-synthetic conditional distribution/PSD/row-column/signal dependence 对照。
6. E5 2x2 factor effect、3-seed、dual-reference、folder bootstrap 和 uncertainty budget。
7. PSNR/SSIM 之外的 gradient ratio、brightness bias、residual/error maps、worst folders。
8. 下一实验的 folder-blocked mismatch-to-transfer linkage；若失败，论文降级为 route 3。

## 15. 下一项唯一实验

执行 **E8：预注册的 folder-blocked synthetic-real mismatch-to-transfer linkage test**，不训练新模型。

对每个 real folder 和 E5 variant，在固定 crop/domain 下构造预注册 mismatch score：strength error、brightness-conditional variance error、tail/kurtosis error、normalized PSD/autocorrelation error、row/column energy error、signal-residual correlation error和 clipping difference。主自变量只有该综合 mismatch score；因变量是三 seed、双 reference 的 PSNR gain、gradient ratio 和 brightness bias。所有标准化和线性系数仅在 9 个 training folders 内确定，用 held-out folder 测试；使用 folder-blocked bootstrap/permutation，不能按 40 个相关 cell 当独立样本。

这个实验直接决定 route 2 是否成立：它检验 mismatch 是否能解释/预测 applicability，而不是再次证明两种分布“看起来不同”。

## 16. Go / No-Go 判据

- **GO route 2**：两份 references 上 LOFO Spearman `rho <= -0.60`，folder-bootstrap 95% CI 不跨 0；mismatch model 相对 training-folder-mean null 的 held-out RMSE 至少降低 25%；方向在 3 seeds 一致；预测效应大于 reference variation，且 mismatch 增大同时对应 PSNR 下降或 gradient loss，不由 brightness bias 单独解释。
- **缩小结论**：方向一致但 CI 跨 0、RMSE 改善 10--25%，或效应小于 seed/reference uncertainty。只能写 descriptive mismatch 与 negative-transfer coexistence，不能写 predictive applicability。
- **STOP route 2 / 转 route 3**：方向不稳定、RMSE 改善 <10%、任一 reference 反向，或结果主要由 folder 5/单一 variant 驱动。停止 mismatch-to-performance 因果表述，保留 characterization + supervision/task boundary。

## 17. 不补采数据情况下的最终计划

1. 冻结 E7 数据门禁、claim vocabulary 和 exclusion rules。
2. 完成 E8 mismatch-to-transfer linkage，按预注册门禁决定 route 2 或 route 3。
3. 统一 E1/E4/E5/E6 的 folder/crop/reference/seed 表，修正“FPN/clean/condition”术语。
4. 生成最小证据链中的图表和 worst-case panels，不增加网络。
5. 用 PaperSpine/research-paper-writing 生成 route-consistent 初稿，再用 reviewer-style audit 检查 claim-evidence alignment。
6. 将 dark/flat/multi-setting acquisition 明确列为局限与未来数据协议，不用算法推断替代缺失标定。
