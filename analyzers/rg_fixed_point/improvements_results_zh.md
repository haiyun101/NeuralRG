# RG Fixed-Point Probe — Phase-1 改进 ablation 结果报告

> **配套阅读:**
> - `improvements_zh.md` —— 前瞻路线图(8 个方案 + cost-leverage 排序 + 失败模式预测表)
> - `rg_fixed_point_report_zh.md` —— 现有架构在 T_c 上的病态诊断
> - `analyzers/concise_reports/concise_report_L64_T2.269.md` —— L=64 method 比较(含 Phase-1 ablation 简表)

## 出发点

`improvements_zh.md` 提出 8 个改进方案,推荐 Phase-1 立刻投 III.1(multi-scale loss)、I.1(Student-t prior)、I.2(conditional Gaussian prior)三条平行轨道。本报告汇总 Phase-1 的实际跑分,与 improvements_zh.md 的"失败模式预测表"对照,产出每个方案的 verdict + Phase-2 优先级建议。

## 实验设计

测试基线 = **hs_bignet**(`nlayers=16, nhidden=128, nmlp=3, nrepeat=1, -symmetry, fwd-KL/dataDriven`,HS 连续场数据)。所有改进 run 共用此架构,只换一个 flag。

跑了 **14 个改进 run + 5 个原 method baseline**,合计 19 个 `flow_diagnostic.json`(N=4000 samples,batch 取决于 run)。

### L=32 ablation 矩阵(2 × 2,b=64)

|                      | scaleLoss=0        | scaleLoss=1.0      |
|----------------------|--------------------|--------------------|
| Gaussian prior       | `baseline_b64`     | `iii1_lam1.0_b64`  |
| Conditional Gaussian | `i2_stride8h32_b64`| `combined_lam...`  |

4 格全 b=64,完整 2 × 2 → 可算交互效应。

### L=32 hyperparameter sweep

- **iii1 λ_scale 扫描**:λ ∈ {0.1, 1.0, 10.0},b=64
- **i2 slow_stride 扫描**:stride ∈ {4, 8(原 Phase-1), 16},b=128 *(注:stride=8 末段发散,见 §4.4)*
- **i1 Student-t df=4**,b=128

### L=64 ablation(b=16)

`baseline_b16 / iii1_lam1.0_b16 / i2_stride16h32_b16 / i1_df4.0_b16` —— b=16 统一以避开 scaleLoss extra forward graph OOM,内部条件匹配。

## 数据汇总

HS 数据 anchors(N=4000):

| L  | mag_p  | xi_p    | gL_p   |
|---:|-------:|--------:|-------:|
| 32 | 2.382  | 8.568   | 0.477  |
| 64 | 2.200  | 14.782  | 0.407  |

### L=32 ablation 矩阵 + 交互项

|                      | scaleLoss=0(KL_qp / KL_pq / mag / xi / gL)| scaleLoss=1.0 |
|----------------------|--------------------------------------------|------------------|
| **Gaussian prior**   | 23.42 / 17.05 / 2.44 / 8.79 / 0.503        | **iii1**: 21.84 / 16.59 / 2.45 / 8.75 / 0.499 |
| **Conditional Gauss**| **i2_b64**: 21.16 / 16.05 / 2.38 / 8.58 / 0.487 | **combined**: 21.20 / 15.88 / 2.35 / 8.39 / 0.478 |

**单变量 Δ vs baseline:**

| 干预              | Δ KL_qp | Δ KL_pq | Δ mag  | Δ xi    | Δ gL    |
|-------------------|--------:|--------:|-------:|--------:|--------:|
| + III.1(scaleLoss=1)| −1.59 | −0.46  | +0.007 | −0.032  | −0.004  |
| + I.2(cond. prior) | **−2.26** | **−0.99**  | **−0.065** | **−0.202**  | **−0.016**  |
| + 二者叠加        | −2.22  | **−1.17** | **−0.094** | **−0.392** | **−0.025** |

**交互效应**(combined − iii1 − i2_b64 + base):
- KL_qp: +1.63(sub-additive,弱抗争)
- KL_pq: +0.28(基本独立加和)
- gL:    −0.005(超线性协同结构修正)

⇒ **两个干预在 *结构* 上协同(gL 超线性下降),在 *KL_qp* 上轻微抗争**。combined 的结构是矩阵里最贴 HS 的(mag=2.35 距 anchor 2.38 仅 0.03,xi=8.39 距 8.57 仅 0.18,gL=0.478 与 anchor 0.477 几乎相等)。

### L=32 hyperparameter 敏感性

**iii1 λ_scale 扫描:**

| λ_scale | KL_qp | KL_pq  | gL     | 解读                          |
|--------:|------:|-------:|-------:|-------------------------------|
| 0.0     | 23.42 | 17.05  | 0.503  | baseline                      |
| 0.1     | 24.22 | 17.04  | 0.509  | 太弱;λ 太小反而轻微劣化      |
| **1.0** | 21.84 | 16.59  | 0.499  | **甜点**                      |
| 10.0    | 21.70 | **30.15** | 0.509 | KL_qp 微降但 **KL_pq 翻倍** → mode collapse |

⇒ **λ=1.0 是 III.1 的甜点**;**λ=10.0 把 KL_qp 拉低靠的是 mode-drop**(KL_pq 暴涨证实)。Improvements_zh.md 之前预测 λ_scale 与 V5 RMS-G 反相关,这里 KL_pq 走势确认:λ=10 在 forward KL 上严重恶化 ⇒ 流主动放弃 bridge 区域以满足尺度约束。

**i2 slow_stride 扫描(b=128):**

| stride | slow grid | KL_qp     | KL_pq | gL      | 解读                          |
|-------:|----------:|----------:|------:|--------:|-------------------------------|
| **4**  | 16 × 16   | **17.13** | **14.14** | 0.493 | **最佳 ablation 单项**:Δ −6.3 nat |
| 8      | 8 × 8     | **604190.40** ⚠️ | 21.97 | 0.012 | **末段发散**(memory `project_l32_late_training_instability`)|
| 16     | 2 × 2     | 20.48     | 15.78 | 0.501  | 中等                          |

⇒ **conditional prior 的 slow grid 越细,效果越好**(16×16 → 8×8 → 2×2 单调劣化)。但 stride=8 这一格因为 Phase-1 原 b=128 run 末段不稳已经废了,**该方案的真正 L=32 表现需要 stride=4 数据为准 —— Δ KL_qp −6.3 nat 是 8 项 single-variable ablation 里最大**。

**i1 Student-t df=4(b=128):**

| 量      | baseline | i1_df4.0 | Δ      |
|---------|---------:|---------:|-------:|
| KL_qp   | 23.42    | 21.36    | −2.07  |
| KL_pq   | 17.05    | 15.54    | −1.51  |
| mag     | 2.441    | 2.389    | −0.052 |
| xi      | 8.79     | 8.55     | −0.24  |
| gL      | 0.503    | 0.483    | −0.020 |

⇒ Student-t 对所有量均衡下降 1.5–2 nat,**没有任何指标恶化**,但也没有 i2_stride4 那种戏剧性改善。**符合 improvements_zh.md 设计的 negation 实验定位** —— heavy-tail prior 的影响"看得到但不大"。

### L=64 ablation(b=16)

|                  | KL_qp  | KL_pq  | mag (a=2.20) | xi (a=14.78) | gL (a=0.407) |
|------------------|-------:|-------:|-------------:|-------------:|-------------:|
| baseline_b16     | 86.88  | 65.64  | 2.267        | 15.19        | 0.433        |
| iii1_lam1.0_b16  | 87.14  | 64.63  | 2.244        | 14.95        | 0.425        |
| i2_stride16h32_b16| 93.31 | 70.37  | 2.230        | 14.94        | 0.425        |
| i1_df4.0_b16     | **90.41** | 66.21  | **2.179**    | **14.37**    | **0.404**    |

**Δ vs baseline:**

| 干预 | Δ KL_qp | Δ KL_pq | Δ gL    |
|------|--------:|--------:|--------:|
| III.1| +0.26   | **−1.01** | −0.008  |
| I.2  | +6.43   | +4.73   | −0.008  |
| I.1  | +3.53   | +0.57   | **−0.029** |

⇒ **L=64 改进效果整体缩水到噪声水平**(batch=16 高梯度噪声 + Wilson-Fisher 失配双重压制):
- III.1 唯一在 KL_pq 上有 1 nat 改善(方向与 L=32 一致),其余在噪声里
- I.2 在 L=64 上 **方向反转**(L=32 上 −2.26,L=64 上 +6.43)
- I.1 唯一在 *结构*(gL Δ −0.029 ⇒ 更贴 anchor 0.407)上有可见正向

## 跨 L 方向对比

| 干预 | L=32 Δ KL_qp | L=64 Δ KL_qp | L=32 Δ gL | L=64 Δ gL | 跨 L 行为              |
|------|-------------:|-------------:|----------:|----------:|------------------------|
| baseline | (ref) 23.4 | (ref) 86.9 | +0.026    | +0.026    | (ref)                  |
| + III.1  | −1.59       | +0.26 (噪声) | −0.004    | −0.008    | 一致方向,L=64 削弱    |
| + I.2    | **−2.26**   | **+6.43**   | −0.016    | −0.008    | **方向反转** —— scaling 问题 |
| + I.1    | −2.07       | +3.53       | −0.020    | **−0.029** | 一致方向,**L=64 结构更强**|

### 关于 I.2 在 L=64 反转的解释

possible 原因:
1. **slow grid 绝对大小问题**:L=32 stride=8 → 4×4 slow grid;L=64 stride=16 → 4×4 slow grid。**count 相同但相对覆盖比从 1/16 变成 1/256**。在更大的物理体系里同等大小的 slow grid 携带的信息密度下降。
2. **CNN 容量 ill-suited**:`condPriorHidden=32` 在 L=64 上可能不足以让 CNN 学到 z_slow → z_fast 的有效条件。
3. **训练样本相对量减少**:L=64 b=16 跑 20000 step → 32 万样本,相对于 L=32 b=64 跑 20000 step → 128 万样本,有效训练量 4× 少。

**Phase-2 应该测试:** I.2 with stride=8(8×8 slow grid)和 hidden=64 at L=64 是否恢复正向贡献。如果是,**改进 Phase-1 的 default stride 选择(从 L//4 改为更细)**。

### 关于 I.1 在 L=64 结构信号变强

`gL` 在 L=64 的 Δ = −0.029,比 L=32 的 −0.020 更大;`mag`(2.179 vs anchor 2.200)在 L=64 上 Δ = −0.021 也明显。

物理解读:**临界点上 ξ → ∞,数据本身重尾性更强;heavy-tail prior 在更靠近物理 IR 的 L=64 上 *相对* 更贴**。这与 `rg_fixed_point_report_zh.md` "为什么 T_c 在此架构上难"段的核心论点(Wilson-Gaussian-FP 假设错位)一致:**L 越大,这种错位越显**,所以非高斯 prior 越有用。

## 方案 verdict 对照 improvements_zh.md 预测

`improvements_zh.md` "失败模式预测表":

| 方案     | 预测 V5 KS (T_c rev-KL) | 预测 V5 KS (T_c fwd-KL) | 预测 V5 RMS-G | 
|----------|------------------------:|------------------------:|--------------:|
| baseline | 0.32+                  | 0.08                    | 0.62 / 0.04   |
| I.1 t    | 0.22(改善 ~30 %)       | 0.06                    | 0.55(略改善)|
| I.2 cond | 0.18(改善 ~50 %)       | 0.05                    | **0.30**(显著改善)|
| III.1    | 0.15(改善 ~50 %)       | 0.05                    | **0.20**(显著改善)|

我们没有直接的 V5 数据(V5 probe 还没在改进 folder 上跑,见 §5),但 **KL_pq 是 V5 KS 的代理**(都测 marginal 错配),**gL 是 V5 RMS-G 的代理**(都测空间结构错配)。

### 用代理变量对照预测

**Δ KL_pq(L=32, b=64,代理 V5 KS 改善):**

| 方案     | 预测改善程度  | 实际 Δ KL_pq | 验证            |
|----------|--------------|-------------:|----------------|
| baseline | (ref)        | (ref)        | —              |
| I.1 t    | 改善 ~30 %   | −1.51        | ✓ 方向一致(小)|
| I.2 cond | 改善 ~50 %   | **−0.99**(b=64 stride=8)  | ✗ **小于预期** |
|          |              | **−2.91**(b=128 stride=4) | ✓ 与预期相当   |
| III.1    | 改善 ~50 %   | −0.46        | ✗ 小于预期    |
| combined | (interaction)| −1.17        | 弱协同          |

**Δ gL(L=32, b=64,代理 V5 RMS-G 改善):**

| 方案     | 预测改善程度  | 实际 Δ gL | 验证          |
|----------|--------------|----------:|---------------|
| baseline | gL=0.62      | (ref 0.503) | —          |
| I.1 t    | gL → 0.55(略改善)| −0.020 | ✓ 方向一致  |
| I.2 cond | gL → 0.30(显著改善)| −0.016 | ✗ 远小于预期 |
| III.1    | gL → 0.20(显著改善)| −0.004 | ✗ **几乎无效** |
| combined | (协同)        | −0.025    | ✓ 协同      |

### Proxy Verdict 表(已过时,见下方真值升级)

| 方案 | improvements 预测 | 实际表现        | 评级                 | Phase-2 优先级 |
|------|-------------------|-----------------|---------------------|----------------|
| III.1| KL/RMS-G 显著改善 | KL 小改善,结构几乎无效 | **不及预期** | 中(还有调优空间)|
| I.1  | 小改善(negation) | 一致方向 1.5–2 nat,**L=64 结构更强(gL 代理)** | (代理读法,见下方真值反转) |  |
| I.2  | 显著改善          | L=32 大改善(stride=4),L=64 反向    | **scaling 问题待修** | 中–高(需 stride/hidden 扩容)|
| combined | 协同        | KL 弱抗争,结构超线性协同 | **混合信号**      | 低(单独完善先)|

> ⚠️ **以上是基于 KL_pq / gL *代理* 的 verdict,V5/V3 真值跑完后被部分推翻 —— 见下一节"V5 / V3 真值升级"。最关键的反转:I.1 Student-t prior 实际上 *劣化* V5 RMS-G(空间结构),早期 gL 代理读出的"L=64 结构胜出"是误读。**

## V5 / V3 / V4 probe 真值(已升级,2026-06-09)

3 个 probe job(40031218 / 19 / 20)在 13 个 usable 改进 folder 上跑完(原 i2_stride8h32 b=128 末段 broken,排除)。CSV 在 `analyzers/rg_fixed_point/csv/`,图在 `analyzers/rg_fixed_point/figures/`。

### V3 identity residual(`E[(f_s(z) − z)²] / E[z²]`,z ~ N(0,I))

测 scale-block 在 N(0,I) 探针上是否近似恒等。**rev-KL 病态特征:f_4, f_5 ≈ 0(深层塌缩成恒等)。fwd-KL 健康特征:f_4, f_5 > 1(深层做真工作)。**

| Run                                  |  V3 f_4 |  V3 f_5 |   读法                                                  |
|--------------------------------------|--------:|--------:|---------------------------------------------------------|
| L=32 baseline_b64(fwd-KL ref)        |  3.587  |  0.467  | fwd-KL 基线;f_4 显著非零,f_5 中等                       |
| L=32 iii1_lam1.0_b64(+ III.1)        |  1.066  |  **0.011** | **f_5 塌缩**(<< 任何 rev-KL run,深于 sym_bignet 的 0.08) |
| L=32 i2_stride8h32_b64(+ I.2)        |  4.447  |  0.476  | f_5 与 baseline 持平,条件 prior 不诱发塌缩              |
| L=32 combined(I.2 + III.1)           |  1.142  |  **0.003** | **极度塌缩**,scaleLoss 主导                            |
| L=32 iii1_lam0.1_b64                 |  2.487  |  0.512  | 接近 baseline(λ 太弱)                                  |
| L=32 iii1_lam10.0_b64                |**0.001**|**0.007**| **f_4, f_5 双塌缩** —— 比 rev-KL sym_bignet 更平凡       |
| L=32 i2_stride4h32_b64(Phase-2)      |  2.133  |  0.742  | f_5 < baseline,但比 i2_stride8h32_b64(0.476)显著高 —— 小 stride 更"忙"|
| L=32 i2_stride8h64_b64(Phase-2)      |  3.489  |**0.235**| **f_5 接近塌缩门槛** —— 大 hidden(64)在 L=32 让 f_5 砍 1/2|
| L=32 i2_stride4h32(b=128)            |  4.597  |  0.356  | f_5 略低于 baseline,**仍非塌缩**                        |
| L=32 i2_stride16h32(b=128)           |  6.882  |  0.616  | f_5 高于 baseline,深层做更多工作                        |
| L=32 i1_df4.0(Student-t b=128)       |  6.404  |**1.473**| **f_5 显著高** —— heavy-tail prior 让深层 block 做实事     |
| L=64 baseline_b16                    |  3.635  |  1.907  | L=64 fwd-KL 基线,f_5 更高(L↑→深层更繁忙)              |
| L=64 iii1_lam1.0_b16                 |  2.277  |  0.998  | f_5 比 baseline 砍半,但未塌缩                          |
| L=64 i2_stride16h32_b16              |  1.797  |  1.049  | 与 L=64 iii1 相近                                       |
| L=64 i2_stride4h32_b16(Phase-2)      |  2.085  |  2.146  | f_5 略高于 baseline,小 stride 在 L=64 上对深层是良性     |
| L=64 i2_stride4h64_b16(Phase-2)      |**4.796**|**3.723**| **f_4 / f_5 是 Phase-2 中最 *活跃* 的** —— 大 hidden 帮 L=64 深层 |
| L=64 i2_stride8h32_b16(Phase-2)      |  3.519  |  1.289  | f_4 ≈ baseline,f_5 略低 —— 健康                          |
| L=64 i2_stride8h64_b16(Phase-2)      |  3.018  |  1.868  | 与 baseline 持平                                          |
| L=64 i1_df4.0_b16                    |  3.972  |  1.956  | 与 baseline 持平,无塌缩                                |
| (参考)L=32 sym_bignet(rev-KL)      |  0.256  |  0.079  | 经典 rev-KL 深塌缩                                      |
| (参考)L=32 pathgrad_bignet(STL)    |  0.157  |  0.018  | STL rev-KL 深塌缩                                       |
| (参考)L=32 hs_bignet(fwd-KL ref)   |  2.759  |  0.300  | fwd-KL 健康                                              |

**关键发现:**

1. **III.1(scaleLoss)的真实机制是 *诱发* rev-KL 病态,而非纠正它**。λ=1 时 f_5 = 0.011(比 sym_bignet 的 0.079 更深),λ=10 时 f_4=0.001 同时崩,这是 *最深的塌缩*。combined 走的也是这条路。
2. **I.1 Student-t 与 I.2 Conditional Gaussian 都 *不诱发* 塌缩**,prior 端的修改对深层 block 是良性。
3. **I.1 在 L=32 上 f_5 = 1.473 是 fwd-KL 阵营最高的**,heavy-tail prior 让最深 scale-block 比 baseline 做更多工作。

### V3 identity 残差(gauge-fixed,2026-06-14 升级)

把每个 block 输出 `f_s(z)` 用 *逐位置* quantile transform 拉成 N(0,1) 边际,再算 `E[(T_s(f_s(z)) − z)²] / E[z²]` —— **只测 copula 层面的恒等偏离**。raw V3 残差里可能有大部分是边际形状变换,gauge 后剩 *真 copula 工作量*。
CSV:`analyzers/csv/rg_v0_v3_gauge.csv`(25 folder)。

| Run                                  | gauge r_4 | gauge r_5 |   读法                                                  |
|--------------------------------------|----------:|----------:|---------------------------------------------------------|
| L=32 baseline_b64                    | 1.153     | 0.014     | fwd-KL 基线;r_4 大,r_5 接近 0                          |
| L=32 iii1_lam1.0_b64                 | 0.060     | 0.0004    | **r_4 砍 19×,r_5 砍 35×** —— III.1 推 copula 接近恒等   |
| L=32 i2_stride8h32_b64               | **1.827** | 0.031     | **r_4 比 baseline 高 60%** —— I.2 让 copula 工作更多      |
| L=32 combined                        | 0.056     | 0.0004    | 同 iii1 路径,copula 塌缩                                 |
| L=32 iii1_lam0.1_b64                 | 1.356     | 0.037     | λ 弱,copula 仍正常                                       |
| L=32 iii1_lam10.0_b64                | **0.0004** | **0.0004** | **★ 完全 copula 塌缩 ★** —— 跟 rev-KL sym_bignet 同特征,III.1 λ=10 真走入 rev-KL 病态相 |
| L=32 i2_stride4h32_b64(Phase-2)      | 0.033     | 0.001     | r_4 砍很多 —— stride=4 在 L=32 b=64 下 copula 工作减少    |
| L=32 i2_stride8h64_b64(Phase-2)      | 0.059     | 0.001     | 大 hidden 也减 copula 工作量                              |
| L=32 i2_stride4h32(b128)             | 0.037     | 0.001     | b=128 sweep:copula 工作减少                              |
| L=32 i2_stride16h32(b128)            | 0.048     | 0.001     | 同上                                                      |
| L=32 i1_df4.0(Student-t b128)        | **1.412** | **0.749** | **Student-t r_5 大 53×** —— heavy-tail 让最深 copula 真正做实事(L=32 上 *唯一* 此模式) |
| L=64 baseline_b16                    | 0.445     | 0.170     | L=64 fwd-KL gauge 基线 —— r_4 小 / r_5 仍可观             |
| L=64 iii1_lam1.0_b16                 | 0.457     | 0.041     | r_5 砍 1/4 —— III.1 在 L=64 上减 copula 工作               |
| L=64 i2_stride16h32_b16              | 0.390     | 0.057     | r_5 砍一半,r_4 持平                                       |
| L=64 i2_stride4h32_b16(Phase-2)      | 0.375     | 0.069     | 与 stride=16 接近                                          |
| L=64 i2_stride4h64_b16(Phase-2)      | **0.135** | 0.030     | **r_4 砍 3.3×** —— 大 hidden 让 copula 工作锐减(同 V5 趋势)|
| L=64 i2_stride8h32_b16(Phase-2)      | 0.359     | 0.041     | Phase-2 winner —— 与 stride=16 同档,copula 工作正常       |
| L=64 i2_stride8h64_b16(Phase-2)      | 0.255     | 0.038     | r_4 砍 1/2(大 hidden)                                    |
| L=64 i1_df4.0_b16(Student-t)         | 0.470     | 0.039     | r_5 比 baseline 砍 4× —— L=64 上 Student-t 与 L=32 反向    |

**关键发现(新)**:

1. **L=32 iii1_lam10.0_b64 是 V3 gauge 上的"完全 copula 塌缩"代表** —— r_4 = r_5 = 0.0004,跟 rev-KL sym_bignet 的 0.0004 完全一致。**直接验证 V5 RMS-G 的"III.1 λ=10 走入 rev-KL 病态"verdict**,从 V3 gauge 这个独立 probe 上 *再次确认*。
2. **L=32 i1_df4.0(Student-t)在 V3 gauge r_5 上 = 0.75** —— **是所有 25 folder 里 r_5 唯一显著大的 cell**。heavy-tail 让最深 block 在 copula 层面真正"做实事"。但这跟 L=64 i1 的 r_5 = 0.04 **方向相反**(L=32 Student-t 让 copula 工作量加倍,L=64 反而减半)。Student-t 的 L 依赖很复杂。
3. **L=64 大 hidden 让 copula 工作锐减**(stride=4h64: r_4=0.135 vs stride=4h32: 0.375)—— 跟 V5 RMS-G 的"大 hidden 劣化"verdict 一致(机制:模型容量过剩 → 内部尺度相互"对齐",但远离真物理 cascade)。

### V4 forward-direction 相邻尺度 KS / RMS-G(2026-06-13 升级)

把 HS 数据正向通过 MERA flow,逐尺度收集 y_s,**比较 zscore(y_s[::2,::2]) 与 zscore(y_{s+1})**(s=2→3 和 s=3→4)。这是 V5 的"上游"探针:V5 比较 *MERA y_s vs block-RG x_s*,V4 比较 *MERA y_s vs MERA y_{s+1}*。

**理想行为**:好的 RG flow 应该在尺度间保持自相似 ⇒ adj_KS / adj_RMS-G 都接近 0。**rev-KL 病态特征**:深层 y_s 塌缩成恒等 ⇒ adj_RMS-G 仍小但 V5 RMS-G 大(尺度间一致但与真物理脱节)。

| Run                                  | adj_KS s2→3 | adj_KS s3→4 | adj_RMS-G s2→3 | adj_RMS-G s3→4 |
|--------------------------------------|------------:|------------:|---------------:|---------------:|
| L=32 baseline_b64                    | 0.075       | 0.058       | 0.053          | n/a            |
| L=32 iii1_lam1.0_b64(+III.1)         | 0.062       | 0.078       | **0.122**      | n/a            |
| L=32 i2_stride8h32_b64(+I.2 cond)    | 0.040       | **0.135**   | **0.110**      | n/a            |
| L=32 combined(I.2+III.1)             | 0.034       | 0.038       | 0.093          | n/a            |
| L=32 iii1_lam0.1_b64                 | 0.084       | 0.054       | **0.120**      | n/a            |
| L=32 iii1_lam10.0_b64                | **0.006**   | **0.013**   | **0.026**      | n/a            |
| L=32 i2_stride4h32_b64(Phase-2)      | 0.050       | 0.033       | 0.096          | n/a            |
| L=32 i2_stride8h64_b64(Phase-2)      | 0.027       | 0.054       | 0.035          | n/a            |
| L=32 i2_stride4h32(b128)             | 0.050       | 0.095       | 0.096          | n/a            |
| L=32 i2_stride16h32(b128)            | 0.056       | 0.052       | 0.052          | n/a            |
| L=32 i1_df4.0(Student-t b128)        | 0.029       | 0.076       | **0.014**      | n/a            |
| L=64 baseline_b16                    | 0.039       | 0.038       | 0.072          | 0.058          |
| L=64 iii1_lam1.0_b16(+III.1)         | 0.021       | 0.038       | 0.066          | 0.053          |
| L=64 i2_stride16h32_b16(+I.2)        | 0.018       | 0.047       | **0.029**      | **0.029**      |
| L=64 i2_stride4h32_b16(Phase-2)      | **0.012**   | 0.045       | **0.019**      | 0.061          |
| L=64 i2_stride4h64_b16(Phase-2)      | 0.033       | 0.027       | 0.048          | 0.040          |
| L=64 i2_stride8h32_b16(Phase-2)      | 0.016       | 0.027       | 0.032          | **0.018**      |
| L=64 i2_stride8h64_b16(Phase-2)      | 0.031       | 0.039       | 0.026          | 0.074          |
| L=64 i1_df4.0_b16(Student-t)         | **0.142**   | 0.033       | **0.013**      | 0.024          |

**关键发现:**

1. **L=32 iii1_lam10.0 的 adj_RMS-G = 0.026 极小**(同时 V5 RMS-G = 0.527 灾难)—— 完美演示 *rev-KL 塌缩特征*:模型内部自尺度高度一致(adj 小),但与真物理 block-RG 完全脱钩(V5 大)。**adj_RMS-G + V5 RMS-G 两个 metric 联合才能识别病态**,单看 adj 看不出来。
2. **L=64 i2_stride16h32_b16 在 V4 上 adj_RMS-G(0.029, 0.029)** 是 Phase-1 L=64 最低之一,与 V5 RMS-G 0.033 / 0.058 的"s=2 winner"一致 —— V4 / V5 在 I.2 上**同向**佐证。
3. **L=64 Phase-2 winner i2_stride8h32_b16** V4 adj_RMS-G(0.032, 0.018)与 V5 RMS-G(0.024, 0.031)**都**显示双 scale 改善,**V4 / V5 在 Phase-2 sweet spot 上联合佐证**。
4. **L=64 i1_df4.0(Student-t)在 V4 adj_RMS-G 上反而 *很低*(0.013, 0.024)**,而 V5 RMS-G *很高*(0.154, 0.206)—— 这是 *第二个* "adj 小 + V5 大"病态模式实例,跟 L=32 iii1_lam10 同病:**Student-t 把 MERA 内部尺度强行拉接近,但代价是远离真 block-RG cascade**(类比 rev-KL 塌缩,只不过是 fwd-KL 路径上的 *边际驱动* 塌缩)。
5. **大 hidden=64 cell 在 V4 上 mixed**:i2_stride4h64 adj 中等,i2_stride8h64 adj 中等,但都在 V5 上劣化 → 与发现 4 同向解读 *大 hidden 拉近 MERA 内部尺度但拉远真物理*。

### V5 KS(标准化边缘 marginal mismatch,scale s=2, 3)

测 `MERA 慢模 y_s` 在标准化后与 Wilson–Kadanoff block-RG 真值 `x_s` 的 marginal 距离。**越小越好**(理想 → 0)。

| Run                                  | KS s=2 | KS s=3 |   读法                                              |
|--------------------------------------|-------:|-------:|-----------------------------------------------------|
| L=32 baseline_b64                    | 0.092  | 0.080  | fwd-KL 基线                                          |
| L=32 iii1_lam1.0_b64                 | 0.112  | 0.085  | 轻微 *劣化* vs baseline                              |
| L=32 i2_stride8h32_b64               | 0.087  | 0.084  | s=2 略改善                                            |
| L=32 combined                        | **0.086** | **0.078** | **本矩阵最小 KS**                                  |
| L=32 iii1_lam0.1_b64                 | 0.113  | 0.084  | 轻微劣化                                              |
| L=32 iii1_lam10.0_b64                | **0.162** | **0.172** | **明显劣化**(mode collapse 副作用)                |
| L=32 i2_stride4h32_b64(Phase-2)      | **0.103** | **0.131** | **KS 劣化** —— stride=4 在 L=32 b=64 上劣化 marginal  |
| L=32 i2_stride8h64_b64(Phase-2)      | 0.095     | 0.103     | 与 baseline 持平,hidden 加大对 L=32 KS 无影响          |
| L=32 i2_stride4h32(b128)             | 0.111  | 0.159  | 在 b=128 上 s=3 偏差大                                |
| L=32 i2_stride16h32(b128)            | 0.099  | 0.078  | s=3 与 combined 持平                                  |
| L=32 i1_df4.0(Student-t)             | **0.175** | **0.167** | **明显劣化** —— heavy-tail 拉远了 marginal           |
| L=64 baseline_b16                    | 0.123  | 0.101  | L=64 baseline                                        |
| L=64 iii1_lam1.0_b16                 | **0.095** | 0.087  | **L=64 上 III.1 *改善* KS**(与 L=32 反向)         |
| L=64 i2_stride16h32_b16              | 0.093  | 0.088  | 与 iii1 持平                                          |
| L=64 i2_stride4h32_b16(Phase-2)      | 0.096     | 0.098     | KS 改善 21–22%                                         |
| L=64 i2_stride4h64_b16(Phase-2)      | 0.093     | 0.101     | 与 stride=16 持平                                      |
| L=64 i2_stride8h32_b16(Phase-2)      | 0.094     | 0.101     | KS 改善 24%                                            |
| L=64 i2_stride8h64_b16(Phase-2)      | **0.067** | 0.095     | **L=64 KS s=2 最佳改善 46%** —— stride=8 + hidden=64 最优 marginal|
| L=64 i1_df4.0_b16                    | **0.185** | **0.172** | **明显劣化** —— heavy-tail 在 L=64 上更糟           |

### V5 RMS-G(空间关联结构 mismatch,scale s=2, 3 —— gauge 真值)

每格点经 quantile transform 拉成 N(0,1) 边际后的 V5 RMS-G(`MERA 慢模 G(r)/G(0)` 与 Wilson `G(r)/G(0)` 的 RMS 距离,只测 copula)。**越小越好**;rev-KL 仍 ≈ 0.54(灾难),fwd-KL baseline ≈ 0.046。
CSV:`analyzers/csv/rg_v5_gauge_compare.csv`。

| Run                                  | gauge s=2 | gauge s=3 |   读法                                                  |
|--------------------------------------|----------:|----------:|---------------------------------------------------------|
| L=32 baseline_b64                    | 0.022     | 0.041     | fwd-KL 基线 gauge                                        |
| L=32 iii1_lam1.0_b64                 | 0.043     | 0.018     | s=2 略劣 / s=3 改善                                       |
| L=32 i2_stride8h32_b64               | 0.047     | **0.003** | **s=3 砍到 1/14** —— **I.2 是真结构 win**                |
| L=32 combined                        | 0.055     | 0.019     | s=2 略劣 / s=3 改善                                       |
| L=32 iii1_lam0.1_b64                 | 0.053     | 0.023     | s=3 持平                                                  |
| L=32 iii1_lam10.0_b64                | **0.393** | **0.349** | **真结构灾难** —— ~ baseline 18×,III.1 λ=10 走入 rev-KL 病态 |
| L=32 i2_stride4h32_b64(Phase-2)      | 0.069     | 0.036     | s=2 劣 / s=3 持平                                          |
| L=32 i2_stride8h64_b64(Phase-2)      | 0.032     | 0.034     | 与 baseline 持平                                           |
| L=32 i2_stride4h32(b128)             | 0.061     | 0.022     | 与 baseline 持平                                           |
| L=32 i2_stride16h32(b128)            | 0.046     | 0.034     | 与 baseline 持平                                           |
| L=32 i1_df4.0(Student-t)             | 0.062     | 0.083     | 与 baseline 持平 —— **不是结构破坏**(verdict 翻)         |
| L=64 baseline_b16                    | 0.046     | 0.042     | L=64 fwd-KL 基线 gauge                                    |
| L=64 iii1_lam1.0_b16                 | 0.040     | 0.032     | 与 baseline 持平                                           |
| L=64 i2_stride16h32_b16              | 0.046     | 0.051     | 与 baseline 持平 —— **不是 Phase-1 winner**(verdict 翻)  |
| L=64 i2_stride4h32_b16(Phase-2)      | 0.057     | 0.061     | s=2 略劣 / s=3 略劣                                        |
| L=64 i2_stride4h64_b16(Phase-2)      | 0.053     | 0.064     | 与 baseline 持平                                           |
| L=64 i2_stride8h32_b16(Phase-2)      | **0.039** | **0.030** | **★ Phase-2 winner ★** —— s=2 改善 15%,s=3 改善 29%(双 scale 真结构 win) |
| L=64 i2_stride8h64_b16(Phase-2)      | 0.046     | 0.064     | 与 baseline 持平                                           |
| L=64 i1_df4.0_b16                    | 0.044     | 0.049     | 与 baseline 持平 —— **不是结构破坏**(verdict 翻)         |

**关键发现:**

1. **III.1 λ=10 在 L=32 上 V5 RMS-G = 0.527,几乎复刻 sym_bignet 的 0.67**。这是 *最强* 的证据表明:**III.1 的内在机制是诱发 rev-KL 病态而非修正它**。
2. **I.1 Student-t prior 在两个 L 上都明显劣化 V5 RMS-G**(L=32 3.5×,L=64 3.7–7.9×)。先前用 gL 代理读出的"L=64 结构胜出"是误读 —— gL 测的是单点关联,V5 RMS-G 测的是空间级 cascade,两者解耦。**Student-t 改善了 marginal,但破坏了 block-RG 可压缩性**。
3. **I.2 Conditional Gaussian 在 L=32 s=3 上 RMS-G = 0.010(基线 0.051),5× 改善**,是 ablation 矩阵里 *唯一显著* 改善 V5 RMS-G 的 single intervention。L=64 上 s=2 仍有 22 % 改善。**I.2 在 V5 RMS-G 上的胜出是 Phase-1 最稳的发现**。
4. **combined 在 V5 上 antagonism on s=2**:III.1 的塌缩干扰了 I.2 的结构改善。**先单独完善 I.2,不要急着 combine**。
5. **Phase-2 capacity scan(2026-06-13)发现新的 L=64 V5 RMS-G winner:`i2_stride8h32_b16` s=2 = 0.024,比 Phase-1 winner `stride=16h32_b16`(s=2 = 0.033)更好 27 %,比 baseline 0.042 改善 43 %**。这是迄今 L=64 上 V5 RMS-G 最佳数字,首次同时 s=2 和 s=3 都不退化(s=3 = 0.031 与 baseline 0.026 持平)。Phase-2 capacity scan 的 (stride, hidden) trade-off **不是单调** —— hidden=64 在 *任意* stride 上都劣化 V5 RMS-G(stride=4: 0.112,stride=8: 0.062),而 hidden=32 + stride=8 是 sweet spot。**I.2 优化方向:窄 stride + 小 hidden,而非宽 stride 或大 hidden**。
6. **L=32 b=64 的 Phase-2 验证显示 stride=4h32 在 L=32 上 *劣化* s=2 RMS-G(0.073 vs baseline 0.053)** —— 与 L=64 stride=8 winner 反向。**stride sweet spot 是 L-dependent**(L=32 在 stride=8,L=64 在 stride=8 配 hidden=32)。

**Gauge-fixed 真值(2026-06-14 更新)—— 两个 verdict 翻转**:

7. **🔄 翻转:I.1 Student-t prior 不破坏结构,是边际伪影**。raw V5 RMS-G "L=32 3.5×,L=64 3.7–7.9× 劣化"在 gauge 下 **全部消失**:
   - L=32 i1: raw 0.187/0.234 → gauge **0.062/0.083**(劣化消失 66/65%)
   - L=64 i1: raw 0.154/0.206 → gauge **0.044/0.049**(与 baseline 0.046/0.042 持平!)
   - **解读**:Student-t 把 marginal 推宽,但空间结构 *与 baseline 一致*。**先前发现 2 撤回:I.1 不破坏结构,只改边际**。

8. **🔄 翻转:I.2 Phase-1 winner stride=16 也是边际伪影**。raw V5 RMS-G "L=64 s=2 改善 22%"在 gauge 下:
   - L=64 i2_stride16: raw 0.033 → gauge **0.046**(与 L=64 baseline gauge 0.046 *完全持平*)
   - **解读**:Phase-1 22% 改善里 100% 是边际形状改变,**0% 是真结构 win**。**先前发现 3 部分撤回:I.2 stride=16 不是真 winner**。

9. **✅ 部分确认:Phase-2 winner i2_stride8h32_b16 仍是真结构 win,只是幅度减小**:
   - L=64 stride=8h32: raw 0.024 → gauge **0.039**(比 baseline gauge 0.046 仍 15% 改善)
   - L=64 baseline: gauge 0.046,Phase-2 winner gauge 0.039
   - **解读**:raw 43% 改善里 ~30% 是边际,~15% 是真结构。**Phase-2 capacity scan 的 winner 立得住,只是 effect size 比 raw 数字暗示的小**。
   - Phase-2 P2.x verdict 推荐"stride=8 hidden=32 + 训练超参 sweep"成立,但**预期 plateau 提升 < 5 nat**(以前预期 22%→43% 是边际伪影夸大了)。

10. **✅ 确认:III.1 λ=10 真灾难**。raw V5 RMS-G 0.527/0.533 → gauge **0.393/0.349**(仍比 baseline gauge ~ 18× 大)。**III.1 高 λ 是真空间结构损坏**,gauge 救不了。

**对比总结**:Phase-1/2 的 raw V5 verdict 共 6 项,gauge 后 **2 项翻转、3 项确认、1 项幅度修正**。所有翻转方向都是 *"raw 看到的破坏 / 改善 = 边际形状改变"*,真空间结构差异远小于 raw 数字暗示的。

### Verdict 表(真值升级)

| 方案 | improvements 预测                  | V3 真实 f_5 | V5 RMS-G 真实 | 评级                                                      |
|------|------------------------------------|------------:|--------------:|----------------------------------------------------------|
| III.1 λ=1.0 | KL/RMS-G 显著改善           | **0.011**(塌缩) | s=2 劣化,s=3 改善 | **机制错位** —— 实际诱发 rev-KL 病态,不是 fwd-KL 改良 |
| III.1 λ=10.0 | 更紧约束 → 更好            | **0.007**(极度塌缩)| **0.527**(灾难)| **走入 rev-KL 病态相** |
| I.1 Student-t | 小改善(negation)        | **1.473**(健康) | **0.187(劣化 3.5×)**| **negation 确认 + 主动恶化空间结构** |
| I.2 cond. (b=64 stride=8) | 显著改善          | 0.476(健康)| **s=3 = 0.010(5× 改善)** | **本表唯一显著改善 V5 的方案** |
| I.2 cond. (b=128 stride=4)| (sweep 极端)      | 0.356(健康)| s=3 = 0.017 | 与 stride=8 同向但稍弱 |
| combined I.2+III.1 | 协同              | 0.003(塌缩)| s=2 antagonism | **III.1 的塌缩反噬 I.2 的结构改善** |
| **I.2 stride=8 hidden=32 b=16 L=64(Phase-2)** | (capacity scan,新做) | 1.289(健康) | **s=2 = 0.024(43% 改善)** | **★ Phase-2 winner ★** —— L=64 上首次同时改善 s=2 和不退化 s=3 |
| I.2 stride=4 hidden=32 b=16 L=64(Phase-2)   | (capacity scan)    | 2.146(健康) | s=2 略劣化 | stride=4 在 L=64 反不如 stride=8 |
| I.2 stride=4 hidden=64 b=16 L=64(Phase-2)   | (capacity scan,反例)| 3.723(过活跃)| **s=2 = 0.112(3× 劣化)**| **大 hidden 反例** —— hidden 拉大反劣化 V5 |
| I.2 stride=8 hidden=64 b=16 L=64(Phase-2)   | (capacity scan)    | 1.868(健康) | s=3 = 0.132 劣化 | hidden=64 + stride=8 仍劣化 |
| I.2 stride=4 hidden=32 b=64 L=32(Phase-2)   | (cross-L 验证)     | 0.742(健康) | s=2 略劣化,s=3 改善 53% | L=32 上 stride=4 反例(L=64 stride=4 也劣) |
| I.2 stride=8 hidden=64 b=64 L=32(Phase-2)   | (cross-L 验证)     | **0.235**(接近塌缩门槛)| **s=2 = 0.086 + s=3 = 0.095 双劣化**| **大 hidden 在 L=32 上 V3 接近塌缩 + V5 双劣** —— hidden=64 在两个 L 上都失败 |

**最重要的两个修正(vs Phase-1 中段代理读法):**

1. **I.1 Student-t 不是"超出 negation",而是 negation 加恶化**。先前 gL 代理读出的"L=64 上 I.1 结构胜出"被 V5 RMS-G 真值反转。**Student-t prior 应该从 Phase-2 路线图中下调,而非提升**。
2. **I.2 Conditional Gaussian 是 Phase-1 唯一 V5 RMS-G 真值改善 的方案**。这一发现独立于 b=64 vs b=128 比较,在 L=32 和 L=64 上方向一致(L=32 s=3 5× 改善,L=64 s=2 22 % 改善)。**I.2 应该上升为 Phase-2 P2.0 顶级优先级,优先做 capacity scan + cross-L 验证**。

**Phase-2 capacity scan 引出的第三个修正(2026-06-13):**

3. **L=64 I.2 的 (stride, hidden) 不是单调 trade-off,而是有 sweet spot(stride=8, hidden=32)**。前期假设"小 stride 总更细 = 更好"和"大 hidden 总更强 = 更好"两条都被推翻:
   - **小 stride 反例**:stride=4 在两个 L 上 V5 RMS-G s=2 都 *劣化*(L=32:0.073;L=64:0.059)
   - **大 hidden 反例**:hidden=64 在两个 L 上 V3 / V5 都 *劣化*(L=32 V3 f_5 = 0.235 近塌缩;L=64 V5 s=2 = 0.112 三倍劣)
   - **唯一胜出 cell**:`L=64 stride=8 hidden=32` 双 scale 同向改善,且 V4 / V5 一致佐证
   - **机制猜想**:hidden 加大让 MERA 内部相邻尺度过度对齐(adj_RMS-G 小),但代价是远离真物理 block-RG cascade(V5 大)—— 与 I.1 Student-t 同病机制
   - **Phase-2 路线图调整**:I.2 优化方向从"扫 (stride, hidden) 2D 网格"改为"以 stride=8 hidden=32 为种子,只扫 *训练超参*(batch、lr、epoch)" —— 架构 sweet spot 已经定。

**Gauge-fixed 引出的第四 / 五次修正(2026-06-14)**:

4. **🔄 I.1 Student-t 从"主动恶化结构"翻成"边际伪影,结构与 baseline 一致"**(详见上方关键发现 7)。**verdict 表里 I.1 行的"主动恶化空间结构"标签 *撤回*。**
   - 新评级:I.1 改边际,不改结构;在 V5 RMS-G 上 *中性*
   - 实操影响:I.1 不再是"禁区",但也不解决空间结构问题 —— 仍不优先(因为不解决核心 plateau)
5. **🔄 I.2 stride=16 Phase-1 winner 从"L=64 22% 改善"翻成"边际伪影"**(详见关键发现 8)。**Phase-1 winner 撤回**。
   - 新评级:I.2 stride=16 在 L=64 上 V5 RMS-G 不改善结构
   - **真 winner 是 Phase-2 stride=8 hidden=32**(关键发现 9),但 effect 减小到 ~15%
   - 实操影响:Phase-2 路线图里"以 stride=8 hidden=32 为种子"仍成立,但预期 plateau 提升下调到 < 5 nat

**总结表(2 raw winner / loser 翻转,1 Phase-2 winner 幅度下调)**:

| 方案                          | raw verdict                | gauge verdict                | 净结果        |
|-------------------------------|----------------------------|------------------------------|---------------|
| I.1 Student-t                 | 主动恶化结构(劣 3.5–7.9×)| 边际伪影,结构 ≈ baseline    | **🔄 撤回原verdict**|
| I.2 stride=16(Phase-1 winner)| L=64 s=2 改善 22%           | 与 baseline gauge 持平        | **🔄 撤回原winner** |
| I.2 stride=8h32(Phase-2 winner)| L=64 s=2 改善 43%          | L=64 s=2 改善 15%             | ✅ 真 winner,幅度下调 |
| III.1 λ=10                    | 灾难(0.527)              | 灾难(0.393)                 | ✅ verdict 保持 |
| III.1 λ=1                     | s=2 劣化 / s=3 改善        | s=2 *劣化撤回*,s=3 改善减半 | 半翻转        |
| rev-KL(sym/STL)              | 灾难(0.67)                | 灾难(0.54)                 | ✅ verdict 保持 |

## 推荐 Phase-2 优先级(V5 真值升级后)

修改 `improvements_zh.md` 的 Phase-2 / Phase-3 顺序:

### 立刻(P2.0,1 周内)

1. ✓ **跑 V3/V4/V5 on 13 改进 folder** —— 已完成,见上方真值表
2. **V.1 Gauge-fixed Layer-by-Layer**(**新增**,与 I.2 并行):
   - 后训练分析,无需重新训练。`analyzers/rg_fixed_point/gauge_fix.py`
     已实现 per-site quantile transform(等价 1D Spline Flow,可微可逆)
   - **解开 V3 报告里的判读盲区**:`f_3 ↔ f_4` 类似的 "V1 MSE 中等
     + V3 residual 大" 模式无法区分"真不动点(结构同但 marginal 不同)"
     vs "不同非平凡变换"。Gauge-fixed V1 直接拉到 N(0,1) 共同 marginal
     再算 MSE,**ratio < 1 ⇒ 真不动点候选**
   - 类比地把同样 transform 套进 V4/V5 的 zscore step → gauge-fixed
     KS / RMS-G,可独立确认 I.1 Student-t 的"L=64 gL 改善"是不是仅
     marginal 形变(预测:gauge-fixed 下消失,确认 Phase-1 V5 真值反转)
   - **demo 已提交**:job `40162202`(`hs_bignet`,CPU,~5 min);后续
     在 19 folder 上跑 ~2h 内完事
3. **I.2 capacity scan + cross-L**(优先级 *上调*):
   - 测 `stride=8, hidden=64`,`stride=4, hidden=64` 在 L=64 上是否进一步压低 V5 RMS-G(s=2 已经改善 22 %,看能不能拉到 50 %+)
   - 重要:验证 I.2 在 L=32 b=64 vs b=128 的"sweet stride"是否一致
   - **I.2 是 Phase-1 唯一确认改善 V5 RMS-G 的方案,Phase-2 的核心任务是稳定它**

### 中期(P2.1,2–3 周)

3. **III.1 finer λ < 1.0 sweep**(降级,优先级 *下调*):
   - λ=0.3, 0.5 看是否能保留 s=3 改善而不触发 f_5 塌缩
   - 若 λ 必须很小才避免塌缩,则 III.1 "诱发-rev-KL-病态"机制根本错;直接弃
4. **II.1 学习的 kept-fraction**(优先级 *上调* 到 P2.1):
   - I.2 的成功来自"把空间结构推到 prior 端";II.1 是"把空间结构推到 dispatch 端"的对应方案
   - 两者都改架构内的空间归纳偏置,值得并行测

### 中期(P2.2,3–6 周)—— 重做物理基线

5. **I.4 粗化 Ising prior**:
   - 比 I.3 物理上更直接;让 prior 自身是一个小 Ising 分布,跳过 EBM 拟合的中间环节
6. **重新评估 I.3 EBM/φ⁴**(优先级 *下调*):
   - 原本因 I.1 的"L=64 结构胜出"被推上去;现 V5 真值反转,**I.1 反而 *破坏* 结构 ⇒ EBM 路线的原动力消失**
   - 仅在 I.2 capacity scan 失败、I.4 也不够时再回看

### 长期(Phase-4)

7. **II.2 self-similarity 框架**(scheme C):博士后级独立项目

### 移除(确认)

- ~~I.1 Student-t 进一步推广~~:V5 RMS-G 反向证据(L=32 3.5×,L=64 3.7–7.9× 劣化)。**直接关闭这一支线**
- ~~I.5 学到的非高斯 prior~~:I.3/I.4 物理更明确
- ~~III.2 V5-as-loss~~:V5 RMS-G 是核心瓶颈已确认,但 V5-as-loss 会让 V5 失去 *独立判官* 地位;**仅在 I.2/II.1/I.4 三条路全部失败时回看**
- ~~III.1 combined 路线~~:V3/V5 真值都显示 combined 是 antagonism on s=2;不再值得作为 Phase-2 顶级方案

## 相关文件

- `improvements_zh.md` —— 原 8 方案路线图
- `rg_fixed_point_report_zh.md` —— 病态诊断(本报告假设其结论作为出发点)
- `concise_report_L64_T2.269.md` —— L=64 改进 ablation 简版
- `data/{32,64}Ising_T2.269_hsBignet_*/flow_diagnostic.json` —— 本报告所有数值的原始 JSON
- `shell/run_L32_iii1_single.sh` / `shell/run_L32_i2_single.sh` / `shell/run_L32_i1_single.sh` —— Phase-1 训练脚本(及 L=64 对应 `_b16.sh`)
- `shell/analyze_L32_single.sh` / `shell/analyze_L64_single.sh` —— 单 folder diag launcher

## 关键留存的不确定性

1. ✓ **V3/V4/V5 真值已到位** —— 见上方真值表,代理 verdict 已被部分推翻(关键反转:I.1 实际 *劣化* V5 RMS-G)
2. **i2_stride8h32 b=128 Phase-1 原 run 完全发散** —— `KL_qp = 604190`(末段不稳)。该 folder 无法用于任何结论;实际 i2 b=128 评估靠 stride=4/16 两个 sweep 点
3. **L=64 改进信号普遍在噪声里** —— b=16 限制下大部分 Δ 落在 ±1 nat。Phase-2 应该考虑 effective-batch 提高(梯度累加)以分离信号
4. **V4 forward-direction probe 真值已写入 `analyzers/rg_fixed_point/csv/rg_v4_dataforward.csv`** —— 上方 verdict 主要用 V5 (block-RG cross-comparison) 的 13 行,V4 的真值可作补充;两个 probe 在原 6 method 上一致(rev-KL 极大 / fwd-KL 中等),改进 folder 上同样模式
5. **训练收敛性扫过(2026-06-09)** —— 14 改进 run 训练 LOSS 末 100 ep 全部稳定;唯 `i2_stride8h32 b=128` 末段 checkpoint 采样 broken(训练 LOSS 健康但 latest .saving 产 KL_qp = 604K),已在所有表里排除
4. **跨 L 比较未做 per-site 归一化** —— per-site KL 跨 L 时 baseline 本身有 FSS scaling(α ≈ 2.20),严格分析时要做(参考 `project_fss_critical_scaling` memory)

---

## 附录 P2.x:L=64 plateau 联合 verdict(2026-06-14)

Phase-2 启动时假设 L=64 bignet plateau(LOSS ≈ 7686 nat)由 **架构容量** 或 **数据量** 限制。两个独立实验同时收尾,**两条假设全部被推翻**。

### 实验 A — megabignet 容量上调(nhidden 128 → 192,1.5× params)

| 配置                          | last 100 (ep ≈ 19800) | best 300-smoothed | Δ vs bignet 7686 |
|-------------------------------|----------------------:|------------------:|-----------------:|
| bignet baseline(nhidden=128) | 7687.80 ± 23.50       | 7686.x            | (ref)            |
| megabignet(nhidden=192, lr=5e-4, gc5.0)| **7723.15 ± 22.95**   | **7724.27**       | **+37 nat**      |

⇒ 1.5× 参数量 + 调过的 lr/gradClip 仍**无法达到** bignet plateau。**容量不是瓶颈**。

### 实验 B — dataset 大小消融(N ∈ {50K, 100K, 200K},架构 = bignet, epochs = 20K)

| N(MCMC samples)| last 100 mean | std    | Δ vs N=200K |
|:----------------:|--------------:|-------:|-----------:|
| 50K              | **7687.51**   | 20.12  | −0.29       |
| 100K             | **7690.20**   | 23.65  | +2.40       |
| 200K(参考)      | 7687.80       | 23.50  | (ref)       |

⇒ 三者**全部在噪声以内**(差 ±2 nat,std 20–24)。**N 从 50K 翻 4× 到 200K,plateau 不动**。**数据量不是瓶颈**。

### 联合结论

| 假设                                  | 测试范围                | 证据                           | 状态(严格) |
|---------------------------------------|------------------------|--------------------------------|:-----------:|
| 在 *bignet* 上加 width 立刻就能破 plateau | nhidden 128 → 192,**1 cell**(lr=5e-4 + gc=5.0 + batch=16) | megabignet plateau +37 nat 劣  | **未确认有效**(see 限定 ①) |
| *bignet* 50K 数据已不够               | N ∈ {50K, 100K, 200K} | bignet 三者同 plateau ±2 nat   | **对 bignet 成立** ⇒ 对其它架构未知 (see 限定 ②) |
| 当前 LR schedule + 20K epoch 已够长   | 单段固定 lr           | best-smoothed 早已稳定         | **当前 schedule 下成立**(see 限定 ③) |

**⇒ 严格 verdict:**bignet baseline 已在"低维 single-knob 调参"(单独加 width / 单独加 data / 单独加 epoch,所有其他超参不动)上**全面饱和**。
**不等于**:任何 width / data / epoch 都不能再压 plateau。

### 三条限定(reviewer 的反驳路径)

**限定 ①(megabignet 未充分扫):**
- 仅测了 1 cell(nhidden=192, lr=5e-4, gc=5.0, batch=16)
- 未扫:lr ∈ {1e-4, 2e-4} × β2 ∈ {0.98, 0.999} × batch ∈ {16, 32} × warmup(线性 0→peak 在 1–5K epoch)
- 严格关 megabignet 需要至少 ~ 12 cell sweep
- megabignet 的 `eff_indep/params` 仅 0.11(bignet 0.24)→ Chinchilla 视角下 *数据饿训*,与限定 ② 耦合

**限定 ②(数据饱和仅对 bignet 成立):**
- "bignet 50K 已饱和" ≠ "更大架构也 50K 饱和"
- 经典 Chinchilla:大模型需要更多 data;megabignet 在 50K 上很可能 *饿训*
- 测试 megabignet 时 *必须* 同步上调 N(N=500K 或 1M),否则 width 实验和 data 实验互相搞错变量
- **结果**:扩 dataset 不是"已被证伪",而是"对单独 bignet 没用,但 *与 megabignet/NSF 联合测试* 时是 must"

**限定 ③(epoch 仅在固定 LR schedule 下饱和):**
- 仅测过 20K epoch + 固定 lr
- 未试:cosine annealing(1e-3 → 1e-5),阶梯衰减,cyclic LR
- memory `project_l32_late_training_instability` 表明末段本来就 noisy ⇒ 退火 LR 潜在压 2–5 nat
- memory `project_resume_optimizer_state` 表明 `-load` 不复原 Adam 矩 ⇒ stage 化训练在 codebase 上 broken,需先修

### Phase-2 路线图(基于严格 verdict 重写)

**降级(单 knob 已饱和,留作组合实验的副产品):**
- 单独加 megabignet 不再作为独立路线;**只作为 NSF 实验的伴随产物**(NSF coupling 本来就需更大 hidden + 更小 LR + gradClip)
- 单独扩 dataset 不再作为独立路线;**仅在测试 megabignet/NSF 时同步扩**(N=500K 或 1M;**本次同时启动扩数据**,见下方扩数据说明)
- 单独加 epoch 不再作为独立路线;**与 LR schedule 改动联动**(优先 cosine annealing)

**关闭(P1 真值已直接反对):**
- ~~I.1 Student-t prior~~:Phase-1 V5 真值显示 *劣化* 3.5–7.9×,不论 L
- ~~III.1 combined~~:V3/V5 都显示 antagonism on s=2
- ~~III.1 λ ≥ 10~~:走入 rev-KL 病态相(L=32 实测 RMS-G 0.527)

**新优先级(P2.x):**
1. **换 architecture family**(真正改 family,不只是改 width):
   - NSF coupling(rational quadratic spline 替 affine)—— code 已有;**与扩 dataset (N=500K) 同步上**
   - Z2-equivariant RNVP(memory `project_z2_equivariance_todo`)
   - 学到的 prior(I.4 粗化 Ising prior / I.3 EBM)—— 物理动机:Gaussian latent ≠ Wilson-Fisher
2. **Multi-L 联合训练** —— weight-tied MERA 共享 L=8/16/32/64 参数;借小 L 数据提升 L=64 effective N
3. **D₄ 数据增强** —— 8× 效样本(对称性归纳偏置,与 limit ② 正交)
4. **II.1 学习的 kept-fraction** —— Phase-2 未做,空间归纳偏置正交改动
5. **LR schedule 改动**(限定 ③ 的廉价 follow-up):cosine annealing 20K epoch on bignet,看是否压 2–5 nat;**便宜单 GPU 一次实验**

**保留(等数据):**
- gauge_v5_compare(job 40261906)将给 25 folder gauge-fixed RMS-G,**判定 I.2 L=64 s=2 22 % 改善是结构 win 还是 marginal artefact**

### 该 verdict 的(严格)含义

L=64 hs_bignet 的 LOSS 7686 与 H(p_HS) = 7637.6 之间的 **48 nat gap(per-site ~0.012)**:
- 在**当前 family + 当前优化路径 + 当前数据规模**下饱和,**至少需要 2 个轴联动**(width × data,或 family × LR schedule)才有可能再压;
- 不能再靠"单独 *加宽* / 单独 *加 data* / 单独 *加 epoch*"这种 single-knob 调参砍下去 —— 这是 Phase-2 第一阶段被实验确认的事实;
- 仍可能是 MERA + affine RNVP + Gaussian prior 与临界 Ising 真实复制器的结构失配,但需 NSF + 大数据联合实验才能证实(单 family 内打不开是 *必要* 不是 *充分* 条件)。

**Phase-2 投入应:(a) NSF / Z2-equiv 至少 1 条 family-level 改动 + 同步扩 dataset 到 N=500K;(b) bignet 上做一次廉价 cosine LR schedule 收尾实验。**

### 实验 C — 双轴联动验证(2026-06-14 完成)

P2.x 严格 verdict 提出"至少需要 2 个轴联动才有可能再压"。Track B + C 是该 verdict 的直接测试:

**Track B**(限定 ③ 测试):megabignet ep 19800 *续训* 10K epoch,加 **cosine LR 5e-4 → 5e-6** annealing,N=200K(同 megabignet 原数据)
**Track C**(限定 ② 测试):megabignet ep 19800 *续训* 10K epoch,固定 lr=5e-4,**N=500K**(2.5× 数据,parallel 链生成)

| 配置                                          | last 100 mean | std    | Δ vs bignet 7686 | Δ vs 原 megabignet 7723 |
|-----------------------------------------------|--------------:|-------:|-----------------:|------------------------:|
| bignet baseline(nhidden=128,固定 lr,N=200K)| 7687.80       | 23.50  | (ref)            | —                       |
| 原 megabignet(nhidden=192,固定 lr,N=200K) | 7723.15       | 22.95  | +37 nat          | (ref)                   |
| **Track B(megabignet + cosine + N=200K)**    | **7715.86**   | 24.32  | **+30 nat**      | **−7 nat**(轻微改善)   |
| **Track C(megabignet + N=500K + 固定 lr)**   | **7728.58**   | 25.81  | **+43 nat**      | **+5 nat**(噪声内劣化)|

**联合结论(2026-06-14,P2.x 终结)**:

1. **cosine LR schedule 在 megabignet 上单给 ~7 nat 改善** —— 比预测的 2–5 nat 略高,但**远不足以拉到 bignet plateau**(还差 30 nat)。Cosine LR 是真实但 *小* 改善。
2. **2.5× 数据(N=200K → N=500K)在 megabignet 上 *没有* 改善**(差 5 nat,std 内)。**megabignet 在 N=200K 上已饱和**,不饿训。**限定 ② 在该 cell 被实验否定**。
3. **架构 × LR schedule × 数据三轴联动(Track B + Track C)合并潜在改善 < 12 nat,仍 30+ nat 不到 bignet plateau**。
   ⇒ **现 architecture family 内,组合调参已经 *彻底饱和***。
4. **bignet baseline 7686 不仅是 single-knob 饱和,也是 *多 knob 联动饱和*** —— P2.x verdict 严格化到 *硬下限* 等级。

### Phase-2 路线图(P2.x 最终版)

```
× 单 knob 加 width / data / epoch                (已被单独实验否定,2026-06-13)
× 单 knob 加 LR schedule                          (Track B 单 +7 nat,不够)
× 多 knob 联动(width × LR × data)              (Track B+C 联合 < 12 nat,仍 30 nat 缺口)
× I.2 Phase-1 winner stride=16h32              (gauge-fixed verdict 显示是边际伪影)
× I.1 Student-t prior                           (gauge-fixed 显示不破坏结构,但也无改善)

✓ 唯一存活的 in-family 改动: I.2 Phase-2 sweet spot stride=8 hidden=32
    - gauge V5 RMS-G s=2 从 0.046 → 0.039(15% 真结构 win)
    - 预期 plateau 提升 < 5 nat(基于改善幅度 / Wilson α scaling 推算)

⇒ 主要 budget 应转向 family-level 改动:
    1. NSF coupling(architecturally 更强 prior)
    2. Z2-equivariant RNVP(architecturally 注入对称归纳偏置)
    3. 学到的 prior(I.4 粗化 Ising / I.3 EBM)
    4. Multi-L 联合训练(scale-invariance 利用小 L 数据)
```

### 实验 D — i2 + nrepeat=2 协同(2026-06-25 完成,**verdict 推翻**)

P2.x 之前 verdict 说 "in-family 调参彻底饱和"。**这条结论被 i2 + nrepeat=2 联用 *推翻***。

设计 4-cell 矩阵看协同:

| Cell | i2 cond. Gaussian prior | nrepeat | 物理直觉攻击点 |
|------|:-----------------------:|:-------:|----------------|
| A baseline | ❌ | 1 | reference |
| B i2 only(Phase-1 P1 winner)| ✅ stride8h32 | 1 | 改 *target*(latent prior 把 fast mode 目标改成 conditional Gaussian)|
| C nr=2 only | ❌ | 2 | 加 *forward capacity*(同尺度 2 层 affine 串行)|
| **★ D i2 + nr=2 ★** | ✅ stride8h32 | 2 | **双攻**(prior + forward 都改)|

#### L=64 4-cell 完整诊断对比

| Cell | LOSS plateau | **KL_qp** | KL_pq | \|M\| | gL | xi |
|------|-------------:|----------:|------:|------:|---:|---:|
| A baseline | 7686 | 86.88 | 65.64 | 2.267 | 0.433 | 15.190 |
| B i2 only | 7695 | 93.20 | 69.53 | 2.287 | 0.439 | 15.292 |
| C nr=2 only | 7736 | **156.36** ❌ | 131.95 | 2.351 | 0.456 | 15.792 |
| **D i2 + nr=2 ★** | **7666** | **51.33** ✅ | **42.38** | **2.254** | **0.418** | **14.923** |

HS data anchors:`|M|_p = 2.200,  gL_p = 0.407,  xi_p = 14.782`(L=64 T_c)

**关键观察**:
1. **D KL_qp 51.33 比 A 87 改善 41%** —— L=64 上 *首次* 真正破 baseline plateau
2. **D 结构匹配近乎 anchor**:gL 偏差 < 3%,xi 偏差 < 1%,mag 偏差 < 2.5%
3. **C 单独 *灾难***(+69 nat) —— 印证 megabignet 规律:加 capacity 不约束反劣化
4. **B 单独 *不改善* KL_qp**(+6 nat) —— Phase-1 winner 标签基于 V5 gauge,KL_qp 角度从未 break

#### 超加性协同 quantified

| 模型 | KL_qp 预测 |
|------|-----------|
| 独立加和(A + Δ_C + Δ_B)| 87 + 69 + 6 = **162** |
| 实际 D | **51** |
| **净协同** | **−111 nat 超出加性预测** |

⇒ **机制是 *正交* 双攻**:
- **i2 改 target**:让 fast mode latent prior *从严格 N(0,I) 改成 conditional Gaussian*,匹配 Ising fluctuation 的局部耦合性
- **nrepeat=2 加 forward capacity**:同尺度 2 层 affine *串行* 吸收高阶矩残差,*精细* Gaussianize fast mode
- **联用 = 把 target 拉近来 + 提升到达 target 的能力**,两个攻击点 *不互相挡道*

#### L=32 cross-L 印证

| Cell | KL_qp | 改善 |
|------|------:|-----:|
| A baseline | 23.42 | (ref)|
| B i2 only | 21.16 | −2.3 |
| **D i2 + nr=2** | **17.69** | **−5.7(24%)** |

L=32 也是 D 最优,但改善幅度小(L=32 已接近 H(p_HS) limit)。

**Cross-L 一致性**:
- L=32 改善 24%
- L=64 改善 41%
- ⇒ **L 越大改善幅度越大**(因为 L=64 baseline plateau 受 FSS 临界标度(α≈2.20)更严重 → 改善空间更大)

#### P2.x verdict 重大修正

| 旧 verdict(2026-06-14) | 新 verdict(2026-06-25) |
|--------------------------|--------------------------|
| in-family 调参彻底饱和 | **错误。i2 + nrepeat=2 联用 break 7686** |
| 主要 budget 转 family-level 改动 | **family-level 仍优先,但 in-family 组合改动 *也* 有空间** |
| stride=8 hidden=32 i2 是 "唯一存活" | i2(8,32) **配 nrepeat=2** 才是 真正的 in-family winner |

**新 Phase-2 路线图(2026-06-25)**:

```
✓ in-family 突破方案(确认有效):
    i2(stride=8, hidden=32) + nrepeat=2
    - L=64 KL_qp 砍 41%,LOSS plateau 砍 20 nat
    - 结构匹配:gL/xi/mag 距 anchor < 3%
    - 协同 mechanism 物理上正交(prior 端 + forward 端)

✓ 接下来值得做(in-family 续扩):
    1. nrepeat=3,4 sweep(D 的 nrepeat 单调性)
    2. D + cosine LR(stack 已知改善方向)
    3. D + N=500K(stack 数据扩展;之前 megabignet 失败的实验在 D 上重试)
    4. D 跟 V0-V5 + gauge 机制分析(probes 已投,等结果)

✓ family-level 仍优先(独立改进方向):
    1. NSF coupling(更强 prior)
    2. Z2-equivariant RNVP
    3. 学到的 prior(I.4 / I.3)
    4. Multi-L 联合训练
```

### V0–V5 + gauge probes on D cells(2026-06-26 完成)

3 个 P2.x cell(D32 = L=32 i2+nr=2,C64 = L=64 nr=2 only,D64 = L=64 i2+nr=2)的 V1/V2b/V3/V4/V5 probes 现已完成(`logs/L*_diag_40677*.out`、`analyzers/csv/rg_*.csv` 2026-06-26 更新)。

#### V3 raw vs gauge(单 block 恒等残差)

| Cell | raw r_5 | **gauge r_5** | raw r_6 | **gauge r_6** | raw / gauge ratio @ f_5 |
|------|--------:|--------------:|--------:|--------------:|------------------------:|
| L=32 baseline_b64       | 0.30    | 0.014         | n/a     | n/a           | ~22× |
| L=32 i2_stride8h32_b64 (B P1) | 0.49 | 0.031        | n/a     | n/a           | ~16× |
| **L=32 D32(i2+nr=2)** | **0.77** | **0.003**    | n/a     | n/a           | **~260×** ⚠ |
| L=64 baseline_b16       | 0.30    | 0.17          | 0.0064  | 0.006         | ~2× |
| L=64 i2_stride8h32_b16(B bignet)| 0.49 | 0.041   | 0.0014  | 0.001        | ~12× |
| **L=64 C64(nr=2 only)**| **2.66** | **0.004** | **2.66** | **0.002**    | **~600×** ⚠ |
| **L=64 D64(i2+nr=2)** | **7.15** ⚡ | **0.028**  | **2.72** | **0.0008**   | **~250×** ⚠ |

**关键发现**:
- **C64 / D64 raw r_5 / r_6 都 *巨大***(2.66 / 7.15)但 **gauge 都很小**(~0.001-0.03)
- **raw/gauge ratio 在 P2.x cells 上是 *250-600×***,baseline 只 ~2-22×
- ⇒ **nrepeat=2 让深 block 做 *大量 marginal 工作*** —— 把 fast mode 边际形状重塑成 N(0,1)。这是 *nrepeat 物理意图* 的 *直接* 实证
- gauge r_5 / r_6 都很小(~0.001-0.03)⇒ copula 层面深 block 接近恒等(real RG fixed-point 行为)

#### V5 raw RMS-G(空间结构 vs Wilson)

| Cell | raw s=1 | raw s=2 | raw s=3 | raw s=4 |
|------|--------:|--------:|--------:|--------:|
| L=32 baseline | 0.06 | 0.05 | 0.05 | n/a |
| **L=32 D32** | **0.20** | **0.14** | **0.16** | n/a |
| L=64 baseline | 0.07 | 0.04 | 0.03 | 0.03 |
| **L=64 C64** | 0.11 | 0.06 | 0.20 | **0.62 ⚠** |
| **L=64 D64** | 0.16 | 0.17 | 0.15 | 0.31 |

**关键反转 ⚠**:**D cells V5 raw *显著劣化***(D32 s=2 = 0.14 vs baseline 0.05 = 3× 劣;D64 s=2 = 0.17 vs baseline 0.04 = 4× 劣)。
但 *同 cells* KL_qp 改善(D32: 17.7 vs 23.4;D64 投影 ~7700 vs A 7686)。

⇒ **D cells *不是* Wilson cascade 的 winner,而是 *学到 *不同的* fixed point***:
- KL_qp 角度好(总体分布距离 p_HS 近)
- V5 Wilson 角度差(空间 cascade 跟 Wilson 不一致)

**对比 D bignet(原 Phase-2 winner B stride=8h32 nr=1)**:V5 gauge s=2 = 0.039(比 baseline 0.046 *改善*)+ KL_qp 改善 → *Wilson + KL_qp 双 win*。
**D32 / D64 是 KL_qp win + V5 *劣化* 反向** —— 跟 L=64 Student-t、L=64 i2 Phase-2 i2(4,32)/(8,64) *同模式*("internal self-similar but external worse")。

#### V5 gauge RMS-G(空间结构 *gauge-fixed*)

| Cell | gauge s=1 | gauge s=2 | gauge s=3 |
|------|----------:|----------:|----------:|
| L=32 baseline | 0.041 | 0.022 | 0.041 |
| L=32 B i2 only | 0.063 | 0.047 | 0.003 |
| **L=32 D32** | **0.117** | **0.076** | 0.086 |
| L=64 baseline | 0.071 | 0.046 | 0.042 |
| L=64 B i2 only | 0.067 | 0.039 | 0.030 |
| **L=64 C64**(待 gauge_transforms.pt)| pending | pending | pending |
| **L=64 D64**(待 gauge_transforms.pt)| pending | pending | pending |

L=64 P2.x cell 的 gauge_v5 待 `40677193`(C64)和 `40677194`(D64)的 `gauge_fix_demo` 跑完后再 compute。**等 L=64 gauge V5 数字 → 确认 D64 是否 "raw 劣 + gauge 也 劣"(单纯不像 Wilson)还是 "raw 劣 + gauge 仍好"(D bignet 同 mechanism)**。

#### 机制 verdict(基于已有数据)

**i2 + nrepeat=2 联用学到 *不同* fixed point**:
1. **深 block 做大量 marginal / fast-mode 工作**(raw V3 r_5 = 0.77-7.15 比 baseline 大 3-20×)
2. **gauge V3 r_5 / r_6 接近 identity**(0.001-0.03 — 真 RG 行为)
3. **V5 raw 跟 Wilson cascade 不像**(s=2 大 3-4×)
4. **KL_qp 改善**(总体距离 p_HS 近)

这是个 **新发现的 fixed point** —— 不是 Wilson cascade,而是 *p_HS-adapted* fixed point:
- 总体分布 *更接近* HS 数据(KL_qp ↓)
- 但 cascade 结构 *不是* Wilson cascade(V5 ↑)
- 像是 "data-driven fixed point" 而非 "Wilson fixed point"

⇒ **i2 + nrepeat=2 *突破* 了 bignet plateau,但代价是 *偏离* Wilson cascade 物理目标**。
**对 *KL_qp 任务* 这是 win;对 *RG fixed-point 任务* 这是 *不同方向***(不能既要既要)。

未来工作:研究"D 学到的新 fixed point"的物理含义。可能是 *p_HS 的 attractor*,跟 Wilson critical attractor 不同。

**主要 plateau 突破必须换 family,不能在 MERA + affine RNVP + Gaussian prior 这组里再磨**。

#### 补遺 —— "不同 fixed point"框架不准确(2026-06-26 修正)

上面"D 学到不同 fixed point"的写法 *不严谨*,**两个 fixed point 其实是同一个**。理论上 `p_HS` *就是* 2D Ising 在 T_c 的 Wilson critical attractor —— "p_HS-adapted"和"Wilson"在物理上不是两件事。

真正发生的是:**i2 + nrepeat=2 把 *同一个* Wilson fixed point 的工作 *拆给* 了 MERA 跟 CNN-prior 两块,V5 只测 MERA 那一块,所以 *漏看* 了 CNN 接住的部分**。

具体:

- baseline(Gaussian prior):loss 要求 `z = MERA.forward(x) ~ N(0, I) iid`。MERA *必须* 把 Ising 短程耦合完全 decouple 到 isotropic latent ⇒ V5 看 MERA 中间 y_s,**MERA 做全部 Wilson 工作 ⇒ V5 公平**。
- i2(conditional Gaussian prior):loss 要求 `z_fast | z_slow ~ N(μ(z_slow), σ²(z_slow))`,**CNN 学 μ, σ**。MERA 可以把 z_fast 跟 z_slow 留有局部耦合(CNN 接得住、给它正确 log-prob)⇒ MERA 不必把 Ising 短程耦合做完 ⇒ V5 看 MERA 中间 y_s 偏离 Wilson,*但物理没丢*,只是 *被 CNN 接走了*。
- V5 漏看 CNN:V5 只调用 `mera = fw.flow if hasattr(fw, "flow") else fw`,prior 没传进来。即便传进来,在 normalizing flow 里 prior 也不变换 sample,只 score。所以 V5 *结构上* 无法把 CNN 接走的物理算回来。

⇒ **D cells 跟 Wilson 是 *同一个* fixed point**,V5 raw 偏离 *不代表* 学到不同物理,代表 **CNN-prior 容量帮 MERA 分担了短程耦合**。当 CNN 容量大(megabignet × i2)时分担越多,V5 raw 偏 Wilson 越明显。

**判别实验(方法论已撤销,见新报告)**:CNN offload 定量分析首版尝试(V6,`rg_v6_cnn_offload.py`, job `40757108`)使用了错误的 target(把 z_fast 的 marginal 拿去跟 N(0,1) 比,忽略了 i2 model 里 z_fast target 本就是条件分布 mixture 而非 N(0,1))。**详细方法论批评 + 从 loss 直接推导的正确定量框架,见 [`prior_offload_analysis_zh.md`](./prior_offload_analysis_zh.md)**。

**主要 plateau 突破必须换 family,不能在 MERA + affine RNVP + Gaussian prior 这组里再磨**。
