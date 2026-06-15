# RG Fixed-Point 探针 —— 改进方向报告

> 配套阅读:`rg_fixed_point_report_zh.md` —— 现有架构在 T_c 上的
> 病态诊断;本报告是面向"如何修"的前瞻路线图。

## 出发点:结构性失配

`rg_fixed_point_report_zh.md` "为什么 T_c 在此架构上难"小节给出
的诊断可总结为一句:

> **Gaussian-prior MERA 流在每一尺度上强制 1/4 慢模 + 3/4 独立
> N(0,1) 快模的分解,这是 trivial Gaussian fixed point 的几何;
> 而 2D Ising 在 T_c 的真实吸引子是 Wilson–Fisher CFT(η = 1/4,
> 非高斯算子谱),不存在任何"3/4 模可独立高斯化"的尺度分离。
> 架构隐含的不动点与物理真实的不动点结构性不兼容。**

V3 深层 block 塌缩、V4 慢模膨胀 / 收缩、V5 KS-vs-RMS-G 两轴失败,
都可追溯到这一根源。要修这套架构,有三类正交干预可做:
**换 prior、换架构(dispatch 几何)、换损失(训练信号)**;
此外还有一类"换论述"的零成本退路。

本报告整理 8 个具体方案,按维度归类,给出实现位置、预期 V5 表现、
成本与风险。最后一节给出 cost-leverage 推荐顺序。

---

## 改进方向总览

| 编号 | 方案                                          | 类别      | 预期 leverage | 工程成本 | 主要风险             |
|------|-----------------------------------------------|-----------|---------------|---------|----------------------|
| I.1  | Student-t prior                               | prior     | 低–中         | 低(1d) | 治标不治本           |
| I.2  | **方案 A** 条件高斯 P(z_fast \| z_slow)        | prior     | 中–高         | 中(1w)| 实现复杂,采样设计    |
| I.3  | **方案 B** 相互作用先验(energy-based)         | prior     | 高            | 中–高(2w)| MCMC / 归一化常数 |
| I.4  | 粗化 Ising prior(literal RG decimation)        | prior     | 高            | 高(2w)| 需 prior 数据 + 可微 |
| I.5  | 学到的非高斯 prior(AR / 小流嵌套)             | prior     | 中–高         | 高(2-3w)| 参数膨胀,训练不稳   |
| II.1 | 可学 kept-fraction                            | 架构      | 中            | 中(1w)| 改 dispatch 几何     |
| II.2 | **方案 C** 自相似约束 + 不舍弃                | 架构+损失 | 高(若成功)   | 极高(1m+)| 偏离 NeuralRG 框架 |
| III.1| 多尺度损失                                    | 损失      | 高            | 低(1w)| 调 λ_scale 超参      |
| III.2| Block-RG 监督训练(V5-as-loss)                | 损失      | 中–高         | 中(2w)| V5 数据生成成本      |
| IV.1 | 重新论述                                      | 论述      | —             | 零      | 放弃 fixed-point 框架|

---

## I. Prior 维度

### I.1  Student-t prior(否定实验,先做)

**动机.** 最便宜的"轻量非高斯"尝试。Student-t 有 heavy tail,
能容纳一定的非高斯性而无需改变 prior 的可分离性
(`p(z) = ∏_i p(z_i)`)。

**实现.**
- `source/gaussian.py` 旁加一个 `source/student_t.py`,实现
  `logProbability(x)`(可用 `torch.distributions.StudentT.log_prob`
  求和)和 `sample(n)`。
- `train/learn.py:57` 把 `s = source.Gaussian(...)` 改为
  `s = source.StudentT(df=4, shape=...)`。
- 训练流程其余完全不动。

**预期 V5.** 深层 V3 残差会变,KS 列(尤其 rev-KL T_c 的 0.32+)
应改善 ~30%;**G(r) 破坏(RMS-G ≈ 0.62)未必修好** —— 因为
heavy-tail 只能容纳"边缘非高斯",但 Wilson–Fisher 的根本困难是
**长程相关性的空间结构**,不是边缘分布形状。

**判定.** 如果做了 I.1 病态依旧,可正式排除"prior 是瓶颈"这条
解释,把后续投资转向架构(II)或损失(III)。这是一个**重要的负面
结果**,价值在排除假说。

---

### I.2  方案 A —— 条件高斯先验(Hierarchical / Conditional Priors)

**动机.** 不再假设 3/4 的快模 `z_fast` 与 1/4 的慢模 `z_slow` 独立,
改为:

```
P(z) = P(z_slow) · P(z_fast | z_slow)
```

其中 `P(z_fast | z_slow)` 仍是 Gaussian,但 **均值或方差是 z_slow
的函数**。物理上:被冻结的快模虽然不再参与后续宏观 RG 流,但与
宏观慢模之间的物理纠缠在局部尺度上 *被保留*。架构不再强迫
"快模与慢模解耦",缓解了强行解耦的矛盾。

**实现位置.**
- `source/` 新增 `conditional_gaussian.py`,提供
  `logProbability(z, condition=z_slow)`。
- 关键改动在 `flow/hierarchy/template.py`:`forward`/`inverse`
  循环要在评估 prior 时把 latent 拆成 `(z_slow, z_fast)`,
  其中 `z_slow = z[..., ::2^depth, ::2^depth]`,`z_fast` 是
  其余位置。
- prior 的条件结构可以参数化为:
  - **最简**:`P(z_fast | z_slow) = N(μ(z_slow), σ²(z_slow))`,
    `μ, σ` 用一个共享 CNN。
  - **更强**:层级条件 —— 每个 fast 位置 `(i,j)` 条件于
    *它所属粗粒度 cell 的慢模值*(局部条件,几何上自然)。
- 训练:`flow.logProbability(x)` 在 latent 端按
  `log P(z_slow) + log P(z_fast | z_slow) + log|det J|` 计算。

**预期 V5.**
- V5 边缘 KS 应改善 ~50%(因为 fast 位置的边缘从硬性 N(0,1) 松到
  "条件 N(μ(z_slow), σ²)" —— marginal 不再被强制单峰 N(0,1))。
- V5 RMS-G 应明显改善(因为 z_slow → z_fast 的耦合留了一条
  "慢模诱导的空间相关"通道,流不必为了把 G(r) 压平而塌掉关联)。
- V3 深层 residual 应该 *从下方靠拢中部*(rev-KL 不再有动力塌
  identity,因为塌了 fast 就被强制 N(0,1) 与慢模独立,违反新 prior 的结构)。

**风险.** `μ, σ` 网络容量不足 ⇒ 条件结构退化为 unconditional;
容量过大 ⇒ prior 把流的工作全抢走(prior 就足以拟合数据,流
学到的双射变恒等)。需要 ablation 找合适规模。

**与现有架构兼容性.** 高 —— 完全可以与现有 MERA、weightTying、
haarPrior 并存。是最值得先试的"换 prior"路线。

---

### I.3  方案 B —— 引入相互作用的先验(Energy-Based Prior)

**动机.** 彻底放弃无相互作用的高斯假设。在 RG 的每一步,为 3/4
的"被丢弃"自由度设定一个 **含相互作用项** 的先验,例如基于 φ⁴
理论的局部能量:

```
−log P(z_fast) = ∑_i [½ m² z_i² + ¼ λ z_i⁴] + ∑_⟨ij⟩ J z_i z_j
```

物理上:**承认"即使被丢弃的模,也受临界波动的支配"**,而不是
把它们当成与体系无关的热噪声。`(m², λ, J)` 可作为可学参数,让
prior 自适应温度。

**实现位置.**
- `source/phi4.py` 新增,实现:
  - `logProbability(x) = -E_phi4(x)` (up to log Z 常数)。
  - `sample(n)` 通过 HMC 或 Langevin。注意:**因为只用于 prior
    评估,sample 函数实际上只在 `q.sample()` 路径用,reverse-KL
    可以直接调用 HMC,fwd-KL 则不需要 sample,只需 logProb**。
- 训练:与 I.2 类似,在 latent 端把 prior 替换为 φ⁴ 能量。
  无需归一化常数 log Z(它是常数,对梯度无贡献),但要小心
  reverse-KL 损失的 `E_q[log q − log p]` 里 `log p_target − log p_prior`
  的 log Z 差会显式出现 → 需要 thermodynamic integration 或
  退而求 log Z 比值。**fwd-KL 不受此困扰**(MLE 不需要归一化常数),
  所以方案 B 应该 **先在 fwd-KL/dataDriven 上做**。

**预期 V5.**
- 最激进的修正,**理论上最贴近物理**:prior 本身在 latent 端就
  带 Wilson-Fisher 风味的 IR 行为(取 `m² < 0, λ > 0` 在临界点
  附近)。
- V5 全部三个指标(KS、W1、RMS-G)都应该改善;特别是 rev-KL 的
  RMS-G 0.62 → 应该跌到 0.1 级别(深层 block 不再有动力塌
  identity,塌了违反 prior 的内置相关性)。
- V4 慢模级联应该真正匹配 block-RG 的 std 3.51 → 2.46
  和 kurt → −1.84(因为 prior 端已经 *本来* 双峰,流不必伪造)。

**风险.**
- 工程复杂度高:φ⁴ 的能量梯度计算与训练循环耦合,需要小心
  数值稳定性(`m² < 0` 区域有双井势)。
- reverse-KL 上 log Z 处理不当会使损失偏移失去意义 ⇒ 强烈建议
  只跟 fwd-KL/dataDriven 组合做。
- φ⁴ 不一定 *是* 2D Ising 普适类的精确连续描述(2D 里 φ⁴ 与
  Ising 在 RG 流上是相连的,但不是字面同一);可作为合理近似。

**与现有架构兼容性.** 中 —— 需要训练循环改造,但 MERA 本体可
保留。这是 *物理上最严肃* 的方案,值得长期投入。

---

### I.4  粗化 Ising prior(literal RG decimation)

**动机.** "真正的 RG 不动点 prior" —— 让 latent 端直接是一个更小
尺寸的 Ising 分布 `Ising(L/2^depth, T)`。流字面变成 L → L/2^depth
的 RG decimation 映射,物理失配 *从根上* 消除。

**实现.**
- 预先 MCMC 生成 `Ising(L=2 或 4, T_c)` 的数据集 / 评估器。
- `source/ising.py` 已有 `logProbability` 的高斯近似形式;
  可直接接进 prior 端。
- 更干净的实现:**prior 本身是另一个、较小尺寸的 NeuralRG 流**
  (嵌套式)。这接近 MERA 文献里的"层级 RG 网络"思想。

**预期 V5.** 与方案 B 类似,因为本质上 Ising prior 也是
energy-based 的相互作用 prior,但更"字面"。RMS-G 应能压到 0
(因为流不再被强迫破坏长程关联)。

**风险.** 需要 prior 数据集与 prior `logProbability` 的可微实现;
若用嵌套流,训练时间成倍增长。比 I.3 工程量更大但物理上更
"教科书"。

---

### I.5  学到的非高斯 prior

**动机.** 不预设 prior 形式,用一个更小的 autoregressive 模型
或 normalizing flow 学一个 prior。最灵活但最远离物理 first
principles。

**实现.** Prior 是另一个可训练模块,例如 PixelCNN 或更小的 RNVP
栈。`flow.prior.logProbability` 与 `flow.prior.sample` 都通过这个
子模块。

**预期 V5.** 可改善但缺乏物理可解释性。**不推荐先做** —— 在 I.2,
I.3, I.4 之前,先把"结构化 prior"路线推到极限,再考虑 fully
learned。

---

## II. 架构维度

### II.1  可学 kept-fraction

**动机.** 现在 `im2col.getIndeices` 把 dispatch 的"快/慢比"固化为
1/4 kept、3/4 dropped。这是一个写死的几何归纳偏置。让它可调:

- 改 `flow/hierarchy/im2col.py:getIndeices` 加 `keep_fraction` 参数。
- 或者更激进:让每尺度的 stride 不是 2,而是更小(比如 stride √2
  几何上不可实现,但可以用 *channel-wise 拆分* 模拟 2/4 kept)。
- 最简单的可学版本:**每尺度上把 2 个 RNVP block 都在 stride=1 上
  运行**(不降采样),只用尺度间的 RNVP 层数加深 → 等价于 1/4 → 1
  kept fraction。

**预期 V5.** T_c 处 1/2 kept 可能比 1/4 kept 显著好转,因为
Wilson-Fisher 的"慢模"维度其实不止 1/4。非临界相反过来:1/4
kept 够用甚至更好。这本身是一个 *干净的物理 signal*。

**风险.** 改 dispatch 几何会触及整个 MERA 设计的核心;需要
重新跑所有 baseline 才能对比。

---

### II.2  方案 C —— 自相似约束 + 不舍弃

**动机.** 真正的临界 RG 不一定要把变量扔进高斯垃圾桶。如果设计
一种连续映射,**仅做尺度 Rescaling**(不丢弃任何模),并约束
映射前后的系统能量函数 *分布形式不变*(寻找真正的鞍点),而不是
用 KL 散度去逼近一个平凡吸引子 —— 这或许是更原教旨主义的
Wilson RG 路径。

**架构形式.**

```
x_L (L×L 场)
     │
     ▼  scale rescaling map Φ: x_L → x_{L/b}
     │  (b 是 RG scale factor;Φ 是 *不可逆* 的尺度变换 +
     │   重排一种连续映射,保 b² 倍维数降维)
     │
     ▼
x_{L/b}  ← 这就是输出,不再有 latent
```

**关键约束 / 损失.**

```
loss = D( E_θ(x_L) , E_θ'(Φ(x_L)) )
```

其中:
- `E_θ` 是参数化的能量函数(例如局部多项式 ansatz)。
- 训练时 *同时* 学 Φ 与 E_θ;要求 Φ 把 E_θ 的分布形式保持不变
  (但允许 θ → θ',即 RG 流动)。
- 找 `θ = θ'` 的鞍点 ⇒ 真正的 Wilson RG fixed point。

**实现.** 这是 **大改** —— 偏离 NeuralRG 的可逆双射框架,接近
"NN ansatz for RG transformations" 这类工作(参考 Koch-Janusz &
Ringel 2018, Lenggenhager et al. 2020, Hou & Wang 2023)。
不再用 `flow.logProbability`,改为参数化能量 + 蒙特卡洛估计
"distribution preservation"。

**预期.** 若成功,**这是字面意义上的 Wilson RG fixed point 探针**,
而不是"模仿"。但工程量与现有 NeuralRG 代码无法复用,基本是
新项目级别。

**与现有架构兼容性.** 几乎没有。值得作为 **长期方向**(博士后 / 论文),
不作为当前 sweep 的 patch。

---

## III. 损失维度

### III.1  多尺度损失(**强烈推荐先做**)

**动机.** 当前流的对数概率 **只在 latent 端**(prior 看到的最深层)
做对比 —— 中间尺度的 y_s 没有任何直接训练信号。这是为什么
rev-KL 可以在浅层乱拟合然后在深层塌 identity:**深层根本看不到
G(r) 是否被破坏**。

**实现.**
- 在 `flow/hierarchy/template.py` 的 `forward` 里返回中间产物:

```python
def forward_with_intermediates(self, x):
    intermediates = []
    for no in range(len(self.indexI)):
        ...
        if no % 2 == 1:   # 每尺度结束记录一次
            intermediates.append(x.clone())
    return x, forwardLogjac, intermediates
```

- 在 `train/learn.py:learnInterface` 里加跨尺度惩罚:

```python
ys = intermediates                       # [y_0, y_1, ..., y_4]
scale_loss = 0
for s in range(len(ys) - 1):
    a = zscore(ys[s][..., ::2, ::2])
    b = zscore(ys[s+1])
    scale_loss += KS_distance(a, b) + W1_distance(a, b)
loss = main_loss + lambda_scale * scale_loss
```

或者更物理:**让 y_s 的标准化 G(r)/G(0) 接近 r^(-η)**(Onsager
精确 η = 1/4):

```python
G_emp = compute_G(ys[s])
G_theory = r ** (-0.25)
scale_loss += MSE(G_emp / G_emp[0], G_theory)
```
后者直接把临界普适性写进损失。

**预期 V5.** 直接修 RMS-G 0.62 那条 —— rev-KL 没法再"塌深层"
逃跑,因为塌了就违反尺度不变性惩罚。预测 RMS-G 跌到 0.1–0.2,
KS 也跟着改善。

**成本.** 只需要改 `learn.py` 和 `template.py`(暴露中间输出),
**1 周以内**。无需重训整个 sweep:挑 2 条流(`sym_bignet` 和
`hs_bignet`)做 ablation 就足以判定。

**为什么排第一:**
- 成本最低,leverage 不低于换 prior。
- 直接攻击 V3/V4/V5 报告里 *所有* 病态的同一根源(深层缺少训练
  信号)。
- 是 I 类、II 类大改之前的 **必要前置实验**。如果加了多尺度损失
  病态还在,才有充分理由动 prior 或架构;否则架构改了也白改。
- 产出一个可发表的 clean ablation:**有 / 无尺度损失对 V5 RMS-G
  的影响**。

---

### III.2  Block-RG 监督训练(V5-as-loss)

**动机.** V5 已经做了 Wilson–Kadanoff block-RG 真值
(`rg_v5_blockRG_compare.py`)。直接训练流去匹配 V5 输出:

```python
# 预先在 HS 数据上跑 V5: x_s = AvgPool2d(2)^s(x_data)
# 训练时:
for s in range(num_scales):
    loss += lambda_s * KL(q_ys || x_s)   # 每尺度都对齐
```

把 V5 从 **诊断工具** 升级为 **训练监督信号**。深层 block 没法再
塌,因为得在每尺度都跟一个具体的、已知非高斯的分布对齐。

**实现.** 中等成本:V5 cascade 已经写好,需要把 numpy 部分搬到
可微 torch ops,并预计算 block-RG 真值数据集(L=32 的话 8000–
20000 个样本,~GB 级)。

**预期 V5.** 极强的对齐信号 ⇒ V5 RMS-G 应该 *按定义* 压到 0
(因为这是损失直接优化的对象)。但要小心:**这会让 V5 不再是
独立的诊断**(因为它进了损失),所以同时需要保留 V4(forward-
direction probe)作为未污染的诊断。

---

## IV. 论述维度

### IV.1  重新框定(零成本)

**动机.** 如果以上都做不动,把这套架构在 T_c 上的结果重新框为:

> "Gaussian-prior MERA 流在 T_c 上能多大程度伪装 Wilson–Fisher"

这本身是一个有意义的科学问题(也是 LDM 原始论文实际隐含的问题)。
把"RG fixed point"字眼从所有 T_c 章节里拿掉,改成 "approximate
Gaussian-FP attempt at T_c"。然后专注 T = 2.15、T = 2.40 的非临界
结果作为 *干净的成功案例*。

**实现.** 仅改 `rg_fixed_point_report.md` 和 `_zh.md` 的小节标题
与"acceptable claims"段。无代码改动。

**何时考虑.** 如果 III.1 做了发现确实是架构根本性问题,
short-term paper 应该走 IV.1 这条;长期再上 I.3 / II.2。

---

## V. 探针 / 解读维度

### V.1  Gauge-fixed Layer-by-Layer Interpretation

**动机.** Phase-1 V1/V2 探针使用 *per-sample zscore*(减均值/除标准差)再
算 MSE。zscore **只移除一阶 + 二阶矩**,留下高阶矩(skewness、kurtosis、
多模态)的差异。临界点的 RG 不动点应该 *允许* marginal 在不同尺度上不
同(反常维数 η = 1/4 意味着每个尺度的 marginal *确实* 不同);所以当
前 V1/V2 把 "marginal 形变" 和 "结构变化" 混在一起报告。

**做法(后训练分析,不改训练).**

1. **训练正常完成**(目前优先 Forward KL)。
2. **冻结参数,网络作为生成器运行**:输入 `z ~ N(0, I)`,大量采样,
   经验估计每一层中间潜变量 `y_s` 的真实分布(或反向:正向 HS 数据,
   收集每尺度中间场 —— 二者效果等价)。
3. **对每个层、每个 site 估出经验边缘 `P_s^{(i,j)}`**;拟合一个 per-site
   的 **1D Spline Flow**(rational-quadratic spline 或更简单的
   piecewise-linear quantile transform),把每个 site 的 marginal 强制
   映射到 `N(0, 1)`:

   ```
   T_s :  y_s^{(i,j)}  ↦  z_s^{(i,j)} ~ N(0, 1)
   ```

   这是 *per-site* 一维双射,可微 + 可逆。
4. **在层间插入恒等映射 `T_s^{-1} ∘ T_s`**:这是数学恒等,**网络的实际
   计算不变**;但它在每层的"gauge-fixed slot" 拉出一个 N(0,1)-marginal
   的视角,允许 *跨层比较*。

   网络在新规范下的等效作用:
   ```
   L'_{s+1} ≡ T_{s+1} ∘ L_{s+1} ∘ T_s^{-1}
   ```
   两端的 marginal 都是 N(0,1) ⇒ 任何 L'_s 与 L'_{s+1} 之间的差异都是
   *结构*(joint dependence / copula)差异,不是 marginal-shape 差异。

**预期对现有 V3 报告的影响.**

V3 报告里 `f_3 ↔ f_4` 的判读 *现在是开放问号* —— V1 MSE = 1.92 既可能
是 "两层做不同非平凡工作"(不是不动点),也可能是 "两层结构相同但
marginal 不同"(临界 η-反常下的真不动点)。**Gauge-fixed V1 直接区分**:

| 情景                          | Gauge-fixed V1 MSE | 物理解读                |
|-------------------------------|--------------------|------------------------|
| 两层结构相同,marginal 不同   | **≈ 0**            | ✓ **真 RG 不动点**     |
| 两层做不同非平凡结构          | ~原 V1 值           | 不是不动点             |
| 两层都 ≈ identity             | ≈ 0(不变)          | rev-KL 塌缩(V3 已区分)|

V3 identity residual + Gauge-fixed V1 MSE **联用** 给出完备的 2 × 2 判别:

|                               | V3 大,gauge-V1 小 | V3 大,gauge-V1 大 | V3 小,gauge-V1 小 |
|-------------------------------|-------------------|-------------------|-------------------|
| 物理含义                      | ✓ **真不动点**     | 不同非平凡变换     | rev-KL 塌缩       |

**实现位置(已就绪).**
- `analyzers/rg_fixed_point/gauge_fix.py` —— per-site quantile transform
  + 收集器 + adjacent-scale MSE 对照
- `shell/gauge_fix_demo.sh` —— Slurm 包装(CPU 1h wall;~ 5 min 单 folder)

**预期 V5.** 用 Gauge-fixed V1/V2 数据(以及把同样 transform 套在 V4/V5
的 zscore step 上,**得到 gauge-fixed KS / RMS-G**)能区分:
- I.2 cond. prior 真改善的是 *结构*(应在 gauge-fixed RMS-G 上同样改善)
- I.1 Student-t 在 marginal 上的 "改善" 应在 gauge-fixed 下 *消失*(只是
  marginal-shape 差异),确认 V5 RMS-G 劣化是真"破坏结构"
- 任何疑似 RG 不动点的层对(hs_bignet `f_3 ↔ f_4`、I.2 cond. 深层)都
  可以从 V3-large + gauge-V1-small 模式真正确认

**与现有架构兼容性.** 完全兼容 —— 纯后训练分析,不改 prior、不改
loss、不改架构。任何 19 个 Phase-1 改进 folder 都可以直接 demo。
唯一前置:训练循环必须暴露 `forward_with_intermediates`(已在
`flow/hierarchy/template.py` 实现,Phase-1 III.1 已经用过)。

---

## 推荐执行顺序

按 **cost-leverage** 排序的实验路径:

### Phase 1 —— 1 周内可完成

0. **V.1 Gauge-fixed Layer-by-Layer**(P2.0 同期立刻可做,与 I.2 并行)
   - 在 19 个 Phase-1 改进 folder 上跑 `shell/gauge_fix_demo.sh`(每个
     ~5 min CPU,顺序跑 ~2h 内完事;并行更快)
   - **优先 demo**:`hs_bignet` 已经在跑(job `40162202`),验证 V1 gauge-
     ratio 是否真在 fwd-KL baseline 上呈现"η-anomalous = 1/4"的预期模式
   - **后续**:把 transform 套进 `rg_fixed_point_robustness.py` 加
     `--gauge-fix` flag,把 gauge-fixed V1/V2 写进 CSV;`improvements_results`
     表格里加新列"gauge-fixed adjacent MSE"

1. **III.1 多尺度损失**(必做,先做这个)
   - 选 `sym_bignet` 与 `hs_bignet` 两条 baseline,各加 `lambda_scale`
     扫 {0, 0.1, 1.0, 10.0} 共 8 个 run。
   - 跑完跑一次 V5 看 RMS-G 与 KS 的改善方向。
   - **判定:** 若 RMS-G 显著下降(< 0.3),说明深层信号缺失是
     主因 ⇒ 继续 Phase 2 的 prior 改造;若不下降,说明问题更深
     ⇒ 直接跳到 Phase 3。
2. **I.1 Student-t prior**(并行,作为否定实验)
   - 用 df = 4 跑一条 `hs_bignet` 复制实验。
   - **判定:** 若 V5 KS 改善但 RMS-G 不动,确认"边缘 vs 空间结构"
     是两个独立的轴,prior 只能影响边缘。

### Phase 2 —— 2–4 周

3. **I.2 方案 A 条件高斯 prior**
   - 设计 `P(z_fast | z_slow) = N(μ(z_slow), σ²(z_slow))`,
     `μ, σ` 用 1–2 层 CNN。
   - 用 fwd-KL/dataDriven 训练(避开 log Z 问题)。
   - 与 Phase 1 结果对照:若 RMS-G 进一步改善,说明 prior 端
     的硬性独立性是另一条独立的 bug。
4. **III.2 V5-as-loss** 作为 Phase 2 的对照实验
   - 验证"理论上 V5 改善的上界"是什么。

### Phase 3 —— 1–3 个月

5. **I.3 方案 B energy-based prior(φ⁴)**
   - 这是 *物理上最严肃* 的方案,值得长期投入。
   - 先在 fwd-KL/dataDriven 上做,reverse-KL 留到 log Z 处理
     方案确定后。
6. **I.4 粗化 Ising prior** 作为 I.3 的"字面版"对照。

### Phase 4 —— 长期(论文 / 博士后级别)

7. **II.2 方案 C 自相似约束**
   - 离开 NeuralRG 现有框架,是一个独立项目。
   - 接近 Koch-Janusz & Ringel 2018 / Lenggenhager 2020 的方向。

### 平行可做

8. **II.1 可学 kept-fraction** 作为 ablation 系列
   (1/4 / 1/2 / 3/4 kept 三组对照),可在 Phase 2 期间并行。

---

## 失败模式预测表

| 方案     | 预测 V5 KS (T_c rev-KL) | 预测 V5 KS (T_c fwd-KL) | 预测 V5 RMS-G | 风险:不变之处          |
|----------|------------------------:|------------------------:|--------------:|-------------------------|
| baseline | 0.32+                   | 0.08                    | 0.62 / 0.04   | —                       |
| I.1 t    | 0.22                    | 0.06                    | 0.55 / 0.04   | 空间结构(RMS-G)       |
| I.2 cond | 0.18                    | 0.05                    | 0.30          | 仍未匹配 block-RG kurt  |
| I.3 EBM  | 0.10                    | 0.04                    | 0.10          | 实现复杂度              |
| I.4 Ising| 0.08                    | 0.04                    | 0.05          | prior 数据 + 可微       |
| II.1 1/2 | 0.20                    | 0.06                    | 0.40          | dispatch 重设计         |
| III.1    | 0.15                    | 0.05                    | 0.20          | scale_loss 调参         |
| III.2    | 0.05                    | 0.04                    | 0.02          | V5 不再独立诊断         |

(预测均为粗估,误差 ±50%。真正的判定靠 Phase 1 结果。)

---

## 相关工作 / 参考思路

- **Koch-Janusz & Ringel, *Nature Phys.* 2018**
  *Mutual information, neural networks and the renormalization group.*
  方案 C 的思想源头。
- **Lenggenhager et al., *Phys. Rev. X* 2020**
  *Optimal renormalization group transformation from information theory.*
  自相似约束的具体实现。
- **Marchand, Wang, Ringel 2024**
  *Wavelet conditional renormalization group.*
  方案 A 的小波版本。
- **Hou & Wang 2023**
  *Renormalization group flow as optimal transport.*
  方案 C 的 optimal-transport 路线。
- **Bachtis et al., *PRR* 2021**
  *Phase transitions in machine learning models.*
  对 normalizing flow 在临界点学习能力的系统讨论。

---

## 见 / 配套

- `rg_fixed_point_report_zh.md` —— 病态诊断与 V1–V5 检查
- `rg_fixed_point_report.md` —— 英文版
- `analyzers/rg_fixed_point/rg_v5_blockRG_compare.py` —— V5 实现
- `analyzers/rg_fixed_point/rg_fixed_point_v4_dataforward.py` —— V4 实现
- `flow/hierarchy/im2col.py` —— dispatch 几何(II.1 / II.2 改这里)
- `source/gaussian.py` —— 当前 prior(I.1–I.5 改这里)
- `train/learn.py:learnInterface` —— 训练循环(III.1 / III.2 改这里)
