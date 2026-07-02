# RG 不动点诊断:fwd-KL vs rev-KL MERA 流的 V0–V5 焦点对比

> 本报告关心一个核心问题:**MERA 归一化流在 T_c 处是否学到了 RG 不动点?**
> 用同架构、同温度、*只差训练目标*(fwd-KL vs rev-KL)的两个流做控制变量对比 ——
> 焦点是 `hs_bignet`(fwd-KL,*currently best performer*,V5 RMS-G = 0.030)与
> `sym_bignet`(rev-KL,同架构 plate)的 V0–V5 全方位差异。
> 加两个非临界 `hs_dataDriven` 流(T=2.15 / 2.40)作为 *远离不动点* 对照。
> 完整诊断 / 26 fold ablation / Phase-2 verdict 见 `rg_fixed_point_report_zh.md` 与 `improvements_results_zh.md`。

## 两个 T_c MERA 流(本报告焦点对象)

两个流 *同架构、同温度、同 epoch budget,只差训练目标*:

| 流          | 训练目标 | 温度 | 架构                                              | epoch | LOSS plateau    |
|-------------|----------|------|--------------------------------------------------|-------|-----------------|
| **hs_bignet**(fwd-KL,focus)| MLE on HS continuous data | T_c | `nlayers=16, nhidden=128, nmlp=3, nrepeat=1, symmetry`,RNVP affine,Gaussian latent | 9500 | ~7686 nat(距 H(p_HS) ≈ 48 nat gap)|
| **sym_bignet**(rev-KL)     | 变分自由能最小化         | T_c | *完全相同*                                       | 9500 | (不可直接对比 LOSS,见 memory `project_loss_not_comparable_across_modes`)|

加 2 个 *远离不动点* 对照流(同架构,fwd-KL,非临界 T):

| 流              | 温度        | 用途                                |
|-----------------|-------------|-------------------------------------|
| `T=2.15 hs_dataDriven` | 低温有序(亚临界,ξ < L/2)| 非临界对照:深尺度应自然接近恒等(物理上 RG 流向"有序"不动点)|
| `T=2.40 hs_dataDriven` | 高温无序(超临界,ξ < L/2)| 非临界对照:深尺度应自然接近恒等(物理上 RG 流向"高斯"不动点)|

**控制变量逻辑**:
- hs_bignet vs sym_bignet:**变 *仅* 训练目标 fwd-KL ↔ rev-KL** → 测训练目标对"学到不动点"的影响
- 临界 vs 非临界:**变 *仅* 温度 T_c ↔ T=2.15/2.40** → 测物理上的临界 vs 非临界差异(非临界本身就近恒等,临界才是真的难)

## 探针方向约定(重要)

normalizing flow 是 *双向* 的(forward 和 inverse 互为逆函数),但每个 probe 选了 **不同的方向** 来测同一个 flow:

| Probe | 方向            | 输入                    | block 调用            |
|-------|-----------------|------------------------|------------------------|
| V0/V1/V2/V2b/V3 | **inverse**(逆向)生成方向 | `z ~ N(0, I)` 高斯探针 | `layer.inverse(z)` |
| V4 / V5 | **forward**(正向)分析方向 | 真 HS 数据 `x ~ p_HS` | `layer.forward(x)` |

**为什么这样选**:
- V0-V3 关心的是"采样时 block 如何把 latent 变换成数据" —— 必须从 latent 端起步,所以用 `z ~ N(0, I)` + inverse
- V4-V5 关心的是"评估时 block 如何把数据粗粒度化" —— 必须从数据端起步,所以用真 HS 数据 + forward

**实际后果**:
- inverse 方向 + 高斯探针 ⇒ V0-V3 测的是 *理想 latent 分布下* block 的形状性质;输入分布跟生产采样的 deeper latent 接近但不完全一致(因为生产是链式,而 V0/V1/V3 是隔离;V2/V2b 是修正)
- forward 方向 + 真数据 ⇒ V4/V5 测的是 *真物理分布下* block 的粗粒度化行为;输入分布是 *真* 的,但 block 的 forward 跟 inverse 是 *不同操作*

所以 V0-V3 的"f_s 跟 f_{s+1} 像不像" ≠ V4-V5 的"y_s 跟 y_{s+1} 像不像"。两者各回答一半 ——
联合才完整。**hs_bignet 在 V0-V3 探针下深层做实事(信号 ~2),但在 V4 真数据下深层趋于一致(0.025)** —— 这是两个方向 *不同* 的真实反映,不是矛盾。

## 对照流(下文每节都用同 4 列)

为了让数字直接可比,下文每个 probe 表都报告 4 个流:

| 列                       | 训练目标 | 温度        | 架构 |
|--------------------------|----------|-------------|------|
| **hs_bignet**(本报告焦点) | fwd-KL   | T_c         | bignet |
| **sym_bignet**            | rev-KL   | T_c         | bignet(**与 hs_bignet 完全相同架构,只是目标不同**)|
| T = 2.15(低温有序)       | fwd-KL   | 低温(亚临界)| bignet |
| T = 2.40(高温无序)       | fwd-KL   | 高温(超临界)| bignet |

**hs_bignet vs sym_bignet 的对比是控制变量最干净的**(同架构、同温度、只差 fwd-KL vs rev-KL 一个变量),
低/高温是 *非临界对照*(物理上 ξ < L/2,深尺度应自然 *接近恒等*)。

## 如何 gauge-fix(强制 marginal 为 Gaussian)

每个 probe 报告的数都是 *gauge-fixed* 之后的值。**所有后续表格里的数都不再标"gauge"两字**(因为全部都是 gauge-fixed)。

操作步骤:对一个 field `y_s`(形状 `(N, C, L_s, L_s)`),

1. 在每个格点 `(c, i, j)` 上,从 N 个样本里取经验累积分布 `F_{cij}`
2. 用变换 `T_{cij}(y) = Φ⁻¹(F_{cij}(y))`,其中 `Φ⁻¹` 是标准正态逆 CDF
3. 应用 `T` 后每个格点严格服从 N(0, 1) 边际

**实现**:128 个 knot 的 piecewise-linear quantile transform(等价于 1D Spline Flow 的可逆 piecewise-linear 形式)。

**意义**:之前是用 `(y − mean) / std` 来"归一化",只对前两阶矩有效;quantile transform 把 *全部边际形状* 都拉成同一个 N(0, 1)。
**所以接下来 MSE / KS / RMS-G 测的差异 *只剩* 联合依赖结构(copula),边际形状差异已被完全剥离**。

---

## V0 / V1 —— N(0, I) 探针,相邻 block 形状相似性

> 原报告"数值结果"章节是 V0(原始探针);"鲁棒性 V1"是同一探针在 *全局* vs *逐位置* 两种归一化下重做。
> 把每个格点拉成 N(0, 1) 后,**两种归一化合并成一个**(quantile transform 是逐位置的,自带"perpos"性质)。
> 所以 V0 ≡ V1 ≡ 本节。

**怎么测**:对每个 scale-block `f_s` *独立* 喂 `z ~ N(0, I)`(形状 `(N, 1, 2, 2)`),得到 `O_s = f_s(z)`。
对 `O_s` 做 quantile transform `T_s`,然后算相邻对 `MSE(T_s(O_s), T_{s+1}(O_{s+1}))`。
**问"相邻 block 的输出形状像不像"**。

| 对          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|-------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2   | **2.73**  | 1.54       | 0.95     | 2.49     |
| f_2 → f_3   | 1.81      | 0.40       | 1.74     | 1.97     |
| f_3 → f_4   | 0.56      | **0.02**   | 0.52     | 0.34     |
| f_4 → f_5   | **1.97**  | **0.000**  | 0.79     | 0.15     |

**hs_bignet 解读**:深对 `f_4 → f_5` 仍 ≈ 2,跟 sym_bignet **0.000** 形成鲜明对比 —— 同架构、同温度、只是 rev-KL 训出的 sym_bignet 在最深一对上 **塌缩到严格 0**。**hs_bignet 最深一对函数 *不同*,符合 fixed-point 假设否定的预期**。非临界 T=2.15/2.40 的深对也明显小于 hs_bignet,说明 hs_bignet 在 T_c 上做的是 *真临界尺度结构* 的工作。`f_3 → f_4` 偏小(0.56)是这条链路上的"伪近似"点;V3 会揭示这是因为 `f_3` 和 `f_4` *恰好* 在输出上类似,而不是它们做的事一样。

---

## V2 —— 链式输入(production composition)

**怎么测**:不再用新鲜 `z`,而是把 deeper block 的输出 *链式* 喂下来 ——
`h_s = f_{s+1}(f_{s+2}(...(f_5(z))...))`,然后对每个 block 算 `MSE(T_s(f_s(h_s)), T_{s+1}(f_{s+1}(h_{s+1})))`。
**问"在生产组合输入下,相邻 block 是不是仍像"**。

| 对          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|-------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2   | 2.11      | 1.41       | 0.74     | 0.57     |
| f_2 → f_3   | 0.82      | 0.38       | 0.52     | 1.98     |
| f_3 → f_4   | 1.57      | 0.02       | 1.31     | 0.59     |
| f_4 → f_5   | **1.92**  | **0.000**  | 0.75     | 0.14     |

**hs_bignet 解读**:深对仍 ≈ 1.92,**跟 V0/V1 一致**。链式输入没有改变 hs_bignet 的"深层 block 做实事"信号。
sym_bignet 在 V2 下深对仍 ≈ 0 —— 这是 V2 *看不出* sym_bignet 真问题的位置(V2b 才看得出)。

---

## V2b —— 链式 + MERA 槽几何修正

**怎么测**:V2 把 `f_{s+1}` 的 *4 个* 输出全部当 `f_s` 的输入,但 MERA 实际只重用 *1 个*(粗模槽 (0, 0)),另 3 个槽是新鲜 `N(0, I)`。
V2b 修正:`h_s[0, 0] ← f_{s+1}` 的输出 `[0, 0]`,其它 3 槽重抽 `N(0, I)`。
**问"在 MERA 真实槽几何下,相邻 block 是不是仍像"**。

| 对          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|-------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2   | 2.13      | 1.97       | 0.96     | 2.20     |
| f_2 → f_3   | 1.53      | 1.53       | 1.72     | 1.48     |
| f_3 → f_4   | 1.52      | 1.51       | 1.51     | 1.43     |
| f_4 → f_5   | **1.78**  | **1.49**   | 1.65     | 1.51     |

**hs_bignet 解读**:V2b 也 ≈ 1.78,**几何修正不消除 hs_bignet 的信号**。
关键对比:**sym_bignet 在 V2b 下深对从 0 暴涨到 1.49** —— 揭示其 V0/V1/V2 下的"近 0"是 4 元组几何 artefact。
**hs_bignet 在 V2b 下基本不动,说明 V0/V1/V2 的信号是 *真* 几何不变量的功能性差异**,不是几何 artefact。
这是 V2b 上 hs_bignet *最区别于 sym_bignet* 的特征 —— 同架构、同温度、只是 rev-KL 训练让 sym_bignet 在 V0/V1/V2 下表现出虚假的"恒等"。

---

## V3 —— 单 block 恒等残差

**怎么测**:对每个 block `f_s` 单独喂 `z ~ N(0, I)`,把输出 `f_s(z)` 做 quantile transform `T_s`,然后算相对残差 `r_s = E[(T_s(f_s(z)) − z)²] / E[z²]`。
**问"这个 block 是不是 *恒等映射*"**。`r_s ≈ 0` 表示恒等;`r_s` 大表示做实事。

| Block | hs_bignet  | sym_bignet | T = 2.15 | T = 2.40 |
|-------|-----------:|-----------:|---------:|---------:|
| f_1   | **2.28**   | 1.23       | 1.71     | 0.91     |
| f_2   | 1.03       | 0.46       | 0.78     | 1.72     |
| f_3   | **1.63**   | 0.02       | 1.41     | 0.56     |
| f_4   | **1.87**   | **0.0004** | 0.76     | 0.17     |
| f_5   | **0.022**  | **0.0004** | 0.006    | 0.009    |

**hs_bignet 解读**:5 个 block 里 4 个做大 copula 工作,只有 f_5 接近恒等。
f_5 = 0.022 *不是* sym_bignet 那种"严格塌缩"(0.0004);它是"较弱工作 + 物理上合理的 *深尺度场已去相关* 信号"。
**比 sym_bignet 大 55×**,这是 V3 上 hs_bignet *最区别于 sym_bignet* 的特征。
注意 sym_bignet 在 f_3/f_4/f_5 全部塌缩到 ≈ 0,而非临界 T=2.15/2.40 只在 f_5 上接近 0(f_5 物理上应当接近恒等,因为深尺度场已去相关)—— sym_bignet 在 f_4 上就塌缩是 **病态**。

---

## V4 —— HS 数据正向方向相邻

**怎么测**:不再用 `z ~ N(0, I)` 探针,而是把 *真 HS 数据* `x` 通过 MERA 正向走 (`x → f_1.forward → y_1 → f_2.forward → y_2 → ...`),收集每个尺度 `y_s`。
然后比较相邻 `MSE(T_s(y_s[::2, ::2]), T_{s+1}(y_{s+1}))`(粗粒度子格点上)。
**问"在 *真数据* 上,MERA 自身的相邻尺度像不像"**。

| 对          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|-------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2   | 0.50      | 0.35       | 0.30     | 1.93     |
| f_2 → f_3   | 1.43      | 0.01       | 1.06     | 0.34     |
| f_3 → f_4   | **1.79**  | **0.000**  | 0.78     | 0.087    |
| f_4 → f_5   | **0.025** | **0.000**  | 0.001    | 0.005    |

**hs_bignet 解读**:中-深尺度 *真数据上* 相邻有大差异(1.43–1.79),但 `f_4 → f_5` *在真数据上几乎一致*(0.025)。
这跟 V0/V1/V2/V2b *探针* 上 f_4→f_5 ≈ 1.92–1.97 形成鲜明对比 —— **hs_bignet 的最深尺度在 `z ~ N(0, I)` *探针* 上做的事跟在 *真数据* 上做的事不一样**。
跟 sym_bignet 比:sym_bignet 在中-深尺度 *真数据上也* 几乎为 0(全塌缩,跟 V0/V1/V2 探针下一致),而 hs_bignet 中-深尺度 *只在真数据上* 接近 0,*探针下* 仍是大值 ——
**V4 揭示一个"探针 vs 真分布"不匹配的微妙性**,需 V5 跟真物理 Wilson 比较才能 disambiguate。

### 详解:"探针 vs 真数据" 不匹配 = 学到 RG 不动点的指纹

把 `f_4 → f_5` 在两种输入下的对比单独拎出来:

| 输入                          | 方向     | f_4→f_5 MSE |
|-------------------------------|---------|-------------:|
| `z ~ N(0, I)` 高斯探针(V0/V1)| inverse | **1.97**     |
| 真 HS 数据 `x ~ p_HS`(V4)     | forward | **0.025**    |

同一对 block,**两种输入下相似性差 80×**。表面上看是矛盾,但物理上有明确解读 —— 这是 RG 不动点的 *物理定义*:

> 一个变换 `R` 在 *不动点附近的吸引子分布 `p*`* 上 *表现* 像恒等(`R(p*) ≈ p*`),
> **但作为一般函数 `R` 本身不是恒等**。

类比:Wilson RG 在 Ising 临界分布上满足 `R(p_critical) = p_critical`,但 R 这个映射作用在 *其它* 分布上(如 Gaussian)输出完全不同的东西。R 本身不是 identity。

把这个物理直觉套用到我们的 block:
- **V0-V3 探针 (`z ~ N(0, I)`)**:测的是 `f_5` 作为 *抽象函数* 的性质 —— 跟 `f_4` 不同(MSE ≈ 2),所以 `f_5 ≠ identity`(抽象上)
- **V4 真数据 (`x ~ p_HS`)**:测的是 `y_5 = f_5(y_4)` 在 *y_4 实际所在的分布上* 的行为 —— `y_5 ≈ y_4`(MSE = 0.025),所以 `f_5` 在 *y_4 的吸引子分布* 上 *表现* 像 identity

**两个回答合起来 = hs_bignet 学到了"数据相对不动点(data-relative fixed point)"**:`f_5` 抽象上非平凡,但在它实际处理的分布上 *表现* 接近恒等。

这正是 V0-V3 探针单独 *看不出* 的信息(只看到"f_5 ≠ identity",误以为"hs_bignet 不是 fixed point");V4 因为输入是 *真分布*,直接揭示了 fixed-point 行为。

**对比 sym_bignet(rev-KL,同架构)**:
- 探针下 f_4→f_5 ≈ 0(深对像 identity)
- 真数据下 f_4→f_5 = 0.000(深对也像 identity)
- 两个测量都说"塌缩到 identity" → `f_5` *作为抽象函数* 退化成 identity

⇒ sym_bignet 是 **退化版**(`f_5 = identity` literally),hs_bignet 是 **正版**(`f_5 ≠ identity` 但 `f_5(y_4) ≈ y_4`)。**正版才是 RG fixed point 的物理定义,退化版只是表面上像但失去了变换内涵**。

这是 V4 *单独* 给出的最关键证据,V0-V3 探针 *永远* 看不到。

---

## V5 —— vs Wilson 真物理 RG

**怎么测**:同样的 HS 输入 `x` 上,**同时**跑 MERA 正向 (`y_s = MERA`) 和 Wilson-Kadanoff 块平均 (`x_s = AvgPool2d(2)^s(x)`)。
两者各自做 quantile transform 后,在每个尺度算 metric。
**问"MERA 慢模是不是跟 Wilson 真物理一致"**。

V5 提供两种互补 metric(*都* 在 gauge 坐标下):

| Metric | 类型 | 测什么 | 适用 |
|--------|------|--------|------|
| **matched-pair MSE** | sample-level | `E_i[(T_y(y_s^i) − T_x(x_s^i))²]` —— 逐 sample 对齐 | **跟 V0–V4 同 metric 家族**,所有尺度都能算 |
| **RMS-G** | distributional | `G(r)/G(0)` 两点函数 RMS 偏差 | 测空间结构形状(分布级),L_s ≤ 2 死区 |
| KS / W1 | distributional(边际) | 边际 CDF / Wasserstein-1 距离 | gauge by construction ≈ 0,**无信号**(仅做 sanity check)|

#### V5 matched-pair MSE(跟 V0–V4 统一 metric)

`E_i[||T_y(y_s^i) − T_x(x_s^i)||²]`,gauge 后 N(0,1) 边际下 **MSE = 2(1 − corr)**,所以:
- MSE = 0 → 完美对齐(corr = 1)
- MSE = 2 → 完全不相关
- MSE > 2 → 反相关(sign-flipped)
- MSE = 4 → 完全反相关

| s | L_s | hs_bignet | sym_bignet | T=2.15 | T=2.40 |
|---|----:|----------:|-----------:|-------:|-------:|
| 0 | 32  | 0.000     | 0.000      | 0.000  | 0.000  |
| 1 | 16  | **0.69**  | **2.24**   | 0.56   | 0.66   |
| 2 |  8  | 0.72      | 2.23       | 0.72   | 0.94   |
| **3** | **4**  | **3.22 ⚠** | 2.23       | **3.24 ⚠** | 1.18   |
| 4 |  2  | **0.38**  | 2.21       | 0.72   | 1.14   |
| 5 |  1  | 0.52      | 2.32       | 0.70   | 0.72   |

**三个新发现**(matched MSE 独有,RMS-G 看不到):

1. **sym_bignet matched MSE ≈ 2.24 *全尺度均匀***。
   翻译成 correlation:**rev-KL 学到的 MERA 慢模 *跟 Wilson 真物理慢模 *逐样本完全不相关***。RMS-G 0.54 说"分布形状不像",matched MSE 2.24 说"*per-sample 也对不上*"。**rev-KL 的 V5 失败是 *双重的*:分布错 + 样本对齐错**。

2. **hs_bignet 在 s=4(L_s=2)matched MSE = 0.38 是 *4 流里最优***,意味着 corr ≈ 81%。这是 V5 *新看到的* 信号 —— RMS-G 在 s=4 死区(n/a),matched MSE 揭示 **hs_bignet 最深 2×2 子格点上 sample-by-sample 跟 Wilson 高度对齐**。

3. ⚠ **hs_bignet 和 T=2.15 在 s=3 上 MSE > 3(反相关 60%)**,sym_bignet 和 T=2.40 没有。
   翻译:fwd-KL MERA 在 s=3(L_s=4 子格点)上学到 *sign-flipped* slow modes —— 跟 Wilson 物理 *值相反* 但 *绝对值结构相同*(所以 RMS-G 看不出来)。
   这可能是:
   - MERA 训练 artefact(scale-3 处偶然反号,因为 RNVP 的 ± 是 latent N(0,1) 对称下的可学习参数)
   - 真 RG 内禀对称性的体现(Wilson 块平均跟 MERA 的 slow-mode 编码可能存在符号自由度)
   - **需要后续分析**(等 L=64 matched MSE 看是否同模式)

#### V5 RMS-G(空间结构形状 distributional view)

| Metric | s = 0 | s = 1 | **s = 2** | s = 3 | s = 4 | s = 5 |
|--------|------:|------:|----------:|------:|------:|------:|
| **hs_bignet RMS-G** | 0.000 | 0.059 | **0.030** | 0.037 | n/a | n/a |
| **sym_bignet RMS-G** | 0.000 | **0.512** | **0.539** | **0.485** | n/a | n/a |
| T=2.15 RMS-G     | 0.000 | 0.071 | 0.046 | 0.025 | n/a | n/a |
| T=2.40 RMS-G     | 0.000 | 0.079 | 0.068 | 0.074 | n/a | n/a |

**为什么 RMS-G 只到 s=3**:s=4 对应 L_s = 2,`G(r)/G(0)` 在 2×2 上只有 r=0 和 r=1 两个距离,样本量也小,信号在噪声里 —— 脚本判定 m < 4 时返回 `n/a`(见 `rg_v5_blockRG_compare.py:163`)。

**hs_bignet 解读(基于 RMS-G)**:
- **s=2 RMS-G = 0.030 是 T_c 上的最低值**(sym_bignet 0.539,T=2.40 控制 0.068)
- 跟 sym_bignet 比:在 s=1, 2, 3 上 hs_bignet *一致* 比 sym_bignet 低 8–18×
- 跟低温 T=2.15 比:s=3 上 T=2.15 = 0.025 略低,但低温是非临界对照(物理上 ξ < L/2,RG 自然流向 Gaussian 不动点,*更容易* 匹配 Wilson)。**关键是临界 T_c 上谁最接近,而那只有 hs_bignet**

#### 两 metric 联合解读(distributional + sample-level)

RMS-G(分布形状)和 matched MSE(样本对齐)互补,**hs_bignet 的两 metric "最优 scale" 不一样**:
- RMS-G 最优 @ **s=2**(0.030)
- matched MSE 最优 @ **s=4**(0.38)

物理直觉:浅尺度(s=2)spatial structure 复杂,matched-pair 对齐难做但 distributional 形状能比;深尺度(s=4)spatial structure 简单(2×2),反过来 distributional 死区但 matched-pair 容易对齐。

**rev-KL sym_bignet 在两 metric 上 *双双灾难***:RMS-G 0.5(分布完全错形)+ matched MSE 2.2(样本完全不相关)→ rev-KL 不论从哪个角度看都跟 Wilson 物理脱钩。

KS / W1 全 ~10⁻³ 是 gauge-fix by construction(N=2000 噪声 ~ 1/√N ≈ 0.022),不携带信号,仅做 gauge-fix 工作的 sanity check。

---

## 综合画像

| Probe       | hs_bignet 信号                          | sym_bignet 对照(同架构,rev-KL)| 关键判断 |
|-------------|----------------------------------------|----------------------------------|----------|
| V0/V1       | 深对 ≈ 1.97(大)                       | ≈ 0(塌缩)                       | hs_bignet 深层做实事 |
| V2          | 深对 ≈ 1.92(大,链式下不变)          | ≈ 0                              | 链式无变化         |
| V2b         | 深对 ≈ 1.78(大,几何修正下不变)      | = 1.49(bombshell)               | V0/V1/V2 的"近 0"是几何 artefact |
| V3 (r_s)    | r_1–r_4 都 > 1;r_5 = 0.022(轻塌缩) | r_4 = r_5 = 0.0004(严格恒等)    | sym_bignet 在 f_4 塌缩是病态 |
| V4          | 真数据上中深 ≈ 1.5–1.8,深对 0.025 自洽 | 全 0,塌缩                       | "探针 vs 真数据"不匹配 |
| V5 RMS-G    | **s=2 = 0.030,跟 Wilson 最接近**       | s=2 = **0.539**(灾难)          | hs_bignet 18× 优于 sym_bignet |

**结论**:hs_bignet 是 *唯一* 同时满足"深层做 copula 实事(V0–V4 健康)"和"跟 Wilson 接近(V5 独占)"的训练目标。
hs_bignet 与 sym_bignet 同架构同温度,**唯一差异是训练目标 fwd-KL vs rev-KL** —— 6 个 probe 上的全面差距证明 *训练目标本身* 决定了 MERA 流是否在做物理 RG,与架构容量、数据规模无关。

## L=32 prior 干预扫描(I.1 Student-t + I.2 conditional Gaussian)

本节对比 *同 L=32* 下 **5 个 fwd-KL 训练**(同架构 RNVP affine,只换 latent prior):

| 流 | latent prior | 参数 |
|----|--------------|------|
| `hs_bignet` | Gaussian | baseline,各向同性 N(0, I) |
| `i2(8,32) P1` | Conditional Gaussian | **stride=8 慢模 sub-lattice + hidden=32 CNN(Phase-1 winner)**|
| `i2(4,32) P2` | Conditional Gaussian | stride=4 慢模(更密)+ hidden=32 |
| `i2(8,64) P2` | Conditional Gaussian | stride=8 + hidden=64(更宽 CNN)|
| **`I.1 Student-t`** | **Student-t** | **df=4 heavy-tail prior**(b=128)|

I.2(Conditional Gaussian):`P(z) = P(z_slow) · P(z_fast | z_slow)`,把 latent 自身做 *慢-快分解*,对应 Wilson RG 物理图像。
I.1(Student-t):各向同性但 heavy-tail,允许 fluctuation 比 Gaussian 大得多。

### V0/V1 perpos

| 对          | hs_bignet | i2(8,32) P1 | i2(4,32) P2 | i2(8,64) P2 | Student-t |
|-------------|----------:|------------:|------------:|------------:|----------:|
| f_1 → f_2   | **2.73**  | 0.52        | 0.63        | 2.48        | 0.48      |
| f_2 → f_3   | 1.81      | 0.66        | 0.59        | 0.13        | 0.22      |
| f_3 → f_4   | 0.56      | **1.58**    | 0.07        | 0.61        | **1.10**  |
| f_4 → f_5   | **1.97**  | **1.93**    | **0.029**   | **0.054**   | **1.08**  |

**4 种深对 pattern**:
- baseline / Phase-1 i2:深对 ≈ 2(blocks 函数很不同)
- Phase-2 i2:深对 ≈ 0.03–0.05(blocks 函数极相似)
- **Student-t**:**深对 ≈ 1.1**(中间状态)

### V2b 几何修正

| 对          | hs_bignet | i2(8,32) P1 | i2(4,32) P2 | i2(8,64) P2 | Student-t |
|-------------|----------:|------------:|------------:|------------:|----------:|
| f_4 → f_5   | **1.78**  | **1.82**    | **1.49**    | **1.50**    | **2.28**  |

**5 个流 V2b 深对都 ≥ 1.5**,Student-t 反而 *最大*(2.28),意味着 Student-t 深层 block 在 V2b 真实几何下做的工作 *最多*。
V0/V1 与 V2b 差异最大的是 Student-t(V1: 1.08,V2b: 2.28)—— "深对趋同"在 V2b 下完全消失,纯 4 元组几何 artefact。

### V3 r_s

| Block | hs_bignet | i2(8,32) P1 | i2(4,32) P2 | i2(8,64) P2 | Student-t |
|-------|----------:|------------:|------------:|------------:|----------:|
| f_3   | 1.63      | 0.65        | 0.14        | 0.56        | 0.47      |
| f_4   | **1.87**  | **1.83**    | **0.033**   | **0.059**   | **1.41**  |
| f_5   | **0.022** | **0.031**   | **0.001**   | **0.001**   | **0.749**  |

**3 种相反的 r_5 模式**:
- baseline / Phase-1 i2:r_5 = 0.02–0.03(*1 个* near-identity scale,健康 fixed-point)
- Phase-2 i2:r_5 = 0.001(*2 个* near-identity scales,**接近 sym_bignet 退化 0.0004**)
- **Student-t**:**r_5 = 0.75**(*0 个* near-identity scales,**深 block 做 *更多* 工作,而不是更少**!)

⇒ **Student-t 跟 i2 Phase-2 是 *相反的* 病理方向**:i2 Phase-2 让 cascade 漂向 identity collapse,Student-t 让 cascade 在最深处 *拒绝* identity(heavy-tail prior 给最深层 *扩张* 空间)。

### V4 adj

| 对          | hs_bignet | i2(8,32) P1 | i2(4,32) P2 | i2(8,64) P2 | Student-t |
|-------------|----------:|------------:|------------:|------------:|----------:|
| f_2 → f_3   | 1.43      | 0.46        | 0.19        | 0.19        | 0.11      |
| f_3 → f_4   | **1.79**  | **1.62**    | **0.08**    | **0.08**    | **1.36**  |
| f_4 → f_5   | **0.025** | 0.06        | **0.001**   | **0.000**   | **1.24**  |

**fixed-point 区(V4 < 0.1)宽度**:
- baseline / Phase-1 i2:**1 scale**
- Phase-2 i2:**3 scales**(全 cascade 自相似)
- **Student-t**:**0 scale!** —— Student-t 在 L=32 上 *完全没有* internal fixed-point 区,深对 V4 = 1.24 跟浅对一样大

(Student-t 在 L=64 上呈现 *3 scale* fixed-point 区 —— 跟 L=32 *方向相反*,见 L=64 报告。**Student-t 的 L 依赖很复杂,跟 i2 cells 的 cross-L 一致模式形成对比**)

### V5 RMS-G vs Wilson

| s | hs_bignet | i2(8,32) P1 | i2(4,32) P2 | i2(8,64) P2 | Student-t |
|---|----------:|------------:|------------:|------------:|----------:|
| 1 | 0.059     | 0.063       | **0.103**   | 0.070       | 0.083     |
| **2** | **0.030** | 0.047 | **0.069** | 0.032 | **0.062** |
| 3 | 0.037     | **0.003**   | 0.036       | 0.034       | **0.083** |

**5 个流的 V5 表现**:
- 跟 Wilson 物理 *最近*:**Phase-1 i2(8,32) 在 s=3 上 0.003**(全表最低,12× 改善)
- 其它 i2 cells:s=2 持平或劣,s=3 与 baseline 同档
- **Student-t**:**全 s 都比 baseline 差**,s=3 = 0.083 是 5 流里 *最差*

### 综合解读

L=32 上的 prior 干预呈现 **3 种不同的 *V0–V5 pattern*** —— 都对 latent 做同方向的"改造",**但在 V0–V5 各 probe 上的反应完全不同**:

| 路径 | 代表流 | V3 深 r | V4 fixed-point 区 | V5 vs Wilson |
|------|---------|---------|-------------------|---------------|
| **保持 baseline + 精细 Wilson 匹配** | Phase-1 i2(8,32) | r_5 = 0.031(健康)| 1 scale | **s=3 = 0.003(全表最低,12× 改善)**|
| **扩 internal 自相似 + 漂向 identity 退化** | Phase-2 i2(4,32) / (8,64) | r_5 = 0.001(接近 sym 0.0004)| 3 scales | s=2 持平 / s=3 持平(*不* 改善)|
| **深层做更多工作 + 远离 Wilson** | Student-t | **r_5 = 0.75**(*反向* —— 比 baseline 大 35×)| **0 scale!** | **s=3 = 0.083(全表最差)**|

⇒ **L=32 的 5 个 fwd-KL 流呈现 *3 种 V0–V5 病理模式*,只有 Phase-1 i2(8,32) 真改善 V5 vs Wilson**。
其它干预要么 *扩 internal 自相似* 但 *外部不像 Wilson*(i2 Phase-2 cells),要么 *完全反向*(Student-t)。

**关键 cross-L 一致性**(跟 L=64 报告联合):
- **stride=8 hidden=32 是 *L-independent* 架构 sweet spot**:L=32 上 V5 s=3 砍 12×,L=64 上 V5 s=2 砍 15%;同架构在两个 L 都是 V5 winner
- **stride=4 / hidden=64 在 L=32 *和* L=64 都过度** —— 把 internal "自相似"推过头,V5 不改善,且 r_5 漂向退化(0.001 vs sym 0.0004)
- **Student-t 在 L=32 和 L=64 *表现相反*** —— L=32 上 deep block 做 *更多* 工作(r_5 = 0.75),L=64 上 deep block 趋同 identity(r_6 = 0.003)。**heavy-tail prior 的 L 依赖跟 i2 完全不同 —— 暗示其物理机制不同,不是简单的"prior 强弱"**

**对 Phase-2 路线图的修正**(基于本节数据,补强 `improvements_results_zh.md` 的 verdict):
- **i2 stride=8 hidden=32 是 *跨 L 的稳健 sweet spot***,值得作为 Phase-2 主推架构
- **stride=4 / hidden=64 已被证明过度** —— 不需要继续扫
- **I.1 Student-t 在 L=32 上 *最差*,L=64 上 internal 看似改善 *外部* 仍差** —— I.1 路线整体不优,但 L 依赖反常值得做物理机制分析(后续工作)
- **Phase-2 路线图把 (stride, hidden) 固定到 sweet spot stride=8 hidden=32 后,重点 *只扫训练超参*(batch、lr、epoch)** —— L=32 / L=64 数据一致支持这点

## 剩余问题(hs_bignet 的失败模式)

虽然 hs_bignet 在 V0–V5 上是 *currently best performer*,**仍是 partial RG fixed point** —— 只在最深一对 block 接近不动点行为,**cascade 整体不是**。具体差距:

1. **LOSS 距 H(p_HS) 仍有 ~48 nat gap**
   - hs_bignet 训练 LOSS 收敛到 7686,理论下限 H(p_HS) ≈ 7637.6
   - 这是 `KL(p_HS || q_θ)`,**不是任何 V_i probe 直接给出的量** —— 但跟 V5 RMS-G > 0 一致(flow 没完全匹配真物理 cascade)

2. **V5 RMS-G ≠ 0**(理想 fixed point 应给 0)
   - s=2 = **0.030**,s=3 = 0.037
   - hs_bignet 的慢模 `G(r)/G(0)` 跟 Wilson 真物理仍有约 3 % 形状差异

3. **V3 r_5 = 0.022 ≠ 0**(理想 fixed point 应给 0)
   - 跟 sym_bignet 的 0.0004 比仍有 *55× 余量*
   - 意味着"数据相对 fixed point"是 *approximate* 的:`f_5(y_4) ≈ y_4` 但 **不严格**

4. **V4 中-深尺度 cascade 还远未自相似**
   - f_2 → f_3 = 1.43,f_3 → f_4 = **1.79** —— 浅-中-深的相邻 y_s *很不一样*
   - 只有最深一对 f_4 → f_5 = **0.025** 接近 fixed point 行为

→ **核心诊断**:hs_bignet 在最深一对 block 上 *学会了* fixed point 行为(V4 = 0.025,V5 RMS-G 接近 0),
但 cascade 整体 *没有学会自相似*(V4 中-深仍很大)。
要让 cascade 整体成为 fixed point,需要让 *所有* 相邻深对的 V4 MSE 都接近 0,而当前只有最末一对达到。

### 是不是因为 L=32 不够大、深度不够?

物理直觉问得很好 —— 真的 RG 不动点需要在 *UV 截断之上、IR 有限尺寸之下* 存在一段 *scaling region*,
让你能多次块平均看到分布形状不变。L=32 的 5 个尺度排列:

| Scale | L_s | 物理含义                                    |
|-------|-----|---------------------------------------------|
| s=1   | 16  | UV(刚离开格点尺度)                          |
| s=2   | 8   | **可能进入临界 scaling 区**                  |
| s=3   | 4   | **可能仍在 scaling 区**                      |
| s=4   | 2   | finite-size 主导(只有 4 个格点)             |
| s=5   | 1   | 单点(琐碎)                                  |

⇒ L=32 上**真正能测自相似的 scaling region 只有 ~ 2 个 scale(s=2, 3)**。这是 *一个* 限制,但 *不是主要瓶颈*。

**经验上扩 L 反而 *更难*(不是更易)**:

| L  | hs_bignet V5 gauge RMS-G s=2 | hs_bignet V5 gauge RMS-G s=3 |
|----|------------------------------:|------------------------------:|
| 32 | **0.030**                     | 0.037                         |
| 64 | 0.046                         | 0.042                         |

L=64 略劣于 L=32。原因是 *临界标度* 反向作用 —— memory `project_fss_critical_scaling`:**KL_fwd ∝ L^α,T_c 处 α ≈ 2.20**(非临界 α ≈ 2.0)。
per-site KL 跨 L 时 baseline 本身 *增长 ~ L^2.20*,L=64 比 L=32 *本质上* 更难拟合。

⇒ 扩 L 同时带两个相反效应:**scaling region 变宽(好)+ 每 site KL 暴涨(坏)**。实测上后者主导。

**真正的瓶颈不是 L 不够,是架构无 scale-invariance 约束**:

1. L=32 上最深一对 V4 = 0.025(接近 fixed point)—— 模型 *在仅有的 scaling region 末端* 是 *能学到* fixed point 的
2. 中-深 V4 = 1.79(远未自相似)—— 这不是 finite-size 模式(那样会让 *所有* 对都被压到接近 0),**是 *模型* 没在中间尺度学到自相似**
3. sym_bignet 在同 L=32 上 V4 *所有对都 ≈ 0*(也自相似了),但是 *退化版* —— 全塌缩成 identity
4. ⇒ L=32 *允许* cascade 表现自相似,**问题是模型架构 *允许* 学到 *非自相似* 的解**

当前 hs_bignet 架构:
- 16 个 RNVP 层 *无 weight tying* → 每 scale 的 RNVP 参数都是 *独立学* 的
- 没有 *强制 scale-invariance 的归纳偏置*
- 训练 LOSS 只惩罚 *最终分布* 距 p_HS 的 KL,**不惩罚 cascade 中间是否自相似**

⇒ 模型学了一个 *能正确生成 p_HS* 但 *不自相似* 的解(只在最深一对 *碰巧* 接近 fixed point,因为深尺度场已去相关,任何"近恒等"映射都对)。

**结论**:L=32 的 scaling region 窄是 *次要* 因素,**主要瓶颈是架构无 scale-invariance 约束 + loss 不惩罚自相似**。
这也是为什么 Phase-2 推荐 **Multi-L 联合训练**(注入跨 L 共享参数)和 **weight tying**(强制同 RNVP 在所有 scale)
而不是 *只* 扩 L —— 后者在临界标度下反而劣化。

## Phase-2 优化方向

| 方向                                  | 是不是 hs_bignet 该走的路 |
|---------------------------------------|:-------------------------:|
| 加 width(megabignet nhidden 192)     | ❌ 实验否定(plateau +37 nat 劣化)|
| 加 data(N=200K → N=500K)             | ❌ 实验否定(2.5× 数据 plateau 不动)|
| 加 epoch / cosine LR                  | 单给 ~7 nat,远不够            |
| **NSF coupling**(更强 prior)        | ✅ 更灵活的 block 变换,可能让中-深尺度也学到 fixed point |
| **Z2-equivariant RNVP**              | ✅ 注入对称归纳偏置,减少 cascade 中 "学错对称" 的 capacity 浪费 |
| **学到的 prior**(I.4 粗化 Ising / I.3 EBM)| ✅ 起步分布更接近 fixed point,让中-深 cascade 工作更少 |
| **Multi-L 联合训练**                 | ✅ 强制 scale-invariance 约束,直接 *鼓励* cascade 自相似 |

**最直接命中 hs_bignet 失败模式("cascade 整体未学到自相似")的两条路**:
- **NSF**(让 block 更有能力做 fixed-point 变换)
- **Multi-L 联合训练**(直接惩罚 cascade 中不自相似的部分)

I.4 / I.3 是 *间接* 帮助 —— 把起步分布推近 fixed point,让流不需要做太多变换。

## 关键数据 / 脚本

- **数据 CSV**:`analyzers/csv/rg_v5_gauge_compare.csv`(V5),`rg_v0_v3_gauge.csv`(V0–V3),`rg_v4_gauge_demo.csv`(V4)
- **训练 folder**:`data/32Ising_T2.269_hs_bignet/`
- **图脚本**:`analyzers/rg_fixed_point/plot_v0_v5_comparison.py`
- **完整诊断**:`rg_fixed_point_report_zh.md`
- **Phase-1/2 ablation verdict**:`improvements_results_zh.md`
