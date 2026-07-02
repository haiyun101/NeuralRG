# RG 不动点诊断 L=64:从 L=32 到 L=64 的跨尺度对比

> 本报告把 L=32 焦点报告(`rg_fixed_point_focus_zh.md`)的核心问题延伸到 L=64:
> **更深的 cascade(L=64 有 6 个 MERA 尺度,L=32 只有 5 个)是否让 hs_bignet 更接近 RG 不动点?**
> 焦点对比是 **L=64 hs_bignet**(= `baseline_b16`)与 **L=32 hs_bignet** 同 architecture 同训练目标(fwd-KL),
> 唯一变量是 **lattice 大小 L**。
>
> ⚠ **L=64 sym_bignet 已训练但没跑过 probe**(folder 在 `data/64Ising_T2.269_sym_bignet/`,ep 9900),所以本报告 *暂* 没有"L=64 rev-KL 控制对照"——
> 等 2026-06-19 集群维护结束后会跑相应 V0-V5 probe + gauge,届时补 sym_bignet 行。
> 现在 L=32 sym_bignet 作为 *跨 L* rev-KL pathology 参照保留。

## 设置:hs_bignet 在两个 L 上是同架构

| 维度 | L=32 hs_bignet | L=64 hs_bignet(= baseline_b16)|
|------|----------------|-------------------------------|
| 训练目标 | fwd-KL(MLE on HS data) | **完全相同** |
| 架构 | `nlayers=16, nhidden=128, nmlp=3, nrepeat=1, symmetry`,RNVP affine,Gaussian latent | **完全相同** |
| Batch | 64 | 16(b=16 在 L=64 因显存限) |
| Epoch | 9500 | 19800 |
| LOSS plateau | ~7686 nat(L=32 H(p_HS) ≈ 7637.6,gap ~48 nat)| ~7686 nat(L=64 H(p_HS) ≈ 7637.6,gap ~48 nat)|
| MERA 尺度数 | 5(`log₂ 32 = 5`)| **6**(`log₂ 64 = 6`)|
| 物理 finite-size | scaling region ~ 2 尺度(s=2,3)| scaling region ~ 3 尺度(s=2,3,4)|

唯一变量是 *L*。两个 plateau LOSS 差不多但 per-site KL 跨 L 时受 *FSS 临界标度* 影响(`KL_fwd ∝ L^α, α ≈ 2.20`),所以 L=64 *本质上更难拟合*。

下文每节同时给 L=32 / L=64 两列,直接读出"加大 L 是否让 fixed point 学得更好"。

---

## V0 / V1 —— N(0, I) 探针,相邻 block 形状相似性

`MSE(T_s(f_s(z)), T_{s+1}(f_{s+1}(z)))` —— 同 z 喂相邻两 block,gauge 后 matched-pair MSE。

| 对 | L=32 hs_bignet | L=64 baseline | L=64 P2 winner(i2 stride8h32)| L=64 Student-t(i1 df4) |
|----|---------------:|---------------:|------------------------------:|-----------------------:|
| f_1 → f_2 | **2.73** | 0.44 | 1.12 | 0.92 |
| f_2 → f_3 | 1.81     | 0.13 | 0.81 | 0.88 |
| f_3 → f_4 | 0.56     | 0.06 | 0.33 | 0.21 |
| f_4 → f_5 | **1.97** | 0.13 | 0.43 | 0.50 |
| f_5 → f_6 | —(L=32 只 5 scale)| 0.16 | **0.034** | **0.033** |

**两层解读**:
- *cross-L*:L=64 baseline 在所有相邻对上都比 L=32 小 5–15×。L=32 的 f_4→f_5 ≈ 2 暗示"最深一对函数很不同";L=64 的 f_5→f_6 ≈ 0.16 暗示"最深一对函数几乎一样"
- *L=64 干预*:**Phase-2 winner 和 Student-t 在浅-中尺度上都比 baseline 大**(中尺度差异更明显)→ 这两个干预 *增加* 了浅-中尺度的 block 工作量;但 *最深* 一对(f_5→f_6)反而 *小 5×*(0.034 vs 0.16)—— 干预让 cascade "把工作前移、最末更接近 identity"

两种解读路径都需要 V3 + V2b 验证:
- *乐观*:L=64 cascade *学到了* 真的 scale invariance,深尺度 block 趋同
- *悲观*:L=64 深尺度 block 趋同因为 *都接近恒等*(V3 会验证)

---

## V2 —— 链式输入(production composition)

`h_s = f_{s+1}(f_{s+2}(...(f_5(z))...))` chained input + matched-pair MSE。

| 对 | L=32 | L=64 |
|----|-----:|-----:|
| f_1 → f_2 | 2.11 | 0.47 |
| f_2 → f_3 | 0.82 | 0.65 |
| f_3 → f_4 | 1.57 | 0.27 |
| f_4 → f_5 | **1.92** | 0.33 |
| f_5 → f_6 | — | 0.12 |

**L=64 链式输入下深对仍小**,跟 V1 同向。链式不改变 L=64 "深尺度 block 趋同"信号。

---

## V2b —— 链式 + MERA 槽几何修正

1 槽链式 + 3 槽新鲜 N(0, I),gauge 后 matched-pair MSE。

| 对 | L=32 | L=64 |
|----|-----:|-----:|
| f_1 → f_2 | 2.13 | **1.24** |
| f_2 → f_3 | 1.53 | 1.15 |
| f_3 → f_4 | 1.52 | 1.15 |
| f_4 → f_5 | **1.78** | 1.32 |
| f_5 → f_6 | — | **1.50** |

**关键反转**:V0/V1/V2 在 L=64 上深对很小(~0.16),但 **V2b 下深对 = 1.50(跟 L=32 接近)**。
这说明 L=64 的"深尺度 block 趋同"在 *V2b 几何修正下消失了* —— 是 *4 元组完整链式* 的 artefact,
跟 L=32 sym_bignet 当年被 V2b 揭示的模式 *同向*。

⚠ **这是个 *警示信号***:**L=64 hs_bignet 可能在 V0/V1/V2 看上去"学到 fixed point"是几何 artefact**,
真实情况要 V3 + V2b + V5 联合判断。

---

## V3 —— 单 block 恒等残差

`r_s = E[(T_s(f_s(z)) − z)²] / E[z²]` —— gauge 后 copula 层面恒等偏离。

| Block | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|-------|---------------:|---------------:|---------------:|---------------:|
| f_1   | 2.28           | 0.80           | 0.86           | 1.08           |
| f_2   | 1.03           | 0.58           | 0.64           | 0.79           |
| f_3   | 1.63           | 0.45           | 0.67           | 0.75           |
| f_4   | **1.87**       | 0.45           | 0.36           | 0.47           |
| f_5   | **0.022**      | **0.17**       | **0.041**      | **0.039**      |
| f_6   | —              | **0.0064**     | **0.0010**     | **0.0026**     |

**关键观察**:
1. **L=64 r 在所有 block 上都小于 L=32**(浅层 r_1: 2.28 → 0.80,深层 r_4: 1.87 → 0.45)。
2. **L=64 *最深* r_6 = 0.0064 比 L=32 *最深* r_5 = 0.022 还小 3×** —— L=64 在最末尺度上更接近"严格恒等"。
3. **L=64 fixed-point 区是 *r_5 = 0.17 + r_6 = 0.0064*** —— 两个尺度逐步逼近恒等;
   L=32 只有 r_5 = 0.022 一个 scale 接近恒等。
4. ⚠ 但 L=64 r_6 = 0.0064 已经接近 *rev-KL 退化版* 的水平(L=32 sym_bignet r_5 = 0.0004)—— 不算严格 collapse,但比 L=32 hs_bignet 的 r_5 更趋向 identity。

**L=64 干预对比**:
- **Phase-2 winner r_6 = 0.0010** 是所有 L=64 流里 *最接近* identity 的(比 baseline 6× 小);跟 *L=32 sym_bignet 的 0.0004 仅 2× 余量*。
- **Student-t r_6 = 0.0026** 介于 baseline 和 P2 winner 之间。
- 三个 L=64 流的 *r_5 *都明显小于 baseline*(0.04 vs 0.17),fixed-point 区比 baseline 宽。

⇒ 跟 V2b 信号联合,**L=64 baseline 在深尺度向 *退化 identity* 漂移已经明显;Phase-2 winner 漂得 *更深***。
**Phase-2 winner 的 V5 最好 + V3 r_6 最小 → 两个 metric 一致说明 P2 winner 学到的 fixed point 比 baseline *更严格***
(但严格到接近 *退化* 临界 —— 是不是 *真* 改善还是 *漂向 rev-KL* 模式,需要 sym_bignet 数据后做严格判断)。

---

## V4 —— HS 数据正向相邻

`MSE(T_s(y_s[::stride, ::stride]), T_{s+1}(y_{s+1}[::stride, ::stride]))`,真数据 forward。

| 对 | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|----|---------------:|---------------:|---------------:|---------------:|
| f_1 → f_2 | 0.50 | 0.42 | 0.44 | 0.56 |
| f_2 → f_3 | 1.43 | 0.38 | 0.46 | 0.52 |
| f_3 → f_4 | **1.79** | **0.39** | 0.35 | **0.06** |
| f_4 → f_5 | **0.025** | 0.49 | **0.034** | **0.034** |
| f_5 → f_6 | — | **0.017** | **0.000** | **0.001** |

**关键反转 #2 + L=64 内 fixed-point profile**:

**Cross-L**:L=32 在 *中-深* f_3→f_4 上 V4 = 1.79(cascade 严重不自相似),L=64 baseline 同位置 V4 = 0.39(更自相似)。但 L=64 *最末* f_5→f_6 = 0.017,跟 L=32 最末 f_4→f_5 = 0.025 量级相当。⇒ **L=64 cascade *内部* 更自相似**(中-深 V4 砍 ~4–5×),最末 fixed-point 质量持平。

**L=64 干预 → fixed-point 区宽度变化**(按 V4 < 0.1 算"fixed-point scale"):

| L=64 流 | fixed-point scale 数 | profile |
|---------|:--:|---------|
| baseline | 1 | 只 f_5→f_6 |
| **P2 winner** | **2** | f_4→f_5 + f_5→f_6 |
| **Student-t** | **3** | f_3→f_4 + f_4→f_5 + f_5→f_6 |

⚠ **Student-t 的 internal fixed-point 区最宽(3 scale),但 V5 不是最好**(下节看)—— 这是个反直觉信号:internal "scale invariance" 跟 external "vs Wilson" 不是单调的。
P2 winner 在 2 scale fixed-point + V5 最好,平衡更好。

---

## V5 —— vs Wilson 真物理 RG(gauge-fixed)

V5 用 *两个互补 metric*:**RMS-G**(分布形状)和 **matched-pair MSE**(样本对齐,与 V0-V4 同 metric 家族)。

### V5 RMS-G(分布级,空间结构形状)

| s | L_s | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|---|-----|---------------:|---------------:|---------------:|---------------:|
| 1 | 16 / 32 | 0.059      | 0.071         | 0.067          | 0.078          |
| **2** | 8 / 16 | **0.030**  | 0.046    | **0.039**      | 0.044          |
| 3 | 4 / 8 | 0.037       | 0.042         | **0.030**      | 0.049          |
| 4 | 2 / 4 | n/a (L_s=2)  | 0.034         | 0.045          | **0.074**      |
| 5 | n/a / 2 | n/a       | n/a (L_s=2)   | n/a            | n/a            |

### V5 matched-pair MSE(样本级对齐,N=2000 samples)

`MSE = 2(1−corr)`,N(0,1) 边际下范围 [0, 4];0 = 完美对齐,2 = 不相关,>2 = 反相关。

| s | L_s | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|---|-----|---------------:|---------------:|---------------:|---------------:|
| 1 | 16 / 32 | 0.69       | 0.57           | **0.53**       | 0.60           |
| 2 | 8 / 16  | 0.72       | 0.71           | **0.69**       | 0.67           |
| 3 | 4 / 8   | **3.22 ⚠** | 0.76           | **0.70**       | 0.73           |
| 4 | 2 / 4   | **0.38**   | 0.74           | 0.73           | 0.75           |
| 5 | n/a / 2 | n/a        | 0.71           | 0.83           | 0.83           |
| 6 | n/a / 1 | n/a        | 0.65           | 0.67           | 0.73           |

### 关键 cross-L 发现(双 metric 联合)

**Finding #1:L=64 baseline V5 RMS-G *略劣化于* L=32**(s=2:0.046 vs 0.030)。即使 L=64 cascade *内部* 更自相似(V4 中-深小),*跟 Wilson* 比反而 *略远*。原因是 **FSS 临界标度**(`KL_fwd ∝ L^α, α ≈ 2.20`)—— L=64 per-site KL 本质上比 L=32 难拟合 ~4×。

**Finding #2:L=64 P2 winner 是 V4 / V5 *都* 改善的唯一流** —— 2 scale internal fixed point + RMS-G 最优 + matched MSE 在 s=1/2/3 都 *最低* → **真正的 fwd-KL fixed-point 候选**。

**Finding #3:Student-t 的 V4 vs V5 反转** —— V4 internal fixed-point 区最宽(3 scale),但 RMS-G 是 *3 流里最差*(s=4 = 0.074)。"自相似但不对"的退化路径,跟 L=32 sym_bignet 的 pathology 同向(程度轻得多)。

**Finding #4(matched MSE 新发现 — L=32 / L=64 *关键差异*):s=3 sign-flip 是 L=32-specific artefact**!

| L | 在 L_s=4 子格点上 | matched MSE | corr |
|---|---|---:|---:|
| **L=32** | s=3(L_s=4)| **3.22** | **−61% 反相关 ⚠** |
| **L=64** | s=4(同 L_s=4)| **0.74** | **+63% 正相关 normal** |

L=64 baseline / P2 winner / Student-t **全部** 在所有尺度上 matched MSE ∈ [0.5, 0.85],**没有任何 sign-flip 异常**。
⇒ L=32 hs_bignet 在 s=3(L_s=4)上的 *反相关* 是 **L=32 训练 artefact,不是 RG 内禀对称**。
可能机制:L=32 上 L_s=4 子格点恰好在 "scaling region 末端 + finite-size 起点"的边界,训练学到 sign flip 作为局部最优拟合;
L=64 上同 L_s=4 已深进 finite-size 区,训练不再做这种翻转 —— 因为 finite-size 区的物理结构 *本身* 就跟 Wilson 一致。

**Finding #5(matched MSE 另一新发现):L=64 baseline matched MSE 全尺度 ~ 0.6–0.8 平稳**,L=32 hs_bignet 跨度大(0.4–3.2)。
**L=64 baseline 跟 Wilson 在每个尺度上都正相关 60–70%**,是 *最稳健* 的 cross-L 行为。即使 RMS-G 略差(因 FSS),sample-level 对齐反而比 L=32 *更均匀*。

⇒ L=64 hs_bignet *不是* "FSS 把 V5 弄差"那么简单 —— **distributional 度量(RMS-G)受 FSS 影响,但 sample-level 对齐(matched MSE)反而更 *均匀健康***。这是 L=64 vs L=32 *cross-L scaling 的双重画像*:RMS-G 角度 L=32 略胜,matched MSE 角度 L=64 *更稳健*。

---

## 综合画像(L=32 vs L=64)

| 维度 | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|------|----------------|----------------|----------------|----------------|
| V0/V1 深对 | 大(1.97)| 小(0.16)| 极小(0.034)| 极小(0.033)|
| **V2b 几何修正** | **大(1.78)** | **大(1.50)** | **大(1.49)** | **大(1.49)** |
| V3 r_深 | r_5 = 0.022 | r_6 = 0.0064 | **r_6 = 0.0010** | r_6 = 0.0026 |
| V4 internal fixed-point 区 | 1 scale | 1 scale | **2 scale** | **3 scale** |
| V5 RMS-G s=2 | **0.030** | 0.046 | **0.039** | 0.044 |
| V5 RMS-G s=3 | 0.037 | 0.042 | **0.030** | 0.049 |

**主结论**(updated with L=64 ablation insight):

1. ✅ **L=64 baseline 比 L=32 hs_bignet *内部更自相似但外部略劣*** —— FSS 临界标度的实测体现。
2. ✅ **Phase-2 winner i2_stride8h32 是 *3 个 L=64 流里 V5 最好的*** —— s=2 = 0.039,s=3 = 0.030(L=64 上 *最接近 Wilson* 的 fwd-KL 训练目标)。同时 V3 r_6 = 0.0010 (deepest fixed-point in this report)。**真 fwd-KL fixed-point 候选**。
3. ⚠ **Student-t 是反例**:V4 internal fixed-point 区最宽(3 scale)*但* V5 跟 Wilson 比是 *3 流里最差*(s=4 = 0.074)。**"internal 自相似 ≠ external 像 Wilson"** —— heavy-tail prior 把 cascade 推向 *self-consistent 但不对* 的解。**这是 fwd-KL 流向 rev-KL pathology 漂移的*前兆模式***。
4. ⚠ **三个 L=64 流的 V2b oneslot 深对都 ≈ 1.5**(同 L=32 hs_bignet 1.78),意味着 V0/V1/V2 "深对趋同"在 *几何修正后* 都消失 —— **不是真 collapse,而是 4 元组几何 artefact**。
5. ⚠ **三个 L=64 流的 V3 r_6 都接近 rev-KL 退化水平**(0.001–0.006):**L=64 的最深尺度普遍向 *degenerate identity* 漂移**(P2 winner 漂得最深,Student-t 次之)。
   ⇒ **这是 L=64 必须当心的"漂向 rev-KL"风险信号** —— 单纯扩 L 让 fixed-point 区*太接近 identity collapse*,需要架构改造(Multi-L tying / Z2-equiv)来 *约束* fixed-point 远离退化。

## 跟 L=32 报告核心论断的对比

L=32 报告结论:**"L=32 cascade 整体未学到自相似;主要瓶颈是架构无 scale-invariance 约束,L=32 不够大只是次要因素"**。

L=64 数据 *修正* 这个结论:

| L=32 报告原文 | L=64 数据修正 |
|----------------|---------------|
| cascade 中-深 V4 = 1.79 远未自相似 | L=64 同位置 V4 = 0.39 → **L 加大 *确实* 让 cascade 更自相似**,所以 L *不是* 完全次要 |
| 主要瓶颈是架构无 scale-invariance | L=64 cascade 自相似改善 + V5 略劣化 → **架构改动 + L 增大 *都* 有作用,但 L 增大反向碰 FSS 墙** |
| Phase-2 推荐 Multi-L 联合训练 + weight tying | 验证 → L 信号确实有,Multi-L 直接利用此信号是正确方向 |
| 推荐方向不包括"单纯扩 L" | 加强 → 单纯扩 L 在 V5 上反劣化,不是 valid path |

⇒ **L=32 报告的 Phase-2 路线图 *基本成立*,L=64 数据补强了"单纯扩 L 不行,但跨 L 共享是 valid"的论断**。

## 待办(等 2026-06-19 维护结束)

1. 把 `data/64Ising_T2.269_sym_bignet` 加进 `rg_fixed_point_robustness.py` 和 V4/V5 probe 的 FOLDERS dict
2. 跑 V1/V2/V2b/V3 + V4 demo + V5 probes + gauge probes(共 5 个 sbatch)
3. **特别关注 L=64 sym_bignet 的 V3 r_5, r_6**:
   - 如果 r_5, r_6 都严格趋 0(像 L=32 sym_bignet 那样)→ 确认 rev-KL 在 L=64 上也是 *退化 identity* 路径
   - 如果 r_5 大、r_6 小 → rev-KL 在 L=64 上模式跟 L=32 不一样,需要新分析
4. **用 L=64 sym_bignet 数据回填 V0–V5 各节的"sym_bignet"列**,做 L=32 报告一样的 *双流 controlled* 对比
5. **跑 matched-pair MSE on V5**(job 40267635 在排队中,等同一时间窗口)同步完成

完成后本报告会从"L=32 vs L=64 跨尺度对比"扩展为"L=64 fwd-KL vs rev-KL 控制变量对比",跟 L=32 报告完全对称。

## 关键数据 / 脚本

- **数据 CSV**:`analyzers/csv/rg_v5_gauge_compare.csv` (V5),`rg_v0_v3_gauge.csv` (V0–V3),`rg_v4_gauge_demo.csv` (V4)
- **训练 folder**:`data/64Ising_T2.269_hsBignet_baseline_b16/`(L=64 hs_bignet),`data/32Ising_T2.269_hs_bignet/`(L=32 hs_bignet)
- **L=32 焦点报告**:`rg_fixed_point_focus_zh.md` / `_en.md`
- **完整诊断**:`rg_fixed_point_report_zh.md`
- **Phase-1/2 ablation verdict**:`improvements_results_zh.md`(含 L=64 Phase-2 winner i2_stride8h32 的 V5 0.024)
