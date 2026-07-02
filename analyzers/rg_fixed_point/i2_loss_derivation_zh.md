# i2 Conditional Gaussian Prior 的 forward-KL Loss 完整推导

**2026-07-01 建立。**

自包含笔记,从 change-of-variables 一路推到 loss 的每一项,含 CNN 最优解、
数值示例、跟 Gaussian baseline 的差异。

> 与 `prior_offload_analysis_zh.md` §3 相同的框架,但这里专注 *单-CNN* 的
> i2(不是 hierarchical HCG),完整所有细节,含每一步的假设、单位、shape。

---

## 0. 记号约定

| 符号 | 含义 | shape |
|------|------|-------|
| L | lattice linear size | scalar |
| x | 数据(HS field)| `(B, 1, L, L)` |
| z | latent field | `(B, 1, L, L)` |
| T | MERA bijector,`z = T(x)` | 映射 |
| slow_stride, S | slow 网格间距 | scalar |
| slow_mask | 1 at slow sites, 0 elsewhere | `(1, 1, L, L)` |
| fast_mask = 1 − slow_mask | | `(1, 1, L, L)` |
| N_slow = (L/S)² | slow site 数 | scalar |
| N_fast = L² − N_slow | fast site 数 | scalar |
| μ_i, σ_i | CNN 给的 mean、std at fast site i | scalar |

## 1. Flow 的 change-of-variables

MERA 是可逆确定映射 `z = T(x)`,反过来 `x = T⁻¹(z)`。**pushforward density**:

```
q(x)  =  p_prior(z) · |det ∂z/∂x|
       =  p_prior(T(x)) · |det DT(x)|
```

取 log:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    log q(x) = log p_prior(z) + log |det DT(x)|           │
│                       ↑ z = T(x)          ↑              │
│              prior score            MERA Jacobian log-det│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**重要**:

- MERA 是 *体积保持* 类的耦合流(RNVP),`log|det DT|` 每一 RNVP block 都有贡献
- 分类:baseline 跟 i2 只在 `p_prior(z)` 上不同,MERA 那块 (log|det DT|) 完全同

## 2. i2 model 的 p_prior(z)

**Prior 的联合分布** 按条件概率链式法则拆:

```
p_prior(z)  =  p_prior(z_slow, z_fast)
             =  p_slow(z_slow) · p_fast|slow(z_fast | z_slow)
```

**⚠ 注意 marginal 跟 joint 的区别**:

```
q_fast(z_fast) = ∫ p_slow(z_slow) · p_fast|slow(z_fast | z_slow) dz_slow    ← marginal
                ↑                                                              (需要积分)
                严格上是 mixture of conditional Gaussians,不是 Gaussian
```

**loss 里 *从不* 用 q_fast marginal**,只用 joint。这是 KL_raw / KL_whit 那类
"target = N(0,1) for z_fast marginal"指标错在哪里 —— 详见
`prior_offload_analysis_zh.md` §2.2。

### 2.1 slow 部分

```
p_slow(z_slow) = ∏  N(z_i; 0, 1)     — iid 标准正态
                i∈slow

log p_slow(z_slow) = − Σ  [ (1/2) z_i²  +  (1/2) log(2π) ]
                     i∈slow

                    = − (1/2)  Σ  z_i²   −  N_slow · (1/2) log(2π)
                             i∈slow
```

`− Σ (1/2) log(2π)` 是常数,训练时忽略。

### 2.2 fast 部分(CNN 参数化)

```
p_fast|slow(z_fast | z_slow) = ∏   N(z_i; μ_i(z_slow), σ_i²(z_slow))
                              i∈fast

                                                          ↑
                              μ_i, σ_i 是 CNN(z_slow) 在 site i 处的输出
```

CNN(`source/conditional_gaussian.py:50-56`)3-layer conv net:

```
z_slow (mask 掉 fast 位置置 0) → Conv(3×3) → ELU → Conv(3×3) → ELU → Conv(3×3) → 2 channels
                                                                          ↓         ↓
                                                                          μ         log_σ
```

log_σ 被 clamp 到 [−5, 5],防 1/σ 爆:`sigma = exp(log_sigma)`。

单 site 的 log 密度:

```
log N(z_i; μ_i, σ_i²)  =  −(1/2)((z_i − μ_i)/σ_i)²  −  log σ_i  −  (1/2) log(2π)
```

全 fast:

```
log p_fast|slow(z_fast | z_slow) = − Σ  [ (1/2)((z_i − μ_i)/σ_i)²  +  log σ_i ]
                                   i∈fast

                                      −  N_fast · (1/2) log(2π)
```

### 2.3 联合 log 密度

把 2.1 + 2.2 加起来:

```
log p_prior(z) =  − (1/2)  Σ  z_i²                                     ← slow 项
                          i∈slow

                 −  Σ  [ (1/2)((z_i − μ_i)/σ_i)²  +  log σ_i ]         ← fast 项
                   i∈fast

                 −  (N_slow + N_fast) · (1/2) log(2π)
                 └────────── const ──────────┘
```

## 3. 完整 log q(x)

代入 §1:

```
log q(x) =  log p_prior(z)   +   log |det DT(x)|
                z = T(x)
```

展开(去 const):

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  log q(x) =  − (1/2)  Σ  z_i²                                      │
│                       i∈slow                                       │
│                                                                    │
│              −  Σ  [ (1/2)((z_i − μ_i)/σ_i)² + log σ_i ]           │
│                i∈fast                                              │
│                                                                    │
│              +  log |det DT(x)|                                    │
│                                                                    │
│              +  const                                              │
│                                                                    │
│  其中 z = T(x),μ_i = μ_i(z_slow),σ_i = σ_i(z_slow) via CNN         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 4. Forward-KL Loss(i2 cells 实际用)

前 KL 定义:

```
L(θ)  =  KL(p_HS || q_θ)  =  E_{x~p_HS}[log p_HS(x) − log q_θ(x)]

                          =  H(p_HS) − E_{x~p_HS}[log q_θ(x)]
                             └─const─┘

⇒  minimize L(θ) ⟺  minimize  − E_{x~p_HS}[log q_θ(x)]
```

**θ = 所有 MERA 参数 ∪ CNN 参数**,同时优化。

代入 §3:

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  L(θ) = E_{x~p_HS} [                                                  │
│                                                                       │
│           ▼─ L_slow ─▼           ▼──────── L_fast ────────▼           │
│                                                                       │
│        (1/2)  Σ  z_i²   +   Σ  [ (1/2)((z_i − μ_i)/σ_i)² + log σ_i ] │
│              i∈slow        i∈fast                                     │
│                                                                       │
│                     −  log |det DT(x)|                                │
│                        └── L_Jac ──┘                                  │
│                                                                       │
│         ]                                                             │
│                                                                       │
│  where  z = T(x) ∈ R^{L×L},                                          │
│         μ_i, σ_i = CNN_θ(z * slow_mask) at fast site i.               │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

3 项之和:

```
L(θ)  =  L_slow(θ)  +  L_fast(θ)  −  L_Jac(θ)
```

**每项来源和梯度流向**:

| 项 | 谁的贡献 | 梯度往哪 |
|---|---|---|
| L_slow | slow 位置的 z² 项 | 只推 MERA(z 依赖 MERA)|
| L_fast | fast 位置的中心化 + 归一化 + σ 罚项 | **同时推 MERA(z_i 依赖 MERA)+ CNN(μ_i, σ_i 依赖 CNN)** |
| L_Jac | MERA 每 block 的 Jacobian logdet 累积 | 只推 MERA |

**CNN 只被 L_fast 拉动**,通过 μ_i, σ_i 两项。

## 5. 跟 Gaussian Baseline 的差异

**Baseline(Gaussian prior)** 的 loss:

```
L_baseline(θ_MERA) = E_{x~p_HS} [ (1/2) Σ_{i∈全 lattice} z_i²  −  log|det DT(x)| ]
```

**注意 slow ∪ fast = 全 lattice**,所以 baseline 的 (1/2) Σ z_i² 等于 i2 里的:

```
(1/2) Σ_slow z_i² + (1/2) Σ_fast z_i²  ← 都是 baseline 的一部分
        ↑                    ↑
    baseline L_slow    baseline "L_fast" 也是 (1/2) z²
```

**i2 vs baseline 差在 L_fast 那块**:

```
L_fast|i2       =  Σ  [ (1/2)·((z_i − μ_i)/σ_i)²  +  log σ_i ]
                 i∈fast

L_fast|baseline =  Σ  (1/2) z_i²
                 i∈fast
```

**差**:

```
ΔL_fast  ≡  L_fast|baseline − L_fast|i2
         =  Σ  [ (1/2) z_i²  −  (1/2)((z_i − μ_i)/σ_i)²  −  log σ_i ]
           i∈fast
```

这是 **CNN 从 loss 里 *直接接住的* 那部分 nat 数**,即 "CNN offload"
(正确定义,见 `prior_offload_analysis_zh.md` §4)。

**注意符号:训练时 loss L 是 minimize 的**,i2 让 L_fast 变小,则 ΔL_fast > 0
是 CNN 帮忙压 loss 的量。

## 6. CNN 最优解 —— 从 L_fast 的驻点条件求

固定 θ_MERA(以及一批数据 → z 固定分布),问 CNN 的 μ, σ 最优是什么。

**对 σ_i 求偏导**(注意 σ_i 隐式依赖 z_slow,先固定 z_slow):

```
∂L_fast/∂σ_i = E_{z_fast|z_slow} [ − (z_i − μ_i)² / σ_i³  +  1/σ_i ]
              = 0
```

解:

```
              _______________________________
             |                       2
   σ_i*  =  E [ (z_i − μ_i(z_slow))   |   z_slow ]      ← 条件标准差
```

**对 μ_i 求偏导**:

```
∂L_fast/∂μ_i = E_{z_fast|z_slow} [ −(z_i − μ_i) / σ_i² ]
              = 0
```

解:

```
   μ_i*  =  E [ z_i  |  z_slow ]                        ← 条件均值
```

⇒ **CNN 的最优 (μ_i, σ_i) 就是 z_i 给定 z_slow 的条件均值/条件标准差**。

**训练完 CNN 学到的是 z_fast 的 *条件统计量***,不是任何"平凡的先验分布"。

V6 数据(2026-06-26):

| Cell | mean(σ_CNN) | ||μ_CNN||_RMS / ||z||_RMS |
|------|------------:|--------------------------:|
| L=32 B(i2, nr=1)| 1.26 | 0.004 |
| L=32 D32(i2, nr=2)| 1.85 | 0.005 |
| L=64 D64(i2, nr=2)| 2.03 | 0.001 |

⇒ CNN 学到的 σ 显著 > 1,但 μ ≈ 0。物理:2D Ising T_c 的 Z₂ 对称
让条件均值必然为 0,但条件方差 = 局部 susceptibility 是有信息的。

## 7. 数值 walk-through

L=32 slow_stride=8 batch 1:

```
z_slow shape:     4 × 4 = 16 sites          (只保留 slow_mask=1 的位置)
z_fast shape:     32² − 16 = 1008 sites     (剩下的)

CNN 输入:         z * slow_mask   → (1, 1, 32, 32) 但 fast 位置全 0
CNN 输出:         (μ, log_σ)      → (1, 2, 32, 32)
                                    只 fast 位置的值被用于 loss
                                    slow 位置的输出被无视

假设一个样本 z 得到:
  Σ_slow z_i²             = 24         (16 slow 值平方和)
  Σ_fast ((z_i−μ_i)/σ_i)² = 1050        (中心化归一化后的平方和)
  Σ_fast log σ_i          = 620         (D32:mean(log σ)≈0.6 × 1008)
  log|det DT|             = 850         (MERA Jacobian)

L_slow    = (1/2) × 24  = 12
L_fast    = (1/2) × 1050 + 620 = 525 + 620 = 1145
L_Jac     = 850

L per-sample = L_slow + L_fast − L_Jac = 12 + 1145 − 850 = 307
                                                          + 常数 (N/2 log 2π)
```

对比 baseline 同一样本(假设 z 分布同):

```
Σ_all z_i² = Σ_slow + Σ_fast_baseline = 24 + N_fast · Var(z_i)
                                       = 24 + 1008 · 3.4²      (D32 z_fast std ~ 3.4)
                                       ≈ 24 + 11640 = 11664
L_slow_baseline = 12
L_fast_baseline = (1/2) × 11640 = 5820
L_Jac           = 850(不变)

L_baseline per-sample = 12 + 5820 − 850 = 4982
```

⇒ **CNN 让 loss 从 4982 降到 307,差 ~4675 nat**(数量级示意,实际 D32 是 17.7 nat KL_fwd,这里没算 std 归一化项、H(p_HS) 减法)。

## 8. 训练动力学诠释

从 CNN 视角看训练:

1. **Init**:CNN 最后一层零权重零偏置 → μ_i = 0, log σ_i = 0 (即 σ_i = 1) 处处 → i2 loss 完全 = baseline loss
2. **早期**:梯度从 L_fast 通过 μ, σ 流回 CNN。CNN 学到:
   - z_i 的条件均值(基本无信号,因 Z₂ 对称,梯度也小) → μ 保持 ≈ 0
   - z_i 的条件方差(有信号,critical fluctuation) → σ 学到 > 1
3. **收敛**:μ_i ≈ 0,σ_i ≈ σ_i*(条件标准差)
4. **同时**,MERA 因为 CNN 接住 σ,不必把 z 推到 std=1,可以让 z 保留大方差
5. **最终 balance**:MERA 输出 z_fast 的实际 std ≈ CNN 学到的 σ,两边 self-consistent

⇒ 这是 V6 数据观察到的 mean(σ) 单调随 nrepeat 上升(1.26 → 1.85 → 2.03)的机制:
**nrepeat=2 给 MERA 多容量,MERA 更"敢"把 z 留宽,CNN 也随之学到更大 σ**。

## 9. 反 KL(为完整性,i2 没用)

若把 i2 prior 用于 reverse-KL 训练(如 baseline NeuralRG 变分模式):

```
L_reverse(θ) = E_{x~q_θ} [log q_θ(x) − log p_target(x)]
```

**Sample x ~ q_θ** 需要从 prior 采样 z 再过 MERA.inverse:

```
1. Sample z_slow ~ N(0, 1) iid                          (all slow sites)
2. Evaluate μ, σ = CNN(z_slow_masked)   [with torch.no_grad()!]
3. Sample z_fast_i = μ_i + σ_i · ε_i,  ε_i ~ N(0, 1)
4. x = MERA.inverse(z)
```

**⚠ 关键实现细节**(`source/conditional_gaussian.py:131-133`):sample 里 CNN 调用
被 `with torch.no_grad()` 包住,**梯度不流回 CNN 从 sample 那条路径**。梯度只
从 `log q_θ(x)` 那步流回 CNN(那步跟 forward-KL 里一样,CNN 在 score z 时接梯度)。

**这是防止 CNN 用"让自己 sample 出容易 score 的 z"这种 trivial 解的安全阀**。

## 10. 何时该考虑 hierarchical prior(HCG)

i2 的 prior 只在 *一个* stride 上加条件结构。若数据在 *多个* stride 上都有
非平凡条件结构(RG 意义上的临界相关),i2 的单-CNN 只能吸收 *一个* 尺度的
物理,其余尺度的物理仍留给 MERA 消化 —— MERA 变得"忙",CNN "闲"。

**HCG = i2 的 multi-scale 延伸**,把 CNN 复制到每个 stride,让 loss 里
Σ log σ 拆到多个 scale:

```
i2 loss L_fast:            Σ  [ (1/2)((z_i−μ_i)/σ_i)² + log σ_i ]     ← 只 fast at final scale
                          i∈fast

HCG loss L_fast_multi:  Σ    Σ    [ (1/2)((z_i−μ_i^k)/σ_i^k)² + log σ_i^k ]
                        k=1  i∈lvl_k                                    ← 每个 hierarchy level 都有
```

单个 CNN 若 scale-shared,自动 enforce "同一个 conditional-whitening 操作在
所有 scale 上都成立" —— 这就是 RG fixed point 的直接架构表达。详见
`hierarchical_conditional_gaussian.py` 和 `prior_offload_analysis_zh.md` §5.3。

---

## 附录:i2 CNN 参数数量估算

`source/conditional_gaussian.py` 默认 CNN 结构(hidden=32):

```
Conv2d(1, 32, k=3, pad=1)      : 1·32·3·3 + 32   =  320
ELU
Conv2d(32, 32, k=3, pad=1)     : 32·32·3·3 + 32  =  9,248
ELU
Conv2d(32, 2, k=3, pad=1)      : 32·2·3·3 + 2    =  578
                                                   ───────
                                Total CNN params:  10,146
```

跟 MERA(L=32 bignet:~10.9M params)相比,**CNN < 0.1%**。极小的先验模块
承担了 46-58 nat 的 loss 分担(V6 数据)—— **CNN 极度参数高效**。
