# Prior Offload Analysis —— CNN 从 loss 里 *直接* 承担了多少物理

**独立报告(2026-06-26 建立,替代原 V6 CNN-offload probe)。**

> 原 V6(`rg_v6_cnn_offload.py`,job `40757108`)使用 KL_raw / KL_whit 作为
> 定量指标,但 target 用错了。本报告收纳:(1) V6 target 错误的详细批评;
> (2) 从 forward-KL loss *直接* 推导的正确定量框架;(3) 后续正确 probe
> 的实现计划。

---

## 1. 问题背景

Phase-2 winner cell(i2 + nrepeat=2,记 D)相比 baseline 有两个 *看似* 冲突的现象:

1. **KL_qp 改善明显**(L=64:87 → 51,-41%)
2. **V5 raw RMS-G 显著劣化**(D64 s=2 = 0.17 vs baseline 0.05)

早期解读(错的):"D 学到了 *不同的 fixed point*,不是 Wilson,而是 p_HS-adapted attractor"。

正确解读(2026-06-26 提出):**Wilson fixed point 跟 p_HS 本就是同一个**。真正发生的是 **MERA 把 Ising 短程耦合的一部分 *转嫁* 给 CNN-prior(条件高斯),V5 只测 MERA bijector,漏看 CNN 那一块**。V5 raw 偏 Wilson 是"测量偏差",不是"物理偏差"。

⇒ 需要 *直接量化* CNN 在 loss 里实际承担了多少物理,验证这个解读。

---

## 2. 首版尝试(V6)与其方法论问题

### 2.1 V6 定义的指标

`rg_v6_cnn_offload.py` 每 cell 测:

| 指标 | 公式 | 意图 |
|------|------|------|
| `KL_raw` | `KL(z_fast || N(0,1))` 用高斯 gap 近似 | z_fast 距离"平凡先验"多远 |
| `KL_whit` | `KL(z̃_fast || N(0,1))`,`z̃_fast = (z_fast − μ_CNN)/σ_CNN` | 用 CNN 白化后剩多远 |
| `ΔKL = KL_raw − KL_whit` | | CNN 清理掉的 "偏差量" |

V6 表(2026-06-26 数据):

| Cell | mean σ | KL_raw | KL_whit | ΔKL |
|------|-------:|-------:|--------:|----:|
| L=32 D32 | 1.85 | 63.4 | 17.1 | +46.3 |
| L=64 D64 | 2.03 | 74.9 | 16.6 | +58.3 |

结论(当时):"CNN 接住 46-58 nat 的 latent-vs-N(0,I) 偏差,把 D cell 打回 baseline 量级"。

### 2.2 target 错误的批评

**关键问题**:i2 model 的 z_fast target *就不是 N(0,1)*。

- baseline(Gaussian prior)训练目标:z_fast marginal = `N(0, 1) iid` ⇒ KL_raw target 正确
- i2(conditional Gaussian prior)训练目标:z_fast **条件** target 是 `N(μ_CNN(z_slow), σ²_CNN(z_slow))`
  - z_fast 的 **marginal** = 一堆 conditional Gaussian 在 z_slow 上的 mixture
  - **marginal 方差 = E[σ_CNN²]** > 1(因为 σ_CNN 学到 > 1)
  - **本就不是 N(0,1)**,是训练 *故意允许* 的偏差

⇒ **KL_raw 用 N(0,1) 当 target 就错了**。63.4 (D32) 这个数字反映的是 "训练目标里 *允许* 的 marginal 偏离",不是"MERA 学得不好"或"CNN 不到位"。**这个指标应删除**。

**KL_whit 的问题**:target 严格意义上对(若 CNN 是最优的,z̃_fast marginal 应 N(0,1)),但是我的高斯 gap 公式测的是 pooled marginal 距离 —— 只是 *必要不充分* 条件(pooled marginal 不 N(0,1) 说明 CNN 没到位,但 marginal 是 N(0,1) 也不说明 conditional 到位)。而且高斯 gap 只测 mean + variance,漏 skew / kurtosis。

⇒ **KL_whit 不是决定性证据**,充其量是 "high-level marginal check"。

### 2.3 什么保留

V6 里 CNN 强度的三个基础指标 *不依赖* 上面 target 错误,仍然有效:

- `mean(σ_CNN)` — CNN 学到的 z_fast 条件标准差(平均)
- `mean(|log σ_CNN|)` — σ 偏离 1 的量级
- `||μ_CNN||_RMS / ||z_fast||_RMS` — CNN 学 μ 的强度(V6 数据:所有 cell 都 ≈ 0.001-0.005,μ 学不到)

结论仍然成立:**CNN 只学 σ,不学 μ**(物理:局部 susceptibility,不是 mean bias,与 Z2 对称一致)。

---

## 3. 从 loss 直接推导的正确框架

### Step 0. flow 的映射

MERA 是确定性一一映射 T:

```
z = T(x)          (x 是数据,z 是 latent)
z = (z_slow, z_fast)     ← 按空间位置拆两组
```

### Step 1. change-of-variables

flow 的 pushforward 密度:

```
q(x) = p_prior(z) · |det ∂z/∂x|
```

取 log:

```
┌─────────────────────────────────────────────────────┐
│   log q(x) = log p_prior(z) + log|det DT(x)|        │
└─────────────────────────────────────────────────────┘
```

`log|det DT|` 是 MERA 每层 Jacobian log-det 的累积。

### Step 2. i2 model 的 prior 分解

i2 prior 就是 **联合** 分布,按条件概率链式法则拆:

```
p_prior(z_slow, z_fast) = p_slow(z_slow) · p_fast|slow(z_fast | z_slow)
                          └────────┘       └────────────────────────┘
                         iid N(0,1)         CNN 参数化的 conditional
```

**注意**:`q_fast(z_fast) ≠ p_slow(z_slow) · p_fast|slow(z_fast|z_slow)`。RHS
是 *联合* 分布,marginal q_fast 需要积掉 z_slow:

```
q_fast(z_fast) = ∫ p_slow(z_slow) · p_fast|slow(z_fast|z_slow) dz_slow
                = mixture of conditional Gaussians    ← 不是 N(0,1)
```

loss 里 **从来不用 marginal q_fast**,只用联合。

**Slow 部分**(iid 标准正态):

```
                        ┌                              ┐
p_slow(z_slow)  =  ∏    │  N(z_i ; 0, 1)               │
                   i∈slow└                              ┘

log p_slow(z_slow)  =  − Σ    [ (1/2) z_i²  +  (1/2) log(2π) ]
                        i∈slow
```

**Fast 部分**(条件高斯,CNN 参数化):

```
                             ┌                                          ┐
p_fast|slow(z_fast|z_slow) = ∏     │ N( z_i ; μ_i(z_slow), σ_i²(z_slow) )  │
                              i∈fast└                                          ┘

log p_fast|slow(z_fast|z_slow) =
                      ┌                                                     ┐
     − Σ              │ (1/2)·((z_i − μ_i)/σ_i)²  +  log σ_i  +  (1/2)log(2π) │
        i∈fast        └                                                     ┘
```

μ_i, σ_i 是 CNN 在 z_slow 上算的,每个 fast 位置 i 一对。

### Step 3. log q(x) 完整式子

代进 Step 1:

```
log q(x)  =  log p_slow(z_slow)  +  log p_fast|slow(z_fast | z_slow)  +  log|det DT(x)|
              └─── A ───┘        └────────── B ──────────┘         └──── C ────┘
              slow 密度          fast 条件密度                         MERA Jacobian
```

展开(去掉跟 x 无关的 const):

```
log q(x)  =  − Σ    (1/2) z_i²                           ← A
              i∈slow
             − Σ   [ (1/2)·((z_i − μ_i)/σ_i)² + log σ_i ] ← B
              i∈fast
             + log|det DT(x)|                            ← C
             + const
```

### Step 4. forward-KL loss

i2 cell 实际训练目标:

```
L  =  − E_{x ~ p_HS}[ log q(x) ]
```

代入 Step 3(移项调号):

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│         ┌   ▼─L_slow─▼       ▼─────── L_fast ────────▼   ▼─L_Jac─▼  │
│         │                                                            │
│  L = E  │  (1/2) Σ z_i²  +  Σ [ (1/2)·((z_i−μ_i)/σ_i)² + log σ_i ]  │
│         │        i∈slow      i∈fast                                  │
│         │                                                            │
│         │     −  log|det DT(x)|                                      │
│         └                                                          ┘ │
│                                                                      │
│    where z = T(x) = MERA.forward(x),                                 │
│          μ_i = μ_i(z_slow) from CNN,                                 │
│          σ_i = σ_i(z_slow) from CNN.                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

`L = L_slow + L_fast − L_Jac`,三项之和。

### Step 5. 三部分谁被谁的参数推

| 项 | 由哪些参数控制 | 训练梯度往哪里 |
|---|---|---|
| L_slow | MERA 参数(经过 z = T(x)) | 只推 MERA |
| L_fast | MERA(经过 z_i)+ CNN(经过 μ_i, σ_i) | **同时** 推 MERA 和 CNN |
| L_Jac | MERA 参数(logdet 累积) | 只推 MERA |

⇒ **CNN 只被 L_fast 里的 log σ_i 和 1/σ_i² 那块推动**;MERA 三块都动。

**CNN 最优解**(对 σ_i 求导 = 0):

```
                    _____________________________
                   |           2
   σ_i*  =  E [ (z_i − μ_i)   |   z_slow ]         ← 条件标准差
              ↑ MERA 输出的 z_fast 的条件方差
```

⇒ **CNN 学到的 σ_i,就是 z_i 给定 z_slow 时的 *条件标准差***。V6 测到 D64
mean σ_i ≈ 2.03,意思是 "训练完 CNN 学到:z_fast 在 z_slow 条件下,条件标准差
大概是 2.03"。**直接从 L_fast 的最优性读出,不用绕 marginal N(0,1) 或
KL_whit**。

---

## 4. 正确的 offload 定量指标

回到 Step 4 的 L_fast:

```
L_fast  =  Σ [ (1/2)·((z_i − μ_i)/σ_i)²  +  log σ_i ]
          i∈fast
```

**跟 baseline(σ ≡ 1, μ ≡ 0)对比**,CNN 让 L_fast *减少* 的量,就是 CNN
在 loss 里直接接住的物理:

```
ΔL_fast_from_CNN
=  L_fast|baseline  −  L_fast|CNN
=  Σ [(1/2) z_i²]  −  Σ [(1/2)·((z_i−μ_i)/σ_i)² + log σ_i]
   i∈fast           i∈fast
```

分成两部分:

**A. σ 罚项(CNN *引入* 的正项)**

```
+ Σ  log σ_i(z_slow)     — CNN σ > 1 时是正数
  i∈fast
                            V6 已量出的 mean(log σ)
```

**B. 减免的 z² 项 vs 中心化 z² 项(CNN 减掉的负项)**

```
+ Σ  [ (1/2) z_i²  −  (1/2)·(z_i−μ_i)²/σ_i² ]     — CNN μ, σ 到位时正值
  i∈fast
                            期望 = 一半 fast 位置的 pointwise z²
                            减掉 σ 归一化后的 pointwise ((z−μ)/σ)²
```

**合起来:每 sample 的 offload 量**

```
ΔL_fast_per_sample  =  E_data[ L_fast|baseline  −  L_fast|CNN ]
```

这才是"CNN 从 loss 里 *直接接走的物理*",单位是 nat。

### 4.1 用 V6 现有数据的一部分反算

V6 存了 `mean(log σ)` 和 `mean(σ)`,先算 A 项:

| Cell | L_fast (fast site 数) | mean(log σ) | **Σ log σ per sample (nat)** |
|------|:---:|:---:|---:|
| L=32 baseline | 1008 | 0 | 0 |
| L=32 B(i2, nr=1)| 1008 | ~0.23 | **~232** |
| L=32 D32(i2, nr=2)| 1008 | ~0.62 | **~625** |
| L=64 baseline | 4032 | 0 | 0 |
| L=64 B(i2, nr=1)| 4080 | ~0.23 | **~938** |
| L=64 D64(i2, nr=2)| 4032 | ~0.71 | **~2863** |

**⚠ 单独 Σ log σ 是 A 项,不是全部 offload。** 完整 offload 还需要 B 项
(减免的 z² 项),这需要 per-sample 计算 `Σ [(1/2) z_i² − (1/2)·((z_i−μ_i)/σ_i)²]`,
不能只从聚合统计反算。

粗略估计:如果 CNN 学到 σ_i ≈ σ_true,那 B 项 ≈ A 项(高斯的对偶关系),
所以 total offload ≈ 2 × A 项。**D64 total ≈ 5000-6000 nat / sample**(占 loss 总量 7712 的 ~70%)。

需要用新 probe 精确量。

### 4.2 CNN 学 μ = 0 简化 formula

V6 测到 `||μ||_RMS / ||z||_RMS ≈ 0.001-0.005`,所以 μ ≈ 0 在实际 cell 里
成立。设 μ = 0,formula 简化为:

```
L_fast|CNN   ≈  Σ [ (1/2)·z_i²/σ_i²  +  log σ_i ]
                i∈fast

L_fast|baseline = Σ (1/2) z_i²
                   i∈fast

ΔL_fast_from_CNN ≈  Σ [ (1/2) z_i² · (1 − 1/σ_i²)  −  log σ_i ]
                   i∈fast
```

括号里的 `(1/2) z_i² (1 − 1/σ_i²)` 是 "CNN 用 σ_i > 1 把 z² 罚项 *缩小*
的量",减掉 `log σ_i` 罚项。**最优情况(σ_i² = ⟨z_i²|z_slow⟩)时,offload
量正比于 fast site 数 × log(conditional variance)**。

---

## 5. 结论

### 5.1 V6 尝试的贡献 vs 撤销

**保留**(方法论稳):

- CNN 只学 σ,不学 μ(与 Z2 对称一致,物理上对应 local susceptibility)
- mean(σ) 的量化:B ~ 1.26,D ~ 1.85 (L=32) / 2.03 (L=64) —— 单调随 nrepeat 和 L 升
- 定性观察:i2 + nr=2 让 latent std 显著变大(11-12 vs baseline 5-7)

**撤销**(target 用错):

- KL_raw = 63 (D32)、75 (D64) 的具体解读("MERA 严重非高斯")
- KL_whit ≈ baseline 的解读("CNN 打回 baseline 量级")
- 三个"决定性发现"里第 3 条(KL 意义上量化 offload)

### 5.2 正确 offload 定量

**per sample:**

```
ΔL_fast_from_CNN  =  E_data[ L_fast|baseline  −  L_fast|CNN ]
                  =  Σ log σ_i  +  减免的 z²-项
                    i∈fast
```

**per cell 估计(A 项)**:

- L=32 D32 ≈ 625 nat / sample
- L=64 D64 ≈ 2863 nat / sample(占 total loss 的 ~37%)

### 5.3 下一步实现

需要一个新 probe(暂命名 `rg_prior_offload.py`),按 4.1 的 formula 逐 sample 算:

```python
# per sample x:
z = MERA.forward(x)                     # (1, 1, L, L)
mu, log_sigma = CNN(z_slow_only)         # (1, 2, L, L) from prior._mu_logsig(z)

# A 项 per sample (fast 位置总和)
A = (log_sigma * fast_mask).sum()

# B 项 per sample
z_fast = z * fast_mask
z_fast_baseline_sq = 0.5 * (z_fast ** 2).sum()                  # baseline L_fast
z_fast_i2_sq = 0.5 * (((z - mu) / sigma * fast_mask) ** 2).sum() # CNN L_fast
B = z_fast_baseline_sq - z_fast_i2_sq

offload = A + B     # per-sample offload (nat)
```

在 7 cell(A/B/C/D at L=32, A/B/C/D at L=64)跑 N=2000 sample,报告
per-sample offload 均值 + std,画柱状图,直接判断 CNN 分担物理的量级。

---

## 附录:V6 原始 CSV 数据保留

V6 的 CSV 数据仍在 `analyzers/csv/rg_v6_cnn_offload.csv`。**下列列仍可信**:

- `label`, `folder`, `L`, `T`, `epoch`, `cond_prior`
- `cnn_mu_rms_over_z_rms` — CNN μ 强度(V6 有效发现:接近 0)
- `mean_sigma`, `std_sigma`, `mean_abs_log_sigma` — CNN σ 强度
- `z_fast_rms` — MERA 输出的 fast 部分实际 RMS

**下列列应弃**:

- `ks_raw`, `w1_raw`, `kl_gauss_raw` — target 用错(N(0,1) 不是 z_fast 的 target)
- `ks_whit`, `w1_whit`, `kl_gauss_whit` — 高斯 gap + marginal 层面,不严格
- `ks_improvement`, `w1_improvement`, `kl_improvement` — 上面两列的 diff,继承同样问题

新 probe 建议直接换 CSV(`analyzers/csv/rg_prior_offload.csv`),不复用旧文件。
