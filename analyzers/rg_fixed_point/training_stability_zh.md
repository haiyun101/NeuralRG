# 训练稳定性问题:Spike 与 Resume

**2026-07-04 建立。**

记录 Phase-2 HCG / i2 训练里两个反复出现的稳定性问题:**(1) 训练中期 spike**、
**(2) 从 checkpoint resume 后性能退化**。附上机制分析、诊断信号、以及针对性
缓解措施。

---

## 1. 问题概览

### 1.1 训练中期 Spike

三个 fresh HCG nr=2 cell(2026-07-01 提交,无 gradClip)在训练后段被 spike 击杀:

| Cell | 最好 best-200 (pre-spike) | 死亡时刻 | Post-spike last-200 |
|------|:-:|:-:|:-:|
| E32-shared nr=2 (40861282) | 1910.40 @ ep 16000-17999 | ep 18000+ | 1922.0(退化 12 nat)|
| E32-perscale nr=2 (40861558) | 1913.35 @ ep 12000-13999 | ep 14000+ | 1919.2(退化 6 nat)|
| **E64-shared nr=2 (40861283)** | **7666.43** @ ep 16000-17999 | **ep 18000+** | **37,057(灾难性 → 161 billion nat max)** |

共同 pattern:训练稳定 12,000-18,000 ep,然后 outlier batch 引爆,后续无法恢复
到 pre-spike best。

### 1.2 Resume 后性能退化

3 个 well-converged cell resume 后 LOSS 反而更差:

| Cell | Pre-resume LOSS | Resume best-200 | 差距 |
|------|:-:|:-:|:-:|
| D64 (i2+nr=2) → 40873381 | 7663.60 (ep 4600) | 7677.8 | **+14.2 nat 更差** |
| E32-shared nr=2 → 40949390 | 1910.40 (ep 17800) | 1913.3 | +2.9 nat |
| E64-shared nr=2 → 40949391 | 7666.43 (ep 17800) | 7687.9 | +21.5 nat |

Resume 后即便 gradClip=5.0 保护也 recover 不了 pre-load 状态。

---

## 2. 机制分析

### 2.1 Spike 的三个协同触发因素

**触发因素 A:log_sigma clamp 边界跳跃**

`source/conditional_gaussian.py:89` 和 HCG 里 log_sigma 硬 clamp 在 [-5, 5]:

```python
log_sigma = log_sigma.clamp(min=-5.0, max=5.0)   # σ ∈ [0.0067, 148]
```

- 若 CNN 输出 log_sigma → 5,则 σ = 148,`log σ_i × N_fast` 项瞬间加 5 × 1008 = 5040 nat / sample
- Clamp 是 hard(不 smooth),边界处 gradient = 0 → 参数不能 recover
- **Loss 骤跳 → gradient 骤跳 → 参数踢飞**

**触发因素 B:Adam 自适应 lr 在 outlier 时放大**

Adam 效果 lr = `lr / √v`。正常时 v 稳定,步长可控。**Outlier gradient**:

```
g = 巨大(某 site log_sigma clamp 后引发的 loss spike)
v = 历史平均 g²(相对小,因为之前 g 都正常)
effective step ~ lr · g / √v = 巨大
```

参数一步跳到 landscape 危险区。

**触发因素 C:nr=2 深 backprop 累积**

nr=2 = 20-24 层 RNVP。gradient 从 loss 逐层反向传播:

```
∂L/∂θ_layer_1 = ∂L/∂θ_layer_N · Π_{k=N-1..1} ∂θ_k+1/∂θ_k
                                    ↑ 24 层乘积
```

若某层 gradient outlier,累积到浅层 gradient 更大。

### 2.2 Spike 的时间演化(观测数据)

以 E64-shared nr=2 为例(2000-epoch bin):

```
ep     0-1999  : mean= 7972.7   max= 39121.4  ⚠ 早期不稳(Adam warm-up)
ep  2000-3999  : mean= 7705.5   max=  7823.6  (稳定收敛)
ep  4000-5999  : mean= 7693.4   max=  7778.5
...
ep 14000-15999 : mean= 7672.1   max=  7760.6  (best 区)
ep 16000-17999 : mean= 7669.3   max=  7756.9  (**best-200 = 7666.4**)
ep 18000-19999 : mean= 140,125,793.8   max= 161,522,483,200.0  ⚠ CATASTROPHIC
```

Spike 通常发生 **训练后段**(ep 14K-18K),原因推测:

1. 训练 landscape 逐渐窄化 —— 深 minimum 附近参数敏感度增加
2. Adam v 逐步减小(gradient 变小)→ effective lr 相对放大
3. 数据里 rare batch(HS field outlier)概率累积

### 2.3 Resume 退化的机制

`main.py:229` 和 `train/learn.py:415` 之间的 workflow:

```python
# main.py
if args.load:
    saved = torch.load(name)     # 加载 checkpoint
    fw.load(saved)                # 恢复 model weights only(bug!)

# learn.py
optimizer = torch.optim.Adam(params, lr=lr)   # 新 optimizer, m=v=0
# 前 500 epoch Adam v 从 0 累积到合理值 → "burn-in"
```

**Bug**:`fw.load()` 只 restore weights,**不 restore optimizer state**。

**从 well-converged model resume 的第一步**:

```
θ = pre-load 参数(接近好 minimum)
g = 当前 gradient(小,因已 converge)
v = 0(fresh Adam)
v_new = 0.001 · g²
θ_new − θ = − lr · g / √(0.001 · g²)
          = − lr · sign(g) · √1000
          ≈ − 32 · lr        ← 巨大步长!
```

**32× lr 步长把参数直接踢出好 minimum**,后续 500 ep Adam warm-up 期参数持续
漂移。等 v 学到合理时,参数已远离 pre-load 状态。回不去。

这就是为什么 D64 resume 从 7663.6 变成 7677.8(**加了 gradClip 也没用**),因为
gradClip 是防 outlier 引起的爆炸,不能防 fresh Adam 的 warm-up 步长。

---

## 3. 诊断信号

### 3.1 Spike 早期预警

监控 log 每 500 ep 计算 max/mean:

```python
if losses[-500:].max() > losses[-500:].mean() * 2:
    print("⚠ EARLY WARNING: loss max ≫ mean, possibly leading to spike")
```

**阈值**:max > 2× mean 就应报警。正常训练 max/mean < 1.3。

### 3.2 CNN log_sigma 边界摸测

在 diagnostic script 里:

```python
# 每 1000 ep 追踪 CNN 输出
_, log_sigma = prior._mu_logsig(z)
frac_at_boundary = ((log_sigma.abs() > 4.5).float().mean().item())
print(f"  fraction of log_sigma near clamp boundary: {frac_at_boundary:.4f}")
```

**阈值**:> 0.001 (0.1%) 就表示 CNN 有一些 outputs 摸到边界。> 0.01 (1%) 就要
准备 spike 发生。

### 3.3 Resume 后 Adam 状态

Resume 之后 前 500 ep 的 gradient magnitude 会不寻常:

```python
# 训练 loop 里,resume 前 500 ep 追踪
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
if grad_norm > baseline_grad_norm * 3:
    print(f"⚠ resume warmup: grad_norm = {grad_norm:.2f} (baseline ≈ {baseline_grad_norm:.2f})")
```

---

## 4. 缓解措施

### 4.1 Spike 预防

**必用**:`GRADCLIP=5.0`。torch.nn.utils.clip_grad_norm_ 硬性截断:

```
||g||_2 > 5  →  g ← g · 5/||g||   (方向保留,幅度截)
```

- 对正常训练无影响(||g|| 通常 <2)
- 对 outlier batch 有效 —— 单步 update 幅度受控 → 参数不踢飞
- 三个 fresh HCG cell 都因 GRADCLIP=0 死;加了 GRADCLIP=5 的 E64-perscale (40866525) 顺利收敛

**推荐**:所有 nr=2、所有 megabignet、所有 fresh 训练默认加 GRADCLIP=5.0。

**未实现的进一步保护**:

- **Soft clamp on log_sigma**:用 `5.0 * tanh(log_sigma / 5.0)` 替代 hard clamp。梯度光滑,不再突然 = 0
- **Adam 权重衰减**:β2 = 0.99(默认 0.999)让 v 更快跟上 gradient outlier
- **每 500 ep 保存 EMA(model weights average)**:即便训练 spike,best-EMA 保留稳定 model

### 4.2 Resume 修复(2026-07-04 已实现)

**已 fix**:`flow/flow.py:37` 现在 save 时也存 optimizer state。

Change:

```python
# flow/flow.py:37
def save(self, optimizer=None):
    if optimizer is None:
        return self.state_dict()      # 旧格式,backward compat
    return {'model': self.state_dict(), 'optimizer': optimizer.state_dict()}

def load(self, saveDict):
    if isinstance(saveDict, dict) and 'model' in saveDict:
        self.load_state_dict(saveDict['model'])
        return saveDict.get('optimizer', None)    # 返回 optimizer state 给 caller 应用
    self.load_state_dict(saveDict)
    return None
```

`main.py` 捕获 `loaded_optimizer_state = fw.load(saved)`,传给 `learnInterface`。
`learn.py` 收到后 apply 到新创建的 Adam:

```python
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
if optimizerState is not None:
    optimizer.load_state_dict(optimizerState)
    print("[resume] restored Adam state — skipping ~500-epoch warm-up")
```

**结果**:未来 resume(新 checkpoint 保存后)从 pre-load Adam 状态继续,**无
warm-up,无参数漂移**。

**Backward compat**:老 checkpoint(bare state_dict)会被 legacy path 加载,返回
None,行为不变。

**存储 cost**:Adam state ≈ 2× model size(m + v)。D64 mega 之前 ~230MB 存 model →
带 Adam 变 ~700MB。可接受。

### 4.3 已经 spike 或 resume 破坏的 cell 怎么救

- **保留 pre-spike 好 checkpoint**(每 200 ep 存 → 通常 spike 前有 stable 好 saving)
- **删掉 post-spike savings**(见 D64 resume 处理:删掉 ep 4800-6800 后 latest = 4600)
- **必要时重跑,从 pre-spike checkpoint + gradClip + soft clamp**
- **接受 pre-spike best 作为 final** —— 对 fair comparison 目的够用

---

## 5. 已建实践 checklist

Phase-2 / Phase-3 训练 job 提交前:

- [ ] `GRADCLIP=5.0` 除非明确不需要(baseline nr=1 可能不必要)
- [ ] nr=2 及以上 depth 必须 GRADCLIP
- [ ] megabignet / hcgHidden >= 64 必须 GRADCLIP
- [ ] Resume 从新 checkpoint(带 Adam state)会自动无 warm-up
- [ ] Resume 从旧 checkpoint(pre-2026-07-04 保存的)仍会 warm-up,考虑接受 pre-load 状态作为 final
- [ ] 训练 log 定期 check `max(last 500 ep)` vs `mean(last 500 ep)` 早预警
- [ ] 每 200 ep 存 checkpoint 保留 pre-spike backup

---

## 6. 相关 memory / 文件

- [[resume-optimizer-state]] — 原始 TODO(已 fix)
- [[l32-late-training-instability]] — best-smoothed vs final-epoch 的原始观测
- `analyzers/rg_fixed_point/prior_offload_analysis_zh.md` §6 讨论 CNN 学的 σ 随 log_sigma clamp 摸边的动力学
- `analyzers/rg_fixed_point/i2_loss_derivation_zh.md` — loss 的完整推导,含 log σ 项
