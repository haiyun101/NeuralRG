# Physical-observable regularizer (χ + U₄) — full report

> Documents the multi-observable physical regularizer that adds
> `λ_χ · (χ_flow − χ_target)² + λ_U · (U₄_flow − U₄_target)²` to the
> training loss. Sweep at L=32, L=64, L=128 shows the mechanism helps
> susceptibility but degrades magnetization, and is unstable at large λ.

## 1. Motivation

**The problem**: even a "converged" MERA champion (L=32 champion at loss
plateau ~1912) produces samples with χ **37% below the true Ising T_c
susceptibility** (χ_flow ≈ 20 vs χ_GT ≈ 32). Loss-at-entropy-floor does
NOT guarantee correct sample statistics — the flow matches
`E[log q(x)]` on average but can still mismatch higher moments.

Concretely, we observed **χ undershoot at every L**:
- L=32 plain champion: χ_flow ≈ 20 vs χ_GT ≈ 32 (−38%)
- L=64 plain champion: χ_flow ≈ 71 vs χ_GT ≈ 106 (−33%)
- L=128 warm-start: χ_flow ≈ 237 vs χ_GT ≈ 357 (−34%)

Interestingly, this pathology is **L-invariant in relative terms**
(~35% χ undershoot regardless of L), suggesting it's an intrinsic
property of the "match log-likelihood" objective, not a finite-size or
optimization artifact.

**The idea**: directly regularize the training objective with **physical
observable matching**. If the loss doesn't push q(x) toward correct χ,
add an explicit term that does.

## 2. Implementation

### 2.1 Differentiable observables from HS field

The flow output is a continuous HS-transformed field `x ∈ R^(L×L)`, but
Ising observables are defined on discrete spins `s ∈ {±1}`. Naive
`sign(x)` isn't differentiable.

**Solution**: soft-spin proxy `s = tanh(x)`. For typical HS-field
magnitudes at T_c (`|x| ~ 3-11`), `tanh(x)` saturates very close to
`±1`, so it's essentially the discrete spin proxy but with a well-
defined gradient everywhere.

Per-sample magnetization on this soft-spin field:
```
    M_i = mean of tanh(x_i) over the L×L lattice
```

Batch susceptibility (finite-L convention):
```
    chi_batch = N_sites · (E[M²] − E[|M|]²)   where N_sites = L²
```

Binder cumulant:
```
    U4_batch = 1 − E[M⁴] / (3 · E[M²]²)
```

Both formulas differentiable end-to-end w.r.t. flow parameters.

### 2.2 Loss augmentation

Added at `train/learn.py` inside the `dataDriven` branch:

```
    loss = -E_data[log q(x)]                     # standard MLE
         + volumePreservingWeight · (log|det J|)²   # VP penalty
         + physRegWeightChi · (chi_batch − chi_target)²
         + physRegWeightU4  · (U4_batch − U4_target)²
```

Each new term averages over `physRegBatch` fresh samples drawn from the
flow at each step (separate from the data batch).

### 2.3 Target auto-computation

Targets `chi_target`, `U4_target` computed at training startup from a
random subsample of the HS training data (default N_target = 5000). Avoids
users needing exact critical values.

At L=32 startup, auto-detected targets:
```
    [phys-reg] data-derived targets from N=5000: chi = 32.609, U4 = 0.6101
```

Matches Wolff-MCMC exact GT (χ = 31.61, U₄ = 0.611) within 3%.

### 2.4 CLI flags added to `main.py`

```
    -physRegWeightChi <λ_χ>      # coefficient for chi term (default 0)
    -physRegWeightU4  <λ_U4>     # coefficient for U4 term (default 0)
    -physRegBatch <N>            # samples per step (default 128 at L=32,
                                  #                  64 at L=64,
                                  #                  8 at L=128 for memory)
    -physRegTargetChi <val>      # override auto-target (default NaN → auto)
    -physRegTargetU4  <val>      # override auto-target (default NaN → auto)
```

Persisted to `parameters.hdf5` so `-load` correctly resumes.

## 3. L=32 sweep results

### 3.1 Setup

Fixed warm-start from L=32 champion (`fixdil+VP-1e-3 nr=1` @ ep 9500),
train 800 more epochs with 3 different λ values:

| Cell | λ_χ | λ_U4 | Rationale |
|:---:|:---:|:---:|:---|
| 1 | 0.01 | 0.01 | Light — mainly probe for direction |
| 2 | 0.1 | 0.1 | Moderate — balanced trade-off (theory) |
| 3 | 1.0 | 1.0 | Strong — physReg-dominant |

Same warm-start ensures loss trajectories are directly comparable.

### 3.2 Physics results (N=2000 samples per cell)

L=32 T_c ground truth: **|M|=0.6544, χ=31.61, U₄=0.6110**

| Cell | \|M\| | χ | U₄ | Loss (Best-200) |
|:---:|:---:|:---:|:---:|:---:|
| Plain champion | 0.674 (+3%) | **19.8 (−37%)** | 0.626 (+2%) | 1912 |
| **λ=0.01** | **0.639 (−2%)** | **21.5 (−32%)** | **0.618 (+1%)** | 1916 (+4 nat) |
| λ=0.1 | 0.477 (**−27%**) | 33.77 (+7%) | 0.519 (−15%) | 1933 (+21 nat) |
| λ=1.0 | 0.415 (**−37%**) | 32.64 (+3%) | 0.533 (−13%) | 1978 (+66 nat) |

### 3.3 Total error ranking

Sum of relative absolute errors on the 3 observables:

| Cell | Total error | Pareto? |
|:---:|:---:|:---|
| **λ=0.01** | **35.3%** | ✅ **Pareto improvement** over plain champion (all 3 obs improve) |
| Plain champion | 42.7% | baseline |
| λ=0.1 | 49.0% | Worse — trades \|M\| and U₄ for χ |
| λ=1.0 | 52.7% | Worst — physReg dominates, MLE quality lost |

**Verdict**: λ=0.01 is the only **strict improvement**. Higher λ fixes χ
at unacceptable cost.

### 3.4 Mechanism analysis

**Why does high λ break |M|?** Consider the physReg gradient direction:

The χ term `(χ_flow − χ_target)²` has gradient (through soft-spin):
```
    ∂/∂θ (χ_flow − χ_target) ∝ ∂E[M²] / ∂θ − 2 · E[|M|] · ∂E[|M|] / ∂θ
```

The first term (∂E[M²]/∂θ) can be increased in two ways:
1. Make M distribution WIDER around its peak → correct fix
2. Make M distribution PEAKED CLOSER TO ZERO → wrong fix (shrinks |M|)

Both increase χ = N·Var(M). The optimizer finds whichever is easier —
and shrinking peaks is EASIER than widening them (needs less
distributional change).

At λ = 0.01: the light push finds mode #1 (small widening).
At λ = 0.1+: the strong push finds mode #2 (peak collapse toward 0),
which explains the |M| drop.

## 4. L=64 results

### 4.1 Cells trained

| Cell | Init | λ_χ | λ_U4 | fp/bf | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| Plain champion | | | | fp32 | Reference: BP200 = 7659 |
| champion_physReg_bf16 | warm | 0.1 | 0.1 | **bf16** | BP200 = 7817 (+158 nat) |
| champion_physReg_fp32 | warm | 0.1 | 0.1 | fp32 | BP200 = 7756 (+97 nat) ✅ best |
| physReg_fresh_v2 | fresh | 0.05 | 0.05 | fp32 | BP200 = 7867 (+208 nat) |

### 4.2 Key observation — instability without gradClip

**Original L=64 physReg jobs (warm+fp32 and fresh+fp32) DIVERGED** to
loss > 10^24. Root cause: physReg gradient through flow.sample() has
large variance at L=64 (larger lattice, batch=16 means more per-sample
gradient noise), and without gradient clipping, occasional large
gradients drive Adam variance estimate wild → runaway.

**Fix that works**: add `-gradClip 1.0`. All L=64 physReg experiments
that included gradClip stayed stable.

### 4.3 fp32 vs bf16 for physReg

- bf16 warm-start: BP200 = 7817 (+158 nat)
- **fp32 warm-start: BP200 = 7756 (+97 nat)** ← better

fp32 gives lower loss because bf16's precision loss compounds through
the physReg gradient computation. bf16 physReg is stable (unlike bf16
fresh-init MERA training) but sub-optimal.

## 5. L=128 failure

### 5.1 Divergence

L=128 physReg (warm-start from L=64, λ=0.1, batch=8, gradClip=1.0)
**DIVERGED catastrophically**: Best-200 = 44,963 (from theoretical
entropy floor at 30,636). Final loss = 6.5×10^7.

### 5.2 Why L=128 physReg is harder

Two amplification factors vs L=64:
1. **Smaller flow.sample() batch** (physRegBatch=8 at L=128 vs 64 at L=64)
   → 8× more variance in χ_batch estimate → noisier physReg gradient
2. **Larger lattice** (16384 sites vs 4096) → χ_flow itself has wider
   fluctuations per batch → target-tracking harder

Result: physReg gradient overshoots frequently, destabilizes even with
gradClip.

### 5.3 Possible fixes (not yet tried)

- **Larger physRegBatch** (requires more GPU memory — need bigger A100
  or gradient checkpointing to fit)
- **Weaker λ** (0.01 instead of 0.1 — but L=32 shows λ=0.01 gives only
  marginal improvement)
- **Warm-up phase**: freeze MERA for first 500 epochs while CNN + physReg
  align

## 6. Cross-L pattern

Absolute physReg cost (loss above floor) at optimal λ per L:

| L | Optimal λ | Loss cost vs plain | Improvement worth it? |
|:---:|:---:|:---:|:---:|
| 32 | 0.01 | +4 nat | ✅ yes — Pareto win, tiny loss cost |
| 64 | 0.1 (with gradClip) | +97 nat | ⚠ marginal — χ improvement modest |
| 128 | (none stable) | diverges | ❌ no — technique doesn't scale |

**Physical intuition**: as L increases, flow.sample() variance grows
(bigger lattice, harder-to-sample bimodal distribution) while GPU memory
constrains physRegBatch. Signal-to-noise of the physReg gradient
degrades, causing instability.

## 7. Key findings

### 7.1 Trade-off is fundamental, not tunable

The `χ` term rewards `Var(M)` growth. Any parameter change that
increases `Var(M)` gets rewarded — including "peaks shrink toward 0",
which is BAD for physics. The optimizer picks whatever's cheapest, and
peak-shrinking is cheaper than peak-widening for HCG MERA parameters.

Higher λ makes this trade-off worse. **No amount of hyperparameter tuning
avoids the |M| ↔ χ trade-off** with this specific χ regularizer form.

### 7.2 Alternatives that would sidestep this

Would need to add explicit constraint on `E[|M|]`:
```
    loss += λ_M · (E[|M|] − |M|_target)²
```
This 3-observable regularizer would penalize peak-shrinking directly.
Not yet tested.

Or use distributional matching (MMD, W1) between flow M distribution
and data M distribution — matches ALL moments simultaneously. More
expensive but doesn't allow the trade-off.

### 7.3 Best-Pareto recommendation

**Use λ ≤ 0.05 only**. Above that, the trade-off cost exceeds the
benefit.

For L=32 specifically: **λ_χ = λ_U4 = 0.01** with warm-start from
champion. Improves all 3 observables slightly at cost of +4 nat loss.
This is the only physReg cell that survived the cleanup.

For L=64: **λ = 0.05-0.1 with `-gradClip 1.0`**. Marginal improvement,
mostly on χ.

For L=128 and above: **skip physReg**. Current technique doesn't scale;
need alternatives (larger batch or distributional matching).

## 8. Data / code / models retained after cleanup

**Kept models** (26 GB total):
- `data/32Ising_T2.269_physReg_chi0.01_u40.01/` — Only L=32 physReg that
  strictly improves all 3 observables
- `data/L64_T2.269_champion_physReg_chi0.1_u40.1/` — bf16 warm-start
- `data/L64_T2.269_champion_physReg_chi0.1_u40.1_fp32/` — best L=64 physReg
- `data/L64_T2.269_physReg_fresh_chi0.05_u40.05_v2/` — L=64 fresh with λ=0.05

**Deleted** (freed 26 GB):
- L=32 λ=0.1 and λ=1.0 (broke |M|, U₄)
- L=64 fp32_v2 (worse than fp32 v1)
- L=64 fresh λ=0.1 (diverged)
- L=128 physReg (diverged badly)

**Code**:
- CLI: `main.py` lines with `-physReg*` flags
- Loss augmentation: `train/learn.py` inside `dataDriven` branch (lines ~640-670)
- No dedicated report/plot script — used `plot_model_physics.py` in multi-model mode

**Analysis outputs**:
- `figures/physReg/L32/` — 4 plot types + summary.json (kept, historical reference)
- `figures/physReg/L64/` — 4 plot types + summary.json
- `analyzers/physReg/measure_physReg_effect.py` — quick numerical measure

## References

- Related report: `cross_L_transfer_report.md` (transfer + physReg cross-reference)
- Related memory: `cnn-absorbs-variance-not-mean` (CNN's role in σ shaping —
  same intuition applies here: physReg is another way to shape σ)
- Related memory: `entropy-reg-does-not-help` — similar "regularizer trade-off"
  finding for a different observable (entropy)
