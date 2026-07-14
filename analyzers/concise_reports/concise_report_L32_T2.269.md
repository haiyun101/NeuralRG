# Ising L=32 — Concise Report (T=2.269)

## ★ Phase-4 volume-preserving penalty (2026-07-09) — the LOSS metric was concealing the story

Diagnostic finding across all HCG variants: the CNN's predicted σ² is
10–100× smaller than the empirical variance of MERA's z_fast output
(calib_ratio ≈ 0.03–0.10). MERA's log|det J| absorbs the mismatch,
letting total LOSS stay low even though the CNN is not encoding
"physical" per-scale variance. To rescue the interpretability the design
was supposed to give, we add a soft penalty forcing MERA toward
volume-preserving:

```
loss_total = −log q(x) + λ · (log|det J_MERA|)²
                 └────── F ──────┘    └── VP penalty ──┘
                 (pure MLE)         (regularizer term)
```

**IMPORTANT**: only **F** (pure MLE) is comparable across variants.
The training-target LOSS `L = F + λ·penalty` inflates VP runs by the
penalty term itself. All numbers below are **F** unless noted.

### L=32 T_c results with VP penalty (fixdil per-scale nr=1)

Fixdil per-scale gained MOST from VP because its CNN was already trying
to encode conditional variance and MERA had been silently overriding it.

| Variant | F | vs fixdil baseline | vs D nr=2 (1896.16) |
|---|---:|---:|---:|
| fixdil baseline (no VP) | 1903.97 | ref | +7.81 |
| fixdil + VP-1e-5 | 1896.54 | **−7.43** | +0.38 |
| fixdil + VP-1e-4 | 1897.06 | −6.91 | +0.90 |
| **fixdil + VP-1e-3** | **1896.20** | **−7.77** | **+0.04** ← ties D nr=2 at HALF compute |
| fixdil + VP-1e-2 | 1897.21 | −6.76 | +1.05 |

### Shared HCG + VP: nearly free at small λ

| Variant | F | vs shared baseline (1891.10) |
|---|---:|---:|
| shared baseline | 1891.10 | ref |
| **shared + VP-1e-5** | **1891.99** | **+0.89** ← essentially free |
| shared + VP-1e-4 | 1892.44 | +1.34 |
| shared + VP-1e-3 | 1893.52 | +2.42 |
| shared + VP-1e-2 | 1895.30 | +4.20 |

### Interpretation

**VP penalty asymmetry across architectures:**
- **Shared** was cheating hardest (its σ was disconnected from physics).
  Adding VP costs LOSS proportional to how much cheating it stops.
  But at small λ (1e-5) it's essentially free.
- **Fixdil** had a semi-honest CNN (Conv0 alive at all levels, R²(mean) up
  to 0.85 at L1). Adding VP acts as a productive regularizer — it forces
  MERA to be honest, and the CNN takes over the physical variance role
  it was supposed to have. Result: fixdil+VP at nr=1 **ties D at nr=2
  compute**.

### CNN behavior interpretation

The CNN's job is to parametrize `p(z_fast[i] | z_slow) = N(μ_i, σ_i²)`
per site per level. In practice:

- **μ ≈ 0** (Z2-symmetrized training pushes it to 0) — CNN's mean output
  is inert.
- **σ carries all signal** — training shapes it to depend on z_slow.
- At **fine levels** the learned law is interpretable: `σ(z_slow)`
  decreases with local ordering `|mean z_slow|` (R² ≈ 0.6 in nodilate,
  0.6–0.85 in fixdil). This is a susceptibility-like law: model is
  "quiet" in ordered blocks, "loud" near domain walls.
- At **coarse levels** most variants collapse Conv0 → σ becomes constant
  → HCG at those levels degenerates to a plain Gaussian.

Volume-preserving penalty pushes σ closer to reflecting actual conditional
variance instead of being an ignored bookkeeping variable.

### Sample and correlation plots — winning VP variants

**Fixdil + VP-1e-3 nr=1** (best fixdil-arm cell, F=1896.20 — ties D nr=2):

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/flow_samples.png" alt="fixdil VP-1e-3 samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/flow_correlations.png" alt="fixdil VP-1e-3 correlations" width="56%">
</p>

**Shared + VP-1e-5 nr=1** (best shared-arm cell, F=1891.99 — near-free VP):

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-5_b64/flow_samples.png" alt="shared VP-1e-5 samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-5_b64/flow_correlations.png" alt="shared VP-1e-5 correlations" width="56%">
</p>

**Shared + VP-1e-2 nr=1** (strongest VP tested, F=1895.30 — largest departure from cheating):

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-2_b64/flow_samples.png" alt="shared VP-1e-2 samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-2_b64/flow_correlations.png" alt="shared VP-1e-2 correlations" width="56%">
</p>

## ★ Physical observables — sample-based physics at T_c

The forward-KL loss (S = −E_data[log q]) tells us how well the flow
matches the data's log-density. But **it doesn't guarantee that samples
from the flow reproduce the physics** — a model can memorize the batch
while still generating overly-ordered / overly-disordered configurations.
We test this by sampling N=10 000 configurations from each flow and
computing standard Ising observables.

### Definitions

**Energy per site** `⟨E⟩ = −(1/N_sites) Σ_⟨ij⟩ s_i s_j` (nearest-neighbour
sum, J=1). Sensitive to short-range order.

**Absolute magnetization** `⟨|M|⟩ = ⟨|(1/N) Σ s_i|⟩`. Order parameter.

**Susceptibility** — the fluctuation of magnetization:
```
χ = N_sites · (⟨M²⟩ − ⟨|M|⟩²)
```
Physically: **how much the system fluctuates in magnetization from
configuration to configuration**. Small χ ⇒ every sample has similar M
(system is "stiff"); large χ ⇒ samples vary widely in M (system is at
criticality). At the 2D Ising critical point χ **diverges** in the
thermodynamic limit: χ ∝ |T − T_c|^{−γ} with γ = 7/4. On a finite lattice
L=32 the peak is finite — theory expects **χ ≈ 33 at T_c** for L=32
(from Wolff MCMC ground truth).

**Binder cumulant** — a universal ratio of moments:
```
U₄ = 1 − ⟨M⁴⟩ / (3 · ⟨M²⟩²)
```
Physically: **a dimensionless measure of how Gaussian the magnetization
distribution is**. For a Gaussian P(M): U₄ = 0. For a delta-function
(sharp order): U₄ = 2/3 ≈ 0.667. For 2D Ising at criticality the
universal value is **U₄ ≈ 0.611** — independent of L, entirely determined
by the fixed point. Deviations from this value directly measure how far
the model's sampled distribution is from a genuine critical distribution.

**Why U₄ is especially useful**:
- Universal at T_c (any 2D Ising-universality-class model gives the same)
- Robust to finite-N sampling noise (moments cancel in the ratio)
- **A model at T_c that gives U₄ ≠ 0.611 is not producing critical
  configurations**, no matter how good its LOSS looks

### Ground truth (Wolff MCMC, N=10 000, L=32 T_c)

```
⟨E⟩ ≈ −1.504     ⟨|M|⟩ ≈ 0.67     χ ≈ 32.93     U₄ ≈ 0.612
```

### Preliminary model observables (partial — CPU tier1 job timed out)

Only 3 variants completed sampling before the CPU walltime cut. Rerun
with Best-200 anchored checkpoints planned.

| Variant                    | ⟨E⟩    | χ    | Δχ%  | U₄    | ΔU₄  |
|:---------------------------|:-----: |:----:|:----:|:-----:|:----:|
| **GT (Wolff)**             | −1.504 | 32.93|  0   | 0.612 |  0   |
| D = i2 nr=2                | −1.415 | 22.28| −32% | 0.621 | 0.009|
| A = Gaussian nr=1          | −1.422 | 20.05| −39% | 0.627 | 0.015|
| HCG shared nr=1 (LOSS champ) | −1.419 |  **6.88** | **−79%** | 0.649 | **0.037** |

### Physics interpretation

Even with only 3 data points the pattern is stark:

- **HCG shared nr=1** (previously reported as LOSS champion by
  single-epoch min) has **χ = 6.88 vs GT 32.93** — 5× too small. Its
  samples don't reproduce critical fluctuations.
- Its U₄ = 0.649 sits closer to the ordered limit (2/3 = 0.667) than to
  the critical value 0.611 — consistent with it producing over-ordered
  configurations.
- Meanwhile **A** (plain Gaussian nr=1, LOSS 8 nat WORSE than HCG shared)
  produces **3× more physical susceptibility (χ = 20)** with U₄ closer
  to the T_c value (0.627 vs shared's 0.649).

**LOSS ≠ physics dissociation confirmed at L=32** exactly as it was at
L=64. The fair-metric fix (Best-200) applied to L=64 rankings correctly
demoted HCG shared and elevated fixdil+VP; when we complete the L=32
tier1 rerun with Best-200 anchors, we expect the same qualitative story.

Remaining variants to run at Best-200 for L=32: fixdil+VP-1e-{5,4,3,2},
HCG nodilate init nr=2, HCG perscale nr=2, HCG shared nr=2, and the
new T-sweep results (T=2.15, 2.22, 2.32, 2.40) once training completes.

> **⚠ Note**: `make_concise_report.py` regenerates this file from scratch —
> both the Phase-3 AND Phase-4 sections must be re-added after every
> regeneration.

> **⚠ Note**: `make_concise_report.py` regenerates this file from scratch —
> both the Phase-3 AND Phase-4 sections must be re-added after every
> regeneration.

---

## ★ Phase-3 HCG results (2026-07-04) — Hierarchical Conditional Gaussian prior

Follow-up to D's KL_qp win: replace i2's single-CNN conditional Gaussian
prior with a **multi-scale hierarchy** of conditional Gaussians (one per
stride level [16, 8, 4, 2, 1] for L=32). CNN can be *scale-shared* (one
CNN across all levels — hard-codes RG scale-invariance) or *per-scale*
(one CNN per level — flexibility).

### HCG best-200 vs D reference (L=32 T_c)

| Cell | Config | Params | Best-200 | vs D32 |
|------|--------|-------:|---------:|:-:|
| A    | Gaussian baseline nr=1 | 10.94 M | 1919.05 | +8.20 |
| B    | i2 (single CNN) nr=1 | 10.95 M | 1919.31 | +8.46 |
| **D** | **i2 (single CNN) nr=2** | **21.89 M** | **1910.85** | ref |
| E-shared nr=1  | HCG shared, nr=1 | 10.95 M | 1912.80 | +1.95 |
| **E-shared nr=2 (pre-spike)** | HCG shared, nr=2 | 21.89 M | **1910.40** | **−0.45 ★** |
| E-perscale nr=1 (broken dil) | HCG per-scale, buggy dilation | 10.98 M | 1917.02 | +6.17 |
| E-perscale nr=1 (**fixdil**) | HCG per-scale, corrected dilation | 10.98 M | 1922.27 | +11.42 |
| E-perscale nr=2 (**fixdil**) | HCG per-scale, corrected dilation, nr=2 | 21.90 M | 1919.25 | +8.40 |

**Key finding**: **E-shared nr=2 beats D32 by 0.45 nat** — HCG's scale-shared
prior (RG-invariant architectural bias) improves on D at equal MERA depth.

### CNN mechanism check (via `hcg_perscale_similarity.py`)

Conv0 weight L2 at each hierarchy level (per-scale HCG at L=32 T_c):

| Level | Broken dil | Fixed dil (2026-07-04) |
|:-:|:-:|:-:|
| 1 | 1.80 | 0.51 |
| 2 | **0.0000 (dead)** | **0.98 ✓** |
| 3 | 0.0001 (dead) | **1.68 ✓** |
| 4 | 0.0000 (dead) | **2.64 ✓** |

σ output std at Level 2: broken = 0.0000 (constant), fixed = 0.63 (conditional).

### Test 4 — off-critical reversal (T = 2.40) confirms scale-invariance is T_c-specific

| Cell | T_c | T=2.40 |
|------|:-:|:-:|
| **shared** (hard-codes scale-invariance) | **1912.80 win by 9 nat** | 1965.90 lose by 1.7 nat |
| **perscale fixdil** (per-level flexibility) | 1922.27 lose | **1964.21 win ★** |

Reversal confirms scale-invariance is a genuine physics inductive bias for
the T_c fixed point — not just an optimization artifact.

> **⚠ Note**: `make_concise_report.py` regenerates this file from scratch —
> this Phase-3 HCG section must be re-added after every regeneration.

---

## Summary — everything in one table

Superset of the two tables above: free energy / energy / entropy **and**
both KL directions, for exact theory, the two **datasets**, and the best
trained flow of each mode. Discrete rows are grouped first.

Font marks where each number comes from (Markdown has no portable text
colour, so font carries the distinction):

- **bold** — exact theory (Onsager / `exactz.md`).
- *italic* — training-measured, read from the run's HDF5 records. A
  reverse-KL run logs `F/E/S` of the flow; a forward-KL run logs only
  `S` (the MLE loss `-E_data[log q]`) — its `F/E` are N/A.
- plain — sample-measured: a dataset sample-average, or the post-hoc
  flow diagnostic that draws `x ~ q` (the only way to get a forward-KL
  run's model-side `F/E`).

| Source                               |  Picture   |    F (-lnZ)    |       E       |       S       |  KL(q‖p)  | KL(p‖q)  |
| :----------------------------------- | :--------: | :------------: | :-----------: | :-----------: | :-------: | :------: |
| **Exact (theory)**                   |  discrete  | **-952.6481**  | **-668.4678** |  **284.1802** |     —     |    —     |
| MCMC dataset (Wolff)                 |  discrete  |      N/A       |   -647.0935   |      N/A      |     —     |    —     |
| **Exact (theory)**                   | continuous | **-2369.5871** | **-466.6111** | **1902.9760** |   **0**   |  **0**   |
| HS dataset (x ~ p_HS)                | continuous |      N/A       |   -466.6109   |   1902.9762   |     —     |    —     |
| *sym_longer — training*              | continuous |  *-2357.6262*  |  *-535.9175*  |  *1821.7087*  | *11.9609* |   N/A    |
| sym_longer — diagnostic (epoch 1590) | continuous |   -2357.3455   |   -533.7950   |   1823.5504   |    N/A    | 89.4075  |
| *hs_bignet — training*               | continuous |      N/A       |      N/A      |  *1906.6096*  |    N/A    | *3.6336* |
| hs_bignet — diagnostic (epoch 9500)  | continuous |   -2348.3218   |   -421.4914   |   1926.8304   |  21.2653  |   N/A    |

Notes:
- Each flow gets **two rows** — *training* and *diagnostic* — the same run
  as the optimiser logged it vs. as a fresh `x ~ q` sample measures it. For
  a converged reverse-KL run the two should agree.
- **Datasets**: `E` is a plain sample average; `F = -lnZ` cannot be
  estimated from samples (needs the partition function) → N/A. HS
  `S_c = E_p[A] + lnZ_c` is an MC entropy estimate (uses exact `lnZ_c`);
  MCMC gives only `E_d`.
- `KL(q‖p)` / `KL(p‖q)`: each direction appears once per flow. The
  *training* row carries the **on-objective** KL — the one that mode
  minimises, recovered from the loss (reverse-KL `KL(q‖p)=loss+lnZ_c`;
  forward-KL `KL(p‖q)=loss-H(p_HS)`). The *diagnostic* row carries the
  **off-objective** KL, which training cannot see. `—` = not applicable
  (theory-discrete / dataset rows); `0` for continuous theory.
- A **negative** training-row `KL(p‖q)` means the MLE loss dipped below
  the entropy floor `H(p_HS)` — training-set overfitting (seen at L=8/16).
- The per-run breakdown for *all* methods stays in the flow-diagnostic
  table above; this summary keeps only the best of each mode.

## Flow visualizations — configurations + physical observables

_Per method: left = flow samples (configurations, `sigmoid(2x)` render_
_of `flow (q)` vs `HS data (p)`); right = flow correlations (magnetisation_
_distribution P(M) + axial two-point correlation G(r)/G(0), flow vs data)._

### hsBignet_baseline_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_baseline_b64/flow_samples.png" alt="hsBignet_baseline_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_baseline_b64/flow_correlations.png" alt="hsBignet_baseline_b64 flow correlations" width="56%">
</p>

### hsBignet_bridge_w5

<p>
<img src="../../data/32Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_samples.png" alt="hsBignet_bridge_w5 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_correlations.png" alt="hsBignet_bridge_w5 flow correlations" width="56%">
</p>

### hsBignet_combined_lam1

<p>
<img src="../../data/32Ising_T2.269_hsBignet_combined_lam1.0_stride8h32_b64/flow_samples.png" alt="hsBignet_combined_lam1 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_combined_lam1.0_stride8h32_b64/flow_correlations.png" alt="hsBignet_combined_lam1 flow correlations" width="56%">
</p>

### hsBignet_hcg_perscale_fixdil_gc5

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64/flow_samples.png" alt="hsBignet_hcg_perscale_fixdil_gc5 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64/flow_correlations.png" alt="hsBignet_hcg_perscale_fixdil_gc5 flow correlations" width="56%">
</p>

### hsBignet_hcg_perscale_fixdil_nr2_gc5

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64/flow_samples.png" alt="hsBignet_hcg_perscale_fixdil_nr2_gc5 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64/flow_correlations.png" alt="hsBignet_hcg_perscale_fixdil_nr2_gc5 flow correlations" width="56%">
</p>

### hsBignet_hcg_shared_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_b64/flow_samples.png" alt="hsBignet_hcg_shared_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_b64/flow_correlations.png" alt="hsBignet_hcg_shared_b64 flow correlations" width="56%">
</p>

### hsBignet_hcg_shared_nr2_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64/flow_samples.png" alt="hsBignet_hcg_shared_nr2_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64/flow_correlations.png" alt="hsBignet_hcg_shared_nr2_b64 flow correlations" width="56%">
</p>

### hsBignet_i1_df4

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i1_df4.0/flow_samples.png" alt="hsBignet_i1_df4 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i1_df4.0/flow_correlations.png" alt="hsBignet_i1_df4 flow correlations" width="56%">
</p>

### hsBignet_i2_stride16h32

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride16h32/flow_samples.png" alt="hsBignet_i2_stride16h32 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride16h32/flow_correlations.png" alt="hsBignet_i2_stride16h32 flow correlations" width="56%">
</p>

### hsBignet_i2_stride4h32

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride4h32/flow_samples.png" alt="hsBignet_i2_stride4h32 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride4h32/flow_correlations.png" alt="hsBignet_i2_stride4h32 flow correlations" width="56%">
</p>

### hsBignet_i2_stride4h32_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride4h32_b64/flow_samples.png" alt="hsBignet_i2_stride4h32_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride4h32_b64/flow_correlations.png" alt="hsBignet_i2_stride4h32_b64 flow correlations" width="56%">
</p>

### hsBignet_i2_stride8h32

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32/flow_samples.png" alt="hsBignet_i2_stride8h32 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32/flow_correlations.png" alt="hsBignet_i2_stride8h32 flow correlations" width="56%">
</p>

### hsBignet_i2_stride8h32_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32_b64/flow_samples.png" alt="hsBignet_i2_stride8h32_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32_b64/flow_correlations.png" alt="hsBignet_i2_stride8h32_b64 flow correlations" width="56%">
</p>

### hsBignet_i2_stride8h32_nr2_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32_nr2_b64/flow_samples.png" alt="hsBignet_i2_stride8h32_nr2_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h32_nr2_b64/flow_correlations.png" alt="hsBignet_i2_stride8h32_nr2_b64 flow correlations" width="56%">
</p>

### hsBignet_i2_stride8h64_b64

<p>
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h64_b64/flow_samples.png" alt="hsBignet_i2_stride8h64_b64 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_i2_stride8h64_b64/flow_correlations.png" alt="hsBignet_i2_stride8h64_b64 flow correlations" width="56%">
</p>

### hsBignet_iii1_lam0

<p>
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam0.1_b64/flow_samples.png" alt="hsBignet_iii1_lam0 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam0.1_b64/flow_correlations.png" alt="hsBignet_iii1_lam0 flow correlations" width="56%">
</p>

### hsBignet_iii1_lam1

<p>
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam1.0_b64/flow_samples.png" alt="hsBignet_iii1_lam1 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam1.0_b64/flow_correlations.png" alt="hsBignet_iii1_lam1 flow correlations" width="56%">
</p>

### hsBignet_iii1_lam10

<p>
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam10.0_b64/flow_samples.png" alt="hsBignet_iii1_lam10 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hsBignet_iii1_lam10.0_b64/flow_correlations.png" alt="hsBignet_iii1_lam10 flow correlations" width="56%">
</p>

### hs_bignet

<p>
<img src="../../data/32Ising_T2.269_hs_bignet/flow_samples.png" alt="hs_bignet flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hs_bignet/flow_correlations.png" alt="hs_bignet flow correlations" width="56%">
</p>

### hs_dataDriven

<p>
<img src="../../data/32Ising_T2.269_hs_dataDriven/flow_samples.png" alt="hs_dataDriven flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hs_dataDriven/flow_correlations.png" alt="hs_dataDriven flow correlations" width="56%">
</p>

### hs_haarPrior

<p>
<img src="../../data/32Ising_T2.269_hs_haarPrior/flow_samples.png" alt="hs_haarPrior flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hs_haarPrior/flow_correlations.png" alt="hs_haarPrior flow correlations" width="56%">
</p>

### hs_weightTying

<p>
<img src="../../data/32Ising_T2.269_hs_weightTying/flow_samples.png" alt="hs_weightTying flow samples" width="42%">
<img src="../../data/32Ising_T2.269_hs_weightTying/flow_correlations.png" alt="hs_weightTying flow correlations" width="56%">
</p>

### jsLoss_bignet_long_lam0

<p>
<img src="../../data/32Ising_T2.269_jsLoss_bignet_long_lam0.5/flow_samples.png" alt="jsLoss_bignet_long_lam0 flow samples" width="42%">
<img src="../../data/32Ising_T2.269_jsLoss_bignet_long_lam0.5/flow_correlations.png" alt="jsLoss_bignet_long_lam0 flow correlations" width="56%">
</p>

### pathgrad_bignet_long_ext

<p>
<img src="../../data/32Ising_T2.269_pathgrad_bignet_long_ext/flow_samples.png" alt="pathgrad_bignet_long_ext flow samples" width="42%">
<img src="../../data/32Ising_T2.269_pathgrad_bignet_long_ext/flow_correlations.png" alt="pathgrad_bignet_long_ext flow correlations" width="56%">
</p>

### phase2_finetune

<p>
<img src="../../data/32Ising_T2.269_phase2_finetune/flow_samples.png" alt="phase2_finetune flow samples" width="42%">
<img src="../../data/32Ising_T2.269_phase2_finetune/flow_correlations.png" alt="phase2_finetune flow correlations" width="56%">
</p>

### sym

<p>
<img src="../../data/32Ising_T2.269_sym/flow_samples.png" alt="sym flow samples" width="42%">
<img src="../../data/32Ising_T2.269_sym/flow_correlations.png" alt="sym flow correlations" width="56%">
</p>

### sym_bignet

<p>
<img src="../../data/32Ising_T2.269_sym_bignet/flow_samples.png" alt="sym_bignet flow samples" width="42%">
<img src="../../data/32Ising_T2.269_sym_bignet/flow_correlations.png" alt="sym_bignet flow correlations" width="56%">
</p>

### sym_bignet_ext

<p>
<img src="../../data/32Ising_T2.269_sym_bignet_ext/flow_samples.png" alt="sym_bignet_ext flow samples" width="42%">
<img src="../../data/32Ising_T2.269_sym_bignet_ext/flow_correlations.png" alt="sym_bignet_ext flow correlations" width="56%">
</p>

