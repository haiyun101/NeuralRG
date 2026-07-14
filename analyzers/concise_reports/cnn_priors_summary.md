# CNN Prior Variants — Performance Summary

Note on all HCG (Hierarchical Conditional Gaussian) prior variants tested,
their LOSS numbers, and structural findings. Complements the individual
`concise_report_L*_T*.md` files.

Two LOSS columns are reported for VP-penalty variants:
- **F** = pure MLE loss `−E_data[log q(x)]` (fair for cross-variant comparison)
- **L** = training-target loss `= F + λ·(log|det J|)²` (what training optimizes)

For non-VP variants F = L. **Use F for comparisons across variants** —
adding a regularizer to L inflates it artificially.

> **Metric caveat (2026-07-13 update)**: The F numbers in the tables below
> are **single-epoch minima** over the record trajectory. At batch=16 the
> per-epoch noise σ ≈ 47 nat, so single-epoch min systematically flatters
> longer runs (more chances to catch a 3σ dip). The **Best-200 rolling
> mean** in `concise_report_L64_T2.269.md` is the sustained metric and
> should be preferred for ranking. Retracted below: the earlier
> "fixdil+VP-1e-3 nr=2 F=7541 new champion" claim was a 3σ noise dip
> (Best-200=7701, actually WORSE than baseline). The true L=64 champion
> is **fixdil+VP-1e-3 nr=1** at Best-200 = 7658.61.

## What does the CNN actually do?

Every HCG variant has one or more CNNs whose role is to **parametrize the
conditional distribution of "fast" latent variables given "slow" ones**.

Concretely, for each fast site *i* at level *k*, the CNN outputs a pair
`(μ_i, log σ_i) = CNN_k(z_slow_context)[i]` and the HCG prior scores:
```
p(z_fast[i] | z_slow) = N(z_fast[i]; μ_i, σ_i²)
```
The full prior factorizes over levels and over fast sites within a level.

Design intent:
- **μ** captures the *conditional mean* — how the slow background biases the
  fast value at each site.
- **σ** captures the *conditional standard deviation* — how much residual
  uncertainty remains at that site given the slow context.
- CNN's receptive field lets it use local slow-site information; different
  variants differ in how far that context reaches (dilation choices).

**Empirical reality (from σ-law and calibration analysis):**
1. **μ ≈ constant** across most variants — Z2-symmetrized training pushes
   μ toward 0, so the CNN's mean prediction is largely inert.
2. **σ carries all the signal** — this is what training actually shapes.
3. **σ² is 10–100× smaller than empirical Var(z_fast − μ)** — CNN
   systematically under-predicts the marginal variance because MERA's
   log|det J| Jacobian absorbs the mismatch instead (bookkeeping trade-off
   with no LOSS penalty; see finding #4 below).
4. At fine levels the σ-law is interpretable: `σ` decreases with the local
   ordering |mean(z_slow)|, i.e. **the model is quiet in ordered blocks
   and noisy near domain walls** — a susceptibility-like law.
5. At coarse levels (levels 1–2) in most variants Conv0 dies → σ becomes
   a constant → HCG collapses to plain N(μ_const, σ_const²) there. Only
   fine levels (3–4) carry a nontrivial CNN-learned law.

So the CNN's *intended* job is per-scale conditional distribution modeling,
but its *realized* job is more like "per-fine-level σ regulator, with
coarse levels neutered." The MERA Jacobian silently handles the rest.

## Variants tested

| Variant | Architecture | CNN count | Motivation |
|---|---|---|---|
| **D** = **i2 nr=2** (`conditional_gaussian`) | **Single CNN**, `condPriorSlowStride=8`, `hidden=32`, `nrepeat=2` | 1 | Phase-2 reference baseline (single-CNN, no hierarchy) |
| **A (Gaussian nr=1)** (`gaussian`) | Isotropic N(0,I) prior + MERA + RNVP | 0 | No learned prior (upper-bound baseline) |
| **HCG shared** (`hcg_shared`) | Single CNN, reused at every level | 1 | Simplest HCG; single scale-invariant law |
| **HCG shared progdil** | Same, with dilation [1,2,4] or [1,4,16] | 1 | Extend receptive field cheaply |
| **HCG perscale fixdil** | Per-level CNN, dilation=strides[k] | K−1 | Give each level scale-matched context |
| **HCG perscale nodilate** | Per-level CNN, dilation=1 | K−1 | Force each CNN to see nearest-neighbor only |
| **HCG perscale init-shared (E1)** | perscale, MERA+CNN copied from shared | K−1 | Rescue per-scale via good init |
| **HCG perscale init-shared (E2)** | E1 + Adam moments copied | K−1 | Avoid Adam warm-up drift ejecting basin |
| **HCG + VP penalty** | Any of above + λ·(log\|det J\|)² | varies | Force MERA volume-preserving to rescue σ physics |
| **HCG fixdil nr=2 + VP init-shared** *(2026-07-13, running)* | fixdil nr=2 + VP, MERA+shared-CNN+Adam copied from shared nr=2 | K−1 | Isolate whether nr=2 fixdil+VP gap was compute-limited or capacity-mismatched |
| **HCG fixdil nr=2 + VP from-nr=1** *(2026-07-13, running)* | fixdil nr=2 + VP, converted from nr=1 champion (rep-1 identity-init) | K−1 | Doubling test — does the nr=1 champion get better with extra depth from its own basin? |

## L=32 T_c results (best-so-far F; L in parens for VP runs)

| Variant | nr=1 F | nr=2 F |
|---|---|---|
| **D = i2 (single CNN)** | 1903.32 | **1896.16** |
| A = Gaussian baseline | 1899.32 | 1902.89 |
| HCG perscale fresh | 1900.67 | 1895.59 |
| HCG perscale E1 (no Adam) | — | 1916.60 (drift!) |
| HCG perscale fixdil | 1903.97 | 1899.75 |
| HCG perscale nodilate + init-shared + Adam (E2) | 1898.51 | **1891.83** |
| **HCG shared** | **1891.10** | 1894.47 |
| HCG shared + VP-1e-5 | 1891.99 (L=1892.14) | *nr=2 running* |
| HCG shared + VP-1e-4 | 1892.44 (L=1892.56) | — |
| HCG shared + VP-1e-3 | 1893.52 (L=1894.16) | — |
| HCG shared + VP-1e-2 | 1895.30 (L=1896.28) | — |
| **HCG fixdil + VP-1e-3** | **1896.20** (L=1896.75) — **ties D nr=2** | *running* |
| HCG fixdil + VP-1e-5 | 1896.54 (L=1896.57) | *running* |
| HCG fixdil + VP-1e-4 | 1897.06 (L=1897.11) | *running* |
| HCG fixdil + VP-1e-2 | 1897.21 (L=1898.44) | *running* |

## L=64 T_c results (best-so-far F; L in parens for VP runs)

| Variant | nr=1 F | nr=2 F |
|---|---|---|
| **D = i2 (single CNN)** | 7604.32 | **7578.27** |
| A = Gaussian baseline | 7600.73 | 7579.28 |
| HCG perscale fresh | 7614.17 | 7608.92 |
| HCG perscale fixdil | 7627.24 | 7597.41 |
| HCG perscale nodilate | 7599.79 | 7587.17 |
| HCG perscale nodilate (continuation) | 7599.53 | 7585.58 |
| **HCG shared** | 7589.96 | **7576.82** |
| HCG shared progdil [1,2,4] | — | 7609.66 (worse) |
| HCG shared progdil [1,4,16] | — | 7630.75 (worse) |
| HCG fixdil + VP-1e-4 | 7586.86 (L=7587.29) | 7560.40† (single-ep) / 7689.79 Best-200 |
| **HCG fixdil + VP-1e-3** | **7580.44** (L=7583.14) — **beats D nr=1 by 24, within 2.2 nat of D nr=2** | 7541.58† (single-ep) / 7701.63 Best-200 |
| HCG fixdil + VP-1e-2 | 7584.53 (L=7592.10) | 7554.50† (single-ep) / 7716.33 Best-200 |
| **HCG fixdil + VP init-shared nr=2** (2026-07-13, planned) | — | *pending — shell/vp_l64_fixdil_nr2_initshared.sh* |
| **HCG fixdil + VP from-nr=1 nr=2** (2026-07-13, planned) | — | *pending — shell/vp_l64_fixdil_nr2_from_nr1.sh* |

`†` = single-epoch min was originally claimed as champion; Best-200 shows
all three nr=2 fixdil+VP arms are actually WORSE than baseline D nr=2
(7578.27) and hcg_shared nr=2 (Best-200 = 7590.52). See metric caveat at top.

## Key findings

### 1. Simpler wins on LOSS (shared > perscale > fixdil)

`shared` HCG with uniform d=1 has the lowest LOSS in 5 of 8 cells. Per-scale
variants add K−1 CNNs but don't earn back the parameter cost on LOSS. Progressive
dilation ([1,2,4] or [1,4,16]) is **strictly worse** than uniform d=1 for shared.

### 2. `fixdil` (matched dilation) is uniformly worse than `nodilate`

By 5–27 nat across all L=32/L=64 nr=1/2 cells. Reason: matched-stride dilation
lets the CNN reach farther context, but at coarse scales the "far context" is
decorrelated majority-voted noise. CNN fits that noise → hurts LOSS.
`nodilate` lets Conv0 die at coarse levels as a **learned regularizer**;
CNN capacity concentrates on fine levels where local correlations are informative.
Documented in `project_hcg_nodilate_beats_fixdil`.

### 3. init-from-shared works iff Adam state is preserved

- **E1** (MERA+CNN copied, fresh Adam m=v=0) is 20–25 nat WORSE than fresh
  perscale — Adam's first step ≈ lr·sign(g), which ejects the model from the
  loaded basin.
- **E2** (MERA+CNN+Adam state all copied from an adamwarm shared checkpoint)
  is the only per-scale variant that BEATS shared at L=32 nr=2 (1891.83 vs 1894.47).
Documented in `project_resume_optimizer_state`.

### 4. All variants are catastrophically under-calibrated

CNN's predicted σ² is 10–100× smaller than empirical Var(z_fast − μ). MERA's
Jacobian absorbs the marginal-variance mismatch: `log q = log p_prior + log|det J|`,
and any prior miscalibration is compensated by MERA expanding volume. LOSS
stays low but **σ is not a physical variance — just whatever CNN outputs after
MERA has already "handled" the scale**. HCG's design premise (per-scale physical
conditional variance) is undermined by this Jacobian cheating.

### 5. Structural pattern shared across nodilate winners

Both nodilate winners show:
- **CNN**: Conv0 at Levels 1–2 (coarse) → 0 → σ constant. Levels 3–4 alive.
- **MERA**: monotonic L2-norm decay finest→coarsest, 100× drop. Coarse RNVPs near-dead.
- σ-law at fine levels: R²(|mean z_slow|) ≈ 0.6 → **σ decreases with local ordering**,
  a physically-interpretable susceptibility-like law.

### 6. VP penalty at MATCHED compute reaches nr=2 baseline quality (2026-07-09 finding)

The training-target L includes the penalty; use pure MLE **F** for fair
comparison. On F:

- **Shared HCG at small λ (1e-5)**: penalty ≈ 0.15 nat, F=1891.99 vs
  baseline 1891.10 → **only 0.9 nat cost** for restoring some Jacobian
  discipline. Essentially free at this strength.
- **Fixdil per-scale + VP-1e-3 (nr=1)**:
  - L=32: F=1896.20 → matches D nr=2 (F=1896.16) **within 0.04 nat**.
  - L=64: F=7580.44 → **beats D nr=1 (7604.32) by 24 nat**, only 2.2 nat
    behind D nr=2 (7578.27).
  
  So at nr=1 compute (half of D nr=2), VP-regularized fixdil closes the
  gap to D's nr=2 quality. VP acts as a productive regularizer that
  extracts the extra headroom fixdil left on the table.

Note this reverses my earlier reading (before I noticed L≠F): the "VP hurts
LOSS" claim on L was inflated by the penalty itself. On F, VP is nearly
free for shared and *actively helpful* for fixdil.

## Rankings by pure MLE F (fair cross-variant metric)

**L=32 T_c:**

1. HCG shared nr=1 (**F=1891.10**) ← champion
2. HCG nodilate + init-shared + Adam nr=2 (**F=1891.83**)
3. HCG shared + VP-1e-5 nr=1 (F=1891.99) — cheap VP essentially free
4. HCG shared + VP-1e-4 nr=1 (F=1892.44)
5. HCG shared + VP-1e-3 nr=1 (F=1893.52)
6. HCG shared nr=2 (F=1894.47)
7. HCG shared + VP-1e-2 nr=1 (F=1895.30)
8. HCG perscale fresh nr=2 (F=1895.59)
9. **D (i2 single-CNN) nr=2** (**F=1896.16**) ← Phase-2 reference
10. HCG fixdil + VP-1e-3 nr=1 (**F=1896.20**) — ties D at half compute
11. HCG fixdil + VP-1e-5 nr=1 (F=1896.54)
12. HCG fixdil + VP-1e-4 nr=1 (F=1897.06)
13. HCG fixdil + VP-1e-2 nr=1 (F=1897.21)
14. HCG perscale nodilate E2 nr=1 (F=1898.51)
15. A (Gaussian) baseline nr=1 (F=1899.32)
16. HCG perscale fixdil nr=2 (F=1899.75)
17. HCG perscale fresh nr=1 (F=1900.67)
18. HCG perscale fixdil nr=1 (F=1903.97)

**L=64 T_c:**

1. HCG shared nr=2 (**F=7576.82**) ← champion
2. **D (i2 single-CNN) nr=2** (**F=7578.27**) — 1.45 nat behind shared
3. HCG fixdil + VP-1e-3 nr=1 (**F=7580.44**) — beats D nr=1 by 24 nat!
4. HCG fixdil + VP-1e-2 nr=1 (F=7584.53)
5. HCG perscale nodilate E1 continuation nr=2 (F=7585.58)
6. HCG fixdil + VP-1e-4 nr=1 (F=7586.86)
7. HCG perscale nodilate nr=2 (F=7587.17)
8. HCG shared nr=1 (F=7589.96)
9. HCG perscale fixdil nr=2 (F=7597.41)
10. HCG perscale nodilate nr=1 (F=7599.79)
11. A (Gaussian) baseline nr=1 (F=7600.73)
12. **D (i2 single-CNN) nr=1** (F=7604.32)
13. HCG perscale fresh nr=2 (F=7608.92)
14. HCG shared progdil [1,2,4] nr=2 (F=7609.66)
15. HCG perscale fresh nr=1 (F=7614.17)
16. HCG perscale fixdil nr=1 (F=7627.24)
17. HCG shared progdil [1,4,16] nr=2 (F=7630.75)

## Rankings by interpretability (calibration + structural fidelity)

Independent of LOSS — how "honest" is each variant about its physics?

1. **Fixdil + VP** — Conv0 alive at every level, VP constrains MERA
   Jacobian, σ closer to physical variance
2. **Fixdil baseline** — Conv0 alive at every level, but MERA cheats
3. **D (i2 single CNN)** — no per-scale pretense, MERA does all the work
4. **Nodilate** — Conv0 dies at coarse (level-specialization by accident)
5. **Shared** — 1 CNN, coarse levels degenerate to constant σ via
   context density, calibrated worst

## Open questions

- **L=32 fixdil + VP nr=2 (running)**: does the fixdil+VP win at nr=1
  transfer to nr=2 or does D still dominate at matched-compute nr=2?
- **L=64 fixdil + VP nr=2 from-scratch — closed (2026-07-13)**: Best-200
  shows all three from-scratch nr=2 fixdil+VP arms are ~30-115 nat WORSE
  than the nr=1 champion. The earlier "F=7541 nr=2 champion" claim was a
  ~3σ batch-noise dip (σ ≈ 47 nat/epoch at batch=16). Two follow-ups
  launching to distinguish "capacity mismatch" from "compute-limited":
  - `initshared` arm — warm-start nr=2 fixdil+VP from shared nr=2's basin
  - `fromnr1` arm — warm-start nr=2 from the nr=1 champion checkpoint via
    identity-init doubling (rep-1 blocks zero'd s/t + ScalableTanh)
- **Physical observables**: do variants that share similar F score
  differently on Binder cumulant U₄, susceptibility χ, correlator η?
  (Tier 1 sample-observables analysis running on CPU.)
- Preliminary Tier 1 finding: **HCG shared nr=1 (LOSS champ) has U₄=0.649
  vs GT 0.612**, whereas plain-Gaussian nr=1 (LOSS worse) has U₄=0.626 —
  the physically better model may not be the LOSS-best model.

## 2026-07-13 additions: init-shared and from-nr=1 warm-starts

**Motivation.** Cold-start nr=2 fixdil+VP arms all ranked BELOW the nr=1
champion on Best-200 (7690-7716 vs 7658). Two possible explanations:
1. **Capacity mismatch** — extra rep depth doesn't help this task; nr=1
   is enough at this hyperparameter setting.
2. **Compute-limited** — nr=2 needs a better initialization; from-scratch
   nr=2 hasn't found the champion basin yet.

The two new arms discriminate these:

### A. `initshared` — fixdil nr=2 + VP init from shared nr=2
Uses existing `-hcgInitFromShared` plumbing (`main.py:71`):
- Copies shared nr=2's MERA + Symmetrized weights (10.9 M params)
- Duplicates shared CNN into each per-scale CNN slot
- Expands shared Adam moments to all per-scale slots
- Then runs with `-volumePreservingWeight ${λ}` added to loss

Two VP strengths queued: 1e-4, 1e-3. Shell: `shell/vp_l64_fixdil_nr2_initshared.sh`.

**Predicted outcome**: If nr=2 is compute-limited, initshared should reach
or beat the shared nr=2 Best-200 (7590.52) and possibly the nr=1 champion
(7658.61). If it plateaus at 7700+ like cold-start, capacity is the issue.

### B. `fromnr1` — fixdil nr=2 + VP-1e-3 doubled from the nr=1 champion
New converter `analyzers/convert_nr1_to_nr2_saving.py` doubles the MERA
layerList from 2·S to 4·S blocks:
- `nr=1 layer X` → `nr=2 rep-0 slot 4·(X//2) + X%2` (copy weights)
- `nr=2 rep-1 slots` → identity-init (zero final Linear of every t/s
  MLP + zero ScalableTanh scale)

Verified end-to-end: converted nr=2 model's forward pass produces
**bit-identical** output to the nr=1 model on the same input at ep 0.
Then training continues with fresh Adam. Shell: `shell/vp_l64_fixdil_nr2_from_nr1.sh`.

**Predicted outcome**: Best-200 starts at ≈ 7658.61 (the nr=1 champion
value) and monotonically improves (or degrades if the fresh Adam warm-up
kicks it out of basin). Cleanest test of "does depth help this task."

## Future directions — non-factorized conditionals

The current HCG uses a **factorized Gaussian** conditional:
```
p(z_fast | z_slow) = ∏_i N(z_fast[i]; μ_i(z_slow), σ_i²(z_slow))
```
Given z_slow, each fast site is independent. At T_c this is a poor
approximation — critical correlations couple fast sites at the same scale,
and the factorized model can't represent that. MERA's Jacobian silently
absorbs the fast-fast correlation via bookkeeping (finding #4). Non-factorized
conditionals would give the CNN a way to represent this structure directly,
reducing the "MERA cheating" burden.

Four options, in increasing expressivity and cost:

### A. Multivariate Gaussian with full or low-rank covariance
```
p(z_fast | z_slow) = N(z_fast; μ(z_slow), Σ(z_slow))
```
Extra CNN output: a Cholesky factor `L ∈ R^{N_fast × k}` (low-rank
approximation, k ~ 4–8) so that `Σ = L L^T + diag(σ²)`. Captures 2-point
fast-fast correlations. Sampling and log-prob remain O(N k²) — cheap.

**Recommended first try** — smallest architecture change, addresses the
2-point critical correlations that cause most of the miscalibration.

### B. Autoregressive conditional
```
p(z_fast | z_slow) = ∏_i p(z_fast[i] | z_fast[<i], z_slow)
```
Order the fast sites (raster scan). Each CNN takes `[z_slow, z_fast[<i]]`
as input. Captures **all** orders of correlation.

**Downside:** sequential sampling (slow inference), depends on site
ordering, more code.

### C. Conditional normalizing flow on the fast subspace
Add a small conditional RNVP (or NSF) parameterized by `z_slow`:
```
u ~ N(0, I) → z_fast = f(u; z_slow)
```
Fully expressive multi-point structure. Cost: nested flow inside HCG,
~2× training compute.

### D. Copula-based
Separate marginals `p(z_fast[i] | z_slow)` from correlations, coupled via a
Gaussian copula on CDF-transformed values. Rarely used in this context,
more complex to implement correctly.

### Testable prediction

If the "MERA does geometry, CNN does slow↔fast" decomposition (previous
section) is right, then adding **fast↔fast** capacity via option A should:
1. Improve calibration ratio (currently 0.03–0.10) → closer to 1
2. Lower F further, potentially bridging the gap to nr=2 baselines at
   nr=1 compute
3. Especially help at T_c, less so at T ≠ T_c (correlations weaker)

If option A does NOT help, the bottleneck is elsewhere (probably MERA
expressivity or the multi-scale factorization itself), and cranking up
options B/C won't help either.

### Interaction with VP penalty

Currently VP + factorized CNN closes ~half the gap to D nr=2 at nr=1
compute (finding #6). VP + multivariate CNN might close the rest —
because the fast-fast structure MERA was covertly encoding could migrate
into the CNN's `Σ` (where it belongs physically), letting MERA fully
volume-preserve without losing model quality.

Suggested experiment sequence:
1. Implement option A (low-rank multivariate Gaussian) — ~50 LOC in
   `source/hierarchical_conditional_gaussian.py`
2. Sweep at L=32 nr=1 with VP-1e-3, expect F ≈ 1893–1894 (matching
   shared baseline)
3. If A works, extend to L=64 and check whether it beats shared nr=2

## References
- `project_hcg_nodilate_beats_fixdil` — fixdil vs nodilate comparison
- `project_resume_optimizer_state` — Adam state importance for -load
- `project_l32_bignet_fix` — capacity is the L=32 bottleneck, not inductive bias
- `project_l32_late_training_instability` — LOSS drift envelope caveat
- `feedback_nodilate_not_winner` — framing convention (mechanism, not ranking)
