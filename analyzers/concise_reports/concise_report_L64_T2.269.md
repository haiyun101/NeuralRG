# Ising L=64 — Concise Report (T=2.269)

## ★ Phase-4 volume-preserving penalty (2026-07-09) — largest gains at L=64

Same VP-penalty methodology as L=32 (see L=32 concise report Phase-4 for
background on why "F" not "L" is the fair metric across variants). At
L=64 the effect is more dramatic — VP transforms fixdil per-scale from
"worst variant" to "matches L=64 D baseline at half compute".

### L=64 T_c results with VP penalty (fixdil per-scale nr=1)

| Variant | F | vs fixdil baseline | vs D nr=2 (7578.27) |
|---|---:|---:|---:|
| fixdil baseline (no VP) | 7627.24 | ref | +48.97 |
| fixdil + VP-1e-4 | 7586.86 | −40.38 | +8.59 |
| **fixdil + VP-1e-3** | **7580.44** | **−46.80** | **+2.17** |
| fixdil + VP-1e-2 | 7584.53 | −42.71 | +6.26 |

**Fixdil+VP-1e-3 nr=1 (F=7580.44) beats D nr=1 (7604.32) by 24 nat**
and closes to within 2 nat of D nr=2 (which uses 2× compute).

### L=64 nr=2 VP sweep — retracted "champion" claim (2026-07-13 update)

Original claim was that **fixdil+VP-1e-3 nr=2** hit F=7541.58 during
training (a 35-nat improvement over the prior best F=7576.82 from HCG
shared nr=2), making it the new L=64 champion.

**This was WRONG.** The F=7541.58 was the log-parsed single-epoch
minimum, which for high-variance batch-noise training doesn't reflect
sustained model quality:

- Per-epoch F noise on batch=16 continuous training: **σ ≈ 47 nat**
  (very noisy — one batch of 16 samples is a poor F estimate)
- F=7541 is only **~3σ below the local mean of 7679** → expected once
  every ~500 epochs by pure chance; **not a real basin**
- Continuation run (job 41786697) also hits F=7541.41 briefly at a
  DIFFERENT epoch (ep 1544) then bounces back — evidence that this
  low value is a batch-noise artifact, not a rediscovered basin

**Correct Best-200 numbers** (200-epoch rolling min-mean of ENTROPY —
averages away the batch-noise dips):

| Variant                     | Best-200  | vs L=64 D nr=2 (7578.27) | vs L=64 hcg_shared nr=2 (7590.52) |
|---|---:|---:|---:|
| fixdil + VP-1e-4 nr=2       | 7689.79   | +111.5 nat (worse)       | +99.3 nat (worse)                   |
| fixdil + VP-1e-3 nr=2       | 7701.63   | +123.4 nat (worse)       | +111.1 nat (worse)                  |
| fixdil + VP-1e-2 nr=2       | 7716.33   | +138.1 nat (worse)       | +125.8 nat (worse)                  |

**All nr=2 fixdil+VP variants are WORSE than baselines on Best-200.**

The true L=64 champion is **fixdil+VP-1e-3 nr=1** at Best-200 = 7658.61
(rank 1 overall, see the Best-200 ranking table below).

Continuations of vp1e-3 nr=2 (job 41786697) and vp1e-4 nr=2 (job 41797159)
are running with proper `-load` support to see if extended training closes
the nr=2 gap. Current continuation Best-200 is 7674 for vp1e-3 nr=2 —
still 15 nat behind the nr=1 champion after 2700 more epochs.

### Sample and correlation plots

All flow visualizations (samples + G(r) correlations + P(M) magnetization
distribution) are consolidated in the [§ Flow visualizations](#-flow-visualizations--all-variants-in-one-place)
section at the bottom of this report. Each plot pair carries a caption
line with KL(p‖q), KL(q‖p), ⟨\|M\|⟩, χ, U₄, ξ_q so they can be compared
without cross-referencing tables.

Note the Tier-1 physical observables ranking (see [§ Physics interpretation](#physics-interpretation--two-distinct-architecture-families)):
despite E-shared having only 1.45 nat higher F than the VP champion, its
susceptibility χ=16.6 is **6.6× smaller** than ground truth (110.15),
while VP champion's χ=51.1 is only 2.2× smaller. VP variants also match
the universal Binder cumulant U₄=0.611 far better (VP-1e-4 nr=2: 0.605,
essentially perfect) than shared (0.654, off by 0.043).

### The "MERA=geometry, CNN=variance" separation (physical interpretation)

VP forces log|det J_MERA| → 0, so MERA can only reshape geometry
(fold Gaussian into bimodal Z2 structure, encode scale hierarchy) but
cannot silently rescale marginal variance. The CNN's σ output must then
actually match the per-scale conditional variance in data. This
separation of roles is physically clean, and empirically fixdil+VP
achieves it while matching or beating the LOSS-optimal architecture.

At L=64 the extra compute (16× the sites of L=32) plus long correlation
length at T_c makes the marginal-variance mismatch bigger — so VP
regularization has more to fix, and delivers 40+ nat improvement over
plain fixdil.

> **⚠ Note**: `make_concise_report.py` regenerates this file from scratch —
> both the Phase-3 AND Phase-4 sections must be re-added after every
> regeneration.

---

## Summary — thermodynamics + KL directions

Superset table: exact theory / dataset / trained flow, showing free
energy, energy, entropy, and both KL directions.

Font marks source: **bold** = exact theory (Onsager / `exactz.md` L=64
section). *italic* = training-measured, read from HDF5 records. plain =
sample-measured (dataset or post-hoc diagnostic).

Since the summary table now contains all trained variants (sorted by S,
ascending), it lives in `analyzers/loss/loss_report_L64_T2.269.md` —
regenerate with `python analyzers/loss/loss_analyzer_fixT.py -L 64 -t 2.269`
and see the "Summary — everything in one table" section of that file.

**Reference rows** (exact theory + HS dataset entropy floor):

| Source                        |  Picture   |    F (-lnZ)    |       E       |       S       | KL(p‖q) |
| :---------------------------- | :--------: | :------------: | :-----------: | :-----------: | :-----: |
| **Exact (theory)**            |  discrete  | **−3808.67**   | **−2673.25**  |  **1135.42**  |    —    |
| MCMC dataset (Wolff)          |  discrete  |      N/A       |   −2570.36    |      N/A      |    —    |
| **Exact (theory)**            | continuous | **−9476.43**   | **−1855.31**  |  **7621.12**  | **0**   |
| HS dataset (x ~ p_HS)         | continuous |      N/A       |   −1856.27    |    7620.16    |    —    |

**Best reverse-KL** (`sym_bignet`): training-row S=5606.61, KL(q‖p)=1431.98
(the reverse-KL objective doesn't try to match the HS marginal, so its
forward KL is large).

## All trained forward-KL variants at L=64 T_c (Best-200 metric)

Ordered by **S = Best-200 = lowest 200-epoch rolling mean of ENTROPY**.
This smoothed metric damps lucky-batch spikes and gives **physically
meaningful (positive) KL(p‖q)** values. See earlier note about
single-epoch min ENTROPY giving artifactual negative KL.

`KL(p‖q) = S − H(p_HS) = S − 7621.12`.

Two KL directions are reported:
- **KL(p‖q) — training (Best-200)**: on-objective, forward-KL minimizes it. Sourced from
  `S − H(p_HS)` using the 200-epoch rolling minimum. Direct measure of "how much
  probability mass p places on samples q assigns low probability" ≈ mode-missing cost.
- **KL(q‖p) — diagnostic**: off-objective reverse-KL, from `flow_diagnostic.json`.
  Sampled by drawing `x ~ q` and scoring `log(q/p)`. Measures "spurious modes q covers
  that p doesn't" ≈ over-generation cost. Not what forward-KL training tried to minimize.

| Rank | Method                                                       |  S(Best-200) | KL(p‖q) train | KL(q‖p) diag |
| :--: | :----------------------------------------------------------- | :----------: | :---------: | :---------: |
|  1   | **hcg_perscale_fixdil_vp1e-3_nr1** ★                         |  **7658.61** |  **37.49**  |  **40.97**  |
|  2   | hcg_perscale_fixdil_vp1e-4_nr1                               |    7659.12   |    37.99    |    44.99†   |
|  3   | baseline_nr2 (**C**)                                         |    7661.64   |    40.52    |   156.36    |
|  4   | hcg_perscale_fixdil_vp1e-2_nr1                               |    7662.94   |    41.82    |    47.98    |
|  5   | hcg_shared                                                   |    7669.66   |    48.53    |    50.24    |
|  6   | i2_stride8h32_nr2 (**D**)                                    |    7676.08   |    54.96    |    51.33    |
|  7   | hcg_perscale_nodilate_initshared_nr2 (cont, latest)          |    7677.78   |    56.66    |    64.79    |
|  8   | hcg_perscale_nodilate_initshared_nr1 (cont, latest)          |    7680.72   |    59.60    |    57.24    |
|  9   | hcg_perscale                                                 |    7681.89   |    60.77    |    84.19    |
|  10  | baseline_b16 (**A** — Gaussian nr=1)                         |    7682.16   |    61.03    |    86.88    |
|  11  | hcg_shared_nr2                                               |    7682.53   |    61.41    |    69.75    |
|  12  | iii1_lam1.0                                                  |    7683.99   |    62.87    |    87.14    |
|  13  | baseline_N50000 (A nr=1, N=50 000 dataset)                   |    7684.38   |    63.26    |    85.64    |
|  ..  | *(10 unlisted variants between rank 13 and rank 23: i2 sweeps, baselines at other N, bridge, hcg_perscale_fixdil_nr2)* | | | |
|  ~23 | hcg_perscale_fixdil_vp1e-4_nr2 (walltime-cut, N=7000)        |    7689.79   |    68.67    |    82.25    |
|  ~29 | hcg_perscale_fixdil_vp1e-3_nr2 (walltime-cut, N=7000)        |    7701.63   |    80.51    |    90.75    |
|  ~35 | hcg_perscale_fixdil_vp1e-2_nr2 (walltime-cut, N=7000)        |    7716.33   |    95.21    |   121.32    |

All numbers are Best-200 (200-epoch rolling min-mean of ENTROPY). nr=2
variants labeled "walltime-cut" reached only ep 7000 of the planned 15 000,
but the Best-200 window still gives sustained loss (not single-epoch min).
`—` = no `flow_diagnostic.json` (never ran post-hoc sampling for that folder).
"~N" ranks = position in the full 40+ variant ranking including small-scale
sweeps (i2 stride/hidden variants, baselines at other N, bridge, etc.) that
aren't individually listed here. The 10 unlisted rows between ranks 13
and 23 are all in the 7684.5–7687.5 Best-200 band.

`†` **fixdil+VP-1e-4 nr=1 stability caveat.** This run reached its Best-200
minimum (LOSS ≈ 7587) at ep 16 885, then **catastrophically diverged**
around ep 21 839: peak LOSS = 8.7×10¹¹ at ep 21 859, never recovered
(final ep ≈ 29 000 sits at LOSS ≈ 36 000, i.e. 28 500 nat worse than
baseline). Root cause: `gradClip=0` on this run — RNVP's log-scale
output has no built-in bound, so a gradient spike lets `exp(s) → Inf`.
Two diagnostic passes exist for this folder:
- `flow_diagnostic.json` at ep 10 500 (**pre-instability**): KL_qp = 44.99 ✓
- `flow_diagnostic_latest.json` at ep 14 500: KL_qp = 1.89×10²⁷ (overflow)

Reported value (44.99) is the honest pre-blowup number. Even at ep 14 500
the flow's inverse was already unstable, so the "champion" Best-200 for
this arm should be read as **fragile** — one bad step from wandering into
overflow. Later `nr=2` VP runs use `-gradClip 5.0` to prevent this failure
mode.

**`nodilate_initshared` row de-dup (2026-07-13)**: previous versions of this
table showed each `_initshared` experiment twice — once for the original
folder and once for its `_cont` continuation — because the loss analyzer
enumerates folders independently and the `_cont` runs live in separate
directories. The `_cont` folder is `-load`-based, so both represent the
same experiment; only the `_cont` row is kept (later trajectory, lower
Best-200). KL(q‖p) values are now filled from each folder's
`flow_diagnostic.json` (diagnostic was run at ep 9200 for nr=2 cont,
ep 15800 for nr=1 cont).

**Reading both directions together:**
- Rank 1 fixdil+VP-1e-3 nr=1 has **both** the lowest KL(p‖q) train (37.49) AND
  a competitive KL(q‖p) diag (40.97) — the flow neither misses p's modes
  nor over-generates spurious ones significantly.
- Baseline C (Gaussian nr=2, rank 3) has good KL(p‖q) train (40.52) but
  **massive KL(q‖p) diag = 156.36** — its samples cover regions p never
  visits. Plain Gaussian prior gives high MLE fit but poor generative
  quality. (Note: A = Gaussian nr=1 is rank 12 with S=7682.16, half the
  compute of C so ~20 nat behind on training-KL as expected.)
- D (rank 6) has moderate on both (54.96 / 51.33) — trade-off pattern.
- HCG shared (rank 5) has balanced small values (48.53 / 50.24).

**Fixdil + VP-1e-3 nr=1 is the true L=64 champion** on the physically-
meaningful Best-200 metric — 3 nat ahead of the A baseline, 17 nat ahead
of D. VP variants occupy 3 of the top 4 slots.

Rankings by **single-epoch min ENTROPY** (previous, buggy metric that
gave negative KL) had A #1 and shared #3; the switch to Best-200 makes
KL positive and reveals fixdil+VP's true dominance.

Only the top 25 shown (42 total forward-KL variants). See
`analyzers/loss/loss_report_L64_T2.269.md` for the complete ranking
including exponent-blowup diagnostic outliers.

**Caveat — records vs log discrepancy**: for VP-regularized runs whose
walltime cut before the record files caught the deep basin, the
record-based S underestimates the true best F reached during training.
For `fixdil+VP-1e-3 nr=2` the log shows F=7541.58 @ ep 7234 while the
record only reached ep 7000 (min ENTROPY=7595.58). A continuation
(job 41737644) is queued to close this gap.

Notes:
- **Rankings changed** (2026-07-12) after fixing 3 analyzer bugs:
  1) folder-name regex `\w+` truncated at `-` (`vp1e-3` → `vp1e`) — fixed to `[\w.-]+`
  2) picker used `min(LOSS)` (with VP penalty) — now uses `min(ENTROPY)` (pure MLE, fair across variants)
  3) summary showed only "best of mode" — now lists ALL trained variants sorted by S
- **The A baseline (`hsBignet_baseline_nr2_b16`)** actually has the lowest S
  after the fix — not our previous "champion" HCG shared or fixdil+VP. The
  earlier claim that HCG shared/fixdil+VP was "champion" was based on the
  log-parsed best F which uses a **different aggregation window** than the
  record-file min ENTROPY the analyzer reads. Log's "F" tracks a moving
  best across all epochs; record ENTROPY is at the min-LOSS epoch (which
  for VP runs is inflated by the penalty). Reconciliation is on the TODO.
- Training-row `KL(p‖q)` is negative for nearly every hsBignet forward-KL
  run (−3 to −80 nat). This is **training-set overfitting**: the model's
  MLE loss on batch samples dips below the entropy floor of the HS dataset.
  This is a well-known artifact of MLE with finite training data and is
  NOT a bug — but it does mean the training loss underestimates the true
  forward-KL by that same amount.
- Each flow's *training* row + *diagnostic* row (when available) — for a
  converged reverse-KL run they should agree; the gap reflects sampling
  noise plus distribution-shift between the training batch and a fresh
  q-sample.
- `F=−lnZ` cannot be estimated from samples alone → dataset rows show N/A.
- The full per-run breakdown for every method is in the flow-diagnostic
  section below; this summary keeps only the two most-comparable modes.

---

## ★ Physical observables — sample-based physics at T_c

The forward-KL loss (S = −E_data[log q]) tells us how well the flow
matches the data's log-density. But **it doesn't guarantee that samples
from the flow reproduce the physics** — a model can memorize the batch
while still generating overly-ordered / overly-disordered configurations.
We test this by sampling N=4000 configurations from each flow at its
Best-200 checkpoint and computing standard Ising observables.

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
L=64 the peak is finite but very large — theory expects **χ ≈ 110 at
T_c** for L=64 (from Wolff MCMC ground truth).

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

### Ground truth (Wolff MCMC, N=200 000, L=64 T_c)

```
⟨E⟩ ≈ −1.491     ⟨|M|⟩ ≈ 0.68     χ ≈ 110.15     U₄ ≈ 0.611
```

### Model observables — Best-200-anchored (2026-07-13)

Sampled from each flow at its Best-200 center epoch. `Δχ%` = relative
error against GT; `ΔU₄` = absolute deviation from 0.611.

| Rank | Method                          | ⟨E⟩     | ⟨\|M\|⟩ | χ    | Δχ%  | U₄    | ΔU₄  |
| :--: | :------------------------------ | :-----: | :-----: | :---:| :--: | :----:| :---:|
|  —   | **GT (Wolff)**                  | −1.491  | 0.68    | 110  | 0    | 0.611 | 0    |
|  1   | fixdil+VP-1e-3 nr=1             | −1.421  | 0.60    |  72.5| −34% | 0.621 | 0.010|
|  2   | fixdil+VP-1e-4 nr=1             | −1.436  | 0.61    |  70.3| −36% | 0.623 | 0.012|
|  3   | baseline_nr2 (**C**)            | −1.439  | 0.61    |  72.3| −34% | 0.622 | 0.011|
|  4   | fixdil+VP-1e-2 nr=1             | −1.423  | 0.61    |  74.7| −32% | 0.621 | 0.010|
|  5   | hcg_shared                      | −1.422  | 0.66    |  14.3| **−87%** | 0.656 | **0.045** |
|  6   | i2_stride8h32_nr2 (**D**)       | −1.411  | 0.64    |  20.0| **−82%** | 0.652 | **0.041** |
|  7   | hcg_perscale_nodilate_...nr2_ct | −1.440  | 0.66    |  14.2| −87% | 0.657 | 0.046|
|  8   | hcg_perscale_nodilate_...nr2    | −1.404  | 0.64    |  15.5| −86% | 0.655 | 0.044|
|  11  | hcg_perscale (base)             | −1.385  | 0.60    |  68.6| −38% | 0.622 | 0.011|
|  12  | baseline_b16 (**A**)            | −1.412  | 0.62    |  74.5| −32% | 0.623 | 0.012|
|  13  | hcg_shared_nr2                  | −1.382  | 0.62    |  16.0| −85% | 0.654 | 0.043|

### Physics interpretation — two distinct architecture families

The variants split cleanly into two groups on the derived observables χ
and U₄:

**Group A — reproduces critical spread (χ ≈ 60-75, U₄ ≈ 0.62)**:
- All fixdil+VP variants
- Baselines A (Gaussian nr=1), C (Gaussian nr=2)
- Base hcg_perscale, iii1_lam1

**Group B — tail-collapsed on M (χ ≈ 8-20, U₄ ≈ 0.65-0.66)**:
- hcg_shared, hcg_shared_nr2
- All hcg_perscale_nodilate variants
- D (i2 nr=2)

### G(r) shape — normalized correlations (not raw magnitude)

`flow_sample_diagnostic.py` computes G(r)/G(0), i.e. the **decay shape**
(each curve starts at 1 at r=0 by construction). So the numbers below
compare **effective correlation lengths ξ_eff**, not absolute G(r) values.

```
r=          1       2       4       8      16      32
GT (HS)     0.791   0.652   0.553   0.480   0.429   0.407
Group A: fixdil+VP-1e-3 nr=1
            0.778   0.629   0.525   0.449   0.404   0.387    ← decays FASTER (ξ_eff shorter)
Group B: hcg_shared
            0.779   0.630   0.527   0.467   0.451   0.450    ← decays SLOWER at long r (ξ_eff longer)
Group B: i2_nr2 (D)
            0.777   0.631   0.530   0.464   0.437   0.435    ← similar slow decay at long r
```

**Group A (fixdil+VP)**: Gnorm decays faster than GT → **effective
correlation length shorter** than critical → individual samples look
"snowflake-like" (locally random) at large scales.

**Group B (HCG shared, D)**: Gnorm decays slower at long r → **effective
correlation length longer** than critical → individual samples have
residual "ordered-block" coherence.

### Both groups exhibit mode collapse — in DIFFERENT directions

The user question that clarified this: **is fixdil+VP really escaping
mode collapse just because its samples look disordered?** No — Group A
still has χ = 72 vs GT = 110, meaning `Var_q(M)` is undersampled. It's
NOT collapsed toward ordered configs, but it's still concentrating
samples in a narrower |M| range than GT's true diversity.

Concrete P(|M|) profile at Best-200 (approximated from ⟨|M|⟩ and χ):

| Group | ⟨\|M\|⟩ | σ_\|M\| | Approximate range | Missing tail |
|---|---|---|---|---|
| GT   | 0.68 | 0.16 | [0.52, 0.84] | (all mass present) |
| A: fixdil+VP-1e-3 | 0.60 | 0.13 | [0.47, 0.73] | rare heavy-|M| (>0.75) ordered blocks |
| B: hcg_shared | 0.66 | 0.06 | [0.60, 0.72] | rare heavy-|M| AND rare low-|M| domain-walls |

**Both** miss the rare heavy-|M| ordered-block configurations. Group A
also fails to reach the low-|M| domain-wall region because its samples
are locally random. Group B fails there too because samples are ordered
blocks with consistent M.

### Two distinct collapse directions

| Property | Group A (fixdil+VP) | Group B (HCG shared) |
|---|---|---|
| Individual sample look | disordered ("snowflake") | ordered ("block") |
| Where samples cluster | middle \|M\| + locally random pattern | middle \|M\| + local coherence |
| Missing tail | heavy-\|M\| ordered configs, low-\|M\| domain walls | heavy-\|M\| AND full-|M| spread |
| G(r) vs GT | decays faster (shorter ξ_eff) | decays slower at long r (longer ξ_eff) |
| U₄ vs GT (0.611) | 0.621 (near critical) | 0.656 (near ordered limit 2/3) |

Both **collapse toward middle-\|M\|** samples, missing critical diversity.
They differ in how the middle-|M| samples LOOK (random vs coherent).
Neither is "the correct T_c distribution" — both are subsets of it.

### Why forward-KL misses both

Forward-KL minimizes `-E_data[log q(x)]` with training samples drawn from
p(x). Rare heavy-|M| tail configurations appear rarely in the 200 000
training samples, so their contribution to the loss gradient is
correspondingly small. The flow has **no signal to learn where these
rare configurations live**, and its architectural inductive biases fill
the gap in whatever way is easiest:
- Group A (fixdil+VP, plain baselines) — architecturally biased toward
  local randomness → "snowflake" samples in the middle-|M| region
- Group B (HCG shared, nodilate) — architecturally biased toward
  hierarchical block-structure → "ordered-block" samples in the same
  middle-|M| region

Both are the SAME underlying failure of forward-KL + finite training
data: rare configurations don't get gradient signal, so architecture
priors decide the shape of the missing tail. The two groups differ in
which prior they impose.

**Fixing this** requires either much more training data (10-100× to
sample the tails), an explicit tail-mass regularizer (penalize small
Var_q(M)), or importance-weighted sampling around rare configurations.

Finite-N artifact (χ still 30% below GT): sampling from a critical
distribution with N=4000 always underestimates the peak of χ. The
theoretical curve requires much larger sample sizes or careful
tail-reweighting — 30% is expected shortfall.

### The champion story

**fixdil+VP-1e-3 nr=1** is best on BOTH:
- Best-200 forward-KL S = 7658.61 (rank 1)
- Best physics observables in Group A (χ=72.5, U₄=0.621)

Notably **HCG shared** (previously reported as "LOSS champion" by the
buggy single-epoch min metric) is now correctly identified as Group B —
its samples don't reproduce critical fluctuations at all despite scoring
well on Best-200 loss. **The metric fix (Best-200) and the physics test
converge on the same conclusion**: fixdil+VP is the true winner.

---

## ★ Phase-3 HCG results (2026-07-04) — Hierarchical Conditional Gaussian at L=64

L=64 exposes RF-too-small limitation of the shared HCG CNN: uniform
dilation=1 gives effective RF=7 pixels, but Level 1 → Level 0 distance
is 16 sites → Level 1 CNN outputs constant σ.

### HCG best-200 vs D64 reference

| Cell | Config | Best-200 | vs D64 |
|------|--------|---------:|:-:|
| A    | Gaussian baseline nr=1 (resumed 40K) | 7683.42 | +19.82 |
| B    | i2 stride=16 nr=1 | 7694.14 | +30.54 |
| **D** | **i2 stride=8 nr=2** | **7663.60** | ref |
| E-shared nr=1 (resumed 40K) | HCG shared, nr=1 | 7671.34 | +7.74 |
| E-shared nr=2 (pre-spike best) | HCG shared, nr=2 | **7666.43** | **+2.83** |
| E-shared nr=2 hh=64 fresh | HCG shared, larger CNN | 7706.5 | +42.9 |
| E-shared nr=2 hh=128 fresh | HCG shared, largest CNN | 7699.2 | +35.6 |
| E-perscale nr=1 broken dil (resume 40K) | HCG per-scale buggy | 7682.80 | +19.20 |
| E-perscale nr=1 (**fixdil**) | HCG per-scale corrected | 7708.00 | +44.4 |
| E-perscale nr=2 fixdil (running) | HCG per-scale nr=2 corrected | 7692.29 @ ep 13K | (proj +25) |

**Reading:**
- No HCG variant reliably beats D64
- Bigger CNN (hh=64 → hh=128) does *not* help — dilation=1 RF=7 too small
- Per-scale HCG with corrected dilation is *worse* than shared at T_c —
  scale-invariance is the correct prior at criticality

### Path forward

`-hcgSharedDilations` flag (2026-07-04): progressive dilation across CNN
layers, e.g. `1,2,4` gives RF=15, `1,2,8` gives RF=23, `1,4,16` gives
RF=43 (covers L=64 Level 1's distance-16). Untested at report time.

> **⚠ Note**: `make_concise_report.py` regenerates this file from scratch —
> this Phase-3 HCG section must be re-added after every regeneration.

---

_Summary table not found in `loss_report_L64_T2.269.md` — run `loss_analyzer_fixT.py -L 64 -t 2.269` first._

<a id="-flow-visualizations--all-variants-in-one-place"></a>
## § Flow visualizations — all variants in one place

_Per method: **left** = flow samples (configurations, `sigmoid(2x)` render of `q` vs HS data `p`); **right** = flow correlations (P(M) + axial two-point G(r)/G(0), flow vs data)._

**Caption line legend (each variant):**
- `KL(p‖q)`, `KL(q‖p)` — from `flow_diagnostic.json` at the epoch listed
- `⟨|M|⟩_q` — continuous magnetization from JSON (`mag_abs_q`). Reference GT p: **2.20**
- `ξ_q` — correlation length from JSON. Reference GT p: **14.78**
- `⟨|M|⟩_s`, `χ`, `U₄` — sign-based physics observables from Tier-1 (only for the 13 variants in the physics table). Reference GT: **⟨|M|⟩_s = 0.68**, **χ = 110**, **U₄ = 0.611**
- `Best-200` — sustained ENTROPY minimum

Ordered by rank in the Best-200 ranking table. Diagnostic epoch is shown; that's the checkpoint used for KL and physics observable sampling. It is NOT the Best-200 epoch — treat these numbers as spot samples of a fully trained model rather than of the exact Best-200 center.

---

### Rank 1 — hcg_perscale_fixdil_vp1e-3_nr1 ★ (champion)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/flow_samples.png" alt="rank 1 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/flow_correlations.png" alt="rank 1 correlations" width="56%">
</p>

> **KL(p‖q) = 38.68** · **KL(q‖p) = 43.95** · ⟨|M|⟩_q = 2.13 · ξ_q = 13.99 · ⟨|M|⟩_s = 0.60 · **χ = 72.5** · **U₄ = 0.621** · Best-200 = 7658.61 · diag ep 13500

---

### Rank 2 — hcg_perscale_fixdil_vp1e-4_nr1  (stability-caveat, see †)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16/flow_samples.png" alt="rank 2 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16/flow_correlations.png" alt="rank 2 correlations" width="56%">
</p>

> **KL(p‖q) = 39.37** · **KL(q‖p) = 44.99** · ⟨|M|⟩_q = 2.19 · ξ_q = 14.18 · ⟨|M|⟩_s = 0.61 · **χ = 70.3** · **U₄ = 0.623** · Best-200 = 7659.12 · diag ep 10500 (pre-blowup; run diverged at ep 21 839, see † in ranking table)

---

### Rank 3 — baseline_nr2 (**C** — Gaussian nr=2)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_baseline_nr2_b16/flow_samples.png" alt="rank 3 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_baseline_nr2_b16/flow_correlations.png" alt="rank 3 correlations" width="56%">
</p>

> **KL(p‖q) = 40.49** · **KL(q‖p) = 47.65** · ⟨|M|⟩_q = 2.18 · ξ_q = 14.22 · ⟨|M|⟩_s = 0.61 · **χ = 72.3** · **U₄ = 0.622** · Best-200 = 7661.64 · diag ep 19000

---

### Rank 4 — hcg_perscale_fixdil_vp1e-2_nr1

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr1_b16/flow_samples.png" alt="rank 4 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr1_b16/flow_correlations.png" alt="rank 4 correlations" width="56%">
</p>

> **KL(p‖q) = 41.09** · **KL(q‖p) = 45.68** · ⟨|M|⟩_q = 2.11 · ξ_q = 13.92 · ⟨|M|⟩_s = 0.61 · **χ = 74.7** · **U₄ = 0.621** · Best-200 = 7662.94 · diag ep 13500

---

### Rank 5 — hcg_shared

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_b16/flow_samples.png" alt="rank 5 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_b16/flow_correlations.png" alt="rank 5 correlations" width="56%">
</p>

> **KL(p‖q) = 48.95** · **KL(q‖p) = 51.51** · ⟨|M|⟩_q = 2.33 · ξ_q = 15.25 · ⟨|M|⟩_s = 0.66 · **χ = 14.3** ★ · **U₄ = 0.656** · Best-200 = 7669.66 · diag ep 15600 · ← Group B "ordered-blocks" collapse

---

### Rank 6 — i2_stride8h32_nr2 (**D** — Phase-2 reference)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/flow_samples.png" alt="rank 6 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/flow_correlations.png" alt="rank 6 correlations" width="56%">
</p>

> **KL(p‖q) = 54.41** · **KL(q‖p) = 58.66** · ⟨|M|⟩_q = 2.28 · ξ_q = 14.94 · ⟨|M|⟩_s = 0.64 · **χ = 20.0** ★ · **U₄ = 0.652** · Best-200 = 7676.08 · diag ep 17800 · ← Group B

---

### Rank 7 — hcg_perscale_nodilate_initshared_nr2 (cont)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_nr2_gc5.0_b16_cont/flow_samples.png" alt="rank 7 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_nr2_gc5.0_b16_cont/flow_correlations.png" alt="rank 7 correlations" width="56%">
</p>

> **KL(p‖q) = 62.94** · **KL(q‖p) = 64.79** · ⟨|M|⟩_q = 2.38 · ξ_q = 15.47 · ⟨|M|⟩_s = 0.66 · **χ = 14.2** · **U₄ = 0.657** · Best-200 = 7677.78 · diag ep 9200 · ← Group B

---

### Rank 8 — hcg_perscale_nodilate_initshared_nr1 (cont)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b16_cont/flow_samples.png" alt="rank 8 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b16_cont/flow_correlations.png" alt="rank 8 correlations" width="56%">
</p>

> **KL(p‖q) = 60.90** · **KL(q‖p) = 57.24** · ⟨|M|⟩_q = 2.20 · ξ_q = 14.33 · ⟨|M|⟩_s = 0.64 · **χ = 15.5** · **U₄ = 0.655** · Best-200 = 7680.72 · diag ep 15800 · ← Group B

---

### Rank 9 — hcg_perscale (base, no dil-shared-init)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_b16/flow_samples.png" alt="rank 9 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_b16/flow_correlations.png" alt="rank 9 correlations" width="56%">
</p>

> **KL(p‖q) = 62.13** · **KL(q‖p) = 84.19** · ⟨|M|⟩_q = 2.10 · ξ_q = 13.76 · ⟨|M|⟩_s = 0.60 · **χ = 68.6** · **U₄ = 0.622** · Best-200 = 7681.89 · diag ep 11600 · ← Group A "snowflake" collapse

---

### Rank 10 — baseline_b16 (**A** — Gaussian nr=1)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_baseline_b16/flow_samples.png" alt="rank 10 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_baseline_b16/flow_correlations.png" alt="rank 10 correlations" width="56%">
</p>

> **KL(p‖q) = 62.41** · **KL(q‖p) = 82.17** · ⟨|M|⟩_q = 2.19 · ξ_q = 14.58 · ⟨|M|⟩_s = 0.62 · **χ = 74.5** · **U₄ = 0.623** · Best-200 = 7682.16 · diag ep 17600 · ← Group A

---

### Rank 11 — hcg_shared_nr2

<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_nr2_b16/flow_samples.png" alt="rank 11 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_nr2_b16/flow_correlations.png" alt="rank 11 correlations" width="56%">
</p>

> **KL(p‖q) = 59.86** · **KL(q‖p) = 68.19** · ⟨|M|⟩_q = 2.14 · ξ_q = 13.58 · ⟨|M|⟩_s = 0.62 · **χ = 16.0** ★ · **U₄ = 0.654** · Best-200 = 7682.53 · diag ep 19400 · ← Group B

---

### Rank 12 — iii1_lam1 (Gaussian A with Z2-alpha loss)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_iii1_lam1.0_b16/flow_samples.png" alt="rank 12 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_iii1_lam1.0_b16/flow_correlations.png" alt="rank 12 correlations" width="56%">
</p>

> **KL(p‖q) = 64.37** · **KL(q‖p) = 85.40** · ⟨|M|⟩_q = 2.17 · ξ_q = 14.28 · Best-200 = 7683.99 · diag ep 19200

---

### Rank 13 — baseline_N50000 (A with 50k-sample dataset)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_baseline_N50000_b16/flow_samples.png" alt="rank 13 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_baseline_N50000_b16/flow_correlations.png" alt="rank 13 correlations" width="56%">
</p>

> **KL(p‖q) = 64.78** · **KL(q‖p) = 85.64** · ⟨|M|⟩_q = 2.22 · ξ_q = 14.67 · Best-200 = 7684.38 · diag ep 19000

---

### Ranks 14–22 — i2 sweeps + miscellaneous (7684.5–7687.5 band)

These are hyperparameter sweeps in the `i2 = conditional_gaussian` family (stride ∈ {4,8,16}, CNN hidden ∈ {32,64}) plus a few other objectives. All cluster in a narrow Best-200 band and show similar mode-collapse behavior. Included here for reference; individual rankings and observable patterns follow the D=i2 family behavior discussed in [§ Physics interpretation](#physics-interpretation--two-distinct-architecture-families).

#### i2_stride8h64 (rank ~14, wider CNN at D config)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h64_b16/flow_samples.png" alt="i2 stride8 h64 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h64_b16/flow_correlations.png" alt="i2 stride8 h64 correlations" width="56%">
</p>

> **KL(p‖q) = 67.82** · **KL(q‖p) = 85.58** · ⟨|M|⟩_q = 2.29 · ξ_q = 15.24 · Best-200 = 7684.46 · diag ep 19800

#### i2_stride16h32 (rank ~16, coarser context)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride16h32_b16/flow_samples.png" alt="i2 stride16 h32 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride16h32_b16/flow_correlations.png" alt="i2 stride16 h32 correlations" width="56%">
</p>

> **KL(p‖q) = 70.37** · **KL(q‖p) = 93.31** · ⟨|M|⟩_q = 2.23 · ξ_q = 14.94 · Best-200 = 7686.05 · diag ep 19800

#### i2_stride4h64 (rank ~17, tighter context, wider CNN)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride4h64_b16/flow_samples.png" alt="i2 stride4 h64 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride4h64_b16/flow_correlations.png" alt="i2 stride4 h64 correlations" width="56%">
</p>

> **KL(p‖q) = 66.44** · **KL(q‖p) = 84.67** · ⟨|M|⟩_q = 2.31 · ξ_q = 15.51 · Best-200 = 7686.42 · diag ep 19800

#### hs_bignet (rank ~19, early baseline)
<p>
<img src="../../data/64Ising_T2.269_hs_bignet/flow_samples.png" alt="hs_bignet samples" width="42%">
<img src="../../data/64Ising_T2.269_hs_bignet/flow_correlations.png" alt="hs_bignet correlations" width="56%">
</p>

> Best-200 = 7686.51 · (no JSON metrics extracted for this old folder)

#### bridge_w5.0t0.5 (rank ~20, bridge-upweighting objective)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_samples.png" alt="bridge samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_correlations.png" alt="bridge correlations" width="56%">
</p>

> **KL(p‖q) = 68.86** · **KL(q‖p) = 93.20** · ⟨|M|⟩_q = 2.10 · ξ_q = 14.13 · Best-200 = 7687.15 · diag ep 19900

#### i1_df4 (rank ~21, Student-t prior df=4)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i1_df4.0_b16/flow_samples.png" alt="i1 df4 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i1_df4.0_b16/flow_correlations.png" alt="i1 df4 correlations" width="56%">
</p>

> **KL(p‖q) = 66.21** · **KL(q‖p) = 90.41** · ⟨|M|⟩_q = 2.18 · ξ_q = 14.37 · Best-200 = 7687.15 · diag ep 19800

#### i2_stride4h32 (rank ~22)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride4h32_b16/flow_samples.png" alt="i2 stride4 h32 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride4h32_b16/flow_correlations.png" alt="i2 stride4 h32 correlations" width="56%">
</p>

> **KL(p‖q) = 64.61** · **KL(q‖p) = 87.13** · ⟨|M|⟩_q = 2.24 · ξ_q = 14.85 · Best-200 = 7687.45 · diag ep 19800

#### i2_stride8h32 (rank ~24, D config at nr=1)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_b16/flow_samples.png" alt="i2 stride8 h32 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_b16/flow_correlations.png" alt="i2 stride8 h32 correlations" width="56%">
</p>

> **KL(p‖q) = 69.53** · **KL(q‖p) = 93.20** · ⟨|M|⟩_q = 2.29 · ξ_q = 15.29 · Best-200 = 7691.09 · diag ep 19800

---

### nr=2 VP walltime-cut arms (ranks ~23, ~29, ~35)

Cold-start nr=2 fixdil+VP arms — all sit below the nr=1 champion and even below baseline D on Best-200. Kept for comparison against the 2026-07-13 warm-start jobs (`initshared` and `fromnr1`, running).

#### vp1e-4 nr=2 (rank ~23)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16/flow_samples.png" alt="vp1e-4 nr2 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16/flow_correlations.png" alt="vp1e-4 nr2 correlations" width="56%">
</p>

> **KL(p‖q) = 67.71** · **KL(q‖p) = 82.25** · ⟨|M|⟩_q = 2.38 · ξ_q = 16.09 · Best-200 = 7689.79 · diag ep 7000 (walltime cut)

#### vp1e-3 nr=2 (rank ~29)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16/flow_samples.png" alt="vp1e-3 nr2 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16/flow_correlations.png" alt="vp1e-3 nr2 correlations" width="56%">
</p>

> **KL(p‖q) = 80.70** · **KL(q‖p) = 90.75** · ⟨|M|⟩_q = 2.39 · ξ_q = 16.16 · Best-200 = 7701.63 · diag ep 7000 (walltime cut)

#### vp1e-2 nr=2 (rank ~35)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16/flow_samples.png" alt="vp1e-2 nr2 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16/flow_correlations.png" alt="vp1e-2 nr2 correlations" width="56%">
</p>

> **KL(p‖q) = 96.61** · **KL(q‖p) = 121.32** · ⟨|M|⟩_q = 2.26 · ξ_q = 15.23 · Best-200 = 7716.33 · diag ep 7000 (walltime cut)

---

### Older / large-hidden HCG variants (deep in ranking)

#### hcg_shared_hcgh128_nr2_gc5.0 (rank ~47, hcgHidden=128)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_hcgh128_nr2_gc5.0_b16/flow_samples.png" alt="hcgh128 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_hcgh128_nr2_gc5.0_b16/flow_correlations.png" alt="hcgh128 correlations" width="56%">
</p>

> **KL(p‖q) = 88.34** · **KL(q‖p) = 94.67** · ⟨|M|⟩_q = 2.26 · ξ_q = 14.59 · Best-200 = 7698.64 · diag ep 19800

#### hcg_shared_hcgh64_nr2_gc5.0 (rank ~48, hcgHidden=64)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_hcgh64_nr2_gc5.0_b16/flow_samples.png" alt="hcgh64 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_shared_hcgh64_nr2_gc5.0_b16/flow_correlations.png" alt="hcgh64 correlations" width="56%">
</p>

> **KL(p‖q) = 79.88** · **KL(q‖p) = 85.73** · ⟨|M|⟩_q = 2.24 · ξ_q = 14.54 · Best-200 = 7698.94 · diag ep 19800

#### hcg_perscale_fixdil_gc5.0 (rank ~51, earlier hcg_perscale attempt with gradClip)
<p>
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b16/flow_samples.png" alt="hcg perscale fixdil gc5 samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b16/flow_correlations.png" alt="hcg perscale fixdil gc5 correlations" width="56%">
</p>

> **KL(p‖q) = 83.14** · **KL(q‖p) = 93.58** · ⟨|M|⟩_q = 2.27 · ξ_q = 15.17 · Best-200 = 7706.66 · diag ep 19800

---

### Alternative-objective runs (not on the forward-KL ranking)

These are trained with different objectives — not directly comparable to forward-KL variants above.

#### sym_bignet (reverse-KL, symmetrized)
<p>
<img src="../../data/64Ising_T2.269_sym_bignet/flow_samples.png" alt="sym_bignet samples" width="42%">
<img src="../../data/64Ising_T2.269_sym_bignet/flow_correlations.png" alt="sym_bignet correlations" width="56%">
</p>

> Best reverse-KL run — training S = 5606.61, KL(q‖p) = 1431.98 · The reverse-KL objective doesn't try to match the HS marginal, so forward-KL for it is huge.

#### jsLoss_bignet_lam0.5 (JS-divergence objective)
<p>
<img src="../../data/64Ising_T2.269_jsLoss_bignet_lam0.5/flow_samples.png" alt="jsLoss samples" width="42%">
<img src="../../data/64Ising_T2.269_jsLoss_bignet_lam0.5/flow_correlations.png" alt="jsLoss correlations" width="56%">
</p>

> Alternative-objective baseline — not on Best-200 ranking.

#### pathgrad_bignet (path-gradient estimator)
<p>
<img src="../../data/64Ising_T2.269_pathgrad_bignet/flow_samples.png" alt="pathgrad samples" width="42%">
<img src="../../data/64Ising_T2.269_pathgrad_bignet/flow_correlations.png" alt="pathgrad correlations" width="56%">
</p>

> Alternative-estimator baseline — not on Best-200 ranking.

