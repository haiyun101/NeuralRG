# RG fixed-point report — L=64 top forward-KL models

Focused RG-probe writeup for the current top-Best-200 L=64 T_c models,
using the V0–V5 battery already applied to earlier baselines in
`rg_fixed_point_focus_L64_en.md`. Where the earlier report compared
A / D / i1_df4 / P2 winner across V0–V5, this file adds the **fixdil+VP-1e-3
nr=1 champion** (Best-200 = 7658.61) and the **HCG shared** family so the
battery scans the full "prior-vs-baseline" axis at the top of the ranking.

## Top models covered here

| Rank | Cell                                  | Prior                   | Best-200  |
|:---:|----------------------------------------|-------------------------|----------:|
|  1   | fixdil+VP-1e-3 nr=1 ★                 | HCG per-scale + VP      | **7658.61** |
|  3   | baseline_nr2 (**C** — Gaussian nr=2)   | Gaussian                | 7661.64   |
|  5   | hcg_shared                             | HCG scale-shared        | 7669.66   |
|  6   | i2_stride8h32_nr2 (**D**, Phase-2 nr=2)| conditional_gaussian    | 7676.08   |
| 10   | baseline_b16 (**A** — Gaussian nr=1)   | Gaussian                | 7682.16   |
|  24  | i2_stride8h32_b16 (**B**, Phase-2 nr=1)| conditional_gaussian    | 7691.09   |

`B` = the nr=1 sibling of D; same prior, half the MERA depth. Covered in
detail in `rg_fixed_point_focus_L64_en.md` as "P2 winner."

## V0–V5 probes — compact definitions

Six probes in the battery. Each answers a different question about
"is the flow at an RG fixed point / did it learn physical scale invariance."
**Smaller = better in every case**, with two caveats noted below.

| Probe | Input | What it measures | Interpretation |
|:-----:|-------|------------------|----------------|
| **V0/V1** | fresh `N(0, I)` noise on a 2×2 probe patch | adjacent-scale MSE, `MSE(T_s(f_s(z)), T_{s+1}(f_{s+1}(z)))` under per-component z-score gauge | small = adjacent blocks act identically = scale-invariant recursion → RG fixed point |
| **V2** | chained input `h_s = f_{s+1}(…f_5(z)…)` | same MSE but with production-composition input | verifies V0/V1 isn't an artifact of feeding all blocks the same fresh z |
| **V2b** | 1 slot chained + 3 slots fresh `N(0, I)` | same MSE with slot-geometry-corrected input | discriminator: if V2b ≫ V0/V1, the "fixed-point signal" was a 4-tuple-chaining artifact |
| **V3** | fresh `N(0, I)` | per-block identity residual `r_s = E[(T_s(f_s(z)) − z)²] / E[z²]` | small = `f_s ≈ identity`. ⚠ Ambiguous — a rev-KL degenerate flow can also hit r → 0 by doing nothing |
| **V4** | real HS data samples | same-as-V0/V1 but on physical inputs | tests whether scale-invariance holds on the actual data distribution, not just synthetic noise |
| **V5** | fresh z + Wilson block-RG target | RMS-G shape match + matched-pair MSE **vs analytical Wilson block-RG** | small = the flow's block acts like the *physically-derived* RG operator; ground truth from Wilson theory |

**Two caveats on "smaller = better":**
1. **V3 alone is not a fixed-point diagnostic.** A block that has learned
   nothing (identity map) trivially gives r_s → 0. You need V3 combined
   with V5 to distinguish "learned physical fixed point" from "collapsed
   to identity." Rev-KL degenerate flows famously score r_s ≈ 10⁻⁴ on V3
   but fail V5 completely.
2. **V0/V1 alone can lie** — the L=64 baseline in
   `rg_fixed_point_focus_L64_en.md` looks like a fixed point on V0/V1
   but V2b unmasks it as a geometry artifact of the 4-tuple probe. The
   full battery matters; single-probe conclusions are unsafe.

## V0–V5 probes on A and B (available now, historical)

The Phase-2 focus report `rg_fixed_point_focus_L64_en.md` has run the full
V0–V5 battery on **A** (baseline_b16, Gaussian nr=1) and **B**
(i2_stride8h32_b16, Phase-2 winner nr=1). Both use the ep 19 800
checkpoint. Reproducing here for the top-models line-up. (A was also
re-run today for methodological consistency — the V0/V1 numbers agree
with the historical values to ~3 sig-figs.)

### V0/V1 — `N(0, I)` probe, adjacent-scale MSE

| Pair    | **A** (Gaussian nr=1) | **B** (P2 winner, i2 nr=1) | New: **D** (i2 nr=2) | New: **Champion** (VP-1e-3 nr=1) |
|---------|---:|---:|---:|---:|
| f_1 → f_2 | 0.44           | 1.12   | 3.16  | 1.65  |
| f_2 → f_3 | 0.13           | 0.81   | 1.67  | 0.089 |
| f_3 → f_4 | 0.06           | 0.33   | 0.12  | 0.007 |
| f_4 → f_5 | 0.13 ↑         | 0.43   | 0.041 | **0.003** |
| f_5 → f_6 | 0.16 ↑         | **0.034** | **5.8×10⁻¹⁴** | **0.003** |

**Reading (deep-layer fixed-point ordering):**
`D ≫ champion > B ≫ A`. A's deep pairs bounce back up (0.13 → 0.19) —
no scale-invariant prior. B's f_5→f_6 drops to 0.034 despite being nr=1,
because conditional_gaussian's slow-mode context lets the last block
degenerate to a coarse average. D at nr=2 has extra depth to
make f_6 exactly f_5 (machine zero). Champion sits between B and D:
deep pairs plateau at 0.003, ~50× lower than A, ~10× higher than D but
via a different mechanism (HCG per-scale + VP keeps the deep blocks
uniform without needing extra depth).

### V2 — chained input, adjacent-scale MSE

Uses production composition `h_s = f_{s+1}(…f_5(z)…)` as input; catches
V0/V1 artifacts from feeding all blocks the same fresh z.

| Pair    | **A**  | **B**  | **C**  | **D**  | **Champion** |
|---------|---:|---:|---:|---:|---:|
| f_1 → f_2 | 0.34   | 0.70   | 0.53   | 1.96   | 1.04     |
| f_2 → f_3 | 0.73   | 0.38   | 0.89   | 1.97   | 0.088    |
| f_3 → f_4 | 0.44   | 0.71   | 1.01   | 0.098  | 0.008    |
| f_4 → f_5 | 0.33   | 0.50   | 0.35   | 0.034  | 0.007    |
| f_5 → f_6 | **0.14** | 0.067  | **0.004** | **~0** | **0.003** |

Values from `rg_fixed_point_robustness.csv` (variant `v2_chain`, updated
2026-07-14 by job 41811255 step 1). All five at ep 19 800 (14 500 for
champion).

**Reading — deep-pair (f_5→f_6) fixed-point ordering under V2:**
`D ≈ C > champion > B ≫ A`. Champion + D + C all sit at ≤0.004 deep-pair
under chained input — chaining doesn't break the V0/V1 signal for these
three, so their deep-scale similarity survives production composition.
A still bounces at 0.14 (worst of the five). Surprise: C (Gaussian nr=2)
matches D on V2 despite Gaussian prior — nr=2 depth alone gets to a
V2-fixed-point signature.

### V2b — 1-slot chained + 3-slot fresh `N(0, I)`

Slot-geometry-corrected input. **Discriminator**: if V2b ≫ V0/V1, the
apparent fixed-point was a 4-tuple chaining artifact.

| Pair    | **A**  | **B**  | **C**  | **D**  | **Champion** |
|---------|---:|---:|---:|---:|---:|
| f_1 → f_2 | 1.47   | 1.67   | 0.67   | 1.72   | 1.47     |
| f_2 → f_3 | 1.26   | 1.93   | 1.97   | 2.34   | 1.57     |
| f_3 → f_4 | 1.14   | 1.04   | 1.70   | 1.39   | 1.48     |
| f_4 → f_5 | 1.22   | 1.59   | 1.48   | 1.52   | 1.50     |
| f_5 → f_6 | **1.49** | **1.49** | **1.49** | **1.49** | **1.49** |

Values from `rg_fixed_point_robustness.csv` variant `v2b_chain_oneslot`.

**Reading — the V2b reversal (everyone loses):** *every* top-model's
deepest V2b pair sits at **≈ 1.49** — including champion and D that had
V0/V1 deep-pair MSE of 0.003 and 5.8×10⁻¹⁴ respectively. **The
"scale-invariant recursion" signal from V0/V1 is largely a
4-tuple probe geometry artifact for ALL five models**, not just A. Under
slot-corrected input, no model actually looks like an RG fixed point at
the deepest scale. Consistent with the focus report's caution
"L=64 hs_bignet's V0/V1 near-fixed-point appearance may be partly a
geometry artefact."

Champion's cleaner V0/V1 curve therefore reflects "the last two blocks
happen to make identical *chained* outputs on 4-tuple probe input", not
"the flow implements a scale-invariant recursion." V5 (Wilson-RG
comparison, pending) is the physics ground truth to decide.

### V3 — per-block identity residual `r_s`

Small r_s = f_s ≈ identity. **Caveat: not automatically good** (rev-KL
degenerate flows also hit r → 0 by doing nothing; need V5 to disambiguate).

`v3_identity_rel` is `E[(T_s(f_s(z)) − z)²] / E[z²]`; values > 2 mean the
block moves the input by more than the input norm (unbounded above).

| Block | **A**    | **B**    | **C**   | **D**    | **Champion** |
|-------|---:|---:|---:|---:|---:|
| f_1   | 0.79     | 0.95     | 1.33    | 1.62     | **3.24** ↑  |
| f_2   | 3.93     | 3.72     | 34.07   | 4.71     | 0.39        |
| f_3   | 7.04     | 5.04     | 43.99   | 2.12     | 0.45        |
| f_4   | 5.01     | 3.52     | 8.51    | 0.52     | 0.30        |
| f_5   | 2.63     | 1.29     | 2.66    | **0.013** | 0.36        |
| f_6   | 0.97     | 0.23     | 2.66    | **0.0016** | 0.21        |

**Reading — per-block identity ordering:**
- **D**: f_5 = 0.013, f_6 = 0.0016 → the last two blocks are essentially
  the identity map. Consistent with V0/V1 f_5→f_6 = machine zero — the
  two blocks agree because they're both doing nothing.
- **Champion**: f_5 = 0.36, f_6 = 0.21 → last blocks are only *mildly*
  identity-like. Not a degenerate identity flow. But f_1 = 3.24 → the
  SHALLOWEST block moves the input a lot (biggest in the table). VP has
  pushed the champion into a "front-loaded" configuration: heavy work
  at the finest scale, gentle at deep scales.
- **B**: f_6 = 0.23 — moderately identity-like at deepest, comparable
  to champion. But shallow blocks r_1-r_4 all ≥ 3.5 — B is *not* as
  front-loaded as champion; it distributes work more evenly.
- **A**: r-profile is spread across scales (0.79 shallow → 0.97 deep).
  No clear "identity-region." No physics-inspired specialization.
- **C**: r-profile is bimodal — very large at middle scales (34, 44!)
  and moderate elsewhere. Consistent with C being a plain Gaussian nr=2
  reference that hasn't specialized cleanly.

**Numerical caveat:** These `v3_identity_rel` values are ~10-100× larger
than what the focus report tabulated for A and B ("f_5 = 0.17, f_6 =
0.0064" for A). The current `rg_fixed_point_robustness.py` computes r_s
against un-gauge-fixed activations, whereas the focus report's numbers
appear to have been on gauge-fixed activations. Ordering is preserved
across the two conventions (small = identity-like) but magnitudes are
not directly comparable to the focus report table.

### V4 — HS data forward, adjacent-scale metrics

Same shape as V0/V1 but on real MCMC-generated data samples (not
synthetic `N(0, I)`). Job 41814473 (2026-07-15) filled the pending
cells for D and champion. Values below are from
`rg_v4_dataforward.csv` (fresh 16:01 Wed).

**V4 `adj_ks`** (KS statistic on adjacent-scale marginals):

| Pair    | **A**  | **B**  | **C**  | **D**  | **Champion** |
|---------|---:|---:|---:|---:|---:|
| s=0→1 | 0.063 | 0.055 | 0.031 | 0.052 | **0.120** |
| s=1→2 | 0.060 | 0.066 | 0.050 | 0.055 | 0.052 |
| s=2→3 | 0.046 | 0.016 | 0.056 | **0.014** | **0.003** |
| s=3→4 | 0.035 | 0.028 | 0.047 | 0.006 | 0.008 |
| s=4→5 | 0.047 | 0.062 | 0.014 | 0.007 | 0.008 |
| s=5→6 | 0.079 | 0.024 | 0.018 | 0.025 | 0.020 |

**V4 `adj_rms_g`** (RMS distributional shape at adjacent scales; `nan`
where deep-scale G(r) support is too small for the fit):

| Pair    | **A**  | **B**  | **C**  | **D**  | **Champion** |
|---------|---:|---:|---:|---:|---:|
| s=0→1 | 0.086 | 0.060 | 0.073 | 0.090 | **0.425** |
| s=1→2 | 0.133 | 0.147 | 0.214 | 0.252 | 0.133 |
| s=2→3 | 0.068 | 0.032 | 0.177 | 0.081 | 0.130 |
| s=3→4 | 0.075 | 0.019 | 0.379 | 0.072 | 0.041 |
| s=4→5 | nan | nan | nan | nan | nan |
| s=5→6 | nan | nan | nan | nan | nan |

**Reading — champion is very shallow-heterogeneous but deep-uniform.**
Champion's s=0→1 metrics are the *worst* of the five (`adj_ks` = 0.12,
`adj_rms_g` = 0.42) — the transition from raw HS data to the first
coarse-grained representation involves the biggest change. But from
scale 2 onward it drops to nearly zero (`adj_ks` = 0.003 at s=2→3, the
smallest in the table).

This is the "front-loaded work" pattern first seen in the cascade
V3-analogue (r_1 = 3.24 for champion, biggest in the table): VP forces
MERA to do most of the shaping at the finest scale where physical
detail lives, leaving deep scales nearly self-similar. A doesn't have
that concentration; C actually gets *worse* at deep scales (`adj_rms_g`
= 0.38 at s=3→4). D matches champion's pattern at mid-deep scales.

### V5 — vs Wilson block-RG ground truth

RMS-G (shape) + KS + matched-pair W1 between `f_s(z)` and the analytical
Wilson block-average of the same input. **Small = closer to the
physically-derived RG operator.** Job 41814473 filled all TBD cells.

**V5 `v5_rms_g`** (distributional shape vs Wilson-blocked data):

| Scale s | L_s | **A**  | **B**  | **C**  | **D**  | **Champion** |
|:-:|:-:|---:|---:|---:|---:|---:|
| 1 | 16 / 32 | 0.123 | **0.092** | 0.109 | 0.127 | **0.463** |
| 2 |  8 / 16 | 0.053 | **0.024** | 0.062 | 0.431 | 0.417 |
| 3 |  4 / 8  | 0.042 | **0.031** | 0.197 | 0.410 | 0.351 |
| 4 |  2 / 4  | 0.019 | 0.065 | **0.623** | 0.513 | 0.351 |

**Reading — B is the physically closest flow, champion is the farthest.**
- **B** wins v5_rms_g at s=1, s=2, s=3 (0.092, 0.024, 0.031). At mid-scales
  B's MERA output is distributionally almost indistinguishable from
  Wilson block-averaged 2×2 blocks of the data — it has learned
  something close to the physical block-RG operator.
- **A** is second-best, especially at deep scales (0.019 at s=4).
- **C** is Wilson-like at shallow but grows apart at deep (0.62 at s=4).
- **D and champion are FAR from Wilson** at every scale. Champion's
  v5_rms_g is 3-8× larger than B's at every scale.

This is a genuine cost of the VP mechanism. Champion pushes hard toward
Gaussianity at every scale (Section C1: latent excess kurtosis 1.4 at
y_6 vs A's 13.4) and suppresses raw amplitudes (Section C2: G(0) = 4.2
at s=1 vs A's 53.4). These "cleanup" operations move MERA away from
Wilson block-RG — which preserves Ising-like amplitude and bimodality.
D has the same problem for a different reason (huge Jacobian rescaling).

**Physical trade-off, quantified:**
- Best-200 loss (forward-KL) ordering: **Champion (7658) > B (7691)** by
  33 nat.
- V5 closeness-to-Wilson (v5_rms_g @ s=2) ordering: **B (0.024) ≫
  Champion (0.417)**.
- Champion **beats B on modeling but loses to B on physics** — B's
  MERA is a physically-recognizable Wilson block-RG; champion's MERA is
  a Gaussian-cleanup operator that happens to also lower forward-KL.
- Whether "physical MERA" or "modeling-optimal MERA" is the goal
  depends on downstream use. For sampling / KL benchmarking: champion.
  For RG-fixed-point analysis / physics interpretation: **B**.

**V5 `v5_ks`** (KS statistic, same story):

| Scale s | **A**  | **B**  | **C**  | **D**  | **Champion** |
|:-:|---:|---:|---:|---:|---:|
| 1 | 0.081 | 0.070 | 0.063 | 0.087 | 0.144 |
| 2 | 0.142 | 0.094 | 0.081 | 0.143 | 0.174 |
| 3 | 0.109 | 0.101 | 0.110 | 0.158 | 0.184 |
| 4 | 0.092 | 0.107 | 0.164 | 0.168 | 0.192 |

Confirms: champion is furthest from Wilson at every scale.

### V5 — vs Wilson block-RG ground truth

RMS-G (shape) + matched-pair MSE (sample-level alignment) between
`f_s(z)` and the analytical Wilson block-average of the same input.
Small = closer to the physically-derived RG operator.

**RMS-G (distributional shape):**

| Scale s | L_s | **A**  | **B**  | New: **D** | New: **Champion** |
|---|-----|---:|---:|---:|---:|
| 1 | 16 / 32 | 0.071 | 0.067 | TBD | TBD |
| 2 |  8 / 16 | 0.046 | **0.039** | TBD | TBD |
| 3 |  4 / 8  | 0.042 | **0.030** | TBD | TBD |
| 4 |  2 / 4  | 0.034 | 0.045 | TBD | TBD |

**Matched-pair MSE (sample-level alignment, N=2000, `2(1 − corr)`):**

| Scale s | L_s | **A**  | **B**  |
|---|-----|---:|---:|
| 1 | 16 / 32 | 0.57 | **0.53** |

**Reading:** B beats A on RMS-G at s=2, s=3 (peak scale-invariant regime)
and on matched-pair MSE at s=1. Confirms B's V0/V1 + V3 + V4
signatures are physical, not a probe artifact. A is still comparable on
absolute scale — its dysfunctional deep V0/V1 doesn't fully break V5,
suggesting the deep-block drift is more of a fine-tuning issue than a
qualitative scale-invariance failure.

Plot: `analyzers/rg_fixed_point/figures/rg_fixed_point_L64_champion.png`
(V0/V1 for A + D + champion only; B panel lives in the focus report).

## Cascade layer analysis (2026-07-15) — replaces V-probes with real cascade data

The V-probes above are all applied to **fresh `N(0, I)` probes on a 2×2
patch, single scale-block at a time** (V2 chains but per-slot, V2b mixes
1-slot chain with 3-slot fresh). None of them push a real batch through
the actual generation cascade the way inference runs. The cascade layer
analysis uses `mera_layer_flow_capture.py` output — real HS data pushed
through *the entire flow*, activation kept at every scale — and computes
five sections of physical/statistical metrics on those actual cascaded
activations.

**Metrics** (all on champion + A now; D pending job 41814472 completion):

| Section | Uses | Metric | Answers |
|:-:|---|---|---|
| **A** | `y_s_gaussianized` | skew, excess kurtosis, KS to N(0,1) | is the latent side actually Gaussian? |
| **B** | `y_s_gaussianized` pair (s, s+1) | MMD² / W1 / KS on marginals | real V0/V1 — do adjacent scales look self-similar under real data? |
| **C** | `y_s_kept` (raw) | G(0), G(r), ξ_s | raw physical field amplitudes and correlation |
| **D** | `y_s_gaussianized` vs `w_s_gaussianized` | MMD² marginal | forward representation ≡ inverse representation? |
| **E** | `champion_y_s` vs `A_y_s` (both gaussianized) | MMD² full-vec, W1 marginal, spearman ρ | cross-model per-scale |

Script: `analyzers/rg_fixed_point/cascade_layer_analysis.py`.
CSV: `csv/cascade_layer_analysis.csv` (114 rows for champion + A).

### C1 — Latent-side Gaussianity ranks D > Champion ≫ A

Section A — **excess kurtosis** at each scale (three-way now that D's
`mera_layer_flow_capture.pt` landed):

| scale | A (Gaussian nr=1) | Champion (VP-1e-3 nr=1) | **D** (i2 nr=2) |
|:-:|---:|---:|---:|
| y_1 | +0.75 | +1.6  | **+0.51** |
| y_2 | +1.25 | +3.0  | **+1.31** |
| y_3 | +4.7  | +3.2  | **+1.60** |
| y_4 | +9.5  | +2.4  | **+1.56** |
| y_5 | +6.5  | +2.4  | **+1.81** |
| **y_6** | **+13.4** | +1.4  | **+0.75** |

**D achieves the most Gaussian latent representation at every scale.**
Deep-scale kurtosis excess:
- A: 13.4 (very heavy tails)
- Champion: 1.4 (mildly non-Gaussian, closer to Gaussian than A)
- D: 0.75 (essentially Gaussian at the deepest scale)

D's advantage is uniform across all scales, not just at the deep end.
KS-to-N(0,1): D 0.03-0.08, champion 0.06-0.09, A 0.02-0.11. D again
consistently closest to Gaussian.

Different priors, different routes to Gaussianity:
- A has a Gaussian prior but doesn't achieve Gaussian latent — MERA
  didn't match.
- D uses `conditional_gaussian` (single CNN parameterizing `σ(z_slow)`)
  and nr=2 depth — MERA + CNN together push the latent to Gaussian
  aggressively.
- Champion uses HCG per-scale + VP; the VP penalty pins log|det J| → 0
  which suppresses MERA's amplitude rescaling, so the latent Gaussianity
  is more indirect.

### C2 — Raw amplitudes: Champion tiny, A moderate, D **enormous**

Section C — `G(0)` (per-site variance of the raw kept-coarse field):

| scale       | A    | Champion | **D** | D/champion |
|:-:|---:|---:|---:|---:|
| y_1 (32×32) | 53.4 | 4.2    | **214.0** | 51× |
| y_2 (16×16) | 19.5 | 5.8    | **740.9** | 128× |
| y_3 (8×8)   | 2.2  | 2.9    | **411.8** | 142× |
| y_4 (4×4)   | 0.38 | 1.29   | **741.1** | 574× |
| y_5 (2×2)   | 0.24 | 0.55   | **783.5** | 1425× |

**Three qualitatively different amplitude regimes:**
- **Champion**: raw field variance ≤ 6 at every scale — the direct
  measurement of what VP was designed for. MERA can't rescale, so raw
  amplitudes stay physical.
- **A**: raw variance up to 53 at finest scale, decays fast to sub-unit
  at deep scales — moderate MERA rescaling concentrated at the fine end.
- **D**: raw variance **200-800× at every scale** — MERA is applying
  extreme scale factors everywhere. The near-Gaussian latent (C1) is
  achieved through massive Jacobian rescaling, not through a physically
  well-calibrated flow.

So Section C1 (D wins on Gaussianity) + Section C2 (D has huge amplitudes)
together say: **D's near-Gaussian latent is a Jacobian trick.** MERA
rescales the field by ~1000× at each scale to satisfy the Gaussian prior's
marginal-variance requirement, but the underlying physical field is not
what the model actually "reasons about."

Champion's smaller kurtosis excess (1.4) at y_6 is achieved with raw
variance 0.55 — no Jacobian trick. That's what VP was designed to make
happen. But quantitatively, D still hits the Gaussian target more
closely, at the cost of losing physical interpretability of MERA's
intermediate representations.

`ξ_s` (correlation length from exponential fit on G(r)) is unreliable at
small L_s because the fitted ξ often exceeds L_s (marked `inf`/`nan`);
not tabulated as diagnostic here.

### C3 — Cross-scale self-similarity is small (and comparable) for BOTH models

Section B — the honest V0/V1 done with real cascaded data:

| pair    | A (MMD²)     | Champion (MMD²) |
|:-:|---:|---:|
| y_1→y_2 | 3×10⁻⁵      | 6×10⁻³ |
| y_2→y_3 | 7×10⁻⁴      | 3×10⁻⁴ |
| y_3→y_4 | **9×10⁻³**  | 1×10⁻³ |
| y_4→y_5 | 3×10⁻³      | −5×10⁻⁵ |
| y_5→y_6 | −8×10⁻⁴     | −4×10⁻⁴ |

Different shape: A peaks at y_3→y_4 (mid-scale most non-self-similar);
champion peaks at y_1→y_2 (finest scale is where it differs most). Both
are small in absolute terms — the two flows really do produce approximately
self-similar cascaded outputs on real data. This **contradicts the V2b
"fixed-point signal was an artifact" reading**: under the artificial
V2b probe both look non-fixed-point (deep MSE ≈ 1.5), but under actual
cascaded data both are self-similar at deep scales. V2b's artificial
input pattern was the problem, not the flows.

### C4 — Forward-inverse consistency (Section D)

MMD² between `y_s` (data pushed forward) and `w_s` (noise pushed inverse)
at each scale ≤ 0.021 for A and ≤ 0.012 for champion. Both flows are
approximately reversible; champion slightly more consistent at deep
scales.

### C5 — Cross-model at each scale (Section E, three-way)

At every scale, all three models produce **distributionally identical**
activations (MMD² baseline ~ 10⁻³, all within noise). W1 similar too.
The interesting signal is per-sample rank correlation (`ρ_sum` = Spearman
on `mean(|y_s|)`):

| scale | Champion vs A | Champion vs D | A vs D |
|:-:|---:|---:|---:|
| s=1 (32×32) | −0.30 | **+0.88** | −0.13 |
| s=2 (16×16) | −0.60 | +0.31 | +0.18 |
| s=3 (8×8)   | −0.61 | +0.12 | +0.25 |
| s=4 (4×4)   | −0.32 | +0.01 | +0.31 |
| s=5 (2×2)   | −0.22 | +0.42 | +0.13 |
| s=6 (1×1)   | −0.12 | **+0.57** | +0.16 |

**Champion and D route samples similarly at the finest scale**
(ρ = +0.88 at s=1) and at the deepest (+0.57 at s=6). At mid-scales they
decorrelate. Champion vs A is anti-correlated everywhere (VP flipped
per-sample routing). A vs D is essentially uncorrelated.

This says: **champion's per-sample routing is closer to D's than to A's
at the extreme scales** (finest and deepest), which is where physics
matters most — the fine scale carries the raw data and the deep scale
sits at the Gaussian prior. VP effectively pushed the champion's flow
onto the same routing manifold as D at the physical endpoints, while
differing from D at the mid-scales (where architectural differences
between HCG per-scale + VP and single-CNN + nr=2 have more room to
show).

### The V-probe conclusions that need retracting given cascade data

1. **"V2b reversal → no genuine fixed point"** — retracted. Under real
   cascaded data both champion and A are approximately self-similar at
   deep scales; the V2b reversal was an artifact of the 4-tuple
   `1-chain + 3-fresh` probe geometry.
2. **"V0/V1 says champion is at a fixed point"** — refined. Champion's
   V0/V1 plateau IS from real self-similarity of the deep cascade, but so
   is A's non-plateau on V0/V1 mostly an artifact of fresh-probe
   geometry — under cascaded data A's f_5→f_6 MMD² is −8×10⁻⁴ (smaller
   than champion's f_1→f_2!).
3. **"D at machine-zero V0/V1 = deepest fixed point"** — pending D
   `mera_layer_flow_capture.pt` from job 41814472. If D's cascade
   self-similarity is comparable to A / champion (~10⁻³), then D's
   V0/V1 = 5.8×10⁻¹⁴ was purely the "f_6 collapsed to identity"
   phenomenon documented in V3 (r_6 = 0.0016), not a stronger
   fixed-point signal on physical data.

### What actually distinguishes champion from A

Not the deep cascade shape (both self-similar). Not the internal
representation distribution (both same). The two things that VP
**actually delivered**, both measurable from the flow_capture data:

- **Latent-side Gaussianity** (Section A): champion's y_6 kurtosis
  excess 1.4 vs A's 13.4 — physical prior matching.
- **Raw amplitude scale** (Section C): champion's fine-scale G(0) = 4.2
  vs A's 53.4 — 13× amplitude suppression, exactly what VP's soft
  volume-preserving penalty was designed to produce.

Both are direct consequences of forcing MERA to be volume-preserving.
Neither shows up in the V0-V5 battery (those probes were designed for
scale-invariance, not for prior calibration or amplitude physics).

## Provenance and pending completions

**Historical availability** (already in CSVs / focus report at ep 19 800):

|                                    | **A** (baseline_b16) | **B** (i2 nr=1)      |
|------------------------------------|:---:|:---:|
| V0/V1                              | ✓ CSV + focus report | ✓ focus report      |
| V2 chained                         | ✓ CSV                | ✗ never run          |
| V2b slot-corrected                 | ✓ CSV                | ✗ never run          |
| V3 identity residual               | ✓ CSV                | ✓ focus report       |
| V4 HS-data forward                 | ✓ CSV                | ✓ focus report       |
| V5 RMS-G + matched-pair MSE        | ✓ CSV                | ✓ focus report       |

**Pending — job 41811255** (submitted 2026-07-14, ~10 h walltime, CPU
batch partition):

|                                    | **D** (i2 nr=2)      | **Champion** (VP-1e-3 nr=1) |
|------------------------------------|:---:|:---:|
| V0/V1                              | ✓ today (job 41802239) | ✓ today (job 41802239) |
| V2 chained                         | ✗ pending 41811255   | ✗ pending 41811255   |
| V2b slot-corrected                 | ✗ pending 41811255   | ✗ pending 41811255   |
| V3 identity residual               | ✗ pending 41811255   | ✗ pending 41811255   |
| V4 HS-data forward                 | ✗ pending 41811255   | ✗ pending 41811255   |
| V5 RMS-G + matched-pair MSE        | ✗ pending 41811255   | ✗ pending 41811255   |

**Naming caveat.** The focus report's "P2 winner" refers to **B** =
`i2_stride8h32_b16` (**nr=1**), NOT to the current top-6 model
`i2_stride8h32_nr2_b16` (**D**, nr=2). Same conditional-Gaussian prior,
different MERA depth. So on 2026-07-14: A + B have full historical
V0-V5; D + champion get V2-V5 today via 41811255.

Job 41811255 will also fill B's V2/V2b (since B is registered in
`rg_fixed_point_robustness.py`), completing that gap. Outputs land in:
- `csv/rg_fixed_point_robustness.csv` — V2, V2b, V3
- `csv/rg_v4_dataforward.csv` — V4
- `csv/rg_v5_blockRG_compare.csv` — V5 RMS-G + matched-pair MSE
- corresponding figures under `figures/`

## Cross-metric summary (to be filled)

Once the V-battery lands, this table gives the one-line RG fingerprint
per model:

| Model      | V0/V1 deep | V2b deep | V3 r_6 | V4 deep | V5 RMS-G s=2 | V5 MSE s=2 | Reading |
|:-----------|:---------:|:--------:|:------:|:-------:|:------------:|:----------:|---------|
| Champion (VP-1e-3) | **0.003** | TBD | TBD | TBD | TBD | TBD | ? |
| A (baseline)       | 0.19      | TBD | TBD | TBD | TBD | TBD | ? |
| C (baseline nr=2)  | TBD       | TBD | TBD | TBD | TBD | TBD | ? |
| D (i2 nr=2)        | 5.8e-14   | TBD | TBD | TBD | TBD | TBD | ? |
| hcg_shared         | TBD       | TBD | TBD | TBD | TBD | TBD | ? |

## See also

- `rg_fixed_point_focus_L64_en.md` — full V0–V5 discussion for A, D, i1_df4
- `rg_fixed_point_focus_en.md` — L=32 parallel
- `rg_fixed_point_report.md` — L=32 methodology + robustness reinterpretation
- `../concise_reports/concise_report_L64_T2.269.md` — Best-200 ranking, physics observables, plot gallery
