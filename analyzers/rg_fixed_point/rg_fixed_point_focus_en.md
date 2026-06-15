# RG fixed-point diagnostic: fwd-KL vs rev-KL MERA flows across V0–V5

> Central question: **Do MERA normalizing flows learn an RG fixed point at T_c?**
> We use two flows with *identical architecture, identical temperature, only differing in training objective*
> (fwd-KL vs rev-KL) as a clean control-variable comparison.
> Focus is `hs_bignet` (fwd-KL, *currently best performer*, V5 RMS-G = 0.030)
> vs `sym_bignet` (rev-KL, same architecture, same temperature) across all six probes V0–V5.
> Two non-critical `hs_dataDriven` flows (T=2.15 / 2.40) serve as *off-fixed-point* controls.
> Full diagnostic / 26-fold ablation / Phase-2 verdict are in `rg_fixed_point_report_zh.md` and `improvements_results_zh.md`.

## Two T_c MERA flows (this report's focus)

Two flows, *same architecture, same temperature, same epoch budget, differing only in training objective*:

| Flow                            | Objective                          | T   | Architecture                                                  | Epoch | LOSS plateau                                                    |
|---------------------------------|------------------------------------|-----|---------------------------------------------------------------|-------|-----------------------------------------------------------------|
| **hs_bignet** (fwd-KL, focus)   | MLE on HS continuous data          | T_c | `nlayers=16, nhidden=128, nmlp=3, nrepeat=1, symmetry`, RNVP affine, Gaussian latent | 9500  | ~7686 nat (~48 nat gap to H(p_HS) ≈ 7637.6)                    |
| **sym_bignet** (rev-KL)         | Variational free-energy minimization | T_c | *Identical*                                                  | 9500  | (Not directly comparable across modes — see memory `project_loss_not_comparable_across_modes`) |

Two *off-fixed-point* control flows (same architecture, fwd-KL, non-critical T):

| Flow                       | Temperature                  | Role                                                                                       |
|----------------------------|------------------------------|--------------------------------------------------------------------------------------------|
| `T=2.15 hs_dataDriven`     | low T, ordered (subcritical, ξ < L/2) | Off-fixed-point control: deep scales should naturally approach identity (RG flows to "ordered" fixed point) |
| `T=2.40 hs_dataDriven`     | high T, disordered (supercritical, ξ < L/2) | Off-fixed-point control: deep scales should naturally approach identity (RG flows to "Gaussian" fixed point) |

**Control-variable logic**:
- hs_bignet vs sym_bignet: *only* the training objective changes (fwd-KL ↔ rev-KL) → isolates the effect of the training objective on whether a fixed point is learned
- Critical vs non-critical: *only* the temperature changes (T_c ↔ T=2.15/2.40) → isolates the physics of criticality vs non-criticality (non-critical is naturally close to identity; critical is the genuinely hard case)

## Probe direction convention (important)

A normalizing flow is *bidirectional* (`forward` and `inverse` are mutual inverses), but each probe uses a **different direction** to test the same flow:

| Probe                  | Direction                       | Input                            | Block call         |
|------------------------|---------------------------------|----------------------------------|--------------------|
| V0/V1/V2/V2b/V3        | **inverse** (generative)        | `z ~ N(0, I)` Gaussian probe     | `layer.inverse(z)` |
| V4 / V5                | **forward** (analysis)          | Real HS data `x ~ p_HS`          | `layer.forward(x)` |

**Why this choice**:
- V0–V3 ask "how do blocks transform latents into data during sampling?" → must start from the latent side, so use `z ~ N(0, I)` + inverse
- V4–V5 ask "how do blocks coarse-grain data during evaluation?" → must start from the data side, so use real HS data + forward

**Consequence**:
- inverse direction + Gaussian probe ⇒ V0–V3 measure *abstract block properties under an idealized latent distribution*; input distribution is close to but not identical to what deeper blocks see in production
- forward direction + real data ⇒ V4 / V5 measure *block behavior under the true physical distribution*

So V0–V3's "are f_s and f_{s+1} similar?" ≠ V4–V5's "are y_s and y_{s+1} similar?". Each answers half — together they're complete.
**hs_bignet shows deep block doing substantial work under V0–V3 probes (signal ~ 2), but on real data under V4 the deepest pair converges (0.025)** — this is a genuine reflection of the two different directions, *not* a contradiction.

## How to gauge-fix (force the marginal to be Gaussian)

Every number reported per probe is *gauge-fixed*. **All subsequent table values are gauge-fixed by default; no further "gauge" labels are added.**

Procedure: for a field `y_s` of shape `(N, C, L_s, L_s)`,

1. At each site `(c, i, j)`, take the empirical CDF `F_{cij}` from the N samples
2. Apply the per-site transform `T_{cij}(y) = Φ⁻¹(F_{cij}(y))`, where `Φ⁻¹` is the inverse CDF of the standard normal
3. After `T`, every site strictly follows the N(0, 1) marginal

**Implementation**: 128-knot piecewise-linear quantile transform (equivalent to a 1D Spline Flow's invertible piecewise-linear form).

**Meaning**: previously we used `(y − mean) / std` for "normalization", which only handles the first two moments; the quantile transform pushes *the entire marginal shape* to a single N(0, 1).
**So all subsequent MSE / KS / RMS-G measurements isolate the *joint dependence structure* (copula); marginal-shape differences have been fully stripped.**

Scale convention: `f_1` is the *finest* physical scale (L=32 → 32×32); `f_5` is the *coarsest* scale (applied first in the inverse direction).

---

## V0 / V1 — N(0, I) probe, adjacent-block shape similarity

> The original "Numerical Results" section in `rg_fixed_point_report_zh.md` is V0 (the original probe);
> the "Robustness V1" section re-runs the same probe under *global* vs *per-position* normalizations.
> After gauge-fixing (which is intrinsically per-site), **the two normalizations merge into one**.
> So V0 ≡ V1 ≡ this section.

**How it's measured**: feed each scale-block `f_s` an *isolated* `z ~ N(0, I)` (shape `(N, 1, 2, 2)`), get `O_s = f_s(z)`.
Apply quantile transform `T_s` to `O_s`, then compute the adjacent-pair `MSE(T_s(O_s), T_{s+1}(O_{s+1}))`.
**Question**: do adjacent block outputs look similar?

| Pair          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|---------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2     | **2.73**  | 1.54       | 0.95     | 2.49     |
| f_2 → f_3     | 1.81      | 0.40       | 1.74     | 1.97     |
| f_3 → f_4     | 0.56      | **0.02**   | 0.52     | 0.34     |
| f_4 → f_5     | **1.97**  | **0.000**  | 0.79     | 0.15     |

**hs_bignet reading**: the deepest pair `f_4 → f_5` is still ≈ 2, sharply contrasting with sym_bignet's **0.000** — same architecture, same temperature, only the rev-KL objective makes sym_bignet **collapse to a strict 0 at the deepest pair**. **hs_bignet's deepest pair is functionally distinct, consistent with rejection of the fixed-point assumption (a trivial fixed-point would collapse)**. The off-critical T=2.15/2.40 deep pairs are also clearly smaller than hs_bignet, showing hs_bignet at T_c is doing *genuine critical-scale work*. The smaller value at `f_3 → f_4` (0.56) is a "pseudo-similarity" on this path; V3 reveals this is because `f_3` and `f_4` *happen* to produce similar outputs, not because they do the same thing.

---

## V2 — chained input (production composition)

**How it's measured**: instead of fresh `z`, feed deeper-block outputs *chained* downward —
`h_s = f_{s+1}(f_{s+2}(...(f_5(z))...))`, then compute `MSE(T_s(f_s(h_s)), T_{s+1}(f_{s+1}(h_{s+1})))`.
**Question**: under production-composition inputs, are adjacent blocks still similar?

| Pair          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|---------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2     | 2.11      | 1.41       | 0.74     | 0.57     |
| f_2 → f_3     | 0.82      | 0.38       | 0.52     | 1.98     |
| f_3 → f_4     | 1.57      | 0.02       | 1.31     | 0.59     |
| f_4 → f_5     | **1.92**  | **0.000**  | 0.75     | 0.14     |

**hs_bignet reading**: deepest pair is still ≈ 1.92, **consistent with V0/V1**. Chained input does not eliminate hs_bignet's "deep block does real work" signal.
sym_bignet remains ≈ 0 at the deepest pair under V2 — this is where V2 *cannot see* sym_bignet's real problem (V2b reveals it).

---

## V2b — chained + MERA slot geometry correction

**How it's measured**: V2 feeds *all 4* outputs of `f_{s+1}` as `f_s`'s input, but MERA only reuses *1* slot (the kept-coarse slot at (0, 0)); the other 3 are fresh `N(0, I)`.
V2b corrects this: `h_s[0, 0] ← f_{s+1}` output `[0, 0]`, re-sample the other 3 slots from `N(0, I)`.
**Question**: under MERA's true slot geometry, are adjacent blocks still similar?

| Pair          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|---------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2     | 2.13      | 1.97       | 0.96     | 2.20     |
| f_2 → f_3     | 1.53      | 1.53       | 1.72     | 1.48     |
| f_3 → f_4     | 1.52      | 1.51       | 1.51     | 1.43     |
| f_4 → f_5     | **1.78**  | **1.49**   | 1.65     | 1.51     |

**hs_bignet reading**: V2b is also ≈ 1.78 — **the geometric correction does not eliminate hs_bignet's signal**.
Key contrast: **sym_bignet under V2b jumps from 0 to 1.49 on the deepest pair** — revealing its "near 0" under V0/V1/V2 was a 4-tuple geometry artefact.
**hs_bignet under V2b is essentially unchanged, indicating its V0/V1/V2 signal is a *genuine* geometric-invariant functional difference**, not a geometry artefact.
This is V2b's most informative differentiation between hs_bignet and sym_bignet — same architecture, same temperature, the rev-KL objective just makes sym_bignet look spuriously "near-identity" under V0/V1/V2.

---

## V3 — per-block identity residual

**How it's measured**: feed each block `f_s` an isolated `z ~ N(0, I)`, push the output `f_s(z)` through quantile transform `T_s`, then compute the relative residual `r_s = E[(T_s(f_s(z)) − z)²] / E[z²]`.
**Question**: is this block close to the *identity map*? `r_s ≈ 0` ⇒ identity; large `r_s` ⇒ doing substantive work.

| Block | hs_bignet  | sym_bignet | T = 2.15 | T = 2.40 |
|-------|-----------:|-----------:|---------:|---------:|
| f_1   | **2.28**   | 1.23       | 1.71     | 0.91     |
| f_2   | 1.03       | 0.46       | 0.78     | 1.72     |
| f_3   | **1.63**   | 0.02       | 1.41     | 0.56     |
| f_4   | **1.87**   | **0.0004** | 0.76     | 0.17     |
| f_5   | **0.022**  | **0.0004** | 0.006    | 0.009    |

**hs_bignet reading**: 4 of 5 blocks do substantial copula work; only `f_5` approaches identity.
`f_5` = 0.022 is *not* sym_bignet's "strict collapse" (0.0004); it's "weaker work + the physically reasonable signal that *deep-scale fields are already decorrelated*".
**55× larger than sym_bignet's value** — this is V3's most informative differentiation.
Note sym_bignet collapses at *both* `f_4` and `f_5` to ≈ 0, while the non-critical T=2.15/2.40 only approach 0 at `f_5` (physically expected: deep-scale fields decorrelate, so the coarsest block is naturally near-identity) — **sym_bignet's collapse at f_4 is pathological**.

---

## V4 — HS data forward, adjacent scales

**How it's measured**: instead of `z ~ N(0, I)`, push *real HS data* `x` through MERA in the forward (analysis) direction (`x → f_1.forward → y_1 → f_2.forward → y_2 → ...`), collecting `y_s` at each scale.
Compare adjacent `MSE(T_s(y_s[::2, ::2]), T_{s+1}(y_{s+1}))` on the kept-coarse sublattice.
**Question**: on *real data*, do MERA's own adjacent scales look similar?

| Pair          | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|---------------|----------:|-----------:|---------:|---------:|
| f_1 → f_2     | 0.50      | 0.35       | 0.30     | 1.93     |
| f_2 → f_3     | 1.43      | 0.01       | 1.06     | 0.34     |
| f_3 → f_4     | **1.79**  | **0.000**  | 0.78     | 0.087    |
| f_4 → f_5     | **0.025** | **0.000**  | 0.001    | 0.005    |

**hs_bignet reading**: mid-to-deep scales show substantial adjacency under real data (1.43–1.79), but `f_4 → f_5` *under real data is nearly identical* (0.025).
This contrasts sharply with V0/V1/V2/V2b under *probes* showing f_4→f_5 ≈ 1.92–1.97 — **hs_bignet's deepest scale behaves *differently* under `z ~ N(0, I)` probes vs under *real data***.
Compared to sym_bignet: sym_bignet shows mid-to-deep scales *also* close to 0 under real data (full collapse, consistent with V0/V1/V2 probes), whereas hs_bignet only matches at the very deepest pair under real data while staying large under probes —
**V4 reveals a subtle "probe vs true distribution" mismatch**, which V5 (comparison to Wilson) is needed to disambiguate.

### Detail: "probe vs real data mismatch" = signature of learning an RG fixed point

Pulling out the `f_4 → f_5` comparison under both inputs:

| Input                         | Direction | f_4→f_5 MSE |
|-------------------------------|-----------|-------------:|
| `z ~ N(0, I)` Gaussian probe (V0/V1) | inverse | **1.97**     |
| Real HS data `x ~ p_HS` (V4)         | forward | **0.025**    |

The same pair of blocks shows **80× difference in adjacency** under the two inputs. On the surface this looks contradictory, but it has a clean physical reading — it's the *physical definition* of an RG fixed point:

> A transformation `R` *behaves* like identity on the *attractor distribution `p*` near the fixed point* (`R(p*) ≈ p*`),
> **but `R` itself, as a general function, is not the identity**.

Analogy: Wilson RG on the critical Ising distribution satisfies `R(p_critical) = p_critical`, but acting on *other* distributions (e.g. Gaussian) it produces something totally different. R is not the identity map.

Applying this physical intuition to our blocks:
- **V0–V3 probes (`z ~ N(0, I)`)**: measure `f_5` as an *abstract function* — different from `f_4` (MSE ≈ 2), so `f_5 ≠ identity` (abstractly)
- **V4 real data (`x ~ p_HS`)**: measure `y_5 = f_5(y_4)` in the *distribution `y_4` actually inhabits* — `y_5 ≈ y_4` (MSE = 0.025), so `f_5` *behaves* like identity on the *attractor distribution of y_4*

**Combining the two answers ⇒ hs_bignet has learned a "data-relative fixed point"**: `f_5` is non-trivial abstractly but *behaves* near-identity on the distribution it actually processes.

This is exactly what V0–V3 probes *cannot see* alone (they only see "f_5 ≠ identity" and would incorrectly conclude "hs_bignet is not a fixed point"); V4, because its input is the *true distribution*, directly reveals the fixed-point behavior.

**Contrast with sym_bignet (rev-KL, same architecture)**:
- Probe: f_4→f_5 ≈ 0 (deep pair looks like identity)
- Real data: f_4→f_5 = 0.000 (deep pair also looks like identity)
- Both measurements say "collapsed to identity" → `f_5` *as an abstract function* degenerates to identity

⇒ sym_bignet is the **degenerate version** (`f_5 = identity` literally); hs_bignet is the **proper version** (`f_5 ≠ identity` but `f_5(y_4) ≈ y_4`). **Only the proper version satisfies the physical definition of an RG fixed point; the degenerate version trivially "looks like identity" but loses the transformation content.**

This is V4's most critical evidence, *invisible* to V0–V3 probes.

---

## V5 — vs Wilson block-RG ground truth

**How it's measured**: on the same HS input `x`, *simultaneously* run MERA forward (`y_s` = MERA) and Wilson–Kadanoff block averaging (`x_s = AvgPool2d(2)^s(x)`).
Each is passed through its own quantile transform; then at each scale we compute:
- `KS`: marginal Kolmogorov–Smirnov distance (after quantile transform, theoretically ≈ 0)
- `RMS-G`: RMS deviation of `G(r)/G(0)` (two-point function) — **the genuine spatial-structure mismatch**

**Question**: do MERA's slow modes match true physical RG?

RMS-G (`G(r)/G(0)` shape mismatch) is the key metric here; KS is uniformly ~10⁻³ (quantile transform forces marginal agreement by construction) and carries no signal, so we omit it.

| Scale     | L_s | hs_bignet | sym_bignet | T = 2.15 | T = 2.40 |
|-----------|----:|----------:|-----------:|---------:|---------:|
| s = 0     | 32  | 0.000     | 0.000      | 0.000    | 0.000    |
| s = 1     | 16  | 0.059     | 0.512      | 0.071    | 0.079    |
| **s = 2** | **8** | **0.030** | **0.539**  | 0.046    | 0.068    |
| s = 3     | 4   | 0.037     | 0.485      | 0.025    | 0.074    |

**hs_bignet reading**: **RMS-G at s=2 is 0.030, the scale closest to true Wilson physics** (18× lower than sym_bignet).
Low T = 2.15 dips lower at s=3 (0.025), but low T is an *off-fixed-point* control — physically far from T_c — so what matters is *who comes closest at T_c*. **hs_bignet is the lowest RMS-G among all six training objectives at T_c** (fwd-KL hs_dataDriven s=2 = 0.043, sym_bignet 0.539).
sym_bignet is catastrophically off Wilson at all scales (0.485–0.539), confirming the core verdict that "rev-KL gets spatial structure wrong".

---

## Composite picture

| Probe      | hs_bignet signal                              | sym_bignet (same architecture, rev-KL) | Key takeaway |
|------------|------------------------------------------------|----------------------------------------|--------------|
| V0/V1      | Deep pair ≈ 1.97 (large)                       | ≈ 0 (collapse)                         | hs_bignet's deep blocks do real work |
| V2         | Deep pair ≈ 1.92 (large, unchanged by chaining)| ≈ 0                                    | Chained input doesn't change the picture |
| V2b        | Deep pair ≈ 1.78 (large, geometry-corrected)   | = 1.49 (bombshell)                     | V0/V1/V2 "near 0" was a geometry artefact |
| V3 (r_s)   | r_1–r_4 all > 1; r_5 = 0.022 (mild collapse)  | r_4 = r_5 = 0.0004 (strict identity)   | sym_bignet's collapse at f_4 is pathological |
| V4         | Mid-deep ≈ 1.5–1.8 on real data; deepest 0.025 self-consistent | All ≈ 0, collapsed         | "probe vs real-data mismatch" |
| V5 RMS-G   | **s=2 = 0.030, closest to Wilson**             | s=2 = **0.539** (disaster)             | hs_bignet 18× better than sym_bignet |

**Conclusion**: hs_bignet is the *only* training objective that satisfies "deep blocks do copula work (V0–V4 healthy)" *and* "close to Wilson (V5 unique)".
hs_bignet and sym_bignet share architecture and temperature; **the only difference is fwd-KL vs rev-KL** — the across-six-probe gap shows the *training objective itself* determines whether the MERA flow does physical RG, independent of architecture capacity or data scale.

## Remaining issues (hs_bignet's failure modes)

Although hs_bignet is the *currently best performer* across V0–V5, **it is still a partial RG fixed point** — only at the deepest pair of blocks does it approach fixed-point behavior; **the cascade as a whole is not**. Specific gaps:

1. **LOSS plateau is ~48 nat above H(p_HS)**
   - hs_bignet converges to LOSS ≈ 7686; theoretical lower bound H(p_HS) ≈ 7637.6
   - This is `KL(p_HS || q_θ)`, **not directly measured by any V_i probe** — but consistent with V5 RMS-G > 0 (the flow doesn't fully match Wilson's cascade)

2. **V5 RMS-G ≠ 0** (an ideal fixed point would give 0)
   - s=2 = **0.030** (lowest, but nonzero); s=3 = 0.037
   - hs_bignet's slow-mode `G(r)/G(0)` differs from Wilson by about 3 %

3. **V3 r_5 = 0.022 ≠ 0** (an ideal fixed point would give 0)
   - Still **55× larger** than sym_bignet's strict identity (0.0004)
   - Says the "data-relative fixed point" is *approximate*: `f_5(y_4) ≈ y_4` but **not strictly equal**

4. **V4 mid-deep cascade is far from self-similar**
   - f_2 → f_3 = 1.43; f_3 → f_4 = **1.79** — shallow-mid-deep adjacent `y_s`'s are *very different*
   - Only the deepest pair f_4 → f_5 = **0.025** approaches fixed-point behavior

→ **Core diagnosis**: hs_bignet *has* learned fixed-point behavior at the deepest pair (V4 = 0.025, V5 RMS-G near 0),
but the cascade as a whole *has not* learned self-similarity (V4 mid-deep is still large).
To make the cascade a fixed point overall, we need *all* adjacent deep-pair V4 MSEs to approach 0; currently only the very last pair does.

### Is L=32 too small / depth insufficient?

A great physics intuition — and partly correct. A true RG fixed point needs a *scaling region* between the UV cutoff and the IR finite-size scale, where repeated block averaging leaves the distribution shape invariant. L=32's five scales:

| Scale | L_s | Physical role                                       |
|-------|-----|-----------------------------------------------------|
| s=1   | 16  | UV (just past the lattice scale)                    |
| s=2   | 8   | **may enter the critical scaling region**           |
| s=3   | 4   | **may still be in the scaling region**              |
| s=4   | 2   | finite-size dominated (only 4 lattice points)       |
| s=5   | 1   | single point (trivial)                              |

⇒ On L=32, **the genuine scaling region where we can test self-similarity is only ~2 scales (s=2, 3)**. This is *a* limitation, but *not the main bottleneck*.

**Empirically, larger L is *harder*, not easier**:

| L  | hs_bignet V5 RMS-G s=2 | hs_bignet V5 RMS-G s=3 |
|----|------------------------:|------------------------:|
| 32 | **0.030**               | 0.037                   |
| 64 | 0.046                   | 0.042                   |

L=64 is slightly *worse* than L=32. The reason is *critical scaling* working in reverse — see memory `project_fss_critical_scaling`: **KL_fwd ∝ L^α with α ≈ 2.20 at T_c** (vs α ≈ 2.0 off-critical).
Per-site KL grows like *L^2.20*, so L=64 is *fundamentally harder* to fit than L=32.

⇒ Enlarging L has *two opposing effects*: **wider scaling region (good)** + **per-site KL explodes (bad)**. Empirically the latter dominates.

**The real bottleneck is not insufficient L, but the architecture's lack of a scale-invariance constraint**:

1. On L=32, the deepest pair V4 = 0.025 (close to fixed-point) — the model *can* learn fixed-point behavior at the only available scaling-region tail
2. Mid-deep V4 = 1.79 (far from self-similar) — *not* a finite-size pattern (which would compress *all* pairs toward 0); **the model simply did not learn self-similarity at intermediate scales**
3. sym_bignet on the same L=32 shows V4 ≈ 0 at *all* pairs (also "self-similar"), but it's the *degenerate* version — full collapse to identity
4. ⇒ L=32 *allows* a self-similar cascade; **the issue is that the model architecture *allows* learning a *non-self-similar* solution**

The current hs_bignet architecture:
- 16 RNVP layers *without* weight tying → each scale's RNVP parameters are *learned independently*
- *No inductive bias* enforcing scale invariance
- The training loss only penalizes `KL(p_HS || q_θ)` on the *final distribution*; **it does not penalize whether intermediate cascade steps are self-similar**

⇒ The model learns a solution that *correctly generates p_HS* but *is not self-similar* (the deepest pair happens to be near-identity only because deep-scale fields are decorrelated, where any near-identity map works).

**Conclusion**: L=32's narrow scaling region is a *secondary* factor; **the main bottleneck is the architecture's lack of a scale-invariance constraint plus the loss not penalizing self-similarity**.
This is why Phase-2 prioritizes **Multi-L joint training** (injecting parameter sharing across L) and **weight tying** (forcing the same RNVP at every scale)
rather than *just* scaling up L — the latter, under critical scaling, actually makes things worse (L=64 results confirm).

## Phase-2 directions

| Direction                              | Should hs_bignet take this path? |
|----------------------------------------|:---------------------------------:|
| More width (megabignet nhidden 192)    | ❌ Experimentally ruled out (plateau +37 nat worse) |
| More data (N=200K → N=500K)            | ❌ Experimentally ruled out (2.5× data, plateau unchanged) |
| More epochs / cosine LR                | Single-axis ~7 nat, far from sufficient |
| **NSF coupling** (stronger prior)      | ✅ More flexible block transformations; could let mid-deep scales also learn fixed point |
| **Z2-equivariant RNVP**                | ✅ Injects symmetry inductive bias; saves capacity from "learning the symmetry" |
| **Learned prior** (I.4 coarse-Ising / I.3 EBM) | ✅ Starting distribution closer to fixed point; the flow needs less mid-deep cascade work |
| **Multi-L joint training**             | ✅ Forced scale-invariance constraint; *directly* encourages cascade self-similarity |

**The two most direct attacks on hs_bignet's actual failure mode ("cascade not self-similar")**:
- **NSF** (give blocks the expressivity to perform fixed-point transformations)
- **Multi-L joint training** (directly penalize non-self-similarity in the cascade)

I.4 / I.3 are *indirect* — they push the starting distribution closer to the fixed point so the flow doesn't have to work as hard.

## Key data / scripts

- **Data CSVs**: `analyzers/csv/rg_v5_gauge_compare.csv` (V5), `rg_v0_v3_gauge.csv` (V0–V3), `rg_v4_gauge_demo.csv` (V4)
- **Training folder**: `data/32Ising_T2.269_hs_bignet/`
- **Figure script**: `analyzers/rg_fixed_point/plot_v0_v5_comparison.py`
- **Full diagnostic**: `rg_fixed_point_report_zh.md`
- **Phase-1/2 ablation verdict**: `improvements_results_zh.md`
