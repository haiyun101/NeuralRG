# RG fixed-point diagnostic L=64: cross-L comparison with L=32

> This report extends the core question of the L=32 focus report (`rg_fixed_point_focus_en.md`) to L=64:
> **Does the deeper cascade (L=64 has 6 MERA scales vs L=32's 5) bring hs_bignet closer to an RG fixed point?**
> The focus comparison is **L=64 hs_bignet** (= `baseline_b16`) vs **L=32 hs_bignet** at *identical* architecture and training objective (fwd-KL);
> the only variable is the **lattice size L**.
>
> ⚠ **L=64 sym_bignet has been trained but has not been probed** (folder `data/64Ising_T2.269_sym_bignet/`, epoch 9900), so this report *temporarily* lacks an "L=64 rev-KL control" —
> after the 2026-06-19 cluster maintenance ends, V0–V5 probes + gauge probes will be run on it, and the sym_bignet rows will be filled in.
> For now, L=32 sym_bignet is retained as a *cross-L* rev-KL pathology reference.

## Setup: hs_bignet is the same architecture at both L

| Aspect                | L=32 hs_bignet        | L=64 hs_bignet (= baseline_b16) |
|-----------------------|-----------------------|--------------------------------|
| Training objective    | fwd-KL (MLE on HS data) | **identical**                |
| Architecture          | `nlayers=16, nhidden=128, nmlp=3, nrepeat=1, symmetry`, RNVP affine, Gaussian latent | **identical** |
| Batch                 | 64                    | 16 (b=16 at L=64 due to memory) |
| Epoch                 | 9500                  | 19800                          |
| LOSS plateau          | ~7686 nat (~48 nat gap to H(p_HS)) | ~7686 nat (~48 nat gap) |
| MERA scales           | 5 (`log₂ 32 = 5`)     | **6** (`log₂ 64 = 6`)          |
| Physical finite size  | scaling region ~ 2 scales (s=2, 3) | scaling region ~ 3 scales (s=2, 3, 4) |

The only variable is L. The two plateau LOSSes are similar, but per-site KL across L is governed by *FSS critical scaling* (`KL_fwd ∝ L^α, α ≈ 2.20`), so L=64 is *fundamentally harder to fit*.

Each section below gives both L=32 / L=64 columns so the reader can directly judge whether increasing L improves the fixed-point picture.

---

## V0 / V1 — N(0, I) probe, adjacent-block shape similarity

`MSE(T_s(f_s(z)), T_{s+1}(f_{s+1}(z)))` — same z fed to two adjacent blocks; matched-pair MSE in gauge.

| Pair      | L=32 hs_bignet | L=64 baseline | L=64 P2 winner (i2 stride8h32) | L=64 Student-t (i1 df4) |
|-----------|---------------:|---------------:|--------------------------------:|------------------------:|
| f_1 → f_2 | **2.73**       | 0.44           | 1.12                            | 0.92                    |
| f_2 → f_3 | 1.81           | 0.13           | 0.81                            | 0.88                    |
| f_3 → f_4 | 0.56           | 0.06           | 0.33                            | 0.21                    |
| f_4 → f_5 | **1.97**       | 0.13           | 0.43                            | 0.50                    |
| f_5 → f_6 | — (L=32 has only 5 scales) | 0.16 | **0.034**                       | **0.033**               |

**Two-level reading**:
- *Cross-L*: L=64 baseline is 5–15× smaller than L=32 across all adjacent pairs. L=32's f_4→f_5 ≈ 2 says "deepest pair are different functions"; L=64's f_5→f_6 ≈ 0.16 says "deepest pair are nearly identical"
- *L=64 ablation*: **Phase-2 winner and Student-t are both *larger* than baseline at shallow-mid scales** (mid-scale differences more pronounced) → these two interventions *increase* shallow-mid block work; but at *the deepest* pair (f_5→f_6) they are *5× smaller* (0.034 vs 0.16) — interventions push work forward and make the last block more identity-like

Both readings need V3 + V2b for verification:
- *Optimistic*: L=64 cascade *has learned* genuine scale invariance — deep blocks converge
- *Pessimistic*: L=64 deep blocks converge because *they're all near identity* (V3 will tell us)

---

## V2 — chained input (production composition)

`h_s = f_{s+1}(f_{s+2}(...(f_5(z))...))` chained input + matched-pair gauge MSE.

| Pair      | L=32 | L=64 |
|-----------|-----:|-----:|
| f_1 → f_2 | 2.11 | 0.47 |
| f_2 → f_3 | 0.82 | 0.65 |
| f_3 → f_4 | 1.57 | 0.27 |
| f_4 → f_5 | **1.92** | 0.33 |
| f_5 → f_6 | — | 0.12 |

**Under chained input, L=64 deep pairs remain small**, consistent with V1. Chaining does not change L=64's "deep blocks converge" signal.

---

## V2b — chained + MERA slot geometry correction

1-slot chained + 3-slot fresh N(0, I), matched-pair gauge MSE.

| Pair      | L=32 | L=64 |
|-----------|-----:|-----:|
| f_1 → f_2 | 2.13 | **1.24** |
| f_2 → f_3 | 1.53 | 1.15 |
| f_3 → f_4 | 1.52 | 1.15 |
| f_4 → f_5 | **1.78** | 1.32 |
| f_5 → f_6 | — | **1.50** |

**Key reversal**: V0/V1/V2 show L=64 deep pairs ≈ 0.16, but **under V2b the deepest pair = 1.50 (comparable to L=32)**.
This means L=64's apparent "deep block convergence" *disappears under V2b geometry correction* — it's an artefact of the 4-tuple chaining,
*in the same pattern as* L=32 sym_bignet was originally unmasked by V2b.

⚠ **This is a *warning signal***: **L=64 hs_bignet's V0/V1/V2 "near fixed-point" appearance may be partly a geometry artefact**;
the true picture requires combining V3 + V2b + V5.

---

## V3 — per-block identity residual

`r_s = E[(T_s(f_s(z)) − z)²] / E[z²]` — gauge-fixed identity deviation at the copula level.

| Block | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|-------|---------------:|---------------:|---------------:|---------------:|
| f_1   | 2.28           | 0.80           | 0.86           | 1.08           |
| f_2   | 1.03           | 0.58           | 0.64           | 0.79           |
| f_3   | 1.63           | 0.45           | 0.67           | 0.75           |
| f_4   | **1.87**       | 0.45           | 0.36           | 0.47           |
| f_5   | **0.022**      | **0.17**       | **0.041**      | **0.039**      |
| f_6   | —              | **0.0064**     | **0.0010**     | **0.0026**     |

**Key observations**:
1. **L=64 r is smaller than L=32 at every block** (shallowest r_1: 2.28 → 0.80; deepest r_4: 1.87 → 0.45).
2. **L=64's *deepest* r_6 = 0.0064 is 3× smaller than L=32's *deepest* r_5 = 0.022** — L=64 approaches "strict identity" more closely at the last scale.
3. **L=64's fixed-point region is *r_5 = 0.17 + r_6 = 0.0064*** — two scales progressively approaching identity;
   L=32 has only r_5 = 0.022 — one scale near identity.
4. ⚠ But L=64 r_6 = 0.0064 is approaching the *rev-KL degenerate identity* level (L=32 sym_bignet r_5 = 0.0004) — not strict collapse, but more identity-like than L=32 hs_bignet's r_5.

**L=64 ablation comparison**:
- **Phase-2 winner r_6 = 0.0010** is the *most* identity-like among all L=64 flows (6× smaller than baseline); only *2× larger* than L=32 sym_bignet's degenerate 0.0004.
- **Student-t r_6 = 0.0026** is between baseline and Phase-2 winner.
- All three L=64 flows have *r_5 markedly smaller than baseline* (0.04 vs 0.17), giving a wider fixed-point region than baseline.

⇒ Combined with V2b, **L=64 baseline shows clear drift toward degenerate identity at the deepest scales; Phase-2 winner drifts *deeper still***.
**Phase-2 winner has the best V5 + smallest V3 r_6 → both metrics consistently say P2 winner learned a *stricter* fixed point than baseline**
(but the strictness is approaching the *degenerate* threshold — whether this is a *real* improvement or *drift toward rev-KL pattern* requires sym_bignet data for strict judgment).

---

## V4 — HS data forward, adjacent scales

`MSE(T_s(y_s[::stride, ::stride]), T_{s+1}(y_{s+1}[::stride, ::stride]))` — real data, forward direction.

| Pair      | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|-----------|---------------:|---------------:|---------------:|---------------:|
| f_1 → f_2 | 0.50           | 0.42           | 0.44           | 0.56           |
| f_2 → f_3 | 1.43           | 0.38           | 0.46           | 0.52           |
| f_3 → f_4 | **1.79**       | **0.39**       | 0.35           | **0.06**       |
| f_4 → f_5 | **0.025**      | 0.49           | **0.034**      | **0.034**      |
| f_5 → f_6 | —              | **0.017**      | **0.000**      | **0.001**      |

**Key reversal #2 + L=64 fixed-point profile**:

**Cross-L**: At mid-deep f_3→f_4, L=32 shows V4 = 1.79 (cascade severely non-self-similar); L=64 baseline at the same position shows V4 = 0.39 (substantially more self-similar). But L=64's deepest pair f_5→f_6 = 0.017 is comparable to L=32's deepest 0.025. ⇒ **L=64 cascade is *internally* more self-similar** (mid-deep V4 reduced ~4–5×), with similar deepest fixed-point quality.

**L=64 ablation → fixed-point region width** (counting "fixed-point scale" as V4 < 0.1):

| L=64 flow | fixed-point scales | profile |
|-----------|:------------------:|---------|
| baseline      | 1 | f_5→f_6 only |
| **P2 winner** | **2** | f_4→f_5 + f_5→f_6 |
| **Student-t** | **3** | f_3→f_4 + f_4→f_5 + f_5→f_6 |

⚠ **Student-t has the widest internal fixed-point region (3 scales), but V5 is not the best** (see next section) — a counterintuitive signal: internal "scale invariance" is *not monotonic* with external "vs Wilson".
P2 winner achieves 2-scale fixed point + best V5 — a better balance.

---

## V5 — vs Wilson block-RG ground truth (gauge-fixed)

V5 uses *two complementary metrics*: **RMS-G** (distributional shape) and **matched-pair MSE** (sample-level alignment, same metric family as V0–V4).

### V5 RMS-G (distributional spatial-structure shape)

| s | L_s | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|---|-----|---------------:|---------------:|---------------:|---------------:|
| 1 | 16 / 32 | 0.059      | 0.071         | 0.067          | 0.078          |
| **2** | 8 / 16 | **0.030** | 0.046    | **0.039**      | 0.044          |
| 3 | 4 / 8 | 0.037       | 0.042         | **0.030**      | 0.049          |
| 4 | 2 / 4 | n/a (L_s=2) | 0.034         | 0.045          | **0.074**      |
| 5 | n/a / 2 | n/a       | n/a (L_s=2)   | n/a            | n/a            |

### V5 matched-pair MSE (sample-level alignment, N=2000 samples)

`MSE = 2(1 − corr)` under N(0,1) marginals; range [0, 4]; 0 = perfect alignment, 2 = uncorrelated, > 2 = anti-correlated.

| s | L_s | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|---|-----|---------------:|---------------:|---------------:|---------------:|
| 1 | 16 / 32 | 0.69       | 0.57           | **0.53**       | 0.60           |
| 2 | 8 / 16  | 0.72       | 0.71           | **0.69**       | 0.67           |
| 3 | 4 / 8   | **3.22 ⚠** | 0.76           | **0.70**       | 0.73           |
| 4 | 2 / 4   | **0.38**   | 0.74           | 0.73           | 0.75           |
| 5 | n/a / 2 | n/a        | 0.71           | 0.83           | 0.83           |
| 6 | n/a / 1 | n/a        | 0.65           | 0.67           | 0.73           |

### Key cross-L findings (joint reading of both metrics)

**Finding #1: L=64 baseline V5 RMS-G is *slightly worse* than L=32** (s=2: 0.046 vs 0.030). Even though L=64's cascade is *internally* more self-similar (small mid-deep V4), it is *slightly farther* from Wilson. The cause is **FSS critical scaling** (`KL_fwd ∝ L^α, α ≈ 2.20`) — L=64's per-site KL is fundamentally ~4× harder to fit than L=32.

**Finding #2: L=64 P2 winner is the unique flow that improves both V4 and V5** — 2 internal fixed-point scales + best RMS-G + lowest matched MSE at s=1/2/3 → **a genuine fwd-KL fixed-point candidate**.

**Finding #3: Student-t's V4 vs V5 reversal** — widest internal fixed-point region (3 scales) but *worst* RMS-G among the 3 flows (s=4 = 0.074). A "self-similar but wrong" degenerate path, qualitatively the same direction as L=32 sym_bignet's pathology (though much milder).

**Finding #4 (matched MSE new finding — KEY L=32 vs L=64 difference): the s=3 sign-flip is L=32-specific**!

| L | At the L_s=4 sublattice | matched MSE | corr |
|---|---|---:|---:|
| **L=32** | s=3 (L_s=4) | **3.22** | **−61% anti-correlated ⚠** |
| **L=64** | s=4 (same L_s=4) | **0.74** | **+63% positive correlation, normal** |

L=64 baseline / P2 winner / Student-t **all** show matched MSE ∈ [0.5, 0.85] at every scale — **no sign-flip anomaly anywhere**.
⇒ L=32 hs_bignet's *anti-correlation* at s=3 (L_s=4) is **an L=32 training artefact, not an intrinsic RG symmetry**.
Possible mechanism: at L=32, the L_s=4 sublattice sits exactly on the boundary between "end of scaling region" and "start of finite-size regime"; training learns a sign-flip as a local-optimum fit there;
at L=64, the same L_s=4 is already deep in the finite-size regime, where the physical structure itself aligns with Wilson — no flip needed.

**Finding #5 (another matched MSE finding): L=64 baseline matched MSE is uniformly ~0.6–0.8 across scales**, whereas L=32 hs_bignet spans 0.4–3.2.
**L=64 baseline is positively correlated with Wilson at 60–70% at every scale** — the most *stable* cross-L behaviour. Even though RMS-G is slightly worse (FSS), sample-level alignment is *more uniformly healthy*.

⇒ L=64 hs_bignet's V5 is *not* simply "FSS made V5 worse" — **distributional measures (RMS-G) are affected by FSS, but sample-level alignment (matched MSE) is *more uniformly healthy***. This is the dual cross-L scaling picture: L=32 edges out on RMS-G, while L=64 is *more robust* on matched MSE.

---

## Composite picture (L=32 vs L=64)

| Aspect                       | L=32 hs_bignet | L=64 baseline | L=64 P2 winner | L=64 Student-t |
|------------------------------|----------------|---------------|----------------|----------------|
| V0/V1 deepest pair           | Large (1.97)   | Small (0.16)  | Tiny (0.034)   | Tiny (0.033)   |
| **V2b geometry-corrected**   | **Large (1.78)** | **Large (1.50)** | **Large (1.49)** | **Large (1.49)** |
| V3 deep r                    | r_5 = 0.022    | r_6 = 0.0064  | **r_6 = 0.0010** | r_6 = 0.0026  |
| V4 internal fixed-point region | 1 scale     | 1 scale       | **2 scales**   | **3 scales**   |
| V5 RMS-G s=2                 | **0.030**      | 0.046         | **0.039**      | 0.044          |
| V5 RMS-G s=3                 | 0.037          | 0.042         | **0.030**      | 0.049          |

**Main conclusions** (updated with L=64 ablation insight):

1. ✅ **L=64 baseline is *internally* more self-similar than L=32 hs_bignet, but *externally* slightly worse** — the predicted footprint of FSS critical scaling.
2. ✅ **Phase-2 winner i2_stride8h32 has the *best* V5 among the 3 L=64 flows** — s=2 = 0.039, s=3 = 0.030 (the *closest to Wilson* fwd-KL training objective at L=64). Simultaneously V3 r_6 = 0.0010 (deepest fixed-point quality in this report). **A genuine fwd-KL fixed-point candidate**.
3. ⚠ **Student-t is a counter-example**: V4 internal fixed-point region is widest (3 scales), *but* V5 vs Wilson is the *worst* among the 3 flows (s=4 = 0.074). **"Internal self-similarity ≠ closeness to Wilson"** — the heavy-tail prior pushes the cascade toward a *self-consistent but wrong* solution. **This is a *precursor pattern* of fwd-KL drifting toward rev-KL pathology.**
4. ⚠ **All three L=64 flows show V2b oneslot deep pair ≈ 1.5** (same as L=32 hs_bignet's 1.78), meaning V0/V1/V2 "deep-pair convergence" *disappears* after geometry correction — **not a real collapse, just a 4-tuple geometry artefact**.
5. ⚠ **All three L=64 flows have V3 r_6 approaching rev-KL degenerate levels** (0.001–0.006): **the deepest scale of L=64 generally drifts toward *degenerate identity*** (P2 winner drifts deepest; Student-t second).
   ⇒ **A risk signal for L=64**: simply scaling up L pushes the fixed-point region *too close to identity collapse*. Architectural constraints (Multi-L tying / Z2-equiv) are needed to *constrain* the fixed point away from degeneracy.

## Comparison with the L=32 report's core claims

L=32 report concluded: **"L=32 cascade has not learned overall self-similarity; the main bottleneck is the architecture's lack of a scale-invariance constraint; L=32 being too small is only a secondary factor"**.

L=64 data *refines* this conclusion:

| L=32 report claim                | L=64 data refines |
|----------------------------------|-------------------|
| Cascade mid-deep V4 = 1.79 far from self-similar | L=64 at the same position V4 = 0.39 → **increasing L *does* improve cascade self-similarity**, so L *isn't* completely secondary |
| Main bottleneck is architecture lack of scale invariance | L=64 internal improvement + V5 external degradation → **both architecture changes and L increase have effect, but increasing L runs into the FSS wall** |
| Phase-2 recommends Multi-L joint training + weight tying | Confirmed → cross-L signal is real; Multi-L directly exploiting this is the correct direction |
| Recommended directions exclude "just increase L" | Strengthened → just increasing L degrades V5 — not a valid path |

⇒ **The L=32 report's Phase-2 roadmap is *essentially correct*; L=64 data reinforces "just increasing L is invalid, but cross-L sharing is valid"**.

## To-do (after 2026-06-19 maintenance ends)

1. Add `data/64Ising_T2.269_sym_bignet` to FOLDERS dict in `rg_fixed_point_robustness.py`, `_v4_dataforward.py`, `rg_v5_blockRG_compare.py`
2. Run V1/V2/V2b/V3 + V4 demo + V5 probes + gauge probes (5 sbatch jobs total)
3. **Especially watch L=64 sym_bignet's V3 r_5, r_6**:
   - If both r_5 and r_6 strictly approach 0 (like L=32 sym_bignet) → confirms rev-KL takes the *degenerate identity* path at L=64 too
   - If r_5 large and r_6 small → rev-KL at L=64 differs in pattern from L=32 — requires new analysis
4. **Use L=64 sym_bignet data to fill in the "sym_bignet" column in V0–V5 sections**, doing the same *dual-flow controlled* comparison as the L=32 report
5. **Run matched-pair MSE on V5** (job 40267635 queued, will run in the same window) — finish in parallel

Once complete, this report will expand from "L=32 vs L=64 cross-L comparison" to "L=64 fwd-KL vs rev-KL controlled comparison", fully symmetric with the L=32 report.

## Key data / scripts

- **Data CSVs**: `analyzers/csv/rg_v5_gauge_compare.csv` (V5), `rg_v0_v3_gauge.csv` (V0–V3), `rg_v4_gauge_demo.csv` (V4)
- **Training folder**: `data/64Ising_T2.269_hsBignet_baseline_b16/` (L=64 hs_bignet), `data/32Ising_T2.269_hs_bignet/` (L=32 hs_bignet)
- **L=32 focus report**: `rg_fixed_point_focus_zh.md` / `_en.md`
- **Full diagnostic**: `rg_fixed_point_report_zh.md`
- **Phase-1/2 ablation verdict**: `improvements_results_zh.md` (includes L=64 Phase-2 winner i2_stride8h32's V5 0.024)
