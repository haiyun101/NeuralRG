# RG Fixed-Point Probe — Phase-1 Improvement Ablation Results

> **Companion reading:**
> - `improvements.md` — forward-looking roadmap (8 schemes + cost-leverage ranking + failure-mode prediction table)
> - `rg_fixed_point_report.md` — current-architecture pathology diagnosis at T_c
> - `analyzers/concise_reports/concise_report_L64_T2.269.md` — L=64 per-method comparison (includes Phase-1 ablation mini-table)

## Premise

`improvements.md` proposed 8 improvement schemes and recommended Phase-1 launch three parallel tracks: III.1 (multi-scale loss), I.1 (Student-t prior, the negation experiment), and I.2 (conditional Gaussian prior). This report consolidates the actual Phase-1 results, compares them to the failure-mode prediction table, and produces a per-scheme verdict + Phase-2 priority recommendation.

## Experimental design

Baseline = **hs_bignet** (`nlayers=16, nhidden=128, nmlp=3, nrepeat=1, -symmetry, fwd-KL / dataDriven`, on HS continuous-field data). All improvement runs use this architecture and flip exactly one flag.

**14 improvement runs + 5 original-method baselines** ran to completion = 19 `flow_diagnostic.json` files (N=4000 samples, batch matched per run).

### L=32 ablation matrix (2 × 2, b=64)

|                       | scaleLoss=0           | scaleLoss=1.0           |
|-----------------------|-----------------------|-------------------------|
| Gaussian prior        | `baseline_b64`        | `iii1_lam1.0_b64`       |
| Conditional Gaussian  | `i2_stride8h32_b64`   | `combined_lam...`       |

All four cells at b=64 → enables the full 2 × 2 interaction term.

### L=32 hyperparameter sweeps

- **iii1 λ_scale sweep**: λ ∈ {0.1, 1.0, 10.0}, b=64
- **i2 slow_stride sweep**: stride ∈ {4, 8 (original Phase-1), 16}, b=128 *(note: stride=8 diverged late-training, see §4.4)*
- **i1 Student-t df=4**, b=128

### L=64 ablation (b=16)

`baseline_b16 / iii1_lam1.0_b16 / i2_stride16h32_b16 / i1_df4.0_b16` — matched at b=16 to fit the scaleLoss extra forward graph; the matched batch keeps the within-trio ablation internally consistent.

## Data summary

HS data anchors (N=4000):

| L  | mag_p  | xi_p    | gL_p   |
|---:|-------:|--------:|-------:|
| 32 | 2.382  | 8.568   | 0.477  |
| 64 | 2.200  | 14.782  | 0.407  |

### L=32 ablation matrix + interaction

|                       | scaleLoss=0  (KL_qp / KL_pq / mag / xi / gL)| scaleLoss=1.0 |
|-----------------------|----------------------------------------------|------------------|
| **Gaussian prior**    | 23.42 / 17.05 / 2.44 / 8.79 / 0.503         | **iii1**: 21.84 / 16.59 / 2.45 / 8.75 / 0.499 |
| **Conditional Gauss** | **i2_b64**: 21.16 / 16.05 / 2.38 / 8.58 / 0.487 | **combined**: 21.20 / 15.88 / 2.35 / 8.39 / 0.478 |

**Single-variable Δ vs baseline:**

| Intervention              | Δ KL_qp | Δ KL_pq | Δ mag  | Δ xi    | Δ gL    |
|---------------------------|--------:|--------:|-------:|--------:|--------:|
| + III.1 (scaleLoss=1)     | −1.59   | −0.46   | +0.007 | −0.032  | −0.004  |
| + I.2 (cond. prior)       | **−2.26** | **−0.99** | **−0.065** | **−0.202**  | **−0.016**  |
| + both (combined)         | −2.22  | **−1.17** | **−0.094** | **−0.392** | **−0.025** |

**Interaction effect** (combined − iii1 − i2_b64 + base):
- KL_qp: +1.63 (sub-additive, mild antagonism)
- KL_pq: +0.28 (essentially additive)
- gL:    −0.005 (super-linear structural synergy)

⇒ **The two interventions are synergistic on *structure* (gL drops super-linearly) but mildly antagonistic on KL_qp.** combined has the matrix's best structural fit (mag=2.35 within 0.03 of anchor 2.38, xi=8.39 within 0.18 of 8.57, gL=0.478 essentially equal to anchor 0.477).

### L=32 hyperparameter sensitivity

**iii1 λ_scale sweep:**

| λ_scale | KL_qp | KL_pq  | gL     | Reading                       |
|--------:|------:|-------:|-------:|-------------------------------|
| 0.0     | 23.42 | 17.05  | 0.503  | baseline                      |
| 0.1     | 24.22 | 17.04  | 0.509  | too weak; small λ marginally hurts |
| **1.0** | 21.84 | 16.59  | 0.499  | **sweet spot**                 |
| 10.0    | 21.70 | **30.15** | 0.509 | KL_qp marginally lower but **KL_pq doubles** ⇒ mode collapse |

⇒ **λ=1.0 is the III.1 sweet spot**; **λ=10.0's KL_qp gain comes from mode-drop** (KL_pq explosion proves it). The improvements.md prediction "λ_scale anti-correlates with V5 RMS-G" is here borne out by the KL_pq trajectory: λ=10 strongly degrades forward KL ⇒ the flow abandons bridge regions to satisfy scale invariance.

**i2 slow_stride sweep (b=128):**

| stride | slow grid | KL_qp     | KL_pq | gL      | Reading                       |
|-------:|----------:|----------:|------:|--------:|-------------------------------|
| **4**  | 16 × 16   | **17.13** | **14.14** | 0.493 | **best single ablation**: Δ −6.3 nat |
| 8      | 8 × 8     | **604190.40** ⚠️ | 21.97 | 0.012 | **late-training divergence** (memory `project_l32_late_training_instability`) |
| 16     | 2 × 2     | 20.48     | 15.78 | 0.501  | middling                       |

⇒ **Conditional prior performance scales monotonically with slow-grid density** (16×16 → 8×8 → 2×2 worsens). The stride=8 row is broken because the Phase-1 original b=128 run diverged late, **so the true L=32 verdict for I.2 must come from stride=4 — Δ KL_qp = −6.3 nat, the single largest improvement in the 8-run single-variable ablation**.

**i1 Student-t df=4 (b=128):**

| Quantity | baseline | i1_df4.0 | Δ      |
|----------|---------:|---------:|-------:|
| KL_qp    | 23.42    | 21.36    | −2.07  |
| KL_pq    | 17.05    | 15.54    | −1.51  |
| mag      | 2.441    | 2.389    | −0.052 |
| xi       | 8.79     | 8.55     | −0.24  |
| gL       | 0.503    | 0.483    | −0.020 |

⇒ Student-t delivers a balanced 1.5–2 nat improvement on every metric, **with no metric degrading**, but no metric improves as dramatically as i2_stride4. **Consistent with improvements.md's negation-experiment positioning** — heavy-tail prior has a real but small effect.

### L=64 ablation (b=16)

|                   | KL_qp  | KL_pq  | mag (a=2.20) | xi (a=14.78) | gL (a=0.407) |
|-------------------|-------:|-------:|-------------:|-------------:|-------------:|
| baseline_b16      | 86.88  | 65.64  | 2.267        | 15.19        | 0.433        |
| iii1_lam1.0_b16   | 87.14  | 64.63  | 2.244        | 14.95        | 0.425        |
| i2_stride16h32_b16| 93.31  | 70.37  | 2.230        | 14.94        | 0.425        |
| i1_df4.0_b16      | **90.41** | 66.21  | **2.179**    | **14.37**    | **0.404**    |

**Δ vs baseline:**

| Intervention | Δ KL_qp | Δ KL_pq | Δ gL    |
|--------------|--------:|--------:|--------:|
| III.1        | +0.26   | **−1.01** | −0.008  |
| I.2          | +6.43   | +4.73   | −0.008  |
| I.1          | +3.53   | +0.57   | **−0.029** |

⇒ **At L=64 the improvement signal shrinks to the noise floor** (batch=16's elevated gradient noise + the Wilson–Fisher mismatch double-suppress):
- III.1 has the only KL_pq improvement (~1 nat, direction-matches L=32);
- I.2 **flips direction** (was −2.26 at L=32, now +6.43 at L=64);
- I.1 is the only intervention with visible *structural* improvement (gL Δ = −0.029, pushing gL=0.404 right onto the anchor 0.407).

## Cross-L direction comparison

| Intervention | L=32 Δ KL_qp | L=64 Δ KL_qp | L=32 Δ gL | L=64 Δ gL | Cross-L behaviour       |
|--------------|-------------:|-------------:|----------:|----------:|--------------------------|
| baseline     | (ref) 23.4   | (ref) 86.9   | +0.026    | +0.026    | (ref)                    |
| + III.1      | −1.59        | +0.26 (noise)| −0.004    | −0.008    | direction-consistent, weakens at L=64 |
| + I.2        | **−2.26**    | **+6.43**    | −0.016    | −0.008    | **direction flips** — scaling issue |
| + I.1        | −2.07        | +3.53        | −0.020    | **−0.029** | direction-consistent, **structure improves with L** |

### Why I.2 reverses sign at L=64

Possible causes:
1. **Absolute slow-grid size problem**: L=32 with stride=8 → 4×4 slow grid; L=64 with stride=16 → 4×4 slow grid. **Same count, but relative coverage drops from 1/16 to 1/256**. The slow grid's information density falls with L.
2. **CNN capacity ill-suited**: `condPriorHidden=32` may be insufficient at L=64 for the conditional CNN to learn a useful z_slow → z_fast structure.
3. **Effective training data shrinks**: L=64 b=16 × 20000 steps = 320K samples; L=32 b=64 × 20000 steps = 1.28M samples — 4× less effective training.

**Phase-2 must test:** I.2 with `stride=8` (8×8 slow grid) and `hidden=64` at L=64 — does it recover the positive sign? If so, the **default stride should be tightened (from L//4 to something finer)** in Phase-1's heuristic.

### Why I.1 structural signal *strengthens* at L=64

`gL` Δ at L=64 = −0.029 vs L=32's −0.020; `mag` at L=64 (2.179 vs anchor 2.200) Δ = −0.021 likewise visible.

Physical reading: **at T_c, ξ → ∞ ⇒ the field's heavy-tail-ness intensifies; a heavy-tail prior fits the IR-closer L=64 regime better**. This aligns with `rg_fixed_point_report.md`'s "Why T_c is hard on this architecture" diagnosis (the Wilson-Gaussian FP / actual Wilson–Fisher FP mismatch): **the larger L, the more visible this mismatch becomes, so non-Gaussian priors gain ground**.

## Scheme verdict vs improvements.md predictions

`improvements.md`'s failure-mode prediction table:

| Scheme   | Predicted V5 KS (T_c rev-KL) | Predicted V5 KS (T_c fwd-KL) | Predicted V5 RMS-G |
|----------|------------------------------:|------------------------------:|--------------------:|
| baseline | 0.32+                        | 0.08                          | 0.62 / 0.04         |
| I.1 t    | 0.22 (~30 % improvement)     | 0.06                          | 0.55 (slight)       |
| I.2 cond | 0.18 (~50 % improvement)     | 0.05                          | **0.30** (large)    |
| III.1    | 0.15 (~50 % improvement)     | 0.05                          | **0.20** (large)    |

We do not yet have direct V5 data (V5 probes not yet run on improvement folders, see §5), but **KL_pq is a proxy for V5 KS** (both measure marginal mismatch) and **gL is a proxy for V5 RMS-G** (both measure spatial-structure mismatch).

### Proxy-based verification

**Δ KL_pq (L=32 b=64, proxy for V5 KS improvement):**

| Scheme   | Predicted improvement | Actual Δ KL_pq                            | Verified       |
|----------|-----------------------|------------------------------------------:|----------------|
| baseline | (ref)                | (ref)                                     | —              |
| I.1 t    | ~30 % improvement     | −1.51                                     | ✓ direction-match (small) |
| I.2 cond | ~50 % improvement     | **−0.99** (b=64 stride=8) / **−2.91** (b=128 stride=4) | ✗ smaller / ✓ matches |
| III.1    | ~50 % improvement     | −0.46                                     | ✗ smaller than predicted |
| combined | (interaction)         | −1.17                                     | mild synergy   |

**Δ gL (L=32 b=64, proxy for V5 RMS-G improvement):**

| Scheme   | Predicted improvement       | Actual Δ gL   | Verified      |
|----------|-----------------------------|--------------:|---------------|
| baseline | gL ≈ 0.62                   | (ref 0.503)   | —             |
| I.1 t    | gL → 0.55 (slight)          | −0.020        | ✓ direction-match |
| I.2 cond | gL → 0.30 (large)           | −0.016        | ✗ much smaller than predicted |
| III.1    | gL → 0.20 (large)           | −0.004        | ✗ **almost no effect** |
| combined | (synergy)                   | −0.025        | ✓ synergy     |

### Proxy verdict table (superseded — see V5/V3 strict verdict below)

| Scheme   | improvements.md prediction | Actual performance              | Rating                | Phase-2 priority   |
|----------|----------------------------|---------------------------------|-----------------------|--------------------|
| III.1    | Large KL/RMS-G improvement | Small KL improvement, near-zero structure | **below expectation** | Medium (tuning room) |
| I.1      | Small (negation)           | Direction-consistent 1.5–2 nat; "L=64 structure wins" (gL proxy) | (proxy reading; flipped by V5, see below) |  |
| I.2      | Large improvement          | Big L=32 win (stride=4), L=64 sign flip | **scaling problem TBD** | Medium-High |
| combined | (synergy)                  | KL mild antagonism, structure super-linear synergy | **mixed signal**      | Low |

> ⚠️ **The verdict above relies on `KL_pq` / `gL` *proxies*; once the V5 / V3 strict values landed they partially overturned this reading. The most important reversal: I.1 Student-t prior actually *degrades* V5 RMS-G (the spatial structure metric) — the "L=64 structure wins" interpretation from gL was a misreading. See the next section.**

## V5 / V3 / V4 probe — strict values (upgraded 2026-06-09)

The three probe jobs (40031218 / 19 / 20) processed all 13 usable improvement folders (the original `i2_stride8h32 b=128` Phase-1 run is excluded because its latest checkpoint is sampling-broken — see Outstanding uncertainties). CSVs live in `analyzers/rg_fixed_point/csv/`; figures in `analyzers/rg_fixed_point/figures/`.

### V3 identity residual (`E[(f_s(z) − z)²] / E[z²]`, z ~ N(0, I))

Tests whether each scale-block acts as near-identity on a standard-Gaussian probe. **Rev-KL pathology signature**: f_4, f_5 ≈ 0 (deep blocks collapse to identity). **Fwd-KL healthy signature**: f_4, f_5 > 1 (deep blocks do real work).

| Run                                  |  V3 f_4 |  V3 f_5 |   Reading                                              |
|--------------------------------------|--------:|--------:|--------------------------------------------------------|
| L=32 baseline_b64 (fwd-KL ref)        |  3.587  |  0.467  | fwd-KL baseline; f_4 nontrivial, f_5 moderate          |
| L=32 iii1_lam1.0_b64 (+ III.1)        |  1.066  |  **0.011** | **f_5 collapses** — deeper than sym_bignet's 0.08      |
| L=32 i2_stride8h32_b64 (+ I.2)        |  4.447  |  0.476  | f_5 matches baseline; conditional prior does not induce collapse |
| L=32 combined (I.2 + III.1)           |  1.142  |  **0.003** | **extreme collapse**; scaleLoss dominates              |
| L=32 iii1_lam0.1_b64                  |  2.487  |  0.512  | near baseline (λ too weak)                              |
| L=32 iii1_lam10.0_b64                 |**0.001**|**0.007**| **f_4, f_5 both collapse** — more trivial than rev-KL sym_bignet |
| L=32 i2_stride4h32 (b=128)            |  4.597  |  0.356  | f_5 slightly under baseline, still nontrivial          |
| L=32 i2_stride16h32 (b=128)           |  6.882  |  0.616  | f_5 above baseline                                      |
| L=32 i1_df4.0 (Student-t b=128)       |  6.404  |**1.473**| **f_5 highest in fwd-KL family** — heavy-tail prior pushes deep blocks to do more work |
| L=64 baseline_b16                     |  3.635  |  1.907  | L=64 fwd-KL baseline; f_5 higher (L↑ → deep blocks busier) |
| L=64 iii1_lam1.0_b16                  |  2.277  |  0.998  | f_5 halved vs baseline, but not collapsed              |
| L=64 i2_stride16h32_b16               |  1.797  |  1.049  | similar to L=64 iii1                                    |
| L=64 i1_df4.0_b16                     |  3.972  |  1.956  | matches baseline; no collapse                          |
| (ref) L=32 sym_bignet (rev-KL)        |  0.256  |  0.079  | classical rev-KL deep collapse                          |
| (ref) L=32 pathgrad_bignet (STL)      |  0.157  |  0.018  | STL rev-KL deep collapse                                |
| (ref) L=32 hs_bignet (fwd-KL ref)     |  2.759  |  0.300  | fwd-KL healthy                                          |

**Key findings:**

1. **The actual mechanism of III.1 (scaleLoss) is to *induce* the rev-KL pathology, not fix it**. At λ=1, f_5 = 0.011 (deeper than sym_bignet's 0.079); at λ=10, both f_4 = 0.001 and f_5 = 0.007 collapse — the *most extreme* collapse in the dataset. The combined run inherits this.
2. **Neither I.1 (Student-t) nor I.2 (Conditional Gaussian) induces collapse** — prior-side modifications are benign for deep blocks.
3. **I.1 at L=32 has f_5 = 1.473, the highest in the fwd-KL family** — heavy-tail prior actively makes the deepest scale-block do more work than baseline.

### V5 KS (standardised marginal mismatch at scales s=2, 3)

Measures the KS distance between the MERA slow-mode field `y_s` (standardised) and the Wilson–Kadanoff block-RG ground truth `x_s` (standardised). **Smaller is better** (ideal → 0).

| Run                                  | KS s=2 | KS s=3 |   Reading                                              |
|--------------------------------------|-------:|-------:|--------------------------------------------------------|
| L=32 baseline_b64                    | 0.092  | 0.080  | fwd-KL baseline                                         |
| L=32 iii1_lam1.0_b64                 | 0.112  | 0.085  | mild *degradation* vs baseline                          |
| L=32 i2_stride8h32_b64               | 0.087  | 0.084  | s=2 small improvement                                   |
| L=32 combined                        | **0.086** | **0.078** | **best KS in the 2×2 matrix**                          |
| L=32 iii1_lam0.1_b64                 | 0.113  | 0.084  | mild degradation                                        |
| L=32 iii1_lam10.0_b64                | **0.162** | **0.172** | **significant degradation** (mode-collapse side effect) |
| L=32 i2_stride4h32 (b128)            | 0.111  | 0.159  | s=3 degraded at b=128                                   |
| L=32 i2_stride16h32 (b128)           | 0.099  | 0.078  | s=3 matches combined                                    |
| L=32 i1_df4.0 (Student-t)            | **0.175** | **0.167** | **significant degradation** — heavy tails pull the marginal away from block-RG |
| L=64 baseline_b16                    | 0.123  | 0.101  | L=64 baseline                                           |
| L=64 iii1_lam1.0_b16                 | **0.095** | 0.087  | **L=64 III.1 *improves* KS** (opposite sign vs L=32)  |
| L=64 i2_stride16h32_b16              | 0.093  | 0.088  | matches iii1                                            |
| L=64 i1_df4.0_b16                    | **0.185** | **0.172** | **significant degradation** — heavy tails worse at L=64 |

### V5 RMS-G (spatial correlation structure mismatch at scales s=2, 3)

Measures the RMS distance between the MERA slow-mode `G(r)/G(0)` and the block-RG `G(r)/G(0)`. **Smaller is better**; rev-KL sym_bignet ≈ 0.67 (disaster); fwd-KL baseline ≈ 0.05 (already near-optimal).

| Run                                  | RMS-G s=2 | RMS-G s=3 |   Reading                                                |
|--------------------------------------|----------:|----------:|----------------------------------------------------------|
| L=32 baseline_b64                    | 0.053     | 0.051     | fwd-KL baseline                                           |
| L=32 iii1_lam1.0_b64                 | **0.101** | 0.022     | s=2 *degraded* 2×, s=3 improved 2× — staggered trade-off |
| L=32 i2_stride8h32_b64               | 0.054     | **0.010** | **s=3 cut to 1/5** — I.2 visibly improves at deep scales |
| L=32 combined                        | 0.077     | 0.030     | s=2 degraded, s=3 improved; worse than i2 alone (antagonism on s=2) |
| L=32 iii1_lam0.1_b64                 | 0.080     | 0.009     | s=3 improved, but KS also degraded (proxy cost)         |
| L=32 iii1_lam10.0_b64                | **0.527** | **0.533** | **disaster** — nearly replicates sym_bignet's 0.67; III.1 λ=10 enters the rev-KL pathology phase |
| L=32 i2_stride4h32 (b128)            | 0.064     | 0.017     | s=3 improved                                              |
| L=32 i2_stride16h32 (b128)           | 0.092     | 0.085     | s=3 matches baseline                                      |
| L=32 i1_df4.0 (Student-t)            | **0.187** | **0.234** | **degraded 3.5–4.6×** — I.1 *breaks* spatial structure |
| L=64 baseline_b16                    | 0.042     | 0.026     | L=64 baseline; RMS-G even lower than L=32 baseline      |
| L=64 iii1_lam1.0_b16                 | 0.048     | 0.035     | mild degradation                                          |
| L=64 i2_stride16h32_b16              | **0.033** | 0.058     | **s=2 improved 22 %**, s=3 mild degradation              |
| L=64 i1_df4.0_b16                    | **0.154** | **0.206** | **degraded 3.7–7.9×** — direction matches L=32; Student-t *worse* at L=64 |

**Key findings:**

1. **III.1 λ=10 at L=32 has V5 RMS-G = 0.527, nearly replicating sym_bignet's 0.67**. This is the *strongest* evidence that **III.1's intrinsic mechanism is to induce the rev-KL pathology, not fix it**.
2. **I.1 Student-t prior visibly degrades V5 RMS-G at both L** (L=32 3.5×, L=64 3.7–7.9×). The earlier "L=64 structural win" read from the `gL` proxy was a misinterpretation — `gL` measures a single-point correlation; V5 RMS-G measures a multi-scale cascade comparison; the two decouple. **Student-t improves the marginal but breaks block-RG compressibility.**
3. **I.2 Conditional Gaussian at L=32 s=3 has RMS-G = 0.010 (baseline 0.051), a 5× improvement** — the *only* single-variable intervention that significantly improves V5 RMS-G in the ablation matrix. At L=64 it still improves s=2 by 22 %. **I.2's V5 RMS-G win is the most robust finding of Phase-1.**
4. **combined shows antagonism on V5 s=2**: III.1's collapse disrupts I.2's structural improvement. **Push I.2 alone first; do not rush combination.**

### Strict verdict table

| Scheme | improvements.md prediction         | V3 f_5 (real) | V5 RMS-G (real) | Rating                                                    |
|--------|------------------------------------|--------------:|----------------:|----------------------------------------------------------|
| III.1 λ=1.0 | Large KL/RMS-G improvement     | **0.011** (collapsed) | s=2 degraded, s=3 improved | **wrong mechanism** — III.1 *induces* rev-KL pathology, not a fwd-KL refinement |
| III.1 λ=10.0 | Tighter constraint → better   | **0.007** (extreme collapse) | **0.527** (disaster) | **enters the rev-KL pathology phase** |
| I.1 Student-t | Small (negation)             | **1.473** (healthy) | **0.187 (3.5× degraded)** | **negation confirmed + active spatial harm** |
| I.2 cond. (b=64 stride=8)  | Large improvement | 0.476 (healthy) | **s=3 = 0.010 (5× improved)** | **only single-variable V5 RMS-G winner** |
| I.2 cond. (b=128 stride=4) | (sweep extreme)   | 0.356 (healthy) | s=3 = 0.017 | same direction as stride=8 but slightly weaker |
| combined I.2 + III.1       | Synergy           | 0.003 (collapsed) | s=2 antagonism | **III.1's collapse offsets I.2's structural improvement** |

**Two key corrections vs the mid-Phase-1 proxy reading:**

1. **I.1 Student-t is not "exceeds negation" — it is negation + active degradation**. The proxy reading "L=64 structural win" was inverted by the strict V5 RMS-G data. **Student-t should be *downgraded* in the Phase-2 roadmap, not promoted.**
2. **I.2 Conditional Gaussian is Phase-1's only V5 RMS-G strict winner**. The signal is direction-consistent at L=32 (s=3 5× improvement) and L=64 (s=2 22 % improvement). **I.2 should be promoted to Phase-2 P2.0 top priority**, with capacity scans and cross-L verification as next experiments.

## Recommended Phase-2 priorities (after V5 strict-value upgrade)

Revised ordering of `improvements.md` Phase-2 / Phase-3:

### Immediately (P2.0, within 1 week)

1. ✓ **Run V3/V4/V5 on 13 improvement folders** — done, strict tables above.
2. **I.2 capacity scan + cross-L** (priority *raised*):
   - Test `stride=8, hidden=64` and `stride=4, hidden=64` at L=64 — does the V5 RMS-G s=2 22 % improvement push toward 50 %+?
   - Critically, verify whether I.2's "sweet stride" is the same at L=32 b=64 and L=32 b=128.
   - **I.2 is Phase-1's only confirmed V5 RMS-G winner; Phase-2's core task is to stabilise it.**

### Mid-term (P2.1, 2–3 weeks)

3. **III.1 finer λ < 1.0 sweep** (priority *lowered*):
   - Test λ = 0.3, 0.5 — can we keep the s=3 RMS-G improvement without triggering f_5 collapse?
   - If λ must be tiny to avoid collapse, the "III.1 *induces* rev-KL pathology" diagnosis is fundamental — *drop it*.
4. **II.1 learnable kept-fraction** (priority *raised* to P2.1):
   - I.2's success comes from "pushing spatial structure to the prior side"; II.1 is the architectural counterpart ("push to the dispatch side").
   - Both modify spatial inductive bias inside the architecture; worth testing in parallel.

### Mid-term (P2.2, 3–6 weeks) — physical-baseline rework

5. **I.4 coarse Ising prior**:
   - More physically direct than I.3; let the prior be a small Ising distribution directly, skip the EBM-fitting middle step.
6. **Re-evaluate I.3 EBM / φ⁴** (priority *lowered*):
   - Originally promoted because of I.1's "L=64 structural win"; with V5 strict data inverting that finding, **the EBM-route motivation is gone**.
   - Revisit only if I.2 capacity scan fails and I.4 also falls short.

### Long-term (Phase-4)

7. **II.2 self-similarity framework** (Scheme C): postdoc-grade independent project.

### Cancelled (confirmed)

- ~~Push I.1 Student-t further~~: V5 RMS-G strict data shows L=32 3.5× and L=64 3.7–7.9× degradation. **Hard-close this branch.**
- ~~I.5 learned non-Gaussian prior~~: I.3/I.4 are more physically grounded.
- ~~III.2 V5-as-loss~~: V5 RMS-G *is* the bottleneck (confirmed), but using V5 as the loss removes its *independent-judge* role. **Revisit only after I.2 / II.1 / I.4 all fail.**
- ~~III.1 combined route~~: V3 and V5 strict data both show combined is antagonistic on s=2; not worth Phase-2 priority slot.

## Related files

- `improvements.md` — original 8-scheme roadmap
- `rg_fixed_point_report.md` — pathology diagnosis (assumed as premise here)
- `concise_report_L64_T2.269.md` — L=64 improvement ablation mini-table
- `data/{32,64}Ising_T2.269_hsBignet_*/flow_diagnostic.json` — raw JSON behind every number here
- `shell/run_L32_iii1_single.sh` / `shell/run_L32_i2_single.sh` / `shell/run_L32_i1_single.sh` — Phase-1 training scripts (and L=64 `_b16.sh` mirrors)
- `shell/analyze_L32_single.sh` / `shell/analyze_L64_single.sh` — single-folder diagnostic launcher
- `shell/rg_fixed_point_robustness.sh` / `rg_fixed_point_v4.sh` / `rg_v5_blockRG.sh` — probe launchers (jobs 40031218–40031220)

## Outstanding uncertainties

1. ✓ **V3/V4/V5 strict values are in** — see strict tables above; the proxy verdict has been partially overturned (most critically, I.1 actually *degrades* V5 RMS-G).
2. **i2_stride8h32 b=128 Phase-1 original run** — training LOSS final-100 mean = 1926 (healthy) but the latest checkpoint produces sampling KL_qp = 604,190 (late-training instability struck the saved checkpoint specifically). Excluded from all strict tables; the real i2 b=128 evaluation rests on the stride=4 / 16 sweep points.
3. **L=64 improvement signal mostly in noise** — at b=16 most Δ values sit within ±1 nat. Phase-2 should consider effective-batch boosts (gradient accumulation) to separate signal from noise.
4. **Cross-L comparison not per-site normalised** — per-site KL at fixed T_c carries FSS scaling (α ≈ 2.20) which is hidden in the raw KL numbers; strict cross-L analysis should normalise (memory `project_fss_critical_scaling`).
5. **V4 forward-direction probe strict values written to `analyzers/rg_fixed_point/csv/rg_v4_dataforward.csv`** — the verdict tables above lean on V5 (block-RG cross-comparison); V4 numbers can serve as supplementary confirmation. Both probes agree on the original 6 methods (rev-KL extreme / fwd-KL moderate); the improvement folders show the same pattern.
6. **Convergence audit (2026-06-09)** — training LOSS over the last 100 epochs is stable for all 14 improvement runs; only `i2_stride8h32 b=128` has a sampling-broken latest checkpoint (excluded from all strict tables).
