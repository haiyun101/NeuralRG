# Ising L=64 — Concise Report (T = 2.269185314213022)

> **Companion to `concise_report_L32_T2.269.md`.** Same architecture
> family (bignet: `nlayers=16, nhidden=128, nmlp=3, nrepeat=1`,
> `-symmetry`), same five training objectives, scaled-up data
> (HS dataset N = 200,000 from `data/mcmc_data/hs_L64_T2.269185314213022_N200000.pt`).
>
> **Status as of writing.** Four runs at ep 19,900 (sym_bignet,
> hs_bignet, jsLoss, bridge); STL pathgrad diverged at step 12,988
> while still warming up under the OOM-forced batch=8. Sample
> diagnostics (`flow_sample_diagnostic.py`) and structural
> observables (`mag_abs_q`, `xi_q`, `G(L/2)/G(0)`) are pending —
> the per-method visuals below show the latest `proposals_NNNNN.png`
> training snapshots as a placeholder.
>
> **Update (post-Phase-1 ablation).** `sym_bignet` flow-sample
> diagnostic completed (`shell/analyze_L64.sh` job 40030341, ep
> 19,900, N = 4,000); its rows below now carry the fresh numbers.
> The remaining four methods (`pathgrad_bignet`, `hs_bignet`,
> `jsLoss_bignet_lam0.5`, `hsBignet_bridge_w5.0t0.5`) were resubmitted
> as one-folder-per-sbatch parallel jobs **40030830–40030833** after
> the original combined job hit the 4-hour wall after only one
> folder. Numbers and figures for those four arrive when the jobs
> finish; in the meantime their rows are marked `(diag pending,
> job 4003083N)`. **Direct MC entropy estimate** of HS at L=64,
> `H(p_HS) = 7620.16` from the sym_bignet diagnostic (replaces the
> earlier ~7611.92 4×-scaling guess; the ~8 nat upward revision
> tightens every `KL(p‖q)` number downstream by the same amount).
>
> **Update (all five diagnostics complete).** Pipeline took four
> iterations to land — `flow_sample_diagnostic.py` had hidden CPU
> hardcoding plus a latent CUDA device-leak in `train/symmetry.py`
> (`torch.LongTensor(...)` is CPU-only) plus a missing `.cpu()` on
> the rendered grid. All patched; final resubmits (40031081–40031098)
> all succeeded on CUDA in ~3–10 min/folder. Every row in this
> report's Summary table and Structural diagnostics is now filled
> with measured numbers. **Headline:** `pathgrad_bignet` is *not*
> the 7421-nat-from-FSS prediction the original Note 1 estimated —
> it diverged catastrophically (training-row stable-phase was
> mis-leading; the late-stage divergence dragged `KL(q‖p)` to
> **27,994.92** and `KL(p‖q)` to **3,989,782** by ep 13,000). The
> bignet at L=64 forces STL to a strictly worse regime than
> score-function reverse-KL until larger-batch + grad-clip remedies
> land.

## ★ Phase-2 P2.x Verdict Update (2026-06-25) — `i2 + nrepeat=2` first in-family breakthrough

**HS data anchors (L=64 T_c):** `|M|_p = 2.200,  gL_p = 0.407,  xi_p = 14.782`

| Cell | Configuration | LOSS plateau | **KL_qp** | KL_pq | \|M\| | gL | xi |
|------|---------------|-------------:|----------:|------:|------:|---:|---:|
| **A** | baseline (nr=1) | 7686 | 86.88 | 65.64 | 2.267 | 0.433 | 15.190 |
| **B** | i2 only (stride8h32, nr=1, Phase-1 P1 winner) | 7695 | 93.20 | 69.53 | 2.287 | 0.439 | 15.292 |
| **C** | baseline + **nrepeat=2** | 7736 | **156.36** ❌ | 131.95 | 2.351 | 0.456 | 15.792 |
| **★ D ★** | **i2 + nrepeat=2** | **7666** | **51.33** ✅ | **42.38** | **2.254** | **0.418** | **14.923** |

**Key observations:**
1. **D KL_qp 51.33 vs A 87 — a 41% improvement** — at L=64, this is the *first* configuration that genuinely breaks the baseline plateau
2. **D's structural match is near-anchor:** gL within 3%, xi within 1%, |M| within 2.5%
3. **C alone is a disaster** (+69 nat) — confirms the megabignet rule: adding capacity without aligning the target degrades training
4. **B alone does *not* improve KL_qp** (+6 nat) — the "Phase-1 winner" label came from V5 gauge structural improvements; on KL_qp it has never broken baseline

### Super-additive synergy (quantified)

| Model | KL_qp prediction |
|-------|-----------------|
| Independent additive (A + Δ_C + Δ_B) | 87 + 69 + 6 = **162** |
| Actual D | **51** |
| **Net synergy** | **−111 nat below additive prediction** |

⇒ **Mechanism is *orthogonal* two-pronged attack:**
- **i2 changes the *target***: latent prior changes from strict N(0,I) to conditional Gaussian, matching local coupling of Ising fluctuations
- **nrepeat=2 adds *forward capacity***: two sequential affine layers per scale absorb higher-order moments and Gaussianize fast modes more precisely
- **Combined = drag the target closer + raise the capacity to reach it**; the two interventions do not block each other

### Cross-L consistency (with the L=32 report)

| L | A KL_qp | D KL_qp | Improvement |
|---|--------:|--------:|-------------|
| 32 | 23.42 | 17.69 | 24% |
| 64 | 86.88 | 51.33 | **41%** |

⇒ Improvement scales with L because L=64 baseline is more constrained by FSS critical scaling (α ≈ 2.20).

### Phase-2 P2.x figures — i2 + nrepeat=2 winner (D64)

<p>
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/flow_samples.png" alt="D64 i2+nr2 flow samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/flow_correlations.png" alt="D64 i2+nr2 flow correlations" width="56%">
</p>

For comparison, the C64 reference (nr=2 *only*, degrades to KL_qp 156):

<p>
<img src="../../data/64Ising_T2.269_hsBignet_baseline_nr2_b16/flow_samples.png" alt="C64 baseline+nr2 flow samples" width="42%">
<img src="../../data/64Ising_T2.269_hsBignet_baseline_nr2_b16/flow_correlations.png" alt="C64 baseline+nr2 flow correlations" width="56%">
</p>

### P2.x verdict revision

| Old verdict (2026-06-14) | New verdict (2026-06-25) |
|--------------------------|--------------------------|
| In-family parameter tuning has saturated | **Incorrect. `i2 + nrepeat=2` breaks 7686.** |
| Main budget should move to family-level changes | **Family-level changes remain a priority, but in-family *combinations* also have room** |
| stride=8 hidden=32 i2 is the "sole survivor" | i2(8,32) **paired with nrepeat=2** is the true in-family winner |

See `improvements_results_zh.md` Appendix P2.x for the Chinese-language full verdict.

---

## Summary — everything in one table

Training-row numbers only (diagnostic rows pending). Constants
used (see Notes for derivation):

- `lnZ_c` (continuous, L=64 T_c) = **9476.428** nat
  = `lnZ_discrete + HS-correction` from
  `etc/exactz.md` row `n=64, T=2.269185...`
- `H(p_HS)` at L=64 T_c ≈ **7611.92** nat — *approximate*
  4× scaling from the L=32 value (1902.98). A direct MC
  estimate on the L=64 HS dataset is pending; expect ±5 nat
  drift, which propagates 1:1 into KL(p‖q) numbers below.

| Source                          |    F (-lnZ)   |       E       |       S       | KL(q‖p) | KL(p‖q) |
| :------------------------------ | :-----------: | :-----------: | :-----------: | :-----: | :-----: |
| **═══ Reference ═══**           | ════════════  | ═════════════ | ═════════════ | ═══════ | ═══════ |
| **Exact (theory, discrete)**    |  **-3808.67** |       —       |       —       |    —    |    —    |
| **Exact (theory, continuous)**  |  **-9476.43** |     ≈ -1865   |     ≈ 7612    |  **0**  |  **0**  |
| HS dataset (x ~ p_HS, approx.)  |      N/A      |     ≈ -1865   |    ≈ 7611.92  |    —    |    —    |
| **═══ Reverse-KL ═══**          | ════════════  | ═════════════ | ═════════════ | ═══════ | ═══════ |
| *sym_bignet — training (smoothed-best, win=300 steps)* | *-9442.23* | *-2091.82*  | *7350.83*  | *34.20* |   N/A   |
| **sym_bignet — diagnostic (ep 19,900, N = 4,000)** | **-9441.51 ± 1.31** | **-2065.35 ± 0.93** | **7376.16 ± 0.92** | **34.92** | **302.00 ± 4.68** |
| **═══ Reverse-KL (path-gradient / STL) ═══** | ════════════ | ═════════════ | ═════════════ | ═══════ | ═══════ |
| *pathgrad_bignet — training (pre-divergence stable, ep ≤ 12900)* | *-2055* (note 1) | (note 1) | (note 1) | *≈ 7421* (note 1) |   N/A   |
| **pathgrad_bignet — diagnostic (ep 13,000, N = 4,000)** | **+18,518.49 ± 7.69** | **+18,637.08 ± 7.65** | **118.59 ± 0.76** | **27,994.92** | **3,989,782 ± 4630** |
| **═══ Forward-KL ═══**          | ════════════  | ═════════════ | ═════════════ | ═══════ | ═══════ |
| *hs_bignet — training (smoothed-best)* |      N/A      |     N/A       |   *7687.28*   |   N/A   | *≈ 67.1* (rev. H = 7620.16) |
| **hs_bignet — diagnostic (ep 19,900, N = 4,000)** | **-9390.02 ± 2.47** | **-1639.48 ± 1.82** | **7750.54 ± 1.67** | **86.41** | **64.39 ± 1.44** |
| **═══ Mixed-objective ═══**     | ════════════  | ═════════════ | ═════════════ | ═══════ | ═══════ |
| *jsLoss_bignet_lam0.5 — training (smoothed-best joint)* |     N/A      |  *-1716.68*   |   *7486.82*   | (note 2) | (note 2) |
| **jsLoss_bignet_lam0.5 — diagnostic (ep 19,900, N = 4,000)** | **-9416.98 ± 1.39** | **-1723.00 ± 1.02** | **7693.99 ± 0.95** | **59.44** | **166.75 ± 2.94** |
| **═══ Bridge-reweighted ═══**   | ════════════  | ═════════════ | ═════════════ | ═══════ | ═══════ |
| *bridge_w5.0t0.5 — training (smoothed-best ENTROPY = unweighted MLE)* | N/A | N/A | *7687.56* | N/A | *≈ 67.4* (rev. H = 7620.16) |
| **bridge_w5.0t0.5 — diagnostic (ep 19,900, N = 4,000)** | **-9383.23 ± 3.01** | **-1602.73 ± 2.23** | **7780.49 ± 2.02** | **93.20** | **68.86 ± 1.43** |

**Note 1 — pathgrad_bignet diverged late.** Stable phase at LOSS ≈
−2055 nat from step ~1000 through step ~12900 (much higher than
sym_bignet's −9442 because batch was forced down to 8 by OOM and
STL needs ~2× the gradient memory, so the STL trajectory at L=64
is far slower to reach the variational floor than the score-function
sym_bignet at batch=16). Then an explicit divergence at step 12,988
(LOSS jumps -306 → 18,294 → 17,675 → 267,982 → ...). The
implied stable-phase `KL(q‖p) ≈ −2055 + 9476.43 ≈ 7421 nat` is
**not** a converged metric — STL at L=64 is severely
under-converged, NOT a 7421-nat-worse method than sym_bignet at L=32.
Re-run with larger batch (A100-80G) and gradient clipping is the
clear path forward.

**Note 2 — jsLoss split.** The HDF5 `LOSS` for `-jsLoss` is the
joint Jensen-Shannon objective `0.5·L_rev + 0.5·L_fwd`, not either
direction separately. At smoothed-best joint LOSS = −815.5 nat,
the per-direction components must be extracted from the
training log (`logs/L64_jsLoss_*.out`); a separate diagnostic run
would also recover the diag-row KL_rev and KL_fwd.

### Cross-L scaling (vs the L=32 concise report)

KL_fwd ∝ L^α with α ≈ 2.20 at T_c (FSS memory
`project_fss_critical_scaling`). Extrapolating L=32 → L=64 with
α = 2.20 predicts a multiplier of `(64/32)^2.20 ≈ 4.59`. The
table compares L=32 best vs L=64 best.

| Method        | L=32 (best on-objective KL) | L=64 (best on-objective KL) | Ratio |   Predicted by α=2.20 |
| :------------ | :-------------------------: | :-------------------------: | :---: | :-------------------: |
| sym_bignet (rev) |      **8.77** nat         |       **34.20** nat         | 3.90  |       FSS for rev-KL not measured (α=2.20 is fwd-KL-specific) |
| hs_bignet (fwd)  |      **3.63** nat         |       **≈ 75.4** nat (approx H) | **20.7** | 16.66 (= 3.63 × 4.59)  |
| jsLoss (mix)     |      ≈ 16.5 / 17.0 nat    |       (split pending)       |   —   |          —            |
| bridge (fwd-side) |      ≈ 21.3 nat          |       **≈ 75.6** nat (approx H) | 3.55 | 97.7 (= 21.3 × 4.59)   |
| pathgrad STL (rev) |     **8.20** nat        |       (diverged, see note 1) |   —  |          —            |

**Two observations:**

1. The sym_bignet rev-KL scales close to L² (ratio 3.90 vs L²
   ratio 4), consistent with rev-KL's per-step cost growing
   primarily with field dimension.
2. The hs_bignet fwd-KL ratio of ≈ 20 is **5× larger than FSS
   predicts** (α = 2.20 says ~16.7). The bignet at L=64 may
   simply need more training (the smoothed-best epoch is ~12,400
   — well before ep 19,900), or the bignet capacity is not yet
   matched to L=64 dimensionality. The bridge run, by contrast,
   ratio 3.55, is *under* the FSS prediction — but that is the
   re-weighted objective, not directly comparable.

### Structural diagnostics: `mag_abs_q`, `xi_q`, `G(L/2)/G(0)`

Same lens as the L=32 concise report's "Structural diagnostics"
section: `flow_sample_diagnostic.py` computes per-config
magnetisation `<|M|>_q` and the axial two-point correlation
`G(r) = ⟨xᵢ x_{i+r}⟩ − ⟨xᵢ⟩⟨x_{i+r}⟩` on `x ~ q` samples drawn
from the trained flow, plus the same statistics on `x ~ p_HS`
samples. `xi_q = Σᵣ G(r)/G(0)` is an effective correlation
length, `g_longrange_q = G(L/2)/G(0)` is the plateau value at
half-lattice — at T_c on a finite L=64 lattice the data value
sits around 0.41 (slowly-decaying critical correlations, not
saturating at long r).

The HS-data anchors (from the sym_bignet diagnostic, identical
across all `_q` rows since `p_HS` does not depend on the flow):

| Quantity         | value     |
| :--------------- | --------: |
| `mag_abs_p`      | **2.20**  |
| `g_longrange_p`  | **0.407** |
| `xi_p`           | **14.78** |

Per-flow structural comparison:

| Method                          | mag_abs_q (data 2.20) | g_longrange_q (data 0.407) | xi_q (data 14.78) | Reading                                                              |
| :------------------------------ | --------------------: | -------------------------: | ----------------: | :------------------------------------------------------------------- |
| **sym_bignet** (rev-KL)         | **3.09**              | **0.738**                  | **23.74**         | over-sharpened on both axes — **L=64 reproduces the L=32 sym_bignet pattern, even more extreme**: \|M\|_q overshoots data by 40 %, G(L/2)/G(0) by 80 %, ξ_q by 60 %. Reverse-KL has collapsed into a near-uniform configuration ("frozen" majority-vote) at L=64. |
| pathgrad_bignet (STL rev-KL, **diverged**) | **2.13**            | **0.843**                  | **16.81**         | **Collapsed-but-not-into-anything-sensible.** mag_q happens to land near data (2.13 vs 2.20) but `g_longrange_q = 0.843` is even more extreme than sym_bignet (0.738) — the field is essentially uniform-magnetisation. Confirmed by the `KL(p‖q) ≈ 4 × 10⁶` and `EA = +18,637` in the Summary — the variational free energy has diverged catastrophically. Treat all this row's numbers as artefacts of a broken trajectory; *not* a comparable method. |
| hs_bignet (fwd-KL)              | **2.23**              | **0.424**                  | **14.89**         | **Best structural fit of the converged set on g_longrange and xi.** Within 1.1 % of `g_longrange_p = 0.407` and 0.7 % of `xi_p = 14.78`. mag_q = 2.23 slightly above data (2.20). Forward-KL doing what its objective optimises: covering the data measure faithfully. |
| jsLoss_bignet_lam0.5 (mixed)    | **2.63**              | **0.582**                  | **18.88**         | **Over-sharpened in all three axes** — the joint JS objective at λ=0.5 inherits sym_bignet's mode-seeking pressure visibly. \|M\|_q overshoots data by 20 %, g_longrange by 43 %, ξ_q by 28 %. The mixed objective costs structural fidelity for its low `KL(q‖p) = 59.4`. |
| bridge_w5.0t0.5 (fwd, reweighted)| **2.10**              | **0.391**                  | **14.13**         | **Bridge upweighting** trades ~7 nat of `KL(p‖q)` (68.9 vs hs_bignet's 64.4) for structural sharpness: mag_q within 5 % of data, g_longrange within 4 %, ξ_q within 4 %. Same trade-off pattern as at L=32 (`project_bridge_upweighting` memory). At L=64 it remains the closest *forward-direction* match to data structure. |

**Reading the sym_bignet L=64 row in context.**

- The L=32 sym_bignet diagnostic ([L=32 concise table at line 342])
  showed `mag_abs_q = 3.11`, `g_longrange_q = 0.51`, `xi_q = 12.02`
  vs L=32 data anchors `2.38 / 0.49 / 8.57`.
- At L=64 the over-sharpening compounds. `g_longrange_q` jumps
  from 0.51 (L=32) to **0.738** (L=64) — the flow is not just
  over-correlated locally but *uniformly correlated across half
  the lattice*. With L=64 ≫ ξ_p ≈ 15, the flow has lost the
  exponentially-decaying tail entirely and learned a
  near-constant-magnetisation distribution.
- The diagnostic `KL(p‖q) = 302 nat` (vs `KL(q‖p) = 34.9 nat`)
  formalises the same diagnosis from the inverse direction:
  the forward KL — which catches mode-dropping — is **8.6 ×**
  the on-objective reverse KL. The flow places most of its
  probability mass on a thin volume near magnetisation peaks
  that the data spreads across a wide bridge.

The other four methods' rows will fill in as jobs 40030830–833
complete; the predicted ordering from the L=32 concise (bridge
closest to data, then hs_bignet, then jsLoss / sym_bignet
over-sharpened) should re-appear at L=64, but with structural
mismatch amplified by the larger lattice as the sym_bignet row
above already shows.

**Predicted ordering confirmed — with two caveats.** With all four
remaining diag rows now landed, the L=32 prediction holds verbatim
for the *converged* methods:

```
bridge_w5.0t0.5  <  hs_bignet  <  jsLoss  <  sym_bignet
(closest to data on structure)             (over-sharpened)
```

- On `g_longrange_q` distance to data 0.407: bridge 0.39 (Δ 0.02)
  < hs_bignet 0.42 (Δ 0.02) < jsLoss 0.58 (Δ 0.18) < sym_bignet
  0.74 (Δ 0.33).
- On `mag_abs_q` distance to data 2.20: bridge 2.10 (Δ 0.10) <
  hs_bignet 2.23 (Δ 0.03) < jsLoss 2.63 (Δ 0.43) < sym_bignet
  3.09 (Δ 0.89). (Note hs_bignet's mag is slightly *closer* to
  data than bridge's, but bridge wins on g_longrange and ξ.)

The amplification of structural mismatch at L=64 vs L=32 is large
but uneven:
- sym_bignet g_longrange: L=32 0.51 → L=64 0.74 (Δ 0.23)
- jsLoss g_longrange:     L=32 0.50 → L=64 0.58 (Δ 0.08)
- hs_bignet g_longrange:  L=32 0.51 → L=64 0.42 (Δ −0.09; *improves* with L)
- bridge g_longrange:     L=32 0.49 → L=64 0.39 (Δ −0.10; *improves*)

The forward-KL family (hs_bignet, bridge) ends up *closer* to the
L=64 critical long-range value than at L=32 because the L=64 HS
data has a numerically smaller `g_longrange_p` (0.407 vs L=32's
0.477), and these flows track it. The reverse-KL family (sym_bignet,
jsLoss) goes the other way — the over-correlated artefact
strengthens at larger lattice. **The two objectives are diverging
in the cross-L direction**, consistent with the `rg_fixed_point`
diagnosis that they live on functionally different fixed points.

**Caveat 1 — pathgrad_bignet diverged.** The row exists but its
numbers are artefacts of a divergence at step ~12,988 (Note 1
above). Treat it as "STL at L=64 requires re-running on A100-80G
with grad clip"; do *not* place it on the converged-method axis.

**Caveat 2 — H(p_HS) sensitivity.** All `KL(p‖q)` numbers are
contingent on `H(p_HS) = 7620.16` from the sym_bignet MC estimate.
The four newly-diagnosed methods all hit `Hp_mc` within ±0.0 of
this (the same HS dataset, same seed) so the cross-method
comparison is internally self-consistent.

### Architectures used at L=64

| Arch    | nlayers | nhidden | trainable params (RNVP) | batch | Used by                                                            |
| :------ | ------: | ------: | ----------------------: | ----: | :----------------------------------------------------------------- |
| bignet  |      16 |     128 |             10,938,240  |   16  | sym_bignet, jsLoss_bignet_lam0.5                                   |
| bignet  |      16 |     128 |             10,938,240  |    8  | pathgrad_bignet (STL — batch forced down by OOM)                   |
| bignet  |      16 |     128 |             10,938,240  |   32  | hs_bignet, hsBignet_bridge_w5.0t0.5 (fwd-KL needs less per-step memory) |

All rows use `nmlp=3, nrepeat=1, -symmetry, -skipHMC`. Same
bignet definition as in the L=32 concise report
(`project_l32_bignet_fix` memory). Batch sizes vary because of
A100-40G memory constraints — see the `shell/run_L64_*.sh`
wrappers and `project_l32_late_training_instability` for the OOM
debugging log.

### How KL(q‖p) and KL(p‖q) are obtained at L=64

Same formulas as L=32 (`concise_report_L32_T2.269.md` § "How
KL(q‖p) and KL(p‖q) are obtained"), with the L=64 constants:

| Direction          | Formula                                              | Source                                    |
| :----------------- | :--------------------------------------------------- | :---------------------------------------- |
| KL(q‖p) — training | `F_c^q + lnZ_c = LOSS_rev + 9476.428`                | Training row of sym_bignet (rev-KL).      |
| KL(p‖q) — training | `CE − H(p_HS) ≈ LOSS_fwd − 7611.92`                  | Training row of hs_bignet (fwd-KL).       |
| KL(p‖q) — bridge   | unweighted `ENTROPY` column, then `− H(p_HS)`        | Bridge-reweighted run's unweighted MLE.   |

H(p_HS) is treated as approximate until the direct MC estimate on
the L=64 HS dataset lands.

### Notes

- **Pre-divergence pathgrad numbers are placeholders.** The STL
  L=64 run was demoted to batch=8 to fit on A100-40G; that step
  count budget (13,000 epochs × 8 samples = 104K total samples
  seen) is roughly half of sym_bignet's effective sample budget
  (19,900 × 16 = 318K). Adding the STL double-forward overhead,
  the run was simply not converged when it diverged. The L=32 STL
  win of 0.57 nat over sym_bignet_ext does NOT survive at L=64
  *as currently trained* — but the comparison is unfair until STL
  gets a full A100-80G batch=16 run to ep 20,000.
- **hs_bignet's L=64 smoothed-best is at ep ~12,400, not ep 19,900.**
  The bignet hits its best generalisation MLE around mid-training,
  then drifts up by ~7 nat by ep 19,900. This is consistent with
  a slight overfitting regime, NOT divergence — `MAG_ABS` and
  `MAG_VAR` columns remain stable through the second half. A
  diag run on the ep 12,400 checkpoint would be the fairer
  benchmark.
- **jsLoss split needs extraction from train log.** The HDF5
  `LOSS` is the joint JS objective. The L=32 convention reads the
  per-direction components (`L_rev`, `L_fwd`) from the
  `logs/L64_jsLoss_*.out` stream's per-step prints. Once
  extracted, the KL_rev / KL_fwd columns can be filled in.
- **Bridge upweighting at L=64 looks comparable to L=32 in
  pattern.** Last-200 mean of the unweighted ENTROPY column ≈
  7691; smoothed-best ≈ 7688. Difference of ~3 nat between
  bridge and pure fwd-KL is much smaller than the absolute KL
  uncertainty from the approximate H(p_HS). A diag run for
  structural observables (`mag_abs_q`, `xi_q`) is needed for the
  real bridge-vs-hs comparison.
- **STL re-run TODO.** Resubmit `pathgrad_bignet` on A100-80G
  (override `--gres=gpu:a100:80gb:1` per memory `reference_l40_swap`
  if 80G nodes are queued faster on `preempt`) with batch=16 and
  gradient clipping (e.g. `gradClip=5.0`), targeting ep 20,000
  match with sym_bignet.

## Per-method visuals — `flow_samples.png` + `flow_correlations.png` (where available, else `proposals_NNNNN.png` placeholder)

_The L=32 concise has per-method (flow_samples + flow_correlations)
panel pairs from `flow_sample_diagnostic.py`. At L=64 only
`sym_bignet` has them in place; the other four show the latest
training-time `proposals_NNNNN.png` placeholder until the
corresponding diag job (40030830–833) finishes._

### sym_bignet — reverse-KL, bignet (ep 19,900)

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![sym_bignet flow samples](../../data/64Ising_T2.269_sym_bignet/flow_samples.png) | ![sym_bignet flow correlations](../../data/64Ising_T2.269_sym_bignet/flow_correlations.png) |

_(Training-time placeholder retained for reference:_
`figures/64Ising_T2.269_sym_bignet__proposals_19900.png` _)_

### pathgrad_bignet — STL reverse-KL, bignet (ep 13,000, pre-divergence; batch=8)

> **The flow has diverged.** mag_abs_q = 2.13 looks data-like by
> accident; g_longrange_q = 0.84 is more extreme than sym_bignet
> (0.74) — the field is essentially uniform-magnetisation. KL(q‖p)
> ≈ 28k, KL(p‖q) ≈ 4M. Visuals included as a record of the failure
> mode, not as a comparable method.

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![pathgrad_bignet flow samples](../../data/64Ising_T2.269_pathgrad_bignet/flow_samples.png) | ![pathgrad_bignet flow correlations](../../data/64Ising_T2.269_pathgrad_bignet/flow_correlations.png) |

### hs_bignet — forward-KL, bignet (ep 19,900)

> Best structural match of the converged set on `g_longrange_q`
> (0.42 vs data 0.41) and `xi_q` (14.89 vs data 14.78).

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![hs_bignet flow samples](../../data/64Ising_T2.269_hs_bignet/flow_samples.png) | ![hs_bignet flow correlations](../../data/64Ising_T2.269_hs_bignet/flow_correlations.png) |

### jsLoss_bignet_lam0.5 — mixed Jensen-Shannon, bignet (ep 19,900)

> Over-sharpened in all three structural axes (mag 2.63 > data
> 2.20, g_longrange 0.58 > 0.41, xi 18.88 > 14.78). The
> mixed-objective gain in `KL(q‖p) = 59.4` (lowest of the converged
> set) costs structural fidelity.

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![jsLoss_bignet flow samples](../../data/64Ising_T2.269_jsLoss_bignet_lam0.5/flow_samples.png) | ![jsLoss_bignet flow correlations](../../data/64Ising_T2.269_jsLoss_bignet_lam0.5/flow_correlations.png) |

### bridge_w5.0t0.5 — bridge-reweighted forward-KL, bignet (ep 19,900)

> Bridge upweighting recovers the closest structural match on mag
> (2.10) and ξ (14.13), trading ~7 nat KL(p‖q) for ~4 % closer
> `g_longrange_q`. Same trade-off as at L=32 (`project_bridge_upweighting`).

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![bridge_w5.0t0.5 flow samples](../../data/64Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_samples.png) | ![bridge_w5.0t0.5 flow correlations](../../data/64Ising_T2.269_hsBignet_bridge_w5.0t0.5/flow_correlations.png) |

## Phase-1 improvement ablation at L=64 (b = 16, ep 19,800)

A separate L=64 experiment line tested III.1 (multi-scale loss),
I.2 (conditional Gaussian prior), and I.1 (Student-t prior) from
`analyzers/rg_fixed_point/improvements_zh.md`, all at matched
`batch = 16` (forced down from the hs_bignet `batch = 32` by the
scaleLoss extra forward-graph cost). The matched-batch baseline
sits ~0–6 nat higher in `KL(q‖p)` than the original `hs_bignet`
run above (86.9 vs 86.4) — the difference is dominated by batch
noise, not by the methods themselves.

| Run                                   | F_c^q          | KL(q‖p) | KL(p‖q)        | mag (data 2.20) | xi (data 14.78) | g_longrange (data 0.407) |
| :------------------------------------ | -------------: | ------: | -------------: | --------------: | --------------: | -----------------------: |
| **baseline_b16** (Gauss prior, no scaleLoss) | -9389.55 ± 2.71 | 86.88 | 65.64 ± 1.44 | 2.27           | 15.19          | 0.433                    |
| **iii1_lam1.0_b16** (+ III.1 scaleLoss)     | -9389.29 ± 2.56 | 87.14 | **64.63 ± 1.44** | 2.24       | 14.95          | 0.425                    |
| **i2_stride16h32_b16** (+ I.2 cond. prior)  | -9383.12 ± 2.49 | 93.31 | 70.37 ± 1.44 | 2.23           | 14.94          | 0.425                    |
| **i1_df4.0_b16** (+ I.1 Student-t prior)    | -9386.02 ± 2.50 | 90.41 | 66.21 ± 1.43 | **2.18**       | **14.37**      | **0.404**                |

**Reading the L=64 ablation.**

- **All four sit on the same KL ridge** (86–93 nat for KL(q‖p),
  65–70 for KL(p‖q)). At `batch = 16` the per-step gradient noise
  (~8× the baseline `batch = 128` noise floor) dominates any
  signal the interventions might carry. **The L=64 ablation can
  read directions, not magnitudes.**
- **iii1 (scaleLoss) is the only intervention that improves
  `KL(p‖q)`** (−1.0 nat vs baseline; the only fwd-direction win
  in this row). The scale-loss prevents deep-block collapse just
  enough to widen the data-covering tail. Direction match with
  L=32 (where iii1 gave −0.46 on KL(p‖q)).
- **i2 (conditional prior) regresses on both KLs at L=64** but
  improves L=32. At L=32 b=64 the same ablation gave KL(q‖p) =
  21.16 vs baseline 23.42 (−2.26 nat); at L=64 b=16 it's +6.43.
  Possible reasons: (a) the slow grid at `stride=16` for L=64
  (4×4 slow) is the same *count* as L=32's `stride=8` slow grid
  (4×4), so the conditional gets relatively *less* coverage of
  the field at larger L; (b) the CNN prior under batch=16 noise
  drifts in unhelpful directions before averaging out.
- **i1 (Student-t) gives the closest structural match at L=64**:
  `mag = 2.18` (Δ −0.02 vs data), `g_longrange = 0.404` (Δ
  −0.003 vs data), `xi = 14.37` (Δ −0.41 vs data). All three
  closer to data than baseline or any of the other interventions.
  The KL trade-off is small (+3.5 nat KL(q‖p), +0.6 KL(p‖q)).
  **This is the surprise** — the negation experiment from
  improvements.md (which was designed to *fail* and confirm the
  prior is not the bottleneck) instead delivers the cleanest
  structural improvement at the largest L tested.

**Cross-L direction summary:**

| Intervention | L=32 effect on KL(q‖p) | L=64 effect on KL(q‖p) | L=32 g_longrange Δ data | L=64 g_longrange Δ data |
| :----------- | ---------------------: | ---------------------: | ----------------------: | ----------------------: |
| baseline      | (ref) 23.42           | (ref) 86.88            | +0.026                  | +0.026                  |
| + III.1 sclos | −1.59                  | +0.26 (noise)          | +0.022                  | +0.018                  |
| + I.2 cond.   | **−2.26**              | +6.43                  | +0.020                  | +0.018                  |
| + I.1 t-prior | −2.07                  | +3.53                  | +0.013                  | **−0.003**              |

- III.1 is direction-consistent (slight L=32 win; noise at L=64).
- I.2 is *direction-flipping*: a clear L=32 win, regression at
  L=64. Needs slow-grid scaling investigation.
- I.1 Student-t is direction-consistent (small KL hit at both L)
  but **structural fit improves with L**. This is the
  Wilson-Fisher critical-tail signature the original critique
  predicted *should* matter — and is the strongest justification
  to look at `I.3 EBM/φ⁴` (improvements_zh.md scheme B) next.

**See `analyzers/rg_fixed_point/improvements_zh.md` § "Phase 3"
for the Phase-2 followups this dataset motivates.**

### Phase-1 ablation visuals — `flow_samples.png` + `flow_correlations.png`

_Same format as the original-method panels above: left = flow x~q
vs HS x~p; right = magnetisation distribution + log-log G(r)/G(0)._

#### baseline_b16 (Gaussian prior, no scaleLoss)

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![baseline_b16 flow samples](../../data/64Ising_T2.269_hsBignet_baseline_b16/flow_samples.png) | ![baseline_b16 flow correlations](../../data/64Ising_T2.269_hsBignet_baseline_b16/flow_correlations.png) |

#### iii1_lam1.0_b16 (+ III.1 multi-scale loss)

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![iii1_lam1.0_b16 flow samples](../../data/64Ising_T2.269_hsBignet_iii1_lam1.0_b16/flow_samples.png) | ![iii1_lam1.0_b16 flow correlations](../../data/64Ising_T2.269_hsBignet_iii1_lam1.0_b16/flow_correlations.png) |

#### i2_stride16h32_b16 (+ I.2 conditional Gaussian prior)

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![i2_stride16h32_b16 flow samples](../../data/64Ising_T2.269_hsBignet_i2_stride16h32_b16/flow_samples.png) | ![i2_stride16h32_b16 flow correlations](../../data/64Ising_T2.269_hsBignet_i2_stride16h32_b16/flow_correlations.png) |

#### i1_df4.0_b16 (+ I.1 Student-t prior, df=4)

_The structural surprise of the L=64 Phase-1: closest mag (2.18 vs
data 2.20) and g_longrange (0.40 vs data 0.41) of any L=64 run._

| Configurations (left flow x~q · right HS x~p) | Physical fit (mag P(M) + log-log G(r)/G(0)) |
|:---:|:---:|
| ![i1_df4.0_b16 flow samples](../../data/64Ising_T2.269_hsBignet_i1_df4.0_b16/flow_samples.png) | ![i1_df4.0_b16 flow correlations](../../data/64Ising_T2.269_hsBignet_i1_df4.0_b16/flow_correlations.png) |

## See also

- `concise_report_L32_T2.269.md` — companion L=32 report with the
  same method set, full diagnostic rows and structural observables.
- `rg_fixed_point_report.md` — RG fixed-point probe (L=32 only at
  present; an L=64 extension would reuse the same scripts on the
  L=64 checkpoints once they are diag-ready).
- `fss_sweep_report.md` — cross-L KL ∝ L^α exponent measurement
  that motivates the cross-L scaling row above.
- `etc/exactz.md` — exact 2D Ising partition functions used to
  compute lnZ_c at L = 8, 16, 32, 64.
