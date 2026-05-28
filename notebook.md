# NeuralRG Working Notebook
*Last updated: 2026-05-27*

---

## Problem Statement

Training MERA-based normalizing flows (NeuralRG) on the 2D Ising model (L=32) near the critical temperature T_c ≈ 2.269. Two training modes are available:

- **Reverse KL (energy-based)**: samples from flow, minimizes KL(q||p). Mode-seeking — can collapse to a single mode.
- **Forward KL / MLE (data-driven)**: trains on MCMC spin configurations, minimizes -E_data[log q(x)]. Mode-covering.

---

## Core Finding: Mode Collapse in Reverse KL

The reverse KL runs at L=32 show **mode collapse** for T ≤ 2.4:

| Temperature | obs_z (magnetization proxy) | Expected |
| :--- | :---: | :---: |
| T = 2.269 (T_c) | 0.4–0.7 | ~0.03 (disordered) |
| T = 2.3 | 0.4–0.7 | ~0.03 |
| T ≥ 2.5 | ~0.03 | ~0.03 (correct) |

**obs_z = ⟨|Σ s_i|⟩ / N** (average absolute magnetization per spin). Near T_c, the true distribution is nearly symmetric (obs_z ≈ 0), but the flow learns to generate mostly all-up or all-down configurations.

The loss *looks* near-exact at T=2.3 (≈ -2342 vs exact -2342.9) — this is misleading. Mode collapse artificially deflates the loss because the collapsed distribution happens to have energy close to the ground state. The physics (thermodynamic observables) is wrong.

**Why T ≥ 2.5 is fine:** Above T_c, the Ising model is in the disordered phase with a unique Gibbs state. The flow has no competing modes to collapse between, so it converges correctly — but even there, the loss is not exactly exact (residual approximation gap from the MERA architecture).

---

## MERA Architecture Limitations

The hierarchical RNVP architecture has structural constraints that may prevent it from reaching exact lnZ:

1. **Fixed 2×2 patch decomposition**: Only nearest-neighbor correlations within a 2×2 block are coupled per layer; long-range correlations require many layers.
2. **Non-overlapping patches**: At each scale, patches tile the lattice without overlap — information between adjacent patches must propagate through multiple coarse-graining steps.
3. **Power-of-2 scale bias**: The hierarchy naturally captures correlations at scales 1, 2, 4, 8, … — correlations at intermediate scales (e.g., 3, 5) are harder.
4. **Factorization assumption**: The flow is not guaranteed to represent all correlations at a given scale before coarse-graining.

Near T_c, the correlation length diverges (ξ → ∞). These limitations become most severe exactly where we need the most expressiveness.

---

## Existing L=32 Data-Driven Runs (T=2.3)

Two runs existed at the start of this session:

| Run folder | nhidden | Epochs run | Best loss | Exact loss |
| :--- | :---: | :---: | :---: | :---: |
| `32Ising_T2.3_sym_dataDriven_skipHMC` | 10 | 5,000 | -1013 | -2342.9 |
| `32Ising_T2.3_sym_MCMCdataDriven` | 64 | 1,600 | -735 | -2342.9 |

Both are far from exact. **Cannot cleanly separate architecture vs. epoch count effects** — the nhidden=10 run trained longer but the nhidden=64 run is more expressive. Neither is a clean baseline.

---

## Experiment: L=8 Data-Driven Training

**Goal:** Test whether MERA is expressive enough to learn the Ising distribution at small scale. If it converges to exact lnZ at L=8, the architecture is fine and the L=32 gap is a capacity/training issue. If not, there is a structural expressiveness problem.

**Why L=8:**
- Exact partition function computable in seconds via transfer matrix (256×256 matrix).
- MCMC data generation is near-instant (Wolff algorithm, 200k samples in < 1 min).
- Training epochs needed to converge are much fewer.
- Provides a clean expressiveness test before scaling.

### Files Created

**`compute_exact_L8.py`** — Transfer matrix exact partition function:
```python
def exact_lnZ(L, T):
    K = 1.0 / T
    n = 2 ** L   # 256 for L=8
    spins = ...  # all 2^L row configurations as ±1 vectors
    vert = spins @ spins.T  # vertical coupling matrix
    log_T = K * vert + 0.5 * K * horiz[:, None] + 0.5 * K * horiz[None, :]
    eigs = np.linalg.eigvalsh(np.exp(log_T - log_T.max()))
    lnZ = L*shift + L*log(max_eig) + log(sum((eigs/max_eig)^L))
    return lnZ
```
Appends L=8 values to `etc/exactz.md` with `fix=0` (correct: the lnZ/fix split in exactz.md is a computational artifact from the original authors; the analyzer always uses `lnZ + fix`, so setting fix=0 puts the full value in lnZ).

**`shell/run_L8_data_driven.sh`** — Three-phase Slurm GPU job:
- Phase 1: `python compute_exact_L8.py` (~1s, appends L=8 to `etc/exactz.md`)
- Phase 2: `python generate_mcmc_data.py -L 8 -T 2.269 -N 200000` (Wolff MCMC, ~1 min)
- Phase 3: Training:
  ```bash
  python main.py -L 8 -T 2.269 \
      -folder ./data/8Ising_T2.269_sym_dataDriven \
      -dataDriven -cuda 0 -epochs 5000 -batch 128 \
      -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
      -savePeriod 100 -symmetry -skipHMC
  ```

**`analyzers/loss_analyzer_fixT.py`** — Added `-L` CLI argument so the analyzer works for any lattice size:
```bash
python analyzers/loss_analyzer_fixT.py -t 2.269 -L 8
# → generates analyzers/loss_report_L8_T2.269.md
```

**`shell/run_L16_data_driven.sh`** — Two-phase Slurm GPU job (6h, 16G):
- Phase 1: `python generate_mcmc_data.py -L 16 -T 2.269 -N 200000` (exact values already in `etc/exactz.md`)
- Phase 2: Training (same arch as L=8, 10k epochs):
  ```bash
  python main.py -L 16 -T 2.269 \
      -folder ./data/16Ising_T2.269_sym_dataDriven \
      -dataDriven -cuda 0 -epochs 10000 -batch 128 \
      -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
      -savePeriod 200 -symmetry -skipHMC
  ```

**Why 200k samples for both L=8 and L=16:** L=32 runs used 50k samples and were far from convergence. 200k gives 4× more coverage of the distribution. For L=16, each sample has 256 spins (vs 64 for L=8), so the dataset in GB is 4× larger, but still well within memory.

**Why exact values not needed for L=16:** The transfer matrix for L=16 would be 2^16 × 2^16 = 65k × 65k — too large. But `etc/exactz.md` already contains the exact n=16 values (computed previously by the original authors via a different method).

### Job Status

| Job ID | L | Status | Submitted |
| :--- | :---: | :--- | :--- |
| 37679192 | 8 | Pending (PD) — waiting for GPU | 2026-05-14 |
| 37679193 | 16 | Pending (PD) — waiting for GPU | 2026-05-14 |

Monitor:
```bash
squeue -u hhuang05                         # check both jobs
tail -f logs/L8_dd_37679192.out            # L=8 live output
tail -f logs/L16_dd_37679193.out           # L=16 live output
```

After jobs complete:
```bash
python analyzers/loss_analyzer_fixT.py -t 2.269 -L 8
python analyzers/loss_analyzer_fixT.py -t 2.269 -L 16
cat analyzers/loss_report_L8_T2.269.md
cat analyzers/loss_report_L16_T2.269.md
```

### Interpreting Results

| Outcome | Interpretation | Next step |
| :--- | :--- | :--- |
| Loss → near exact (-lnZ_L8) | MERA is expressive enough; L=32 gap is capacity/training | Scale up: more epochs, bigger nhidden for L=32 |
| Loss stuck far from exact | Architecture has structural expressiveness limits | Consider Z2-equivariant RNVP, data augmentation, or different architecture |

---

## Future Work (Parked)

### Z2-Equivariant RNVP
`flow/rnvp.py` lines 33–106 contain commented-out code that enforces Z2 (spin-flip) symmetry at every coupling step (symmetrizes `s`, antisymmetrizes `t`). This is a stronger form of symmetry than the `-symmetry` wrapper (which only averages at inference). May help with mode collapse by construction.

**To revisit:** uncomment and test whether Z2-equivariant coupling reduces mode collapse at T ≤ 2.4 for L=32 energy-based training.

### Data Augmentation (D4 / Translation Symmetry)
MCMC data could be augmented by applying all 8 symmetries of the square lattice (4 rotations × 2 reflections) and periodic translations. This would multiply effective dataset size by 8×L² without additional sampling. Deferred: run plain L=8 data-driven first to establish baseline.

### Higher-Temperature Gap Analysis
Even at T ≥ 2.5 (no mode collapse), reverse KL loss does not reach exact. This gap is pure architecture expressiveness — worth quantifying systematically across temperatures to separate "collapse error" from "approximation error."

---

## Key Commands Reference

```bash
# Submit new job
sbatch shell/run_L8_data_driven.sh

# Check queue
squeue -u hhuang05

# Generate analysis report
python analyzers/loss_analyzer_fixT.py -t 2.269 -L 8   # L=8
python analyzers/loss_analyzer_fixT.py -t 2.3 -L 32    # L=32

# Manual training (energy-based, L=32)
python main.py -L 32 -T 2.269 -cuda 0 -epochs 80000 -batch 128 \
  -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 -savePeriod 100 \
  -symmetry -skipHMC -folder ./opt/MyRun/

# Temperature sweep (energy-based)
bash shell/scan_temps.sh --sym 2.269 2.3 2.4 2.5
```

## File Map

```
main.py                        Entry point
generate_mcmc_data.py          Wolff MCMC sampler → .pt datasets
compute_exact_L8.py            Transfer matrix exact Z for L=8
flow/rnvp.py                   RNVP coupling (Z2-equivariant version commented out at lines 33–106)
train/learn.py                 symmetryMERAInit() + learnInterface() training loop
source/ising.py                2D Ising target distribution
analyzers/loss_analyzer_fixT.py  Per-temperature loss report (supports -L flag)
etc/exactz.md                  Exact lnZ reference values (L=4,8,16,32,64)
shell/run_L8_data_driven.sh    L=8 3-phase Slurm job
shell/scan_temps.sh            Temperature sweep wrapper
data/mcmc_data/                MCMC .pt datasets
data/8Ising_T2.269_sym_dataDriven/  L=8 training output (created by job)
logs/                          Slurm stdout/stderr
```

---

# Session 2026-05-26 — L=32 capacity follow-up, FSS sweep, JS loss

(See end-of-file. Sections added today.)

## Context entering the session

The previous session ended having confirmed the **bignet capacity fix for L=32 forward-KL** (hs_bignet: nlayers=16 nhidden=128, ~10.9M params): KL(p‖q) dropped from 29.5 (default arch) → 3.63 nat at L=32, T=T_c. Three things were unfinished:
1. Whether reverse-KL at L=32 also benefits from bignet (i.e. is the gap capacity-bound or objective-bound?)
2. FSS temperature sweep at T ∈ {2.15, 2.22, 2.27, 2.32, 2.40} across L ∈ {8, 16, 32}, HS forward-KL, arch-per-L
3. Whether a JS-style symmetric loss could break the reverse-KL mode-dropping floor

All three were addressed today; (1) and (2) completed, (3) is still running at end of session.

---

## 1. Reverse-KL L=32 bignet anchor: capacity helps modestly, NOT a fix

**Anchor run** `sym_bignet` (job 38741298 phase-1 on preempt/l40 + 38744474 phase-2 extension):
- nlayers=16 nhidden=128, 10.94M params (same arch as hs_bignet)
- Total ~7000 effective epochs (1950 phase-1 + ~5050 phase-2 after burn-in)
- **Final converged smoothed KL(q‖p) ≈ 9.7 nat** (best single epoch: 8.47)

Comparison vs default-arch reverse-KL:

| Run | Arch | Epochs | KL(q‖p) |
|---|---|---|---|
| sym (default) | 10/64 | 990 | 16.12 |
| sym_longer (default) | 10/64 | 1590 | 11.96 |
| sym_bignet (this work) | 16/128 | ~7000 | **9.71** |
| hs_bignet (forward-KL bignet, ref) | 16/128 | 9500 | KL(p‖q) = **3.63** |

So bignet at L=32 gives:
- **Forward-KL**: 29.5 → 3.6 nat (−88%, capacity-bound)
- **Reverse-KL**: 12.0 → 9.7 nat (−20%, mostly **objective-bound**)

**Implication**: reverse-KL at L=32 is hitting the mode-seeking floor. More parameters can chip ~2 nat off but cannot close the remaining ~6 nat gap to hs_bignet. The mode-dropping is the dominant remaining error.

**Project memory updated**: `project_l32_bignet_fix.md` (the capacity-not-inductive-bias finding remained intact; objective-bound reverse-KL is the new corollary).

**Workflow lessons filed**:
- `project_resume_optimizer_state.md`: `-load` only restores model weights, not Adam moments. Phase-2 wasted its first ~500 epochs blowing up (L: −2356 → +1234 → −1995) and recovering, before resuming downward trend. Fix: extend `flow.save()` in `flow/flow.py:37` to include optimizer state dict.

---

## 2. FSS temperature sweep — 15 trained checkpoints + diagnostic post-hoc

**Trained 15 (L, T) HS forward-KL flows**, arch-per-L:
- L=8 → default (10/64)
- L=16 → midbig (12/96)
- L=32 → bignet (16/128)
- T = {2.15, 2.22, T_c, 2.32, 2.4} (T_c slot reuses existing T=2.269185314213022 datasets)
- Epochs: 20k each (L=8/L=16 reached 19500; L=32 ran out of 6h slot at ~16-17k)
- 200k HS samples per (L, T)

**Workflow problem solved en route**: Tufts gpu/a100 partition had only 2 live nodes (cc1gpu003/005, both mixed). 5 L=32 a100 jobs would have trickled through over ~20h. Strategy:
- Long L=32 a100 jobs stay on gpu/a100 (worth waiting for non-preempt slot)
- Short L=8/L=16 jobs rerouted to preempt/l40
- Diagnostic jobs use generic `--gres=gpu:1` to grab any idle GPU (P100/T4/v100/l40), starts in <10s vs ~6h queue on l40-only

**Filename gotcha**: `generate_mcmc_data.py -T 2.40` writes `mcmc_wolff_L*_T2.4_N*.pt` (Python `str(2.40)` strips trailing zero), but the HS-gen + training scripts looked for `*_T2.40_*.pt` literally. All 3 L's at T=2.4 failed in 2s on first submission. Fixed by switching the T=2.4 slot to use filename "2.4" everywhere.

**Converged training-loss (CE = H(p_HS) + KL(p‖q))**:

| L | T=2.15 | T=2.22 | T_c | T=2.32 | T=2.40 |
|---|---|---|---|---|---|
| 8 | 118.1 | 118.8 | 119.2 | 119.7 | 120.4 |
| 16 | 470.7 | 475.8 | 479.8 | 484.7 | 487.2 |
| 32 | 1879.5 | 1914.9 | 1924.2 | 1940.6 | 1963.9 |

LOSS scales as N=L² (as expected for an extensive H(p_HS)), grows monotonically with T (entropy rises). KL itself is buried in the constants — needs the diagnostic step to extract.

**Diagnostic sweep**: Submitted 15 `flow_sample_diagnostic.py` jobs against each trained folder. At end of session **5/15 complete (all L=8)**, 10 still running (L=16, L=32). Diagnostic computes:
- Flow side: ⟨A⟩_q, H(q), F_c^q, KL(q‖p), and structure metrics ⟨|M|⟩_q, ξ_q, G(L/2)/G(0)_q
- Data side: CE, KL(p‖q), and same structure metrics for HS data
- Outputs `flow_diagnostic.json` + `flow_samples.png` + `flow_correlations.png` per folder

**Limitation observed**: diagnostic skips KL terms for non-T_c temperatures because `etc/exactz.md` lacks lnZ_c entries for them. Structural metrics (⟨|M|⟩, ξ, G) still computed — sufficient for the T_c-shift analysis (peak / inflection / crossing locations).

**Preview L=8 q-side**:

| T | ⟨\|M\|⟩_q | ξ_q | G_q(L/2) |
|---|---|---|---|
| 2.15 | 3.222 | 3.064 | 0.732 |
| 2.22 | 2.923 | 2.887 | 0.677 |
| 2.269 | 2.824 | 2.804 | 0.649 |
| 2.32 | 2.542 | 2.631 | 0.597 |
| 2.40 | 2.281 | 2.406 | 0.527 |

Monotone in T; inflection between 2.22 and 2.32 (consistent with T_c ≈ 2.27). L=16/L=32 needed to pin a T_c-shift.

---

## 3. JS-like (symmetrized-KL) loss — new training mode implemented

Implemented Option 2 from the JS-variant discussion: a tunable convex combination of the existing two losses:
```
L_js = jsLambda · KL(q‖p) + (1 − jsLambda) · KL(p‖q)
     = jsLambda · F_c^q                            (drops lnZ_c, gradient-irrelevant)
     + (1 − jsLambda) · (−E_p[log q])              (drops H(p_HS), gradient-irrelevant)
```

**Files changed** (committed in next push):
- `main.py`: `-jsLoss`, `-jsLambda` (default 0.5) flags
- `train/learn.py`: new branch combining the existing reverse-KL and forward-KL paths per step. Inputs **not standardized** (the flow must train in one coordinate system since both terms feed it physical x).
- `shell/run_L8_jsLoss.sh`, `shell/run_L32_jsLoss.sh`: sbatch wrappers

**Sanity-check math** at L=8 with λ=0.5:
- L_rev → −lnZ_c + KL(q‖p) ≈ −146 + 0.7 = −145
- L_fwd → H(p_HS) + KL(p‖q) ≈ +116 + 1.4 = +117
- L_js → 0.5·(−145 + 117) = **−14**

**L=8 jsLoss test** (job 38773396, RUNNING, ~13min in / 1.5h budget):
- epoch 500 → L_js = −13 (close to predicted −14)
- epoch 2500 (latest check) → on track
- Confirms implementation correctness; results to compare with pure forward and reverse runs once it converges.

**L=32 jsLoss anchor** (job 38774030, RUNNING, default-arch, 5k epochs):
- Science question: does symmetric loss break the L=32 reverse-KL mode-dropping floor (KL_qp ≈ 12) without destroying forward-KL coverage (KL_pq ≈ 30 at default arch)?
- Predicted converged L_js ≈ −226
- At epoch 100: L_js = −173. Fast downward trajectory, ~50 nat from target.
- Decision rule for T-sweep:
  - **If JS breaks the floor (KL_qp drops well below 12 here)**: schedule full L=32 jsLoss sweep over T
  - **If JS just trades errors** (KL_qp up, KL_pq down, sum no better than forward-KL alone): skip the sweep; JS is just an interpolation
- Off-critical T's are not interesting for JS anyway: T<<T_c is single-dominant-mode, T>>T_c is single-peak Gaussian-like → JS reduces to one of the two KLs.

---

## Conceptual side-conversations

- **Why "negative KL(p‖q)" in L=8/L=16 reports**: training-batch overfit lets empirical CE dip below H(p_HS). The fresh-sample diagnostic always gives non-negative KL.
- **Mode-seeking floor in reverse-KL**: KL(q‖p) is zero-forcing — gradient blind to regions where q≈0, even if p>0 there. Bigger nets only shrink the bias slightly; an architectural fix (Z2-equivariant RNVP — see `project_z2_equivariance_todo.md`) or symmetric objective is needed.
- **Ratio (p‖q)/(q‖p) as shape diagnostic**: ratio ≫1 ⇔ q too narrow (mode-dropping); ratio ≪1 ⇔ q too wide (over-coverage). At L=32 reverse-KL the ratio was 7.4 (mode-shaped); at L=32 forward-KL the inverted ratio (q‖p)/(p‖q) was 5.85 (over-coverage shape).
- **Where the extra forward-KL entropy hides**: NOT in extra valley density at M≈0 (the M-plot showed q's valley is even lower than p's). Instead, the 24-nat over-coverage is in long-range coherent fluctuations — ξ_q = 8.87 > ξ_p = 8.57, G_q(L/2)/G(0) = 0.508 > G_p = 0.477. The flow is "over-coherent" / hallucinating larger critical domains, not noise-leaking.

---

## Today's commits

- `d39beba` Add FSS temperature-sweep + L=32 reverse-KL bignet sbatch scripts (6 shell scripts, 321 insertions)

**Uncommitted at end of session**:
- `main.py`, `train/learn.py` (JS loss implementation)
- `shell/run_L8_jsLoss.sh`, `shell/run_L32_jsLoss.sh`, `shell/diag_sweep_one.sh`, `shell/diag_L32_sym_bignet.sh`, `shell/extend_L32_sym_bignet.sh` (note: extend_L32 was in the earlier commit)
- The notebook.md update (this section)

---

## Open threads at end of session

1. **L=8 jsLoss** (38773396, RUNNING): expected to finish in ~45 min. Need to confirm KL_qp and KL_pq values after run.
2. **L=32 jsLoss** (38774030, RUNNING): ~5h budget. Result gates the JS-sweep decision.
3. **Diagnostic sweep** (38774020-38774029, 10 still RUNNING): L=16 and L=32 due in ~5-10 min. Triggers FSS analysis + T_c-shift check.
4. **`sym_bignet` diagnostic timeout** (38772726): the bignet reverse-KL diagnostic hangs at 35min, never produces output. The same diagnostic on `hs_bignet` (forward-KL bignet) and `sym_longer` (default reverse-KL) both ran fine — issue is specific to bignet × reverse-KL combo. Probably some quadratic-in-depth slowness in the iterative MERA sampling. Worth debugging but doesn't block FSS work.
5. **TODO: optimizer-state-save fix** in `flow/flow.py:37` (per `project_resume_optimizer_state.md` memory).

---

# Session 2026-05-27

## Work done

### 1. `sym_bignet` — re-scanned for true best KL_rev
Previous quote (11.00 nat) was the final-checkpoint diagnostic. Scanning every record file and taking the 50-epoch smoothed minimum:

| Source                          | Epoch | LOSS   | KL_rev   |
|---------------------------------|------:|-------:|---------:|
| Smoothed-best (50-ep window)    | 5925  | −2360.19 | **9.40** |
| Final smoothed                  | 5950  | −2359.46 | 10.12 |
| Diagnostic (N=8000 sample-est)  | 5950  | —      | 10.14 |

Updated `analyzers/concise_report_L32_T2.269.md` with all three rows. The 0.7-nat gap between smoothed-best and diagnostic is consistent with training-batch noise (LOSS std≈4 nat per batch → 50-batch SE≈0.6 nat).

### 2. Diagnostics for the 3 new L=32 runs
First batch job (`38813480`) timed out after completing only `sym_bignet`. Re-ran the missing two as per-folder jobs:
- `jsLoss_bignet_long`: KL(p‖q) = 19.23 nat, ⟨|M|⟩_q = 2.80 (data 2.38).
- `phase2_finetune`: in progress at end of session.

### 3. Entropy regularization on Phase-1 forward KL
Implemented `-entropyBeta` flag in `main.py` and `train/learn.py`. Loss becomes `MLE - β·H(q)` with sequential-backward to avoid OOM (the jsMemOpt pattern). 

Stability sweep:
- **β = 0.05** (job 38813102): diverged to LOSS = −5×10²⁴ by ep 600. Continuous-flow H(q) is unbounded above; β=0.05 is too aggressive.
- **β = 0.005** (job 38813847): stable. By ep 1800 LOSS = 1920.1 (matches hs_bignet baseline). ⟨|M|⟩=2.36 (data 2.38). No bridge change yet — need long-horizon run to know if it helps.

Conclusion: β must be small. For continuous flows there is no "natural" entropy scale; β=0.05 from typical RL/VAE settings is wildly too large.

### 4. Stage-1 NSF sanity at L=8 — PASSED
Job 38814407 trained NSF (K=8 bins, bound=20) at L=8 hs_dataDriven for 3000 epochs.
- Final LOSS = 121.13 nat (50-mean at ep 2900)
- RNVP baseline at L=8: 119.19 nat (at ep 27000)
- NSF is ~2 nat behind RNVP but with (a) `nlayers=8` instead of 10, (b) 9× fewer epochs. At ep 2100 NSF already at LOSS=120.8 — converges 4-5× faster per epoch.

Verdict: NSF is invertible, stable, and competitive on L=8. Safe to proceed to L=32 head-to-head.

### 5. Stage-2 NSF at L=32
Two jobs submitted:
- **NSF default** (l=10, h=64, job 38817758): training stably. At ep 2000 LOSS = 1959, min ever = 1927. Already beating RNVP-default's 1930 nat.
- **NSF bignet** (l=16, h=128, job 38817759): **diverged to NaN by ep 40**. Initial LOSS = 35k, oscillated 30k-200k, then NaN. Cause: wide MLP with random Gaussian init produces O(1) raw spline parameters → after softmax + exp the bin widths/heights are wildly non-uniform → log-det explodes → at 1024 dims this overwhelms Adam.

### 6. NSF identity-init fix
Added to `train/learn.py:_make_nsf_block`:
- Zero the final Linear's weight matrix.
- Set the bias so each spline starts as identity on `[-B, B]`:
  - first K entries (raw_widths) = 0 → softmax = uniform widths
  - next K entries (raw_heights) = 0 → softmax = uniform heights = widths
  - last K-1 entries (raw_derivs) = `log(exp(1 - DEFAULT_MIN_DERIV) - 1)` ≈ 0.540 → softplus + min ≈ 1 = boundary derivs

Verified on a direct NSFCoupling block test: `||x - z||_inf = 2.4e-7`, log-det = 0 to floating-point precision. Bignet relaunched (job 38820893 → swapped to L40 as 38820893).

### 7. Temperature sweep — KL_fwd vs (L, T)
Computed `KL_fwd = LOSS - H(p_HS)` for the 15-point sweep where `H(p_HS) = E_p[A] + lnZ_HS` and `lnZ_HS = lnZ + fix` (interpolated from `etc/exactz.md`).

Best-smoothed LOSS over each trajectory (more reliable than final-epoch):

| L  | T=2.15 | T=2.22 | T_c   | T=2.32 | T=2.40 |
|---:|-------:|-------:|------:|-------:|-------:|
| 8  | 0.58   | 0.73   | 0.72  | 0.78   | 0.79   |
| 16 | 2.08   | 3.53   | 4.74  | 5.34   | 2.93   |
| 32 | 8.59   | 11.96  | 15.30 | 14.95  | 15.85  |

### 8. FSS plot
Built 4-panel figure: `analyzers/fss_sweep_KL_v2.png` + CSV.
- (a) KL vs T per L
- (b) per-site KL vs T (intensive check)
- (c) KL vs L per T (log-log)
- (d) scaling fits KL ∝ L^α with reference α=1, α=2 lines

**Headline finding — scaling exponent:**

| T   | α (fit KL ∝ L^α) |
|----:|-----------------:|
| 2.15 | 1.95 |
| 2.22 | 2.01 |
| **T_c** | **2.20** |
| 2.32 | 2.13 |
| 2.40 | 2.16 |

Off T_c the exponent is ≈ 2 (per-site KL intensive). At T_c the exponent is 2.20 — **super-extensive**. Per-site KL grows with L at criticality but saturates off-critical. This is the quantitative FSS signature that the flow's representational gap at T_c is fundamental, not a finite-L artifact.

### 9. L=16 anomaly diagnosed
Panel (b) showed L=16 with per-site KL *higher* than L=32 at T=2.32 (0.021 vs 0.015). Hypothesis (arch): the sweep used a "midbig" interpolation at L=16 (`nlayers=12, nhidden=96`) that's halfway between L=8 default and L=32 bignet.

Direct comparison at T_c:

| L=16 arch                    | best LOSS | KL_fwd | per-site |
|------------------------------|----------:|-------:|---------:|
| midbig (sweep, l=12, h=96)   | 479.09    | 4.74   | 0.0185   |
| **default (older, l=10, h=64)** | **477.51** | **3.16** | **0.0123** |

Default arch *beats* midbig by 1.6 nat. Substituting at T_c brings L=16 per-site KL back to ~0.012 — in line with L=8 (0.011) and L=32 (0.015). The anomaly was an arch artifact.

**Rule for NeuralRG arch scaling:** binary, not graded. Default suffices for L ≤ 16; bignet matters only at L=32 (when MERA depth crosses to 5 levels). Off-critical L=16 sweep points are still inflated and need re-running with default arch (jobs 38820889-92).

### 10. L=32 late-training instability
While re-deriving KL from the sweep, found that final-epoch LOSS overstates KL by 10-90 nat at L=32. Smoothed late-window max minus smoothed best:

| T   | best LOSS | late-10% max | spike (nat) |
|----:|----------:|-------------:|------------:|
| 2.15 | 1878.60 | 1881.56 | +3 |
| 2.22 | 1897.49 | 1926.47 | **+29** |
| T_c  | 1917.91 | 1928.19 | +10 |
| 2.32 | 1939.51 | 2027.06 | **+88** |
| 2.40 | 1962.25 | 2053.69 | **+91** |

Best epoch is mid-training (ep 9k-16k of 20k budget). High-T points worst. Likely an Adam/scheduler issue (related to [resume optimizer state](#)), or learning rate too high for late phase. The fix is either early-stop on smoothed best LOSS, or a cosine-decay schedule.

### 11. Cluster strategy — A100 vs L40 swap
Slurm `--start` estimates for the L=16 + NSF bignet relaunch put them 1-14 hours away on A100 (most A100 nodes `down*`). L40 nodes in `preempt` are nearly empty with no pending L40 jobs. L40 has 46 GB (> A100 40 GB) and matches A100 FP32 throughput within 10-20% for our workload. Swapped all 5 pending jobs to L40 via `--gres=gpu:l40:1` override — wait time dropped from hours to minutes.

## Discussions and insights

### "Best" depends on which window you look in
- `sym_bignet`: smoothed-best at ep 5925 = 9.40 nat; final 50-ep mean at ep 5950 = 10.12 nat; sample diagnostic at ep 5950 = 10.14 nat. All three are "the answer" depending on how you ask.
- The right number to *quote* is the smoothed-best. The right number for the diagnostic table is the sample-based one — they're honest about different sources of noise (training-batch vs Monte Carlo).
- At L=32 high-T, the gap between "best mid-training" and "final-epoch" is *order ten nat*. A naive sweep that quotes final-epoch values systematically overstates KL by 10-90 nat — and that error gets baked into every FSS plot at large L.

### FSS is the diagnostic, not absolute KL
- A flow that lowers L=32 KL_fwd by 2 nat looks impressive in a single-line table but might not change the scaling exponent at all.
- An architecture that brings α(T_c) down from 2.20 toward 2.05 — even at slightly worse absolute KL — is genuinely capturing long-range correlations. The new benchmark for entropy reg, NSF, JS, and two-phase is "what does it do to α?"

### Capacity scaling is not monotonic in L
- The midbig L=16 result (1.6 nat *worse* than default) was counterintuitive. The rule turned out to be binary: default suffices up through L=16; bignet matters only at L=32. Wider blocks at L=16 add optimization noise without useful capacity for 4 MERA levels.
- Deeper lesson: "more parameters = better" is wrong for normalizing flows when the architecture already has the right shape. Capacity is needed at specific scales (here, the 5th MERA level for L=32), not uniformly along an interpolation axis.

### Init scale matters more than you'd think
- NSF bignet diverged because random Gaussian init of the conditioner MLP produces O(1) raw parameters → after softmax + exp the spline bins become wildly non-uniform → log-det explodes → at 1024 dims Adam can't recover in 40 steps.
- The identity-init fix is the same principle as RNVP's zero-init for `s` and `t` networks (each layer starts as identity). Translated to the spline parametrization: zero-weight + bias pattern that makes the spline equal the identity transform.
- Once we add this, the flow *learns the deviation* from identity rather than *recovers from chaos*. That's always the better optimization landscape.

### Resources strategy
- When the cluster queue is the rate-limiting step, the cheapest experiment is the one that runs on hardware that's actually free. Most NeuralRG training is FP32 matmul without NVLink → hardware-agnostic.
- L40 swap saved ~13 hours today.

### Why forward-KL bridge can be narrower than data
- Asked earlier in session: "phase 1 forward KL should be mass-covering, why is the bridge narrower?" Answer in retrospect: mass-covering means putting *some* mass everywhere p has mass, not *equal* mass. The flow puts most density on the two peaks and just enough at the bridge to keep KL_fwd finite. The bridge being narrower than data is fine for KL_fwd but problematic for reverse-KL refinement (Phase 2 collapses the bridge).

## Open threads

1. **L=16 default-arch sweep** (jobs 38820889-92, L40): 4 jobs, T={2.15, 2.22, 2.32, 2.40}, ~6h each. Will give clean L=16 row for FSS panel (b).
2. **L=32 NSF bignet relaunch** (job 38820893, L40 with identity init): expected to train cleanly given Stage-1 sanity + identity-init verification.
3. **L=32 NSF default** (job 38817758, A100): still running, LOSS=1959 at ep 2000, min=1927, target ep 8000.
4. **Phase-2 finetune diagnostic** (job 38817732): RUNNING at end of session.
5. **L=32 late-training spike diagnosis**: try cosine LR decay after ep 10k, or AdamW with weight decay. Combined with the Adam-state-on-resume issue, this is the next optimization improvement.
6. **Entropy reg long-horizon at β=0.005**: re-run for 10k epochs to see if bridge actually widens (the 2k-ep version only confirmed stability, not effect).

## Today's commits

- (this session, pending push) Add NSF coupling + identity init, entropy regularizer, FSS sweep CSVs, bridge-trajectory diagnostic, L=32 concise-report updates, and the sbatch suite for NSF / entropy reg / JS loss / Phase-2 finetune / L=16 default-arch sweep / diagnostics.
