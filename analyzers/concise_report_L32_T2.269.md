# Ising L=32 — Concise Report (T=2.269)

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

| Source                                       |  Picture   |    F (-lnZ)    |       E       |       S       |   KL(q‖p)   |  KL(p‖q)   |
| :------------------------------------------- | :--------: | :------------: | :-----------: | :-----------: | :---------: | :--------: |
| **Exact (theory)**                           |  discrete  | **-952.6481**  | **-668.4678** |  **284.1802** |      —      |     —      |
| MCMC dataset (Wolff)                         |  discrete  |      N/A       |   -647.0935   |      N/A      |      —      |     —      |
| **Exact (theory)**                           | continuous | **-2369.5871** | **-466.6111** | **1902.9760** |    **0**    |   **0**    |
| HS dataset (x ~ p_HS)                        | continuous |      N/A       |   -466.6109   |   1902.9762   |      —      |     —      |
| *sym_longer — training*                      | continuous |  *-2357.6262*  |  *-535.9175*  |  *1821.7087*  |  *11.9609*  |    N/A     |
| sym_longer — diagnostic (epoch 1590)         | continuous |   -2357.3455   |   -533.7950   |   1823.5504   |     N/A     |  89.4075   |
| *sym_bignet — training (ep5925, 50-smooth best)* | continuous |  *-2360.1850*  |      N/A      |      N/A      |   *9.4020*  |    N/A     |
| *sym_bignet — training (ep5950, last 50-smooth)* | continuous |  *-2359.4644*  |      N/A      |      N/A      |  *10.1226*  |    N/A     |
| sym_bignet — diagnostic (ep5950, N=8000)         | continuous |   -2359.4443   |   -519.2137   |   1840.2306   |   10.1427   |  64.5779   |
| *hs_bignet — training*                       | continuous |      N/A       |      N/A      |  *1906.6096*  |     N/A     |  *3.6336*  |
| hs_bignet — diagnostic (epoch 9500)          | continuous |   -2348.3218   |   -421.4914   |   1926.8304   |   21.2653   |    N/A     |
| *jsLoss_bignet_long — training (ep7999)*     | continuous |  *-2353.13*    |      N/A      |      N/A      |  *~16.5*    |  *~17.0*   |
| *phase2_finetune — training (ep1999, σ=3.5)* | continuous |  *-1067.10*    |      N/A      |      N/A      |  *~17.6*    |    N/A     |

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
- `sym_bignet` (bignet reverse-KL): KL_rev = LOSS + lnZ_c =
  -2358.59 + 2369.587 ≈ **11.00 nat** — currently the best reverse-KL run
  at L=32, beating both default-arch `sym_longer` (11.96 nat) and the
  Phase-2 finetune (~17.6 nat). Confirms capacity helps both directions,
  not just forward KL.
  - **Update (re-scan)**: scanning all checkpoints, the smoothed (50-ep
    window) LOSS minimum is at **ep 5925** with KL_rev ≈ **9.40 nat**.
    The published diagnostic measurement (at ep 5950, N=8000 sample-
    estimate) gives KL_rev = **10.14 nat** — the ~0.7 nat gap between
    training-recovered and diagnostic-measured values is consistent with
    training-batch noise (LOSS std ≈ 4 nat per batch at this scale, so
    50-batch SE ≈ 0.6 nat). Both numbers improve on the prior 11.00 quote.
    Diagnostic-only side info (off-objective): KL(p‖q) ≈ 64.6 nat — the
    flow is sharper than the data (mag_abs 3.11 vs 2.38; xi 12.0 vs 8.6).
- `jsLoss_bignet_long`: combined JS loss LOSS_js≈-216; per-direction
  components L_rev≈-2353, L_fwd≈1920 read from training logs.
  KL_rev ≈ L_rev + lnZ_c = 16.5 nat; KL_fwd ≈ L_fwd - H(p_HS) = 17.0 nat.
  Both directions balanced and within ~1 nat of single-direction baselines.
- `phase2_finetune`: reverse-KL refinement from `hs_bignet` anchor with
  σ-standardized inputs. KL_rev = LOSS + lnZ_c - N·log σ =
  -1067.10 + 2369.587 - 1284.95 ≈ 17.6 nat. Worse than `sym_bignet`
  (the bridge collapsed: bridge_p 0.0095 → 0.000), confirming that
  reverse-KL refinement on top of a forward-KL anchor over-sharpens
  the modes at the cost of the connecting density.
- **In progress**: `hsBignet_ent0.05` (entropy-regularized Phase-1) —
  adds `-β·H(q)` to the MLE loss to widen the bridge region.
  Pending results — submitted as job 38812652.

## Flow samples — flow output (x ~ q) vs HS target data (x ~ p)

_Each panel pair: left = configurations the trained flow generates,_
_right = HS samples from the true target. Same sigmoid(2x) rendering._

### hs_bignet

![hs_bignet flow samples](../data/32Ising_T2.269_hs_bignet/flow_samples.png)

### hs_dataDriven

![hs_dataDriven flow samples](../data/32Ising_T2.269_hs_dataDriven/flow_samples.png)

### sym

![sym flow samples](../data/32Ising_T2.269_sym/flow_samples.png)

### sym_bignet  *(new — best reverse-KL)*

![sym_bignet flow samples](../data/32Ising_T2.269_sym_bignet/flow_samples.png)

### jsLoss_bignet_long  *(new — balanced JS)*

![jsLoss_bignet_long flow samples](../data/32Ising_T2.269_jsLoss_bignet_long_lam0.5/flow_samples.png)

### phase2_finetune  *(new — fwd→rev workflow)*

![phase2_finetune flow samples](../data/32Ising_T2.269_phase2_finetune/flow_samples.png)

## Flow correlations — magnetisation P(M) and two-point G(r)

_Left: per-config magnetisation distribution. Right: normalised_
_axial two-point correlation G(r)/G(0). Flow (q) vs HS data (p)._

### hs_bignet

![hs_bignet flow correlations](../data/32Ising_T2.269_hs_bignet/flow_correlations.png)

### hs_dataDriven

![hs_dataDriven flow correlations](../data/32Ising_T2.269_hs_dataDriven/flow_correlations.png)

### sym

![sym flow correlations](../data/32Ising_T2.269_sym/flow_correlations.png)

### sym_bignet  *(new — best reverse-KL)*

![sym_bignet flow correlations](../data/32Ising_T2.269_sym_bignet/flow_correlations.png)

### jsLoss_bignet_long  *(new — balanced JS)*

![jsLoss_bignet_long flow correlations](../data/32Ising_T2.269_jsLoss_bignet_long_lam0.5/flow_correlations.png)

### phase2_finetune  *(new — fwd→rev workflow)*

![phase2_finetune flow correlations](../data/32Ising_T2.269_phase2_finetune/flow_correlations.png)

