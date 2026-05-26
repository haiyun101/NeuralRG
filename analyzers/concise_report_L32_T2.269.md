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

## Flow samples — flow output (x ~ q) vs HS target data (x ~ p)

_Each panel pair: left = configurations the trained flow generates,_
_right = HS samples from the true target. Same sigmoid(2x) rendering._

### hs_bignet

![hs_bignet flow samples](../data/32Ising_T2.269_hs_bignet/flow_samples.png)

### hs_dataDriven

![hs_dataDriven flow samples](../data/32Ising_T2.269_hs_dataDriven/flow_samples.png)

### hs_haarPrior

![hs_haarPrior flow samples](../data/32Ising_T2.269_hs_haarPrior/flow_samples.png)

### hs_weightTying

![hs_weightTying flow samples](../data/32Ising_T2.269_hs_weightTying/flow_samples.png)

### sym

![sym flow samples](../data/32Ising_T2.269_sym/flow_samples.png)

## Flow correlations — magnetisation P(M) and two-point G(r)

_Left: per-config magnetisation distribution. Right: normalised_
_axial two-point correlation G(r)/G(0). Flow (q) vs HS data (p)._

### hs_bignet

![hs_bignet flow correlations](../data/32Ising_T2.269_hs_bignet/flow_correlations.png)

### hs_dataDriven

![hs_dataDriven flow correlations](../data/32Ising_T2.269_hs_dataDriven/flow_correlations.png)

### hs_haarPrior

![hs_haarPrior flow correlations](../data/32Ising_T2.269_hs_haarPrior/flow_correlations.png)

### hs_weightTying

![hs_weightTying flow correlations](../data/32Ising_T2.269_hs_weightTying/flow_correlations.png)

### sym

![sym flow correlations](../data/32Ising_T2.269_sym/flow_correlations.png)

