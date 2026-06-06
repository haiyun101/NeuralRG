# Ising L=32 Thermodynamic Report (T=2.269)
Generated: 2026-05-22 22:17:55

| Method                        |  Picture   |    F (-lnZ)    |       E       |       S       |
| :---------------------------- | :--------: | :------------: | :-----------: | :-----------: |
| **Exact (theory)**            |  discrete  | **-952.6481**  | **-668.4678** |  **284.1802** |
| **MCMC baseline (Wolff)**     |  discrete  |      N/A       | **-647.0935** |      N/A      |
| **Exact (theory)**            | continuous | **-2369.5871** | **-466.6111** | **1902.9760** |
| sym_longer  (best reverse-KL) | continuous |   -2357.6262   |   -535.9175   |   1821.7087   |
| hs_bignet  (best forward-KL)  | continuous |      N/A       |      N/A      |   1906.6096   |

## How to read this table

All numbers are in **nats** (same unit as the training loss). Two thermodynamic
pictures, each with free energy / energy / entropy:

- **Discrete** — the ±1 Ising spins. `F_d = -lnZ_d`, `E_d = U_d/T`, `S_d`.
- **Continuous** — the Hubbard-Stratonovich field x. `F_c = -lnZ_c`,
  `E_c = <A>` with action `A(x)=½xᵀK⁻¹x − Σ log cosh x_i`, `S_c = <A> + lnZ_c`.

The two pictures are the **Picture** column (one theory row each); the
`F`/`E`/`S` columns hold `F_d,E_d,S_d` or `F_c,E_c,S_c` per the row's picture.

Identities (hold within each picture, and across):

```
  per picture :  F = E - S          (i.e. T*S = U + T*lnZ)
  across      :  F_d - F_c = fix    (HS Gaussian normalisation)
```

At T = 2.269185 :  fix = 1416.9390   (check: F_d - F_c = 1416.9390)

Physical (energy units) for the discrete picture:
`U_d = -1516.8774`,  `T*S_d = 644.8576`  (MCMC `U_d = -1468.375`).

**Only the single best run per training mode is shown** (lowest loss); the
full per-run comparison is in the flow-diagnostic table below. What each
kept run reports:

- reverse-KL run (`sym`, `nsym`, …): the flow is sampled, so **all three**
  continuous quantities are measured — `F = loss`, `E = E_q[<A>]`,
  `S = H(q)`, satisfying `F = E - S`. Each is directly comparable
  to the continuous theory row (gap = how far the flow is from p_HS).
- HS data-driven (`hs_dataDriven`): loss → **S** in the continuous picture
  (MLE minimum = `H(p_HS) = S_c`).

Excluded for clarity: `nsym_MCMCdataDriven, sym_MCMCdataDriven, sym_dataDriven, sym_dataDriven_skipHMC, sym_dataDriven_skipHMC_epoch500000` — MLE on **dequantised
  discrete spins**, a different target distribution than the HS field, so
  not comparable to the continuous theory row. (Still appears in the flow
  diagnostic below, which is training-mode agnostic.)

## Flow diagnostic (post-hoc, independent of training mode)

Model side — sample `x ~ q` from the trained flow:
`<A>_q = E_q[A(x)]`, `H(q) = -E_q[log q(x)]`, `F_c^q = <A>_q - H(q)`.
`F_c^q` is a variational upper bound on `-lnZ_c` (Gibbs).

**How the two KL directions are obtained — and which training
mode minimizes each:**
```
  KL(q‖p) = E_q[log q - log p] = F_c^q + lnZ_c      (mode-seeking)
    Obtained from FLOW samples x ~ q : draw x ~ q from the trained
    flow, score log q(x) and the action A(x); KL = F_c^q + lnZ_c.
    >>> This IS the REVERSE-KL (energy-based) training objective:
        reverse-KL loss = F_c^q = KL(q‖p) - lnZ_c, so minimising
        the loss minimises KL(q‖p).  No data needed.

  KL(p‖q) = E_p[log p - log q] = CE - H(p_HS)       (mass-covering)
    CE = -E_p[log q].  Obtained from HS DATA samples x ~ p_HS
    (the hs_L*.pt files): draw x ~ p_HS, score log q(x); subtract
    the MC entropy H(p_HS) = E_p[A] + lnZ_c.
    >>> This IS the FORWARD-KL / MLE (data-driven) objective:
        MLE loss = CE = H(p_HS) + KL(p‖q), so minimising the loss
        minimises KL(p‖q).  Needs samples from p (cannot be done
        reliably by importance-reweighting flow samples).
```
Each training mode minimises only ONE direction; this diagnostic
measures BOTH for every run regardless of how it was trained — so
the *off-objective* KL is the honest cross-check.

`KL(q‖p)` small but `KL(p‖q)` large ⇒ mode-dropping (flow ignores
modes of `p`). Both small ⇒ genuinely good fit. For data-driven runs
this is the only way to see the *model* `H(q)`: the training-time
`ENTROPY` logs the data-side cross-entropy `-E_data[log q]`, which can
dip below `H(p_HS)` purely from training-batch overfitting.

| Method             |       <A>_q        |        H(q)        |     F_c^q      |  KL(q‖p)  | KL(p‖q)  | epoch |
| :----------------- | :----------------: | :----------------: | :------------: | :-------: | :------: | :---: |
| **Exact (theory)** |   **-466.6111**    |   **1902.9760**    | **-2369.5871** |   **0**   |  **0**   |   -   |
| hs_bignet          | -421.4914 ± 0.600  | +1926.8304 ± 0.554 |   -2348.3218   |  21.2653  | 16.0055  |  9500 |
| hs_dataDriven      | -398.6420 ± 0.373  | +1950.3789 ± 0.336 |   -2349.0209   |  20.5662  | 29.4610  | 29500 |
| hs_haarPrior       | -387.4505 ± 0.380  | +1959.2013 ± 0.334 |   -2346.6517   |  22.9353  | 31.9144  |  9500 |
| hs_weightTying     | +4077.0619 ± 9.359 | +1711.1331 ± 0.640 |   +2365.9287   | 4735.5158 | 45.0620  |  9500 |
| nsym               | -535.2985 ± 0.318  | +1807.9476 ± 0.306 |   -2343.2461   |  26.3410  | 83.3031  |  990  |
| nsym_HP            | -555.2869 ± 0.292  | +1784.8471 ± 0.282 |   -2340.1340   |  29.4531  | 165.6845 |  990  |
| nsym_WT            | -529.5789 ± 0.645  | +1736.9738 ± 0.528 |   -2266.5527   |  103.0344 | 201.0364 |  990  |
| nsym_longer        | -533.0444 ± 0.347  | +1812.7222 ± 0.332 |   -2345.7666   |  23.8205  | 65.3606  |  1590 |
| sym                | -536.7518 ± 0.310  | +1816.7179 ± 0.305 |   -2353.4697   |  16.1174  | 110.3590 |  990  |
| sym_longer         | -533.7950 ± 0.316  | +1823.5504 ± 0.312 |   -2357.3455   |  12.2416  | 89.4075  |  1590 |

Standardized data-driven runs (flow trained on `u = x/σ`) are converted
back to physical scale via `log q_X(x) = log q_U(x/σ) - N·logσ`; `σ` is
read from each run's `flow_input_sigma.json`.

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
