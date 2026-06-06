# Ising L=8 Thermodynamic Report (T=2.269)
Generated: 2026-05-21 15:27:01

| Method                           |  Picture   |    F (-lnZ)   |      E       |      S       |
| :------------------------------- | :--------: | :-----------: | :----------: | :----------: |
| **Exact (theory)**               |  discrete  |  **-60.1418** | **-43.4459** | **16.6959**  |
| **MCMC baseline (Wolff)**        |  discrete  |      N/A      | **-42.0611** |     N/A      |
| **Exact (theory)**               | continuous | **-148.6550** | **-30.3860** | **118.2690** |
| sym  (best reverse-KL)           | continuous |   -148.2645   |   -33.8592   |   114.4052   |
| hs_dataDriven  (best forward-KL) | continuous |      N/A      |     N/A      |   116.1207   |

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

At T = 2.269185 :  fix = 88.5132   (check: F_d - F_c = 88.5132)

Physical (energy units) for the discrete picture:
`U_d = -98.5868`,  `T*S_d = 37.8860`  (MCMC `U_d = -95.4443588256836`).

**Only the single best run per training mode is shown** (lowest loss); the
full per-run comparison is in the flow-diagnostic table below. What each
kept run reports:

- reverse-KL run (`sym`, `nsym`, …): the flow is sampled, so **all three**
  continuous quantities are measured — `F = loss`, `E = E_q[<A>]`,
  `S = H(q)`, satisfying `F = E - S`. Each is directly comparable
  to the continuous theory row (gap = how far the flow is from p_HS).
- HS data-driven (`hs_dataDriven`): loss → **S** in the continuous picture
  (MLE minimum = `H(p_HS) = S_c`).

Excluded for clarity: `sym_dataDriven` — MLE on **dequantised
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

| Method             |      <A>_q       |        H(q)       |     F_c^q     | KL(q‖p) | KL(p‖q) | epoch |
| :----------------- | :--------------: | :---------------: | :-----------: | :-----: | :-----: | :---: |
| **Exact (theory)** |   **-30.3860**   |    **118.2690**   | **-148.6550** |  **0**  |  **0**  |   -   |
| hs_dataDriven      | -27.3997 ± 0.101 | +119.9008 ± 0.090 |   -147.3005   |  1.3545 |  0.9568 | 27000 |
| nsym               | -32.8472 ± 0.083 | +113.3930 ± 0.076 |   -146.2402   |  2.4148 |  4.2483 |  9800 |
| sym                | -33.0950 ± 0.079 | +114.8473 ± 0.078 |   -147.9424   |  0.7127 |  2.0001 |  9800 |

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

| Source                                   |  Picture   |    F (-lnZ)   |      E       |      S       | KL(q‖p)  |  KL(p‖q)  |
| :--------------------------------------- | :--------: | :-----------: | :----------: | :----------: | :------: | :-------: |
| **Exact (theory)**                       |  discrete  |  **-60.1418** | **-43.4459** | **16.6959**  |    —     |     —     |
| MCMC dataset (Wolff)                     |  discrete  |      N/A      |   -42.0611   |     N/A      |    —     |     —     |
| **Exact (theory)**                       | continuous | **-148.6550** | **-30.3860** | **118.2690** |  **0**   |   **0**   |
| HS dataset (x ~ p_HS)                    | continuous |      N/A      |   -30.3860   |   118.2690   |    —     |     —     |
| *sym — training*                         | continuous |  *-148.2645*  |  *-33.8592*  |  *114.4052*  | *0.3906* |    N/A    |
| sym — diagnostic (epoch 9800)            | continuous |   -147.9424   |   -33.0950   |   114.8473   |   N/A    |   2.0001  |
| *hs_dataDriven — training*               | continuous |      N/A      |     N/A      |  *116.1207*  |   N/A    | *-2.1483* |
| hs_dataDriven — diagnostic (epoch 27000) | continuous |   -147.3005   |   -27.3997   |   119.9008   |  1.3545  |    N/A    |

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
