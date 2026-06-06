# Ising L=16 Thermodynamic Report (T=2.269)
Generated: 2026-05-21 15:27:04

| Method                           |  Picture   |    F (-lnZ)   |       E       |      S       |
| :------------------------------- | :--------: | :-----------: | :-----------: | :----------: |
| **Exact (theory)**               |  discrete  | **-238.6423** | **-167.4130** | **71.2292**  |
| **MCMC baseline (Wolff)**        |  discrete  |      N/A      | **-163.9458** |     N/A      |
| **Exact (theory)**               | continuous | **-592.8757** | **-118.5950** | **474.2808** |
| sym  (best reverse-KL)           | continuous |   -590.9911   |   -131.9934   |   458.9977   |
| hs_dataDriven  (best forward-KL) | continuous |      N/A      |      N/A      |   471.6035   |

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

At T = 2.269185 :  fix = 354.2335   (check: F_d - F_c = 354.2335)

Physical (energy units) for the discrete picture:
`U_d = -379.8911`,  `T*S_d = 161.6324`  (MCMC `U_d = -372.0234069824219`).

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

| Method             |       <A>_q       |        H(q)       |     F_c^q     | KL(q‖p) | KL(p‖q) | epoch |
| :----------------- | :---------------: | :---------------: | :-----------: | :-----: | :-----: | :---: |
| **Exact (theory)** |   **-118.5950**   |    **474.2808**   | **-592.8757** |  **0**  |  **0**  |   -   |
| hs_dataDriven      | -108.1010 ± 0.230 | +479.8033 ± 0.212 |   -587.9043   |  4.9715 |  3.7871 | 29500 |
| sym                | -129.4680 ± 0.162 | +460.9769 ± 0.161 |   -590.4449   |  2.4308 | 10.8203 |  9400 |

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

| Source                                   |  Picture   |    F (-lnZ)   |       E       |      S       | KL(q‖p)  |  KL(p‖q)  |
| :--------------------------------------- | :--------: | :-----------: | :-----------: | :----------: | :------: | :-------: |
| **Exact (theory)**                       |  discrete  | **-238.6423** | **-167.4130** | **71.2292**  |    —     |     —     |
| MCMC dataset (Wolff)                     |  discrete  |      N/A      |   -163.9458   |     N/A      |    —     |     —     |
| **Exact (theory)**                       | continuous | **-592.8757** | **-118.5950** | **474.2808** |  **0**   |   **0**   |
| HS dataset (x ~ p_HS)                    | continuous |      N/A      |   -118.5949   |   474.2808   |    —     |     —     |
| *sym — training*                         | continuous |  *-590.9911*  |  *-131.9934*  |  *458.9977*  | *1.8846* |    N/A    |
| sym — diagnostic (epoch 9400)            | continuous |   -590.4449   |   -129.4680   |   460.9769   |   N/A    |  10.8203  |
| *hs_dataDriven — training*               | continuous |      N/A      |      N/A      |  *471.6035*  |   N/A    | *-2.6773* |
| hs_dataDriven — diagnostic (epoch 29500) | continuous |   -587.9043   |   -108.1010   |   479.8033   |  4.9715  |    N/A    |

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
