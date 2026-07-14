# Ising L=64 Thermodynamic Report (T=2.269)
Generated: 2026-07-12 15:32:11

| Method                                       |  Picture   |    F (-lnZ)    |       E        |       S       |
| :------------------------------------------- | :--------: | :------------: | :------------: | :-----------: |
| **Exact (theory)**                           |  discrete  | **-3808.6723** | **-2673.2499** | **1135.4223** |
| **MCMC baseline (Wolff)**                    |  discrete  |      N/A       | **-2570.3574** |      N/A      |
| **Exact (theory)**                           | continuous | **-9476.4283** | **-1855.3064** | **7621.1220** |
| sym_bignet  (best reverse-KL)                | continuous |   -8044.4521   |   -2437.8384   |   5606.6133   |
| hsBignet_baseline_nr2_b16  (best forward-KL) | continuous |      N/A       |      N/A       |   7579.2808   |

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

At T = 2.269185 :  fix = 5667.7560   (check: F_d - F_c = 5667.7560)

Physical (energy units) for the discrete picture:
`U_d = -6066.0995`,  `T*S_d = 2576.4837`  (MCMC `U_d = -5832.6171875`).

**Only the single best run per training mode is shown** (lowest loss); the
full per-run comparison is in the flow-diagnostic table below. What each
kept run reports:

- reverse-KL run (`sym`, `nsym`, …): the flow is sampled, so **all three**
  continuous quantities are measured — `F = loss`, `E = E_q[<A>]`,
  `S = H(q)`, satisfying `F = E - S`. Each is directly comparable
  to the continuous theory row (gap = how far the flow is from p_HS).
- HS data-driven (`hs_dataDriven`): loss → **S** in the continuous picture
  (MLE minimum = `H(p_HS) = S_c`).

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

| Method                                      |                                <A>_q                                 |                          H(q)                          |               F_c^q                |              KL(q‖p)              |   KL(p‖q)    | epoch |
| :------------------------------------------ | :------------------------------------------------------------------: | :----------------------------------------------------: | :--------------------------------: | :-------------------------------: | :----------: | :---: |
| **Exact (theory)**                          |                            **-1855.3064**                            |                     **7621.1220**                      |           **-9476.4283**           |               **0**               |    **0**     |   -   |
| hsBignet_baseline_b16                       |                          -1633.3841 ± 2.001                          |                   +7756.1626 ± 1.829                   |             -9389.5467             |              86.8817              |   65.6419    | 19800 |
| hsBignet_baseline_nr2_b16                   |                          -1507.5665 ± 1.902                          |                   +7812.5035 ± 1.490                   |             -9320.0700             |              156.3584             |   131.9481   | 19800 |
| hsBignet_bridge_w5.0t0.5                    |                          -1602.7337 ± 2.225                          |                   +7780.4917 ± 2.023                   |             -9383.2255             |              93.2028              |   68.8565    | 19900 |
| hsBignet_hcg_perscale_fixdil_gc5.0_b16      |                          -1584.6493 ± 1.839                          |                   +7798.1969 ± 1.563                   |             -9382.8461             |              93.5822              |   83.1353    | 19800 |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr1_b16 |                          -1685.4005 ± 1.628                          |                   +7743.0501 ± 1.494                   |             -9428.4506             |              47.9777              |   41.7882    | 14500 |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16 |                          -1630.6204 ± 1.808                          |                   +7724.4875 ± 1.595                   |             -9355.1079             |              121.3204             |   96.6086    |  7000 |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16 |                          -1686.0638 ± 1.621                          |                   +7749.3953 ± 1.489                   |             -9435.4591             |              40.9693              |   35.7133    | 14500 |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16 |                          -1669.2189 ± 1.797                          |                   +7716.4554 ± 1.595                   |             -9385.6743             |              90.7540              |   80.6952    |  7000 |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16 | +1888960208145768036559749120.0000 ± 165373014124171415974313984.000 | +135894075359516753920.0000 ± 61472571561455550464.000 | +1888960072251692723015254016.0000 | 1888960072251692723015254016.0000 |  26474.8904  | 14500 |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16 |                          -1679.1016 ± 1.602                          |                   +7715.0806 ± 1.490                   |             -9394.1822             |              82.2461              |   67.7136    |  7000 |
| hsBignet_hcg_shared_b16                     |                          -1668.3997 ± 1.334                          |                   +7757.7892 ± 1.222                   |             -9426.1889             |              50.2395              |   47.5345    | 19800 |
| hsBignet_hcg_shared_hcgh128_nr2_gc5.0_b16   |                          -1577.9707 ± 1.250                          |                   +7803.7870 ± 1.135                   |             -9381.7577             |              94.6706              |   88.3428    | 19800 |
| hsBignet_hcg_shared_hcgh64_nr2_gc5.0_b16    |                          -1558.3800 ± 1.267                          |                   +7832.3135 ± 1.146                   |             -9390.6935             |              85.7348              |   79.8846    | 19800 |
| hsBignet_hcg_shared_nr2_b16                 |                          -1615.0695 ± 1.388                          |                   +7791.6137 ± 1.262                   |             -9406.6833             |              69.7450              |   60.5627    | 19800 |
| hsBignet_i1_df4.0_b16                       |                          -1628.5525 ± 1.851                          |                   +7757.4678 ± 1.684                   |             -9386.0203             |              90.4080              |   66.2140    | 19800 |
| hsBignet_i2_stride16h32_b16                 |                          -1630.2497 ± 1.835                          |                   +7752.8672 ± 1.678                   |             -9383.1169             |              93.3115              |   70.3689    | 19800 |
| hsBignet_i2_stride4h32_b16                  |                          -1632.2858 ± 1.945                          |                   +7757.0162 ± 1.788                   |             -9389.3020             |              87.1263              |   64.6083    | 19800 |
| hsBignet_i2_stride4h64_b16                  |                          -1662.2133 ± 1.678                          |                   +7729.5413 ± 1.523                   |             -9391.7546             |              84.6737              |   66.4416    | 19800 |
| hsBignet_i2_stride8h32_b16                  |                          -1617.5582 ± 1.963                          |                   +7765.6714 ± 1.781                   |             -9383.2296             |              93.1988              |   69.5326    | 19800 |
| hsBignet_i2_stride8h32_nr2_b16              |                          -1713.1742 ± 1.555                          |                   +7711.9286 ± 1.457                   |             -9425.1028             |              51.3256              |   42.3790    | 19800 |
| hsBignet_i2_stride8h64_b16                  |                          -1657.7424 ± 1.796                          |                   +7733.1048 ± 1.647                   |             -9390.8472             |              85.5811              |   67.8176    | 19800 |
| hsBignet_iii1_lam1.0_b16                    |                          -1624.7109 ± 1.892                          |                   +7764.5771 ± 1.726                   |             -9389.2880             |              87.1404              |   64.6343    | 19800 |
| hs_bignet                                   |                          -1639.4787 ± 1.824                          |                   +7750.5401 ± 1.667                   |             -9390.0187             |              86.4096              |   64.3923    | 19900 |
| jsLoss_bignet_lam0.5                        |                          -1722.9985 ± 1.022                          |                   +7693.9854 ± 0.947                   |             -9416.9839             |              59.4445              |   166.7469   | 19900 |
| pathgrad_bignet                             |                         +18637.0803 ± 7.654                          |                   +118.5905 ± 0.756                    |            +18518.4898             |             27994.9181            | 3989782.2924 | 13000 |
| sym_bignet                                  |                          -2065.3497 ± 0.926                          |                   +7376.1605 ± 0.922                   |             -9441.5103             |              34.9181              |   301.9957   | 19900 |

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

| Source                                                                                            |  Picture   |              F (-lnZ)             |                 E                 |             S              |              KL(q‖p)              |   KL(p‖q)    |
| :------------------------------------------------------------------------------------------------ | :--------: | :-------------------------------: | :-------------------------------: | :------------------------: | :-------------------------------: | :----------: |
| **Exact (theory)**                                                                                |  discrete  |           **-3808.6723**          |           **-2673.2499**          |       **1135.4223**        |                 —                 |      —       |
| MCMC dataset (Wolff)                                                                              |  discrete  |                N/A                |             -2570.3574            |            N/A             |                 —                 |      —       |
| **Exact (theory)**                                                                                | continuous |           **-9476.4283**          |           **-1855.3064**          |       **7621.1220**        |               **0**               |    **0**     |
| HS dataset (x ~ p_HS)                                                                             | continuous |                N/A                |             -1856.2663            |         7620.1620          |                 —                 |      —       |
| *pathgrad_bignet — training*                                                                      | continuous |            *24101.1289*           |            *24101.1289*           |         *259.5755*         |            *33577.5572*           |     N/A      |
| pathgrad_bignet — diagnostic (epoch 13000)                                                        | continuous |             18518.4898            |             18637.0803            |          118.5905          |                N/A                | 3989782.2924 |
| *sym_bignet — training*                                                                           | continuous |            *-8044.4521*           |            *-2437.8384*           |        *5606.6133*         |            *1431.9762*            |     N/A      |
| sym_bignet — diagnostic (epoch 19900)                                                             | continuous |             -9441.5103            |             -2065.3497            |         7376.1605          |                N/A                |   301.9957   |
| *jsLoss_bignet_lam0.5 — training*                                                                 | continuous |            *13518.8184*           |            *3658.2349*            |        *5817.4189*         |            *22995.2467*           |     N/A      |
| jsLoss_bignet_lam0.5 — diagnostic (epoch 19900)                                                   | continuous |             -9416.9839            |             -1722.9985            |         7693.9854          |                N/A                |   166.7469   |
| *hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7658.6147*         |                N/A                |  *37.4927*   |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16 — diagnostic (epoch 14500)                            | continuous |             -9435.4591            |             -1686.0638            |         7749.3953          |              40.9693              |     N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7659.1194*         |                N/A                |  *37.9974*   |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16 — diagnostic (epoch 14500)                            | continuous | 1888960072251692723015254016.0000 | 1888960208145768036559749120.0000 | 135894075359516753920.0000 | 1888960072251692723015254016.0000 |     N/A      |
| *hsBignet_baseline_nr2_b16 — training*                                                            | continuous |                N/A                |                N/A                |        *7661.6393*         |                N/A                |  *40.5174*   |
| hsBignet_baseline_nr2_b16 — diagnostic (epoch 19800)                                              | continuous |             -9320.0700            |             -1507.5665            |         7812.5035          |              156.3584             |     N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-2_nr1_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7662.9412*         |                N/A                |  *41.8193*   |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr1_b16 — diagnostic (epoch 14500)                            | continuous |             -9428.4506            |             -1685.4005            |         7743.0501          |              47.9777              |     N/A      |
| *hsBignet_hcg_shared_b16 — training*                                                              | continuous |                N/A                |                N/A                |        *7669.6559*         |                N/A                |  *48.5339*   |
| hsBignet_hcg_shared_b16 — diagnostic (epoch 19800)                                                | continuous |             -9426.1889            |             -1668.3997            |         7757.7892          |              50.2395              |     N/A      |
| *hsBignet_i2_stride8h32_nr2_b16 — training*                                                       | continuous |                N/A                |                N/A                |        *7676.0814*         |                N/A                |  *54.9594*   |
| hsBignet_i2_stride8h32_nr2_b16 — diagnostic (epoch 19800)                                         | continuous |             -9425.1028            |             -1713.1742            |         7711.9286          |              51.3256              |     N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_nr2_gc5.0_b16_cont — training*                         | continuous |                N/A                |                N/A                |        *7677.7792*         |                N/A                |  *56.6572*   |
| *hsBignet_hcg_perscale_nodilate_initshared_nr2_gc5.0_b16 — training*                              | continuous |                N/A                |                N/A                |        *7680.5499*         |                N/A                |  *59.4279*   |
| *hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b16_cont — training*                             | continuous |                N/A                |                N/A                |        *7680.7183*         |                N/A                |  *59.5963*   |
| *hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b16 — training*                                  | continuous |                N/A                |                N/A                |        *7681.7616*         |                N/A                |  *60.6396*   |
| *hsBignet_hcg_perscale_b16 — training*                                                            | continuous |                N/A                |                N/A                |        *7681.8914*         |                N/A                |  *60.7695*   |
| *hsBignet_baseline_b16 — training*                                                                | continuous |                N/A                |                N/A                |        *7682.1562*         |                N/A                |  *61.0342*   |
| hsBignet_baseline_b16 — diagnostic (epoch 19800)                                                  | continuous |             -9389.5467            |             -1633.3841            |         7756.1626          |              86.8817              |     N/A      |
| *hsBignet_hcg_shared_nr2_b16 — training*                                                          | continuous |                N/A                |                N/A                |        *7682.5282*         |                N/A                |  *61.4063*   |
| hsBignet_hcg_shared_nr2_b16 — diagnostic (epoch 19800)                                            | continuous |             -9406.6833            |             -1615.0695            |         7791.6137          |              69.7450              |     N/A      |
| *hsBignet_iii1_lam1.0_b16 — training*                                                             | continuous |                N/A                |                N/A                |        *7683.9900*         |                N/A                |  *62.8681*   |
| hsBignet_iii1_lam1.0_b16 — diagnostic (epoch 19800)                                               | continuous |             -9389.2880            |             -1624.7109            |         7764.5771          |              87.1404              |     N/A      |
| *hsBignet_baseline_N50000_b16 — training*                                                         | continuous |                N/A                |                N/A                |        *7684.3797*         |                N/A                |  *63.2577*   |
| *hsBignet_i2_stride8h64_b16 — training*                                                           | continuous |                N/A                |                N/A                |        *7684.4589*         |                N/A                |  *63.3369*   |
| hsBignet_i2_stride8h64_b16 — diagnostic (epoch 19800)                                             | continuous |             -9390.8472            |             -1657.7424            |         7733.1048          |              85.5811              |     N/A      |
| *hsBignet_baseline_N100000_b16 — training*                                                        | continuous |                N/A                |                N/A                |        *7684.4898*         |                N/A                |  *63.3679*   |
| *hsBignet_i2_stride16h32_b16 — training*                                                          | continuous |                N/A                |                N/A                |        *7686.0511*         |                N/A                |  *64.9291*   |
| hsBignet_i2_stride16h32_b16 — diagnostic (epoch 19800)                                            | continuous |             -9383.1169            |             -1630.2497            |         7752.8672          |              93.3115              |     N/A      |
| *hsBignet_i2_stride4h64_b16 — training*                                                           | continuous |                N/A                |                N/A                |        *7686.4241*         |                N/A                |  *65.3022*   |
| hsBignet_i2_stride4h64_b16 — diagnostic (epoch 19800)                                             | continuous |             -9391.7546            |             -1662.2133            |         7729.5413          |              84.6737              |     N/A      |
| *hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b16 — training*                                           | continuous |                N/A                |                N/A                |        *7686.4654*         |                N/A                |  *65.3435*   |
| *hs_bignet — training*                                                                            | continuous |                N/A                |                N/A                |        *7686.5099*         |                N/A                |  *65.3880*   |
| hs_bignet — diagnostic (epoch 19900)                                                              | continuous |             -9390.0187            |             -1639.4787            |         7750.5401          |              86.4096              |     N/A      |
| *hsBignet_bridge_w5.0t0.5 — training*                                                             | continuous |                N/A                |                N/A                |        *7687.1508*         |                N/A                |  *66.0289*   |
| hsBignet_bridge_w5.0t0.5 — diagnostic (epoch 19900)                                               | continuous |             -9383.2255            |             -1602.7337            |         7780.4917          |              93.2028              |     N/A      |
| *hsBignet_i1_df4.0_b16 — training*                                                                | continuous |                N/A                |                N/A                |        *7687.1528*         |                N/A                |  *66.0309*   |
| hsBignet_i1_df4.0_b16 — diagnostic (epoch 19800)                                                  | continuous |             -9386.0203            |             -1628.5525            |         7757.4678          |              90.4080              |     N/A      |
| *hsBignet_i2_stride4h32_b16 — training*                                                           | continuous |                N/A                |                N/A                |        *7687.4459*         |                N/A                |  *66.3240*   |
| hsBignet_i2_stride4h32_b16 — diagnostic (epoch 19800)                                             | continuous |             -9389.3020            |             -1632.2858            |         7757.0162          |              87.1263              |     N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7689.7905*         |                N/A                |  *68.6685*   |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16 — diagnostic (epoch 7000)                             | continuous |             -9394.1822            |             -1679.1016            |         7715.0806          |              82.2461              |     N/A      |
| *hsBignet_i2_stride8h32_b16 — training*                                                           | continuous |                N/A                |                N/A                |        *7691.0914*         |                N/A                |  *69.9694*   |
| hsBignet_i2_stride8h32_b16 — diagnostic (epoch 19800)                                             | continuous |             -9383.2296            |             -1617.5582            |         7765.6714          |              93.1988              |     N/A      |
| *hsBignet_hcg_shared_progdil1-2-4_nr2_gc5.0_b16 — training*                                       | continuous |                N/A                |                N/A                |        *7691.9614*         |                N/A                |  *70.8394*   |
| *hsBignet_hcg_perscale_nr2_gc5.0_b16 — training*                                                  | continuous |                N/A                |                N/A                |        *7697.5284*         |                N/A                |  *76.4065*   |
| *hsBignet_hcg_shared_hcgh128_nr2_gc5.0_b16 — training*                                            | continuous |                N/A                |                N/A                |        *7698.6400*         |                N/A                |  *77.5181*   |
| hsBignet_hcg_shared_hcgh128_nr2_gc5.0_b16 — diagnostic (epoch 19800)                              | continuous |             -9381.7577            |             -1577.9707            |         7803.7870          |              94.6706              |     N/A      |
| *hsBignet_hcg_shared_hcgh64_nr2_gc5.0_b16 — training*                                             | continuous |                N/A                |                N/A                |        *7698.9422*         |                N/A                |  *77.8202*   |
| hsBignet_hcg_shared_hcgh64_nr2_gc5.0_b16 — diagnostic (epoch 19800)                               | continuous |             -9390.6935            |             -1558.3800            |         7832.3135          |              85.7348              |     N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7701.6256*         |                N/A                |  *80.5037*   |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16 — diagnostic (epoch 7000)                             | continuous |             -9385.6743            |             -1669.2189            |         7716.4554          |              90.7540              |     N/A      |
| *hsBignet_i2_stride8h32_nh192_nr2_lr5e-4_gc5.0_ga2_b16 — training*                                | continuous |                N/A                |                N/A                |        *7705.2237*         |                N/A                |  *84.1018*   |
| *hsBignet_hcg_perscale_fixdil_gc5.0_b16 — training*                                               | continuous |                N/A                |                N/A                |        *7706.6647*         |                N/A                |  *85.5428*   |
| hsBignet_hcg_perscale_fixdil_gc5.0_b16 — diagnostic (epoch 19800)                                 | continuous |             -9382.8461            |             -1584.6493            |         7798.1969          |              93.5822              |     N/A      |
| *hsBignet_baseline_l16h192_nr2_lr5e-4_gc5.0_ga2_b16 — training*                                   | continuous |                N/A                |                N/A                |        *7710.9558*         |                N/A                |  *89.8338*   |
| *hsBignet_hcg_shared_progdil1-4-16_nr2_gc5.0_b16 — training*                                      | continuous |                N/A                |                N/A                |        *7711.1192*         |                N/A                |  *89.9972*   |
| *hsBignet_hcg_perscale_nr2_b16 — training*                                                        | continuous |                N/A                |                N/A                |        *7711.3255*         |                N/A                |  *90.2036*   |
| *hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16 — training*                                          | continuous |                N/A                |                N/A                |        *7716.3284*         |                N/A                |  *95.2064*   |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16 — diagnostic (epoch 7000)                             | continuous |             -9355.1079            |             -1630.6204            |         7724.4875          |              121.3204             |     N/A      |
| *hsBignet_baseline_l16h192_lr5e-4_gc5.0_b16 — training*                                           | continuous |                N/A                |                N/A                |        *7723.1536*         |                N/A                |  *102.0316*  |
| *hsBignet_baseline_l16h192_lr5e-4_gc5.0_b16_ext_N200000_e10000_cos_lr5e-4_fromEp19800 — training* | continuous |                N/A                |                N/A                |        *7723.1536*         |                N/A                |  *102.0316*  |
| *hsBignet_baseline_l16h192_lr5e-4_gc5.0_b16_ext_N500000_e10000_fromEp19800 — training*            | continuous |                N/A                |                N/A                |        *7723.1536*         |                N/A                |  *102.0316*  |
| *hsBignet_i2_stride8h32_nh192_lr5e-4_gc5.0_b16 — training*                                        | continuous |                N/A                |                N/A                |        *7727.8315*         |                N/A                |  *106.7095*  |
| *hsBignet_baseline_l16h192_b16 — training*                                                        | continuous |                N/A                |                N/A                |        *7779.9192*         |                N/A                |  *158.7973*  |

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
