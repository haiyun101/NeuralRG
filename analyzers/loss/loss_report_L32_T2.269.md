# Ising L=32 Thermodynamic Report (T=2.269)
Generated: 2026-07-12 15:34:25

| Method                              |  Picture   |    F (-lnZ)    |       E       |       S       |
| :---------------------------------- | :--------: | :------------: | :-----------: | :-----------: |
| **Exact (theory)**                  |  discrete  | **-952.6481**  | **-668.4678** |  **284.1802** |
| **MCMC baseline (Wolff)**           |  discrete  |      N/A       | **-647.0935** |      N/A      |
| **Exact (theory)**                  | continuous | **-2369.5871** | **-466.6111** | **1902.9760** |
| sym  (best reverse-KL)              | continuous |   -2355.8455   |      N/A      |      N/A      |
| hsBignet_ent0.05  (best forward-KL) | continuous |      N/A       |      N/A      |   1821.4387   |

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

| Method                                      |           <A>_q           |        H(q)        |     F_c^q      |   KL(q‖p)   | KL(p‖q)  | epoch |
| :------------------------------------------ | :-----------------------: | :----------------: | :------------: | :---------: | :------: | :---: |
| **Exact (theory)**                          |       **-466.6111**       |   **1902.9760**    | **-2369.5871** |    **0**    |  **0**   |   -   |
| hsBignet_baseline_b64                       |     -411.5089 ± 0.570     | +1934.6547 ± 0.524 |   -2346.1636   |   23.4235   | 17.0458  | 19800 |
| hsBignet_bridge_w5.0t0.5                    |     -411.6697 ± 0.669     | +1930.0257 ± 0.610 |   -2341.6953   |   27.8917   | 21.2812  |  1800 |
| hsBignet_combined_lam1.0_stride8h32_b64     |     -409.5754 ± 0.540     | +1938.8071 ± 0.493 |   -2348.3825   |   21.2046   | 15.8801  | 19800 |
| hsBignet_hcg_perscale_fixdil_gc5.0_b64      |     -414.7236 ± 0.560     | +1933.5632 ± 0.489 |   -2348.2868   |   21.3003   | 19.7584  | 19800 |
| hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64  |     -421.7461 ± 0.532     | +1930.5609 ± 0.485 |   -2352.3070   |   17.2801   | 16.1508  | 19800 |
| hsBignet_hcg_perscale_fixdil_vp1e-2_b64     |     -427.9958 ± 0.537     | +1928.9604 ± 0.488 |   -2356.9562   |   12.6309   | 11.4615  |  9500 |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b64 |     -414.9389 ± 0.500     | +1931.3776 ± 0.447 |   -2346.3165   |   23.2706   | 21.1091  |  4000 |
| hsBignet_hcg_perscale_fixdil_vp1e-3_b64     |     -429.0831 ± 0.530     | +1929.5956 ± 0.492 |   -2358.6787   |   10.9084   |  9.9815  |  9500 |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b64 |     -411.0287 ± 0.547     | +1935.9297 ± 0.484 |   -2346.9584   |   22.6287   | 18.0570  |  4500 |
| hsBignet_hcg_perscale_fixdil_vp1e-4_b64     |     -424.9759 ± 0.533     | +1930.0887 ± 0.500 |   -2355.0647   |   14.5224   | 11.6798  |  9500 |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b64 |     -407.0290 ± 0.583     | +1933.4896 ± 0.522 |   -2340.5186   |   29.0685   | 21.9958  |  4500 |
| hsBignet_hcg_perscale_fixdil_vp1e-5_b64     |     -428.6813 ± 0.551     | +1929.2785 ± 0.508 |   -2357.9598   |   11.6273   |  9.8999  |  9500 |
| hsBignet_hcg_perscale_fixdil_vp1e-5_nr2_b64 |     -410.3604 ± 0.600     | +1931.7496 ± 0.560 |   -2342.1100   |   27.4771   | 21.3138  |  3500 |
| hsBignet_hcg_shared_b64                     |     -438.2858 ± 0.427     | +1921.6580 ± 0.402 |   -2359.9438   |    9.6433   |  9.8912  | 19800 |
| hsBignet_hcg_shared_b64_adamwarm            |     -438.2858 ± 0.427     | +1921.6580 ± 0.402 |   -2359.9438   |    9.6433   |  9.8912  | 19800 |
| hsBignet_hcg_shared_nr2_b64                 |     -427.1282 ± 0.424     | +1931.7363 ± 0.400 |   -2358.8645   |   10.7226   | 10.8922  | 19800 |
| hsBignet_hcg_shared_nr2_b64_adamwarm        |     -427.1282 ± 0.424     | +1931.7363 ± 0.400 |   -2358.8645   |   10.7226   | 10.8922  | 19800 |
| hsBignet_hcg_shared_vp1e-2_b64              |     -413.2046 ± 0.442     | +1940.6969 ± 0.401 |   -2353.9015   |   15.6856   | 14.0372  |  9500 |
| hsBignet_hcg_shared_vp1e-3_b64              |     -408.7332 ± 0.446     | +1945.8975 ± 0.405 |   -2354.6307   |   14.9564   | 13.1088  |  9500 |
| hsBignet_hcg_shared_vp1e-4_b64              |     -413.4598 ± 0.441     | +1942.1479 ± 0.403 |   -2355.6077   |   13.9794   | 12.4878  |  9500 |
| hsBignet_hcg_shared_vp1e-5_b64              |     -413.2029 ± 0.444     | +1942.0166 ± 0.405 |   -2355.2195   |   14.3676   | 12.6603  |  9500 |
| hsBignet_i1_df4.0                           |     -418.7517 ± 0.565     | +1929.4765 ± 0.521 |   -2348.2282   |   21.3589   | 15.5443  | 19800 |
| hsBignet_i2_stride16h32                     |     -427.7674 ± 0.556     | +1921.3379 ± 0.516 |   -2349.1053   |   20.4818   | 15.7802  | 19800 |
| hsBignet_i2_stride4h32                      |     -424.9349 ± 0.508     | +1927.5214 ± 0.464 |   -2352.4563   |   17.1308   | 14.1388  | 19800 |
| hsBignet_i2_stride4h32_b64                  |     -414.5572 ± 0.500     | +1935.3539 ± 0.450 |   -2349.9111   |   19.6760   | 15.2565  | 19800 |
| hsBignet_i2_stride8h32                      | +603751.3783 ± 185938.151 | +1930.5675 ± 0.498 |  +601820.8108  | 604190.3979 | 21.9672  | 19800 |
| hsBignet_i2_stride8h32_b64                  |     -419.7325 ± 0.597     | +1928.6915 ± 0.547 |   -2348.4239   |   21.1632   | 16.0546  | 19800 |
| hsBignet_i2_stride8h32_nr2_b64              |     -434.1628 ± 0.527     | +1917.7305 ± 0.487 |   -2351.8933   |   17.6938   | 13.3671  | 15400 |
| hsBignet_i2_stride8h64_b64                  |     -417.8477 ± 0.550     | +1929.0513 ± 0.503 |   -2346.8990   |   22.6881   | 16.4216  | 19800 |
| hsBignet_iii1_lam0.1_b64                    |     -413.1722 ± 0.596     | +1932.1981 ± 0.552 |   -2345.3703   |   24.2168   | 17.0369  | 19800 |
| hsBignet_iii1_lam1.0_b64                    |     -415.4069 ± 0.559     | +1932.3443 ± 0.515 |   -2347.7512   |   21.8359   | 16.5882  | 19800 |
| hsBignet_iii1_lam10.0_b64                   |     -388.6225 ± 0.375     | +1959.2672 ± 0.333 |   -2347.8896   |   21.6975   | 30.1494  | 19800 |
| hs_bignet                                   |     -421.3687 ± 0.605     | +1926.9272 ± 0.557 |   -2348.2959   |   21.2911   | 16.0055  |  9500 |
| hs_haarPrior                                |     -387.4505 ± 0.380     | +1959.2013 ± 0.334 |   -2346.6517   |   22.9353   | 31.9144  |  9500 |
| hs_weightTying                              |     +4077.0619 ± 9.359    | +1711.1331 ± 0.640 |   +2365.9287   |  4735.5158  | 45.0620  |  9500 |
| jsLoss_bignet_long_lam0.5                   |     -464.3189 ± 0.479     | +1887.0193 ± 0.452 |   -2351.3382   |   18.2489   | 19.2334  |  7900 |
| nsym                                        |     -535.2985 ± 0.318     | +1807.9476 ± 0.306 |   -2343.2461   |   26.3410   | 83.3031  |  990  |
| nsym_HP                                     |     -555.2869 ± 0.292     | +1784.8471 ± 0.282 |   -2340.1340   |   29.4531   | 165.6845 |  990  |
| nsym_WT                                     |     -529.5789 ± 0.645     | +1736.9738 ± 0.528 |   -2266.5527   |   103.0344  | 201.0364 |  990  |
| nsym_longer                                 |     -533.0444 ± 0.347     | +1812.7222 ± 0.332 |   -2345.7666   |   23.8205   | 65.3606  |  1590 |
| pathgrad_bignet_long_ext                    |     -513.0215 ± 0.327     | +1848.5133 ± 0.324 |   -2361.5348   |    8.0523   | 55.3730  |  4950 |
| phase2_finetune                             |     -494.3934 ± 0.365     | +1857.2024 ± 0.354 |   -2351.5959   |   17.9912   | 30.8759  |  1500 |
| sym                                         |     -536.7298 ± 0.311     | +1816.7263 ± 0.305 |   -2353.4560   |   16.1311   | 110.3590 |  990  |
| sym_bignet                                  |     -519.1807 ± 0.327     | +1840.2494 ± 0.324 |   -2359.4301   |   10.1570   | 64.5779  |  5950 |
| sym_bignet_ext                              |     -519.1807 ± 0.327     | +1840.2494 ± 0.324 |   -2359.4301   |   10.1570   | 64.5779  |  5950 |
| sym_longer                                  |     -533.7950 ± 0.316     | +1823.5504 ± 0.312 |   -2357.3455   |   12.2416   | 89.4075  |  1590 |

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

| Source                                                                               |  Picture   |    F (-lnZ)    |       E       |       S       |   KL(q‖p)   |    KL(p‖q)    |
| :----------------------------------------------------------------------------------- | :--------: | :------------: | :-----------: | :-----------: | :---------: | :-----------: |
| **Exact (theory)**                                                                   |  discrete  | **-952.6481**  | **-668.4678** |  **284.1802** |      —      |       —       |
| MCMC dataset (Wolff)                                                                 |  discrete  |      N/A       |   -647.0935   |      N/A      |      —      |       —       |
| **Exact (theory)**                                                                   | continuous | **-2369.5871** | **-466.6111** | **1902.9760** |    **0**    |     **0**     |
| HS dataset (x ~ p_HS)                                                                | continuous |      N/A       |   -466.6109   |   1902.9761   |      —      |       —       |
| *sym — training*                                                                     | continuous |  *-2355.8455*  |      N/A      |      N/A      |  *13.7416*  |      N/A      |
| sym — diagnostic (epoch 990)                                                         | continuous |   -2353.4560   |   -536.7298   |   1816.7263   |     N/A     |    110.3590   |
| *nsym — training*                                                                    | continuous |  *-2341.5535*  |      N/A      |      N/A      |  *28.0336*  |      N/A      |
| nsym — diagnostic (epoch 990)                                                        | continuous |   -2343.2461   |   -535.2985   |   1807.9476   |     N/A     |    83.3031    |
| *nsym_HP — training*                                                                 | continuous |  *-2334.6865*  |      N/A      |      N/A      |  *34.9006*  |      N/A      |
| nsym_HP — diagnostic (epoch 990)                                                     | continuous |   -2340.1340   |   -555.2869   |   1784.8471   |     N/A     |    165.6845   |
| *nsym_WT — training*                                                                 | continuous |  *-2273.2793*  |      N/A      |      N/A      |  *96.3078*  |      N/A      |
| nsym_WT — diagnostic (epoch 990)                                                     | continuous |   -2266.5527   |   -529.5789   |   1736.9738   |     N/A     |    201.0364   |
| *phase2_finetune — training*                                                         | continuous |  *-917.2841*   |  *-509.2142*  |   *408.0699*  | *1452.3030* |      N/A      |
| phase2_finetune — diagnostic (epoch 1500)                                            | continuous |   -2351.5959   |   -494.3934   |   1857.2024   |     N/A     |    30.8759    |
| *sym_longer — training*                                                              | continuous |  *-1298.8237*  |   *150.0136*  |  *1448.8372*  | *1070.7634* |      N/A      |
| sym_longer — diagnostic (epoch 1590)                                                 | continuous |   -2357.3455   |   -533.7950   |   1823.5504   |     N/A     |    89.4075    |
| *pathgrad_bignet — training*                                                         | continuous |   *180.3591*   |   *180.3591*  |  *1453.8895*  | *2549.9462* |      N/A      |
| *pathgrad_bignet_b128 — training*                                                    | continuous |    *9.8453*    |    *9.8453*   |  *1451.9790*  | *2379.4324* |      N/A      |
| *pathgrad_bignet_long — training*                                                    | continuous |    *9.8453*    |    *9.8453*   |  *1451.9790*  | *2379.4324* |      N/A      |
| *sym_bignet — training*                                                              | continuous |  *-2322.5054*  |  *-604.9927*  |  *1717.5125*  |  *47.0817*  |      N/A      |
| sym_bignet — diagnostic (epoch 5950)                                                 | continuous |   -2359.4301   |   -519.1807   |   1840.2494   |     N/A     |    64.5779    |
| *nsym_longer — training*                                                             | continuous |  *-1210.1097*  |   *90.7138*   |  *1448.9839*  | *1159.4773* |      N/A      |
| nsym_longer — diagnostic (epoch 1590)                                                | continuous |   -2345.7666   |   -533.0444   |   1812.7222   |     N/A     |    65.3606    |
| *jsLoss_lam0.5 — training*                                                           | continuous |   *714.9796*   |  *-173.1918*  |  *1452.5170*  | *3084.5667* |      N/A      |
| *pathgrad_bignet_long_ext — training*                                                | continuous |  *-602.2075*   |  *-602.2075*  |  *1701.0914*  | *1767.3796* |      N/A      |
| pathgrad_bignet_long_ext — diagnostic (epoch 4950)                                   | continuous |   -2361.5348   |   -513.0215   |   1848.5133   |     N/A     |    55.3730    |
| *sym_bignet_ext — training*                                                          | continuous |  *-2340.1038*  |  *-544.8760*  |  *1795.2279*  |  *29.4833*  |      N/A      |
| sym_bignet_ext — diagnostic (epoch 5950)                                             | continuous |   -2359.4301   |   -519.1807   |   1840.2494   |     N/A     |    64.5779    |
| *jsLoss_bignet_lam0.5 — training*                                                    | continuous |  *5743.3809*   |  *4024.4253*  |  *1452.8545*  | *8112.9680* |      N/A      |
| *jsLoss_bignet_long_lam0.5 — training*                                               | continuous |  *5743.3809*   |  *4024.4253*  |  *1452.8545*  | *8112.9680* |      N/A      |
| jsLoss_bignet_long_lam0.5 — diagnostic (epoch 7900)                                  | continuous |   -2351.3382   |   -464.3189   |   1887.0193   |     N/A     |    19.2334    |
| *hsBignet_i2_stride8h32_nr2_b64 — training*                                          | continuous |      N/A       |      N/A      |  *1910.7498*  |     N/A     |    *7.7738*   |
| hsBignet_i2_stride8h32_nr2_b64 — diagnostic (epoch 15400)                            | continuous |   -2351.8933   |   -434.1628   |   1917.7305   |   17.6938   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-4_b64 — training*                                 | continuous |      N/A       |      N/A      |  *1911.8132*  |     N/A     |    *8.8372*   |
| hsBignet_hcg_perscale_fixdil_vp1e-4_b64 — diagnostic (epoch 9500)                    | continuous |   -2355.0647   |   -424.9759   |   1930.0887   |   14.5224   |      N/A      |
| *hsBignet_hcg_shared_b64 — training*                                                 | continuous |      N/A       |      N/A      |  *1911.9757*  |     N/A     |    *8.9997*   |
| hsBignet_hcg_shared_b64 — diagnostic (epoch 19800)                                   | continuous |   -2359.9438   |   -438.2858   |   1921.6580   |    9.6433   |      N/A      |
| *hsBignet_hcg_shared_b64_adamwarm — training*                                        | continuous |      N/A       |      N/A      |  *1911.9757*  |     N/A     |    *8.9997*   |
| hsBignet_hcg_shared_b64_adamwarm — diagnostic (epoch 19800)                          | continuous |   -2359.9438   |   -438.2858   |   1921.6580   |    9.6433   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-5_b64 — training*                                 | continuous |      N/A       |      N/A      |  *1912.0945*  |     N/A     |    *9.1184*   |
| hsBignet_hcg_perscale_fixdil_vp1e-5_b64 — diagnostic (epoch 9500)                    | continuous |   -2357.9598   |   -428.6813   |   1929.2785   |   11.6273   |      N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_nr2_gc5.0_b64 — training*                 | continuous |      N/A       |      N/A      |  *1912.2661*  |     N/A     |    *9.2900*   |
| *hsBignet_hcg_perscale_fixdil_vp1e-3_b64 — training*                                 | continuous |      N/A       |      N/A      |  *1912.5579*  |     N/A     |    *9.5819*   |
| hsBignet_hcg_perscale_fixdil_vp1e-3_b64 — diagnostic (epoch 9500)                    | continuous |   -2358.6787   |   -429.0831   |   1929.5956   |   10.9084   |      N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_nr2_gc5.0_b64 — training* | continuous |      N/A       |      N/A      |  *1912.6720*  |     N/A     |    *9.6960*   |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64 — training*            | continuous |      N/A       |      N/A      |  *1912.8344*  |     N/A     |    *9.8583*   |
| *hsBignet_hcg_perscale_nr2_b64 — training*                                           | continuous |      N/A       |      N/A      |  *1912.8739*  |     N/A     |    *9.8979*   |
| *hsBignet_hcg_shared_nr2_b64 — training*                                             | continuous |      N/A       |      N/A      |  *1913.0173*  |     N/A     |   *10.0413*   |
| hsBignet_hcg_shared_nr2_b64 — diagnostic (epoch 19800)                               | continuous |   -2358.8645   |   -427.1282   |   1931.7363   |   10.7226   |      N/A      |
| *hsBignet_hcg_shared_nr2_b64_adamwarm — training*                                    | continuous |      N/A       |      N/A      |  *1913.0173*  |     N/A     |   *10.0413*   |
| hsBignet_hcg_shared_nr2_b64_adamwarm — diagnostic (epoch 19800)                      | continuous |   -2358.8645   |   -427.1282   |   1931.7363   |   10.7226   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-2_b64 — training*                                 | continuous |      N/A       |      N/A      |  *1913.1230*  |     N/A     |   *10.1470*   |
| hsBignet_hcg_perscale_fixdil_vp1e-2_b64 — diagnostic (epoch 9500)                    | continuous |   -2356.9562   |   -427.9958   |   1928.9604   |   12.6309   |      N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_nr2_gc5.0_b64 — training*     | continuous |      N/A       |      N/A      |  *1913.4264*  |     N/A     |   *10.4504*   |
| *hsBignet_hcg_shared_vp1e-5_b64 — training*                                          | continuous |      N/A       |      N/A      |  *1915.4982*  |     N/A     |   *12.5222*   |
| hsBignet_hcg_shared_vp1e-5_b64 — diagnostic (epoch 9500)                             | continuous |   -2355.2195   |   -413.2029   |   1942.0166   |   14.3676   |      N/A      |
| *hsBignet_hcg_shared_vp1e-4_b64 — training*                                          | continuous |      N/A       |      N/A      |  *1915.7794*  |     N/A     |   *12.8034*   |
| hsBignet_hcg_shared_vp1e-4_b64 — diagnostic (epoch 9500)                             | continuous |   -2355.6077   |   -413.4598   |   1942.1479   |   13.9794   |      N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_gc5.0_b64 — training*     | continuous |      N/A       |      N/A      |  *1916.4131*  |     N/A     |   *13.4371*   |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_gc5.0_b64 — training*                | continuous |      N/A       |      N/A      |  *1916.5495*  |     N/A     |   *13.5735*   |
| *hsBignet_i2_stride4h32 — training*                                                  | continuous |      N/A       |      N/A      |  *1916.5906*  |     N/A     |   *13.6145*   |
| hsBignet_i2_stride4h32 — diagnostic (epoch 19800)                                    | continuous |   -2352.4563   |   -424.9349   |   1927.5214   |   17.1308   |      N/A      |
| *hsBignet_hcg_shared_vp1e-3_b64 — training*                                          | continuous |      N/A       |      N/A      |  *1916.6099*  |     N/A     |   *13.6339*   |
| hsBignet_hcg_shared_vp1e-3_b64 — diagnostic (epoch 9500)                             | continuous |   -2354.6307   |   -408.7332   |   1945.8975   |   14.9564   |      N/A      |
| *hsBignet_hcg_shared_vp1e-2_b64 — training*                                          | continuous |      N/A       |      N/A      |  *1916.7699*  |     N/A     |   *13.7939*   |
| hsBignet_hcg_shared_vp1e-2_b64 — diagnostic (epoch 9500)                             | continuous |   -2353.9015   |   -413.2046   |   1940.6969   |   15.6856   |      N/A      |
| *hsBignet_hcg_perscale_b64 — training*                                               | continuous |      N/A       |      N/A      |  *1916.8950*  |     N/A     |   *13.9190*   |
| *hsBignet_i2_stride4h32_b64 — training*                                              | continuous |      N/A       |      N/A      |  *1916.9971*  |     N/A     |   *14.0211*   |
| hsBignet_i2_stride4h32_b64 — diagnostic (epoch 19800)                                | continuous |   -2349.9111   |   -414.5572   |   1935.3539   |   19.6760   |      N/A      |
| *hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_gc5.0_b64 — training*         | continuous |      N/A       |      N/A      |  *1917.1461*  |     N/A     |   *14.1701*   |
| *hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b64 — training*                     | continuous |      N/A       |      N/A      |  *1917.6154*  |     N/A     |   *14.6394*   |
| *hsBignet_i1_df4.0 — training*                                                       | continuous |      N/A       |      N/A      |  *1917.7622*  |     N/A     |   *14.7861*   |
| hsBignet_i1_df4.0 — diagnostic (epoch 19800)                                         | continuous |   -2348.2282   |   -418.7517   |   1929.4765   |   21.3589   |      N/A      |
| *hsBignet_i2_stride16h32 — training*                                                 | continuous |      N/A       |      N/A      |  *1917.7696*  |     N/A     |   *14.7936*   |
| hsBignet_i2_stride16h32 — diagnostic (epoch 19800)                                   | continuous |   -2349.1053   |   -427.7674   |   1921.3379   |   20.4818   |      N/A      |
| *hsBignet_i2_stride8h32_b64 — training*                                              | continuous |      N/A       |      N/A      |  *1917.8095*  |     N/A     |   *14.8335*   |
| hsBignet_i2_stride8h32_b64 — diagnostic (epoch 19800)                                | continuous |   -2348.4239   |   -419.7325   |   1928.6915   |   21.1632   |      N/A      |
| *hsBignet_combined_lam1.0_stride8h32_b64 — training*                                 | continuous |      N/A       |      N/A      |  *1917.8740*  |     N/A     |   *14.8980*   |
| hsBignet_combined_lam1.0_stride8h32_b64 — diagnostic (epoch 19800)                   | continuous |   -2348.3825   |   -409.5754   |   1938.8071   |   21.2046   |      N/A      |
| *hsBignet_i2_stride8h32 — training*                                                  | continuous |      N/A       |      N/A      |  *1917.9254*  |     N/A     |   *14.9494*   |
| hsBignet_i2_stride8h32 — diagnostic (epoch 19800)                                    | continuous |  601820.8108   |  603751.3783  |   1930.5675   | 604190.3979 |      N/A      |
| *hsBignet_baseline_b64 — training*                                                   | continuous |      N/A       |      N/A      |  *1918.3207*  |     N/A     |   *15.3447*   |
| hsBignet_baseline_b64 — diagnostic (epoch 19800)                                     | continuous |   -2346.1636   |   -411.5089   |   1934.6547   |   23.4235   |      N/A      |
| *hsBignet_i2_stride8h64_b64 — training*                                              | continuous |      N/A       |      N/A      |  *1918.3388*  |     N/A     |   *15.3627*   |
| hsBignet_i2_stride8h64_b64 — diagnostic (epoch 19800)                                | continuous |   -2346.8990   |   -417.8477   |   1929.0513   |   22.6881   |      N/A      |
| *hsBignet_iii1_lam1.0_b64 — training*                                                | continuous |      N/A       |      N/A      |  *1918.3590*  |     N/A     |   *15.3830*   |
| hsBignet_iii1_lam1.0_b64 — diagnostic (epoch 19800)                                  | continuous |   -2347.7512   |   -415.4069   |   1932.3443   |   21.8359   |      N/A      |
| *hsBignet_iii1_lam0.1_b64 — training*                                                | continuous |      N/A       |      N/A      |  *1918.4601*  |     N/A     |   *15.4841*   |
| hsBignet_iii1_lam0.1_b64 — diagnostic (epoch 19800)                                  | continuous |   -2345.3703   |   -413.1722   |   1932.1981   |   24.2168   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64 — training*                              | continuous |      N/A       |      N/A      |  *1918.8851*  |     N/A     |   *15.9091*   |
| hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64 — diagnostic (epoch 19800)                | continuous |   -2352.3070   |   -421.7461   |   1930.5609   |   17.2801   |      N/A      |
| *hs_dataDriven — training*                                                           | continuous |      N/A       |      N/A      |  *1919.2093*  |     N/A     |   *16.2333*   |
| *hs_bignet — training*                                                               | continuous |      N/A       |      N/A      |  *1919.2093*  |     N/A     |   *16.2333*   |
| hs_bignet — diagnostic (epoch 9500)                                                  | continuous |   -2348.2959   |   -421.3687   |   1926.9272   |   21.2911   |      N/A      |
| *hsBignet_hcg_perscale_initshared_nr2_gc5.0_b64 — training*                          | continuous |      N/A       |      N/A      |  *1919.6833*  |     N/A     |   *16.7073*   |
| *hsBignet_baseline_nr2_lr5e-4_gc5.0_b64 — training*                                  | continuous |      N/A       |      N/A      |  *1919.8356*  |     N/A     |   *16.8596*   |
| *hsBignet_hcg_perscale_fixdil_gc5.0_b64 — training*                                  | continuous |      N/A       |      N/A      |  *1921.8303*  |     N/A     |   *18.8542*   |
| hsBignet_hcg_perscale_fixdil_gc5.0_b64 — diagnostic (epoch 19800)                    | continuous |   -2348.2868   |   -414.7236   |   1933.5632   |   21.3003   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b64 — training*                             | continuous |      N/A       |      N/A      |  *1922.6172*  |     N/A     |   *19.6412*   |
| hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b64 — diagnostic (epoch 4500)                | continuous |   -2346.9584   |   -411.0287   |   1935.9297   |   22.6287   |      N/A      |
| *hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b64 — training*                             | continuous |      N/A       |      N/A      |  *1922.7171*  |     N/A     |   *19.7411*   |
| hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b64 — diagnostic (epoch 4500)                | continuous |   -2340.5186   |   -407.0290   |   1933.4896   |   29.0685   |      N/A      |
| *hsBignet_hcg_perscale_initshared_gc5.0_b64 — training*                              | continuous |      N/A       |      N/A      |  *1922.9738*  |     N/A     |   *19.9978*   |
| *hsBignet_hcg_perscale_fixdil_vp1e-5_nr2_b64 — training*                             | continuous |      N/A       |      N/A      |  *1923.2251*  |     N/A     |   *20.2491*   |
| hsBignet_hcg_perscale_fixdil_vp1e-5_nr2_b64 — diagnostic (epoch 3500)                | continuous |   -2342.1100   |   -410.3604   |   1931.7496   |   27.4771   |      N/A      |
| *hsBignet_bridge_w5.0t0.5 — training*                                                | continuous |      N/A       |      N/A      |  *1923.8990*  |     N/A     |   *20.9229*   |
| hsBignet_bridge_w5.0t0.5 — diagnostic (epoch 1800)                                   | continuous |   -2341.6953   |   -411.6697   |   1930.0257   |   27.8917   |      N/A      |
| *hsBignet_ent0.005 — training*                                                       | continuous |      N/A       |      N/A      |  *1924.8153*  |     N/A     |   *21.8393*   |
| *hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b64 — training*                             | continuous |      N/A       |      N/A      |  *1925.0514*  |     N/A     |   *22.0754*   |
| hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b64 — diagnostic (epoch 4000)                | continuous |   -2346.3165   |   -414.9389   |   1931.3776   |   23.2706   |      N/A      |
| *hs_nsf_bignet_clip5 — training*                                                     | continuous |      N/A       |      N/A      |  *1928.0758*  |     N/A     |   *25.0998*   |
| *hsBignet_iii1_lam10.0_b64 — training*                                               | continuous |      N/A       |      N/A      |  *1932.7540*  |     N/A     |   *29.7779*   |
| hsBignet_iii1_lam10.0_b64 — diagnostic (epoch 19800)                                 | continuous |   -2347.8896   |   -388.6225   |   1959.2672   |   21.6975   |      N/A      |
| *hs_haarPrior — training*                                                            | continuous |      N/A       |      N/A      |  *1935.0308*  |     N/A     |   *32.0548*   |
| hs_haarPrior — diagnostic (epoch 9500)                                               | continuous |   -2346.6517   |   -387.4505   |   1959.2013   |   22.9353   |      N/A      |
| *hs_weightTying — training*                                                          | continuous |      N/A       |      N/A      |  *1946.8197*  |     N/A     |   *43.8437*   |
| hs_weightTying — diagnostic (epoch 9500)                                             | continuous |   2365.9287    |   4077.0619   |   1711.1331   |  4735.5158  |      N/A      |
| *hsBignet_ent0.05 — training*                                                        | continuous |      N/A       |      N/A      | *205106.1156* |     N/A     | *203203.1396* |
| *hs_nsf_bignet_noclip — training*                                                    | continuous |      N/A       |      N/A      |     *nan*     |     N/A     |     *nan*     |
| *hs_nsf_default — training*                                                          | continuous |      N/A       |      N/A      |     *nan*     |     N/A     |     *nan*     |

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
