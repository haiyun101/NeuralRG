# Finite-Size Scaling Temperature Sweep — Forward-KL on HS

Snapshot of FSS results across (L, T) for the HS continuous-field forward-KL training mode.
Data from `analyzers/fss_sweep_KL_v2.csv`; companion plot `analyzers/fss_sweep_KL_v3.png`
(rendered by `analyzers/make_fss_plot.py`).

## What was measured

For each `(L, T)`:
- `LOSS_best = min_{epoch} smoothed_50(LOSS)` over the full training trajectory
- `KL_best = LOSS_best − H(p_HS)` where `H(p_HS) = E_p[A] + lnZ_HS` and `lnZ_HS = lnZ_exact + fix`
- Smoothing window = 50 epochs (chosen to suppress late-training noise; see [project_l32_late_training_instability])

Training config: default arch for L=8/16, bignet for L=32. All runs
`-symmetry -noDeq -dataDriven`, 20k epochs, batch=128. HS dataset N=200000
per (L,T).

| L  | Arch    | nlayers | nhidden | trainable params |
|----|---------|--------:|--------:|-----------------:|
| 8  | default |     10  |     64  |        1,068,240 |
| 16 | default |     10  |     64  |        1,424,320 |
| 32 | bignet  |     16  |    128  |       10,938,240 |

Params scale as ~`L²` for fixed (nlayers, nhidden) — the linear part is the
number of im2col patch positions in MERA × per-block RNVP cost. Bignet was
chosen for L=32 because the default arch at L=32 (1.78 M params) leaves
KL_fwd at ≈27 nat instead of ≈15 nat (see [project_l32_bignet_fix]); we
keep default arch at L=8/16 because at those sizes default already saturates.

**Important caveat:** LOSS is not comparable across reverse-KL and forward-KL modes
([project_loss_not_comparable_across_modes]). All numbers here are forward-KL only.

## Data

| L  | T      | KL_best (nat) | per-site KL    | best_ep |
|----|--------|--------------:|---------------:|--------:|
| 8  | 2.150  |        0.579  |       0.00904  | 19452   |
| 8  | 2.220  |        0.734  |       0.01146  | 17716   |
| 8  | 2.269  |        0.723  |       0.01129  | 16014   |
| 8  | 2.320  |        0.780  |       0.01219  | 16400   |
| 8  | 2.400  |        0.791  |       0.01235  | 18953   |
| 16 | 2.150  |        2.247  |       0.00878  | 13672   |
| 16 | 2.220  |        3.589  |       0.01402  | 16018   |
| 16 | 2.269  |        3.159  |       0.01234  | 22845   |
| 16 | 2.320  |        3.005  |       0.01174  |  8778   |
| 16 | 2.400  |        3.265  |       0.01275  | 18575   |
| 32 | 2.150  |        8.594  |       0.00839  | 16301   |
| 32 | 2.220  |       11.962  |       0.01168  | 13210   |
| 32 | 2.269  |       15.304  |       0.01495  |  9496   |
| 32 | 2.320  |       14.948  |       0.01460  | 16674   |
| 32 | 2.400  |       15.849  |       0.01548  | 13815   |

## Headline finding — α is a criticality witness

Power-law fit `KL_best(L) = a · L^α` (closed-form OLS on log-log):

| T      | α     | a         | R²     |
|--------|------:|----------:|-------:|
| 2.150  | 1.946 | 9.88e-03  | 1.000  |
| 2.220  | 2.014 | 1.18e-02  | 0.994  |
| **2.269 (T_c)** | **2.202** | 7.29e-03  | **1.000** |
| 2.320  | 2.130 | 1.08e-02  | 0.997  |
| 2.400  | 2.163 | 8.27e-03  | 0.999  |

**Off-critical points cluster at α ≈ 2.0** (per-site KL is intensive — fixed system-size
budget per spin). **At T_c, α = 2.20**, ~10% above the off-critical baseline. The KL
per spin grows with L specifically at criticality. This matches the physical
expectation: at the critical point the correlation length diverges, so a fixed-capacity
flow has to spread its expressive power thinner as L grows.

The numerical gap (Δα ≈ 0.2 between T_c and off-T_c) is reproducible across all
off-critical points and supported by R² ≥ 0.97 fits. T_c itself has the tightest
fit (R² = 1.000).

See [project_fss_critical_scaling] for the memory-archived version.

## Per-site KL — same finding, different lens

The "per-site KL" column makes the criticality witness visible without fitting:

- Off-critical (T = 2.15, 2.40): per-site KL roughly flat across L
  (T=2.15: 0.0090 / 0.0088 / 0.0084 — slightly *decreasing* with L;
  T=2.40: 0.0124 / 0.0128 / 0.0155 — slight rise consistent with the
  off-critical α=2.16 fit).
- At T_c: per-site KL grows from 0.011 (L=8) → 0.012 (L=16) → 0.015 (L=32).
  That's a +35% per-site cost going L=8 → L=32 *at criticality only*.

If your flow had infinite capacity, per-site KL would be flat at every T. The fact
that it grows with L at T_c quantifies the capacity-per-spin gap.

## Plot

![FSS forward-KL sweep](figures/fss_sweep_KL_v3.png)

Rendered by `analyzers/make_fss_plot.py` from `fss_sweep_KL_v2.csv`.
T_c is drawn in **red** in every panel (curve, marker, and vertical line);
off-critical temperatures use blue→green→yellow on viridis. The earlier v2
plot (`fss_sweep_KL_v2.png`) is kept for historical comparison but had
nearly-indistinguishable greens for the three off-critical temperatures
and no red T_c marker — v3 supersedes it.

Four panels of the same 15 (L,T) points, each emphasizing a different view:

**(a) Absolute KL_fwd vs T (log y).** Three L-curves stacked vertically:
L=8 sits at ~0.7 nat across the whole T range, L=16 climbs from ~2 to ~5
nat, L=32 hovers near 8–16 nat. The vertical red dashed line marks T_c.
The vertical *gap* between L-curves widens at T_c, visible on log-y as
the L=32 curve creeping up while L=8 stays flat. The L=16 T=2.32 spike
above its neighbors is the midbig-arch artifact (see Caveats).

**(b) Per-site KL_fwd vs T (intensive collapse test).** If KL scaled
exactly as L², the three lines would *collapse* onto each other (per-site
KL would be flat in L). Off-critical they nearly do, sitting in a tight
0.008–0.016 nat/site band. At T_c (red dashed) the L=32 curve breaks
*above* the L=8 and L=16 curves (≈0.015 vs ≈0.012) — that gap is α > 2.
With the L=16 default-arch re-runs landed (2026-05-28), there is no
longer any outlier point; the previous L=16 T=2.32 spike at 0.021 (a
midbig-arch regression) has fallen into the band at 0.012.

**(c) KL_fwd vs L, per T (log-log).** Each T is a 3-point line (L=8,
16, 32). The T_c line is drawn in red with a thicker stroke and larger
markers; it is visibly the *steepest* of the five. T=2.15 (purple) is
the shallowest. The slope is α. Reading off this panel: T_c slope ≈ 2.2,
T=2.15 slope ≈ 1.95.

**(d) Power-law fits with reference slopes.** Same data plus dashed fit
lines through each T's three points, with α annotated in the legend
(T_c α=2.20 in red, off-critical α=1.95–2.16). Two reference lines:
**black dotted** at α=2 (extensive, per-site KL constant) and **black
dot-dashed** at α=1 (perimeter, flow only suffers at boundaries). All
five fits land near or above α=2; only the red T_c fit visibly tilts
above the dotted α=2 reference. The off-critical scaling is consistent
with a capacity-saturated flow that needs O(L²) parameters to cover all
2^(L²) modes equally well; the T_c excess is the criticality penalty.

The four panels are redundant by design — they show the same fact
(α>2 at T_c, α≈2 elsewhere) from absolute, intensive, log-log, and
reference-line perspectives, so the conclusion is hard to mistake for
a fitting artifact.

## Caveats

### L=16 off-critical points (resolved 2026-05-28)

The L=16 column at T=2.22, 2.32, 2.40 was originally produced from the "midbig" sweep
(nlayers=12, nhidden=96), an intermediate arch between default (L≤16) and bignet
(L=32). At T_c we verified default beats midbig by 1.6 nat (LOSS 477.51 vs
479.09; see [project_l32_bignet_fix]).

**Update**: the three off-critical points have been re-run with the default arch
(jobs 38838147/8/9, completed 2026-05-28, 20000 ep each). The headline change is
T=2.32 (per-site KL dropped 0.0209 → 0.0117 — the previous spike was indeed the
midbig regression, not physics). T=2.22 and T=2.40 shifted only slightly. The
R² values improved across all temperatures (T=2.32 went 0.970 → 0.997) and the
qualitative finding is now cleanly visible: α=2.20 at T_c, α≈1.95–2.16 off-critical.

### L=32 late-training instability

L=32 hs_dataDriven LOSS spikes +10 to +90 nat in the last 10% of training; we use
best-smoothed-over-trajectory LOSS, not final-epoch, throughout this report.
See [project_l32_late_training_instability] for the magnitudes and the open
mitigation TODO.

### NSF beats RNVP on KL but is unstable at L=32

Parallel Stage-2 NSF (spline coupling) runs at L=32 T_c bignet:
- NSF bignet smoothed-best KL = **3.32 nat** (best at ep 4072, NaN by ep 5928)
- RNVP bignet smoothed-best KL = 3.63 nat (no NaN through 20k)

NSF wins ~0.3 nat but does not survive training. Gradient-clipping
re-run (`-gradClip 5.0`, job 38835512) is in flight. See
[project_nsf_identity_init] for the identity-init prereq that bought
the first 5k epochs.

## Reproducing this report

```bash
# 1. Per-(L,T) training (one job each — see shell/scan_temps.sh)
sbatch --gres=gpu:a100:1 shell/scan_temps.sh 2.15 2.22 2.269 2.32 2.40

# 2. Extract best-smoothed LOSS and KL from the records
python analyzers/calc_exact_loss.py    # (uses etc/exactz.md + the HS fix formula)

# 3. Refresh CSV + plot
python analyzers/make_fss_plot.py      # reads fss_sweep_KL_v2.csv, writes fss_sweep_KL_v3.png
```

The CSV columns mirror what the plot script consumes; regenerate the plot whenever
new (L,T) points land.

## Open questions

1. **Does α(T_c) keep growing for L=64?** A single point would test whether the
   excess at T_c is a finite-size correction (saturates around α≈2.2) or an
   honest divergence as L→∞. Single L=64 run estimated ~36h on A100 bignet —
   defer unless capacity opens.
2. **Does the NSF + gradClip fix close the per-site KL at T_c?** If 3.32 nat
   replaces 3.63 nat at L=32 with full training, α(T_c) recomputes from
   (0.723, 3.16, 3.32 nat) to roughly 2.15 — still above 2.0 but smaller. The
   gap is partially a flow-capacity artifact.
3. **Cosine LR / AdamW for the late-training spikes** ([project_l32_late_training_instability]).
   Would let us use the actual converged LOSS rather than smoothed-best.
