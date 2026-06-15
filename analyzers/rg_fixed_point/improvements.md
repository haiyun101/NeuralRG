# RG Fixed-Point Probe — Improvement Directions Report

> Companion to `rg_fixed_point_report.md` — the diagnosis of the
> current architecture's pathologies at T_c. This report is the
> forward-looking roadmap of "how to fix it".

## Premise: the structural mismatch

The "Why T_c is hard on this architecture" subsection of
`rg_fixed_point_report.md` distils to one statement:

> **A Gaussian-prior MERA flow forces a 1/4-slow + 3/4-independent-
> N(0, 1)-fast decomposition at every scale — the geometry of the
> trivial Gaussian fixed point. The 2D Ising T_c attractor is the
> Wilson–Fisher CFT (η = 1/4, non-Gaussian operator spectrum), in
> which no scale admits a clean fast/slow separation. The
> architecture's implicit fixed point and the physical fixed point
> are structurally incompatible.**

Every T_c pathology in V3 (deep-block collapse), V4 (slow-mode
inflation / contraction), and V5 (KS-vs-RMS-G two-axis failure)
traces back to this single root cause. Three orthogonal classes
of intervention can attack it: **change the prior, change the
architecture (dispatch geometry), change the loss (training
signal)**. A fourth class — "change the framing" — is the
zero-cost retreat.

This report lays out 8 concrete proposals across these classes,
each with implementation pointers, predicted V5 outcomes, cost,
and risks. A cost-leverage execution order is given at the end.

---

## Overview

| ID   | Scheme                                          | Class      | Expected leverage | Engineering cost | Main risk            |
|------|------------------------------------------------|------------|-------------------|-----------------|----------------------|
| I.1  | Student-t prior                                | prior      | low–med           | low (1d)        | symptom, not cause   |
| I.2  | **Scheme A** conditional Gaussian P(z_f \| z_s) | prior      | med–high          | med (1w)        | implementation depth |
| I.3  | **Scheme B** interacting prior (energy-based)   | prior      | high              | med–high (2w)   | MCMC / log-Z         |
| I.4  | Coarse-grained Ising prior (literal RG decim.)  | prior      | high              | high (2w)       | prior data + diff.    |
| I.5  | Learned non-Gaussian prior (AR / nested flow)   | prior      | med–high          | high (2–3w)     | parameter blow-up    |
| II.1 | Learnable kept-fraction                        | arch       | med               | med (1w)        | dispatch redesign    |
| II.2 | **Scheme C** self-similarity, no discarding     | arch+loss  | high (if it works)| very high (1m+) | leaves NeuralRG      |
| III.1| Multi-scale loss                               | loss       | high              | low (1w)        | λ_scale tuning       |
| III.2| Block-RG supervised training (V5-as-loss)       | loss       | med–high          | med (2w)        | V5 data generation   |
| IV.1 | Reframe                                        | framing    | —                 | zero            | abandon FP framing   |

---

## I. Prior changes

### I.1  Student-t prior (do this as a negation experiment)

**Motivation.** Cheapest possible "lightweight non-Gaussian" test.
Student-t has heavy tails and can absorb some non-Gaussianity
without breaking the prior's separability (`p(z) = ∏_i p(z_i)`).

**Implementation.**
- Add `source/student_t.py` alongside `source/gaussian.py`,
  implementing `logProbability(x)` (via `torch.distributions.
  StudentT.log_prob` summed over sites) and `sample(n)`.
- `train/learn.py:57`: replace `s = source.Gaussian(...)` with
  `s = source.StudentT(df=4, shape=...)`.
- Everything else in the training loop untouched.

**Predicted V5.** V3 deep residuals shift; V5 KS (especially the
rev-KL T_c row at 0.32+) should improve ~30%. **G(r) damage
(RMS-G ≈ 0.62) probably does *not* heal** — heavy tails address
the marginal shape, but Wilson–Fisher's core obstacle is
**long-range spatial structure**, not marginal shape.

**Decision criterion.** If I.1 leaves the pathologies in place,
"prior is the bottleneck" is formally excluded, and the next bets
should move to architecture (II) or loss (III). This is a
**valuable negative result** — its worth is in eliminating a
hypothesis.

---

### I.2  Scheme A — Conditional Gaussian (Hierarchical Prior)

**Motivation.** Drop the assumption that the 3/4 fast modes
`z_fast` are independent of the 1/4 slow modes `z_slow`. Replace
the prior with:

```
P(z) = P(z_slow) · P(z_fast | z_slow)
```

where `P(z_fast | z_slow)` is still Gaussian, but its **mean or
variance is a function of z_slow**. Physically: frozen fast modes
no longer participate in the macroscopic RG flow, but their local
entanglement with the slow modes is *preserved*. The architecture
no longer forces fast/slow decoupling, which softens the
structural conflict.

**Implementation pointer.**
- Add `source/conditional_gaussian.py` exposing
  `logProbability(z, condition=z_slow)`.
- Main change in `flow/hierarchy/template.py`: the `forward` /
  `inverse` loop must split the latent into `(z_slow, z_fast)`
  before prior evaluation, where `z_slow = z[..., ::2^depth,
  ::2^depth]` and `z_fast` is the complement.
- Conditional-structure parameterisations:
  - **Simplest**: `P(z_fast | z_slow) = N(μ(z_slow), σ²(z_slow))`,
    with `μ, σ` a shared CNN.
  - **Stronger**: local conditioning — each fast position `(i, j)`
    conditioned on *the slow mode of the coarse cell containing it*
    (geometrically natural).
- Training: `flow.logProbability(x)` computes
  `log P(z_slow) + log P(z_fast | z_slow) + log|det J|`.

**Predicted V5.**
- V5 marginal KS should improve ~50% (the fast-position marginal
  is relaxed from the hard N(0, 1) constraint to "conditional
  N(μ(z_slow), σ²)" — no longer forced to a unimodal N(0, 1)).
- V5 RMS-G should improve markedly (the slow → fast coupling
  leaves a channel for "slow-mode-induced spatial correlation",
  so the flow needn't destroy G(r) to flatten anything).
- V3 deep residuals should *converge upward from below* (rev-KL
  no longer has an incentive to collapse to identity — collapsing
  forces fast to be N(0,1) independent of slow, violating the new
  prior structure).

**Risks.** `μ, σ` network too small ⇒ conditional structure
degenerates to unconditional; too large ⇒ the prior absorbs the
flow's work (the prior alone fits the data, and the flow's
bijection degenerates to identity). Ablations needed to find the
right capacity.

**Compatibility.** High — coexists with the existing MERA,
`weightTying`, `haarPrior`. **Most worth trying first among the
prior-replacement routes.**

---

### I.3  Scheme B — Interacting Prior (Energy-Based)

**Motivation.** Abandon the non-interacting Gaussian assumption
entirely. At each RG step, set the prior for the 3/4 "discarded"
DOFs to a distribution with **interaction terms** — for example,
a local energy from φ⁴ theory:

```
−log P(z_fast) = ∑_i [½ m² z_i² + ¼ λ z_i⁴] + ∑_⟨ij⟩ J z_i z_j
```

Physically: **acknowledge that even discarded modes are governed
by critical fluctuations**, rather than treating them as thermal
noise unrelated to the system. `(m², λ, J)` can be learnable
parameters that adapt to temperature.

**Implementation pointer.**
- Add `source/phi4.py` with:
  - `logProbability(x) = -E_phi4(x)` (up to log Z constant).
  - `sample(n)` via HMC or Langevin. Note: **sample is only used
    on the `q.sample()` path. Reverse-KL needs HMC; forward-KL /
    MLE needs only logProb, not sample**.
- Training: as in I.2, replace prior at the latent end with the
  φ⁴ energy. No need for log Z (constant, no gradient), but
  beware: in the reverse-KL loss `E_q[log q − log p]`, the
  difference `log p_target − log p_prior` exposes a log-Z
  difference explicitly ⇒ requires thermodynamic integration or
  log-Z ratio estimation. **Forward-KL avoids this entirely**
  (MLE doesn't need the normalising constant), so **scheme B
  should be tried first on fwd-KL / dataDriven**.

**Predicted V5.**
- The most aggressive correction; **theoretically the most
  faithful to the physics**: the prior itself carries Wilson–
  Fisher-flavoured IR behaviour at the latent end (take
  `m² < 0, λ > 0` near criticality).
- All three V5 metrics (KS, W1, RMS-G) should improve;
  particularly rev-KL's RMS-G 0.62 → should drop to ~0.1 (deep
  blocks no longer incentivised to collapse — collapsing violates
  the prior's built-in correlations).
- V4 slow-mode cascade should genuinely match block-RG's
  std 3.51 → 2.46 and kurt → −1.84 (because the prior end is
  *itself* bimodal already; the flow needn't fake it).

**Risks.**
- High engineering complexity: φ⁴ energy gradient integrates with
  the training loop; numerical stability needs care (`m² < 0`
  region is a double-well potential).
- Reverse-KL log-Z mishandling makes the loss offset meaningless
  ⇒ strongly recommend combining only with fwd-KL / dataDriven.
- φ⁴ is not literally the continuum description of 2D Ising
  universality (they're connected on the RG flow but not the
  same field theory). Treat as a reasonable approximation.

**Compatibility.** Medium — training loop reworked, but the MERA
body is preserved. **The most physically serious proposal; worth
long-term investment.**

---

### I.4  Coarse-grained Ising prior (literal RG decimation)

**Motivation.** "The truly RG-fixed-point-correct prior" — set
the latent end to be a smaller-L Ising distribution
`Ising(L/2^depth, T)`. The flow then literally implements the
L → L/2^depth RG decimation map; the structural mismatch is
eliminated **at the source**.

**Implementation.**
- Pre-generate MCMC samples / evaluator for `Ising(L = 2 or 4,
  T_c)`.
- `source/ising.py` already exposes a Gaussian-approximation
  `logProbability`, which can be plugged into the prior end.
- A cleaner implementation: **make the prior another (smaller)
  NeuralRG flow** (nested). This is close to the MERA-literature
  "hierarchical RG network" idea.

**Predicted V5.** Similar to scheme B — at heart this is also an
energy-based interacting prior, just more literal. RMS-G should
drop to ≈ 0 (the flow is no longer forced to break long-range
correlations).

**Risks.** Requires a prior dataset and a differentiable prior
`logProbability`; nested flows multiply training cost. More
engineering than I.3 but more "textbook" physically.

---

### I.5  Learned non-Gaussian prior

**Motivation.** Don't preset the prior form; learn it with a
smaller autoregressive model or a normalising flow. Most flexible
but most distant from physical first principles.

**Implementation.** The prior is another trainable module — say,
a PixelCNN or a smaller RNVP stack.
`flow.prior.logProbability` and `flow.prior.sample` both route
through this submodule.

**Predicted V5.** Can improve, but lacks physical interpretability.
**Not recommended first** — push the "structured prior" route
(I.2, I.3, I.4) to its limit before resorting to fully learned.

---

## II. Architectural changes

### II.1  Learnable kept-fraction

**Motivation.** Right now `im2col.getIndeices` hard-codes the
dispatch "fast/slow ratio" at 1/4 kept, 3/4 dropped. This is a
hand-fixed geometric inductive bias. Make it adjustable:

- Modify `flow/hierarchy/im2col.py:getIndeices` to take a
  `keep_fraction` parameter.
- Or more aggressively: let each scale's stride be smaller than 2
  (stride √2 is geometrically impossible, but *channel-wise
  splitting* can emulate 2/4 kept).
- Simplest learnable version: **run both RNVP blocks at each
  scale on stride=1** (no downsampling), deepening only via more
  RNVP layers per scale → equivalent to 1/4 → 1 kept fraction.

**Predicted V5.** At T_c, 1/2 kept may substantially beat
1/4 kept, because Wilson–Fisher's slow-mode dimension exceeds
1/4 of the field. Off-T_c the opposite: 1/4 kept is enough, maybe
even better. The contrast itself is a *clean physical signal*.

**Risks.** Changing the dispatch geometry touches the core of the
MERA design; baselines must be re-run for fair comparison.

---

### II.2  Scheme C — Self-Similarity Without Discarding

**Motivation.** A genuine critical RG transformation needn't toss
variables into a Gaussian wastebasket. If we design a continuous
map that **performs only a scale rescaling** (no discarding) and
constrain the system's energy-function form to be *invariant*
before and after the map (look for an actual saddle point),
rather than using KL divergence to approach a trivial attractor —
that is arguably the more orthodox Wilson RG route.

**Architectural form.**

```
x_L (L×L field)
     │
     ▼  scale-rescaling map Φ: x_L → x_{L/b}
     │  (b is the RG scale factor; Φ is an *irreversible*
     │   scale-transformation + rearrangement map, b²-fold
     │   dimension reduction)
     │
     ▼
x_{L/b}  ← the output; no latent stack
```

**Key constraint / loss.**

```
loss = D( E_θ(x_L) , E_θ'(Φ(x_L)) )
```

where:
- `E_θ` is a parameterised energy function (e.g., local
  polynomial ansatz).
- Training learns Φ and E_θ jointly; demands that Φ preserves the
  *form* of E_θ (but permits θ → θ', i.e. RG flow).
- The saddle point `θ = θ'` is the genuine Wilson RG fixed point.

**Implementation.** This is a **major break** — abandons the
invertible-bijection NeuralRG frame, closer to "NN ansatz for RG
transformations" (cf. Koch-Janusz & Ringel 2018, Lenggenhager et
al. 2020, Hou & Wang 2023). `flow.logProbability` is gone;
replaced by a parameterised energy + Monte Carlo estimate of
"distribution preservation".

**Predicted outcome.** If it works, **this is a literal Wilson
RG fixed-point probe**, not an "imitation". But the engineering
cannot reuse the existing NeuralRG code — essentially a new
project.

**Compatibility.** Almost none. Worth pursuing as a **long-term
direction** (postdoc / paper-grade), not as a patch on the
current sweep.

---

## III. Loss changes

### III.1  Multi-scale loss (**do this first — strongly recommended**)

**Motivation.** Currently the flow's log-prob is evaluated **only
at the latent end** (where the prior sits). The intermediate
scales y_s receive **no direct training signal**. That is why
rev-KL can fit the bulk in the shallow blocks and then collapse
the deep blocks to identity: **the deep blocks never see whether
G(r) is broken**.

**Implementation.**
- In `flow/hierarchy/template.py`, return intermediate outputs:

```python
def forward_with_intermediates(self, x):
    intermediates = []
    for no in range(len(self.indexI)):
        ...
        if no % 2 == 1:   # record once per scale
            intermediates.append(x.clone())
    return x, forwardLogjac, intermediates
```

- In `train/learn.py:learnInterface`, add a cross-scale penalty:

```python
ys = intermediates                       # [y_0, y_1, ..., y_4]
scale_loss = 0
for s in range(len(ys) - 1):
    a = zscore(ys[s][..., ::2, ::2])
    b = zscore(ys[s+1])
    scale_loss += KS_distance(a, b) + W1_distance(a, b)
loss = main_loss + lambda_scale * scale_loss
```

Or more physically: **drive y_s's normalised G(r)/G(0) toward
r^(-η)** (Onsager exact η = 1/4):

```python
G_emp = compute_G(ys[s])
G_theory = r ** (-0.25)
scale_loss += MSE(G_emp / G_emp[0], G_theory)
```

The latter writes the critical universality directly into the
loss.

**Predicted V5.** Direct attack on RMS-G 0.62 — rev-KL can no
longer escape by collapsing deep, because collapsing violates the
scale-invariance penalty. Predict RMS-G falls to 0.1–0.2; KS
follows.

**Cost.** Modify `learn.py` and `template.py` (expose
intermediate outputs); **under a week**. No full sweep rerun
needed: 2 flows (`sym_bignet` and `hs_bignet`) suffice for the
ablation.

**Why this ranks first:**
- Lowest cost; leverage at least as high as prior replacement.
- Directly attacks the shared root cause (deep blocks have no
  training signal) of *all* V3/V4/V5 pathologies.
- A **necessary prerequisite** to class-I and class-II changes:
  if the pathology persists after the multi-scale loss, there is
  finally a strong reason to change the prior or the architecture;
  otherwise such changes are firing in the dark.
- Yields a publishable clean ablation: **with / without scale
  loss, effect on V5 RMS-G.**

---

### III.2  Block-RG supervised training (V5-as-loss)

**Motivation.** V5 already provides the Wilson–Kadanoff block-RG
ground truth (`rg_v5_blockRG_compare.py`). Train the flow to match
V5's output directly:

```python
# Pre-compute V5 on the HS data:  x_s = AvgPool2d(2)^s(x_data)
# Then at training time:
for s in range(num_scales):
    loss += lambda_s * KL(q_ys || x_s)
```

V5 graduates from **diagnostic tool** to **training signal**. Deep
blocks can no longer collapse, because they must match a specific
non-Gaussian distribution at every scale.

**Implementation.** Medium cost: the V5 cascade already exists in
numpy; needs porting to differentiable torch ops, plus a
pre-computed block-RG dataset (for L=32, 8000–20000 samples ≈ GB
scale).

**Predicted V5.** Strong alignment signal ⇒ V5 RMS-G should *by
construction* drop to 0 (it is the direct training target).
Caveat: **this contaminates V5 as a diagnostic**, so V4
(forward-direction probe) must be retained as the unpolluted
witness.

---

## IV. Framing

### IV.1  Reframe (zero cost)

**Motivation.** If none of the above are tractable, reframe the
current architecture's T_c results as:

> "How well can a Gaussian-prior MERA flow fake Wilson–Fisher?"

This is itself a meaningful scientific question (and is in fact
the question the original LDM paper implicitly answers). Remove
the "RG fixed point" phrasing from every T_c section; rewrite as
"approximate Gaussian-FP attempt at T_c". Then foreground the
T = 2.15 and T = 2.40 results as **clean success cases**.

**Implementation.** Edit `rg_fixed_point_report.md` and `_zh.md`
section headings and the "acceptable claims" block. No code
changes.

**When.** If III.1 reveals the issue is genuinely architectural,
the short-term paper should take route IV.1; long-term work then
moves to I.3 / II.2.

---

## Recommended execution order

Cost-leverage prioritised path:

### Phase 1 — within 1 week

1. **III.1 multi-scale loss** (mandatory first step)
   - Pick `sym_bignet` and `hs_bignet` as baselines; sweep
     `lambda_scale ∈ {0, 0.1, 1.0, 10.0}` for 8 runs total.
   - After training, run V5 to read off changes in RMS-G and KS.
   - **Decision:** if RMS-G falls noticeably (< 0.3), missing
     deep-scale signal is the primary cause ⇒ proceed to Phase 2
     for prior reform. If not, the issue is deeper ⇒ jump to
     Phase 3.
2. **I.1 Student-t prior** (in parallel, as a negation
   experiment)
   - One `hs_bignet` replica with df = 4.
   - **Decision:** if V5 KS improves but RMS-G does not, this
     confirms that "marginal vs spatial structure" are two
     independent axes and the prior can only act on the marginal.

### Phase 2 — 2–4 weeks

3. **I.2 Scheme A conditional Gaussian prior**
   - Design `P(z_fast | z_slow) = N(μ(z_slow), σ²(z_slow))` with
     `μ, σ` from a 1–2-layer CNN.
   - Train with fwd-KL / dataDriven (sidesteps the log-Z issue).
   - Compare against Phase 1 results: if RMS-G improves further,
     prior-side hard independence is a *second*, independent bug.
4. **III.2 V5-as-loss** as a Phase-2 control
   - Establishes the theoretical V5 ceiling improvement.

### Phase 3 — 1–3 months

5. **I.3 Scheme B energy-based prior (φ⁴)**
   - The most physically serious route; worth a long investment.
   - Start on fwd-KL / dataDriven; reverse-KL waits until log-Z
     handling is settled.
6. **I.4 Coarse-grained Ising prior** as the "literal" counterpart
   to I.3.

### Phase 4 — long-term (paper / postdoc grade)

7. **II.2 Scheme C self-similarity constraint**
   - Leaves the existing NeuralRG framework; a standalone project.
   - Aligns with Koch-Janusz & Ringel 2018 / Lenggenhager et al.
     2020.

### In parallel

8. **II.1 learnable kept-fraction** as an ablation series
   (1/4 vs 1/2 vs 3/4 kept), runnable alongside Phase 2.

---

## Failure-mode prediction table

| Scheme   | Predicted V5 KS (T_c rev-KL) | Predicted V5 KS (T_c fwd-KL) | Predicted V5 RMS-G | Risk: what stays broken     |
|----------|-----------------------------:|-----------------------------:|-------------------:|------------------------------|
| baseline | 0.32+                        | 0.08                         | 0.62 / 0.04        | —                            |
| I.1 t    | 0.22                         | 0.06                         | 0.55 / 0.04        | spatial structure (RMS-G)    |
| I.2 cond | 0.18                         | 0.05                         | 0.30               | still misses block-RG kurt   |
| I.3 EBM  | 0.10                         | 0.04                         | 0.10               | implementation complexity    |
| I.4 Ising| 0.08                         | 0.04                         | 0.05               | prior data + differentiability|
| II.1 1/2 | 0.20                         | 0.06                         | 0.40               | dispatch redesign            |
| III.1    | 0.15                         | 0.05                         | 0.20               | scale_loss hyper-tuning      |
| III.2    | 0.05                         | 0.04                         | 0.02               | V5 no longer independent     |

(All predictions are rough estimates, ±50%. The real verdict
comes from Phase-1 results.)

---

## Related work / reference directions

- **Koch-Janusz & Ringel, *Nature Phys.* 2018**
  *Mutual information, neural networks and the renormalization group.*
  The thought-seed of Scheme C.
- **Lenggenhager et al., *Phys. Rev. X* 2020**
  *Optimal renormalization group transformation from information theory.*
  Concrete realisation of the self-similarity constraint.
- **Marchand, Wang, Ringel 2024**
  *Wavelet conditional renormalization group.*
  A wavelet-based realisation of Scheme A.
- **Hou & Wang 2023**
  *Renormalization group flow as optimal transport.*
  An optimal-transport take on Scheme C.
- **Bachtis et al., *PRR* 2021**
  *Phase transitions in machine learning models.*
  Systematic discussion of normalising-flow capability at
  criticality.

---

## See also / companion files

- `rg_fixed_point_report.md` — pathology diagnosis + V1–V5 checks
- `rg_fixed_point_report_zh.md` — Chinese edition of the same
- `analyzers/rg_fixed_point/rg_v5_blockRG_compare.py` — V5 code
- `analyzers/rg_fixed_point/rg_fixed_point_v4_dataforward.py` — V4 code
- `flow/hierarchy/im2col.py` — dispatch geometry (touched by II.1 / II.2)
- `source/gaussian.py` — current prior (touched by I.1–I.5)
- `train/learn.py:learnInterface` — training loop (touched by III.1 / III.2)
