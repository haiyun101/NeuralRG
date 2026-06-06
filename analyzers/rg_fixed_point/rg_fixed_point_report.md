# RG Fixed-Point Probe of the L=32 MERA Flow

> **Status (post-V3 robustness check).**
> The principal conclusion of the original analysis — that
> reverse-KL flows at T_c satisfy our hypothesis because
> `MSE(f_4, f_5)` ≈ 10⁻⁵ on z-scored outputs — was **overturned
> by the V3 identity-residual check**: the rev-KL deep blocks
> are individually near-identity (`f_4`, `f_5` residuals ≈ 0.02–0.3
> on N(0, I)), so two near-identity functions trivially produce
> matching outputs on the same probe. The MSE-on-z-scored-outputs
> probe alone cannot distinguish a learned scale-invariant
> fixed-point map from "the optimizer left these blocks as
> approximate identities". The forward-KL "deep MSE ≈ 2" remains a
> real functional asymmetry (V3 residuals 2.8–15.4 at f_4 — large,
> non-trivial work).
>
> The original Motivation, Methodology, Numerical results, and
> Interpretation sections below are preserved as the **historical
> record**; the corrected conclusions are in the **"Robustness
> checks (V1 / V2 / V3) — and the major reinterpretation they
> force"** section near the end. Read both: the original is what
> the probe measured; the robustness section is what the
> measurement actually means.
>
> **V2b addendum (one-slot chain).** V2's chain-input probe feeds
> the entire 4-element output of the previous block as the next
> block's 2×2 patch. The production MERA composition only re-uses
> 1 of the 4 patch positions (the kept-coarse slot); the other 3
> are fresh N(0, I). V2b fixes this: it sets
> `h_s[..., 0, 0] = o_{s+1}[..., 0, 0]` and draws the remaining
> 3 patch positions fresh. Result: rev-KL `MSE(f_4, f_5)` jumps
> from V2's 0.0000 to V2b's **1.489**, identical to the off-T_c
> controls. The original "rev-KL satisfies our hypothesis"
> reading was therefore an artefact of two stacked geometry
> mismatches with production (the V3 identity-on-N(0,I) finding
> AND V2's full-tuple chain). Neither original-V2 nor original-V4
> survives the geometry-correction.
>
> **V4 addendum (data-forward direction, kept-coarse only).** A
> complementary probe runs real HS data through the MERA forward
> (analysis) direction and compares the **slow-mode sub-lattice**
> (stride `2^s` at scale s) between adjacent intermediates `y_s`,
> `y_{s+1}`. A first pass that averaged over the full L × L
> `y_s` is retracted — it mixed the slow mode with `1 − 1/4^s`
> frozen-latent positions that are trained-to-N(0, 1), so the
> reported "rev-KL std ≈ 1.1 at deep scales" was the latent floor.
>
> On the kept-coarse sub-lattice the V4 picture is sharper:
> reverse-KL flows shrink the slow-mode std monotonically
> 3.51 → 0.94 with kurtosis → 0 (slow mode is being **erased**, not
> coarse-grained) and develop *anti-correlated* nearest-neighbours
> on the coarse lattice (G(1)/G(0) negative); forward-KL flows
> instead inflate the slow-mode std to 8× its input value at
> intermediate scales (T_c hs_dataDriven: std 3.51 → 28.94 at y_3)
> before partial relaxation. Block-RG ground truth (V5) gives
> std 3.51 → 2.46 and a bimodal kurt ≈ −1.84 — neither family
> matches it. V4-corrected therefore confirms V3 (rev-KL deep
> blocks are doing nothing physical) AND adds two new pathologies:
> rev-KL slow-mode collapse-to-Gaussian, and rev-KL anti-correlated
> spatial structure.
>
> **V5 addendum (Wilson–Kadanoff block-RG ground truth).** The
> block-average cascade `x_s = AvgPool2d(2)^s(x)` is run on the
> same HS input as V4 and compared scale-by-scale to the
> MERA-subsampled slow-mode field. Two-axis verdict: **fwd-KL gets
> spatial structure right but amplitude wrong** (RMS deviation of
> G(r)/G(0) is 0.02–0.07 at coarse depth for hs_bignet, but the
> marginal std blows up 4–9× above block-RG); **rev-KL gets
> amplitude near-prior but spatial structure entirely wrong**
> (RMS-G ≈ 0.62–0.67 at every coarse scale, an order of magnitude
> worse than every fwd-KL row; KS marginal distance grows
> monotonically 0.13 → 0.20). The off-T_c control at T = 2.40
> matches block-RG in both axes (KS → 0.04, RMS-G → 0.07 at
> deep scales). V5 is therefore the external benchmark V4 needed:
> no flow on this architecture does Wilson-style RG at T_c, and
> the training objective dictates which axis breaks.

## Motivation

In traditional physics, when a system is at a critical phase
transition, zooming out (coarse-graining) leaves the physics
unchanged. If a normalizing flow has genuinely learned the physics
of the Ising model, the intermediate layers of its network should
reflect this: the mathematical transformations applied at deeper
layers should become identical to one another.

This is the **RG fixed-point** signature. We design a probe that
measures how the per-scale block transformations of a trained MERA
flow evolve with depth, on a controlled standard-Gaussian input.

If the L=32 T_c flow has internalised the scale-invariant physics
at criticality, its deep scale-blocks should converge to a common
functional form (adjacent-scale MSE drops and plateaus). Off-T_c
flows are not scale-invariant and should not show this convergence.

> **[V3 note on the methodology premise.]** The above prediction
> is necessary but not sufficient: adjacent-MSE → 0 can happen
> *either* because both blocks implement the same non-trivial
> scale-invariant map (the intended reading) *or* because both
> blocks are near-identity (a triviality). V3 below uses the
> per-block identity residual `E[(f_s(z) − z)^2]` to break this
> degeneracy. None of the probed flows passes the joint test
> "low adjacent-MSE AND large identity residual"; the rev-KL
> flows take the trivial path.

## Methodology

### Step 1: Isolate the functional blocks

The comparison is between the *functions* (neural network blocks)
themselves, not the final generated images.

For L=32 the MERA architecture has

    depth = log2(L) * nrepeat * 2 = 5 * 1 * 2 = 10

RNVP modules arranged into **5 physical coarse-graining scales**
(one offset-0 + one offset-1 mask per scale). We group every two
adjacent RNVP modules into a single *scale-block* `f_s`, giving us
the 5 transformations to compare:

    f_1, f_2, f_3, f_4, f_5

Each block contains its internal `s(·)` and `t(·)` networks
(`nlayers` coupling layers × `nmlp` MLP hidden layers per network).

The probe applies each block in the **inverse direction** (latent
→ physical), matching how MERA generates samples.

### Step 2: Standardised probing batch

Different layers operate on latents whose marginal scale may have
drifted, so a raw-data input is not a fair comparison. We use a
controlled probe:

- Draw `Z ~ N(0, I)` of shape `(N, 1, 2, 2)`, `N = 10000`
- Pass `Z` through each scale-block independently (bypassing the
  spatial dispatch / collect of MERA, since the block is a pure
  function on 2×2 patches)

### Step 3: Functional-distance MSE

Define

    O_s = f_s(Z)               # raw block output
    O_s_norm = (O_s - mean(O_s)) / std(O_s)
                               # per-element z-score across the batch

The z-scoring strips trivial per-block scale factors (block 3
multiplying by 2 while block 4 multiplies by 0.5 wouldn't reveal
real functional difference; we care about *shape*, not scale).

Compute adjacent-scale MSE on z-scored outputs:

    MSE(O_s, O_{s+1}) = mean[ (O_s_norm - O_{s+1}_norm)^2 ]
                     for s = 1..4

### Step 4: Expected signatures

| Region | Expectation | Why |
|---|---|---|
| Early scales (f_1 → f_2) | high MSE | first scale-block converts the prior into the leading-order field structure; bulk of "easy" non-Gaussianisation happens here, so the function looks fundamentally different from the rest |
| Deep scales at T_c (f_3 → f_4, f_4 → f_5) | **low MSE, plateaued** | RG fixed-point signature — same function repeated at all scales |
| Off-T_c (T=2.15 or T=2.4) | **not plateaued** | system has a finite correlation length, no scale invariance, blocks should not converge to a common form |

> **[V4/V5 note on probe-input scope.]** Steps 1–3 define each
> `f_s` as a function `R⁴ → R⁴` and evaluate it on a `(N, 1, 2, 2)`
> standard-Gaussian patch. This is a **function-property** probe,
> not a production-input probe. In the production forward pass,
> `f_s` sees a patch carved out of the *slow-mode field* at scale
> `s` by the dispatch pattern; per V5, that field has std
> ≈ 3.5 → 6–29 (fwd-KL) or 3.5 → 1 (rev-KL), nothing like `N(0, I)`
> until you hit the deepest latent end of the inverse direction.
> The original probe's "low deep MSE = rev-KL learned the fixed
> point" reading is therefore an `N(0, I)`-on-patch statement; V4
> and V5 (kept-coarse field probes) are what tests whether the
> production slow mode is actually scale-invariant. See the
> Robustness section's "Probe-input scope" subsection for the
> full version of this caveat.

## Implementation

- Script: `analyzers/rg_fixed_point.py`
- Job: `shell/rg_fixed_point.sh` (batch CPU, ~30 min budget; actually
  finishes in seconds since the probe is just 10000 forward passes
  per scale on 2×2 patches)
- Flows probed (L=32 `hs_dataDriven`, default arch, 20000-ep
  forward-KL training):
  - `data/32Ising_T2.15_hs_dataDriven` — low T, ordered phase
  - `data/32Ising_T2.269185314213022_hs_dataDriven` — T_c
  - `data/32Ising_T2.4_hs_dataDriven` — high T, disordered phase
- Second T_c flow added as a robustness check:
  - `data/32Ising_T2.269_hs_bignet` — independent T_c training,
    same architecture (`nlayers=16, nhidden=128, nmlp=3`), shorter
    schedule (10000 ep, latest saving at ep 9500). If the deep-MSE
    T_c signature is real and not a single-seed artefact, this run
    should reproduce it.
- Cross-method comparison at T_c — all five methods from the
  `concise_report_L32_T2.269.md` comparison table, plus the STL
  path-gradient extension, probed with the same bignet architecture:
  - `sym_bignet` (rev-KL)
  - `pathgrad_bignet_long_ext` (STL — reverse-KL via path gradient)
  - `hs_bignet` (forward-KL reference, reused from baseline panel)
  - `jsLoss_bignet_long_lam0.5` (mixed JS = 0.5·rev + 0.5·fwd)
  - `phase2_finetune` (rev-KL warmup → fwd-KL second stage)
  - `hsBignet_bridge_w5.0t0.5` (bridge-reweighted forward-KL)

The script extracts `Symmetrized.flow.layerList` (10 RNVP modules),
groups into 5 scale-blocks of 2 modules each, and applies each
group sequentially via repeated `inverse()` calls. The MERA's
spatial dispatch/collect is intentionally bypassed — we want the
raw per-patch transformation, not the spatially-weaved composition.

## Numerical results

Probed all three L=32 `hs_dataDriven` checkpoints with N=10000
N(0, 1) 2×2 patches each. All flows: same architecture
(`nlayers=16, nhidden=128, nmlp=3, nrepeat=1`, no weightTying, RNVP
coupling, `-symmetry`). Checkpoint epochs taken at the latest saved
state in each folder (16500 / 16000 / 17000).

Adjacent-scale MSE on z-scored outputs:

| pair | T = 2.15 (ordered) | T = 2.269 (T_c, hs_dataDriven, ep 16000) | T = 2.269 (T_c, hs_bignet, ep 9500) | T = 2.40 (disorder) |
| :--- | ---: | ---: | ---: | ---: |
| f_1 → f_2 | 0.96 | 0.88 | **2.73** | **2.48** |
| f_2 → f_3 | 1.74 | 0.35 | 1.86 | 1.96 |
| f_3 → f_4 | 0.52 | **1.92** | 0.59 | 0.34 |
| f_4 → f_5 | 0.79 | **1.92** | **1.98** | **0.15** |

Indexing convention: `f_1` is the LAST applied scale-block in the
generative direction (z → x), i.e. the finest physical scale (after
all the coarse-scale work is done). `f_5` is the FIRST applied
scale-block (coarsest scale, closest to the latent prior). MERA's
`inverse()` iterates `for no in reversed(range(...))`, so layerList
indices 0–1 (our `f_1`) are the bottom of the generative stack and
indices 8–9 (`f_5`) are the top.

## Interpretation

The signature does NOT match the simple "T_c MSE plateaus at deep
scales" prediction from the methodology section. The actual pattern
is the **opposite** for T_c vs T=2.40:

- **T = 2.40 (disordered)**: MSE drops sharply at deep layers
  (0.34 → 0.15). The deepest scale-blocks act *almost identically*
  on the probe — the signature of convergence to a **trivial
  Gaussian fixed point**. Off-T_c the correlation length is short
  (ξ ≈ 10 lattice units, < L/2), so at the coarsest MERA scales the
  field is effectively decorrelated and the flow has nothing
  non-trivial left to do.

- **T = 2.269 (T_c)**: MSE stays *high* at deep layers (1.92, 1.92).
  The deepest scale-blocks remain functionally distinct. ξ → ∞ at
  T_c means there is no scale beyond which the field looks
  uncorrelated — every scale carries real critical structure, and
  the flow must do non-trivial work at each scale to handle the
  long-range correlations. The second T_c flow (`hs_bignet`, ep
  9500) reproduces the high deepest-pair value to 3% (1.98 vs 1.92),
  so this feature is a real T_c signature rather than a
  single-trajectory artefact — see the **Robustness check** section
  below. *(V3 update: this fwd-KL reading is the one that survives —
  V3 residuals at f_3, f_4 are 2.6–15.4, confirming the deep blocks
  do real non-identity work in fwd-KL.)*

- **T = 2.15 (ordered)**: non-monotone, MSE oscillates 0.96 → 1.74
  → 0.52 → 0.79. No clean trivial-fixed-point plateau either. The
  ordered phase has long-range order (correlation length cut by
  finite L), so deep scales still see structure but of a simpler
  kind than at T_c — hence intermediate MSE values, neither the
  large-T plateau nor the T_c divergence.

This is the **inverse-RG signature** rather than the "RG fixed
point with identical blocks" picture our hypothesis
predicted. Both readings reach the same physical conclusion
(criticality is qualitatively different from off-T_c), but through
opposite numerical signatures:

| | predicted by methodology | observed |
|---|---|---|
| T_c | low / plateaued MSE | **high** MSE at deep layers |
| Off-T_c (T=2.40) | non-trivial blocks at every depth | **low** MSE at deep layers |

### Robustness check — second T_c flow (`hs_bignet`)

We re-ran the probe on a second, independently-trained T_c flow
(`data/32Ising_T2.269_hs_bignet`, latest saving at ep 9500) with the
same bignet architecture. The motivation: pin down which features of
the four-MSE profile are robust criticality signatures vs which are
artefacts of a single training trajectory.

Per-pair comparison of the two T_c runs:

| pair | hs_dataDriven (ep 16000) | hs_bignet (ep 9500) | Δ | robust? |
| :--- | ---: | ---: | ---: | :---: |
| f_1 → f_2 | 0.88 | 2.73 | +1.85 | no |
| f_2 → f_3 | 0.35 | 1.86 | +1.51 | no |
| f_3 → f_4 | 1.92 | 0.59 | -1.33 | no |
| **f_4 → f_5** | **1.92** | **1.98** | **+0.06** | **yes** |

The deepest-pair MSE — **f_4 → f_5, the coarsest physical scale,
closest to the latent prior** — is reproducible across the two
runs to 3% (1.92 vs 1.98) and stays an order of magnitude above
T = 2.40 (0.15) and a factor of ~2.5 above T = 2.15 (0.79). This
is the robust T_c signature: at the scale that lives closest to
the prior, both T_c flows refuse to converge toward a trivial
identity-like map, while the disordered-phase flow converges
sharply (0.34 → 0.15 across the last two pairs).

The shallower pairs (f_1 → f_4) disagree substantially between
the two T_c runs. Likely reasons:

- The two trainings are at different points along their loss
  trajectory (16000 vs 9500 ep). Shallow scales handle the
  bulk of "easy" non-Gaussianisation, where the loss landscape
  is relatively flat with many near-degenerate solutions —
  different initial seeds and optimiser histories find different
  routes through the easy modes.
- Z-scoring strips amplitude, leaving shape, but shape at shallow
  scales is driven by the residual local structure of the HS
  field, which the two flows can encode differently while
  matching the same target marginal.

The deepest pair sees the long-range structure that is the
*defining* feature of criticality (ξ → ∞), so it has no such
easy-degeneracy escape — both flows are forced to do non-trivial
work at the coarsest scale, and the MSE lands at the same high
value.

Take-away: the **f_4 → f_5 MSE** is the cleanest single-number
T_c witness extracted from this probe. The four-pair *shape* is
not a robust fingerprint, but the coarsest-pair *value* is.

### Cross-method analysis at T_c (concise_report methods)

The deepest-pair MSE robustness check above used two **forward-KL**
T_c flows (`hs_dataDriven`, `hs_bignet`). To test whether the
"high deep-MSE at T_c" pattern is a property of the T_c phase or
a property of the forward-KL objective, we probed all five methods
from the `concise_report_L32_T2.269.md` cross-method comparison
table (plus the STL path-gradient run; same bignet arch,
nlayers=16, nhidden=128, all at T_c):

| Method                                       | Objective                                    | f_1 → f_2 | f_2 → f_3 | f_3 → f_4 |    f_4 → f_5 |
| :------------------------------------------- | :------------------------------------------- | --------: | --------: | --------: | -----------: |
| sym_bignet (rev-KL)                          | reverse-KL `E_q[A + log q]`                  |      1.60 |      0.40 | **0.019** | **3.4 × 10⁻⁵** |
| pathgrad_bignet_long_ext (STL)               | reverse-KL via path gradient                 |      1.68 |      0.49 | **0.018** | **5.8 × 10⁻⁵** |
| hs_bignet (fwd-KL, forward-KL reference)     | forward-KL `−E_data[log q]`                  |      2.73 |      1.86 |      0.59 |     **1.98** |
| jsLoss_bignet_long (mixed JS, λ=0.5)         | Jensen-Shannon = 0.5·rev-KL + 0.5·fwd-KL     |      1.66 |      0.21 |      0.59 |         1.19 |
| phase2_finetune (rev-KL warmup → fwd-KL)     | 2-stage objective switch                     |      2.41 |      1.20 |      0.35 |         0.97 |
| bridge_w5.0t0.5 (bridge-reweighted fwd-KL)   | weighted forward-KL                          |      0.86 |      1.05 |      2.18 |         2.21 |

Sorted by objective family, the deepest-pair MSE is **stunningly
clean**:

| Family             | f_4 → f_5 range | Behaviour |
| :----------------- | :-------------- | :-------- |
| **Reverse-KL** (sym, STL)              | **~ 10⁻⁵** | **adjacent deep scales literally identical** — fixed-point signature |
| **Mixed** (jsLoss, phase2_finetune)    | ~ 1.0–1.2  | intermediate, between fwd and rev |
| **Forward-KL** (hs_bignet, bridge)     | ~ 1.9–2.2  | high, inverted signature (this report's original surprise) |

The reverse-KL flows — `sym_bignet` and the STL path-gradient
variant — DO satisfy our hypothesis. Their two deepest
scale-blocks act on the standard-Gaussian probe in a way that
agrees to 5–6 significant figures. **MSE(f_4, f_5) ≈ 3 × 10⁻⁵ and
6 × 10⁻⁵** for sym and STL respectively, and MSE(f_3, f_4) ≈ 0.02
— i.e. the three coarsest scale-blocks converge to a common
function, exactly the "RG fixed point" the methodology predicted.

> **[V3-corrected]** The "converge to a common function" reading
> is wrong. V3 below shows the rev-KL `f_4` and `f_5` are each
> near-identity (residuals 0.08–0.26 for sym, 0.018–0.16 for STL).
> Two near-identity functions trivially agree on any input. The
> finding is "the rev-KL optimiser left the coarsest blocks as
> approximate identities", not "the flow learned a scale-invariant
> fixed-point map".

The mixed-objective methods (`jsLoss`, `phase2_finetune`) land
**between** the two extremes — consistent with their loss being a
literal interpolation of forward and reverse KL.

The forward-KL family (`hs_dataDriven`, `hs_bignet`, `bridge`) is
the one that defies the prediction; the bridge run, which
upweights MLE on bridge-domain samples, has the highest
deepest-pair MSE in the whole sweep (2.21).

### Reinterpretation: the probe distinguishes training objectives, not phases

> **[V3-corrected]** This subsection's "objective fingerprint"
> reading remains correct **descriptively** — rev-KL and fwd-KL
> flows produce qualitatively different per-block behaviour at
> T_c. But the mechanistic explanation below ("the coarsest
> rev-KL blocks have nothing to do and collapse onto a common
> identity-like map") is what V3 *directly verifies*: those
> blocks ARE near-identity. The previous-section claim that this
> constitutes "the RG fixed point" is what V3 overturns. Read the
> mechanistic-reading bullets as a description of *what* the
> optimiser does, not as evidence for a learned RG fixed point.

Combining the original four-flow baseline panel with the methods
panel, the cleanest summary of the data is:

- **The "deep MSE → 0" signature is a property of reverse-KL
  training**, not of being at T_c per se. Both rev-KL flows
  collapse their coarsest scale-blocks to an identical map even
  though they are at T_c (where the field is most correlated).

- **The "deep MSE → ~2" signature is a property of forward-KL
  training**, not (only) of T_c either. The hs_bignet and bridge
  runs both sit at this high-deep-MSE plateau.

- The off-T_c forward-KL flows (T = 2.15, T = 2.40) **do not** stay
  at MSE ≈ 2 at the deepest pair — T = 2.40 drops to 0.15. So
  forward-KL training plus T_c is required to lift the deep-MSE
  high. Forward-KL plus off-T_c lets the deep scales become
  trivial; reverse-KL at T_c also makes them trivial-identical.

A mechanistic reading:

- **Reverse-KL** trains the flow by sampling from `q` and matching
  energies. The optimiser sees how each scale-block contributes
  to the energy of a sample drawn from the flow itself. The
  coarsest scales — closest to the latent prior — see almost-
  Gaussian inputs (the deep z's haven't been pushed far by the
  later coarse-graining yet, in the q-distribution view). Once
  the flow has fitted the bulk Gaussian-ish structure with the
  shallower blocks, the coarsest blocks "have nothing to do" and
  collapse onto a common identity-like map. The cleanest
  reverse-KL solution at T_c happens to be one where the deepest
  scales are functionally identical.

- **Forward-KL** trains by scoring data samples (`-log q(x_data)`).
  The optimiser must explain *real* HS-field configurations at
  every scale, including the long-wavelength modes that carry the
  T_c critical correlations. The coarsest scales see the
  longest-range field structure and have to do non-trivial work
  per scale; they cannot collapse onto a common form without
  destroying the long-range fit. Hence the high deep-MSE.

- **Mixed objectives** (jsLoss, phase2) blend the two pressures
  and land in the middle.

So the probe is, in retrospect, a **training-objective fingerprint
at T_c**, not a phase indicator. The original four-flow baseline
panel happened to compare two forward-KL T_c flows against a
forward-KL off-T_c flow, so it picked up only the
forward-KL-specific T_c lift. The methods panel reveals the full
two-dimensional structure (objective × temperature).

### Why our hypothesis inverts here

> **[V3 note]** Reading the original report in retrospect: the
> "fwd-KL inverts our hypothesis" framing was based on the
> rev-KL flows producing the predicted MSE ≈ 0. V3 shows that
> "MSE ≈ 0" was a near-identity triviality, so neither rev-KL
> nor fwd-KL is at a learned RG fixed point. The inversion
> language below should be read as "fwd-KL deep blocks do
> non-trivial work, rev-KL deep blocks do near-nothing" — not
> as "the hypothesis holds for one direction but not the other."

Our hypothesis assumes the flow has *explicitly* learned
a scale-invariant fixed-point map and applies it repeatedly. That
would happen with **weight tying** (`-weightTying`, all scales share
parameters). These flows have `weightTying = False`, so each scale
has its own parameters and can pick whatever transformation it
needs at that scale.

Without weight tying, the criticality signature flips: critical
flows *need* different transformations at different scales (because
fluctuations span all scales at T_c), while off-critical flows let
the deep scales become trivial (because the field is decorrelated
beyond ξ).

### Caveats and reproducibility

1. **Z-scoring removes scale information.** Two blocks that differ
   only by a global rescaling produce MSE = 0 on z-scored outputs.
   That's intentional — we want shape, not amplitude — but means
   the probe is blind to magnitude differences. If the deep T_c
   scale-blocks differ mainly in shape (which they do here), the
   probe sees it.

2. **2×2 patch is the natural unit.** The MERA's RNVP modules
   operate on 2×2 patches via the `kernelShape` parameter; probing
   on a different patch size would not match what the modules
   actually see at training time.

3. **N(0, 1) probe vs the actual scale-block input.** Deep scale-blocks
   in production see outputs of shallower scale-blocks, not pure
   noise. The probe is "what would block s do if it received
   standardised noise?" — a controlled stress test, not a faithful
   re-enactment. The interpretation is "do they act the same way on
   the same controlled input?", which is the cleanest functional
   distance measurement.

4. **`-weightTying` test** would be the strongest follow-up: train
   an L=32 flow at T_c with the weight-tied prior and re-probe.
   With weights shared across scales, the adjacent-MSE should
   collapse to ~0 by construction at every pair (since the
   functions are literally identical) — that gives a sanity-check
   floor and tests whether weight tying actually helps at T_c.

5. **L=8 / L=16 controls** — repeat the same probe at smaller L
   to see how the deep-MSE response scales with system size. If
   the T_c divergence at deep scales is a genuine criticality
   signature (and not arch-specific), it should grow with L.

6. **Training-progress sensitivity.** The `hs_bignet` run was probed
   at ep 9500 (vs ep 16000 for `hs_dataDriven`), so its shallow-pair
   MSE values reflect a less-converged state. Repeating at matched
   epoch (e.g. take both at the latest common checkpoint) would
   sharpen the cross-run comparison. The fact that the deepest
   pair already agrees to 3% at this epoch mismatch indicates the
   coarsest scale-block locks in early during training.

## Robustness checks (V1 / V2 / V3) — and the major reinterpretation they force

After the cross-method analysis, we ran three robustness checks
(`analyzers/rg_fixed_point_robustness.py`, output:
`rg_fixed_point_robustness.csv` + three PNGs) on the most
informative subset of flows. Two checks confirm the original
qualitative pattern; the third one **overturns the principal
"reverse-KL satisfies our hypothesis" reading** and shows it was a
numerical triviality, not a learned scale-invariance.

### Probe-input scope (what V1 / V2 / V3 actually test)

V1, V2 and V3 all share the original probe's geometry: the input
is `z ~ N(0, I)` shaped `(N, 1, 2, 2)` — a single 2×2 patch per
batch element — and each scale-block's RNVP is applied to it
directly via `layer.inverse(z)`. The dispatch / collect plumbing
is bypassed entirely. So these probes measure each scale-block as
**a function `R⁴ → R⁴` on a standard-Gaussian patch**, not as it
acts inside the production forward / inverse pass on the full
`L × L` field.

In production:

- A scale-block in the **forward** direction receives a 2×2 patch
  carved out of the slow-mode field at that scale by the dispatch
  pattern. Per V5, the slow-mode field has std ≈ 3.5 (input) →
  growing to std ≈ 6–29 at intermediate scales for fwd-KL flows,
  or → shrinking to std ≈ 1 for rev-KL flows. Not `N(0, I)` at
  any intermediate scale.
- A scale-block in the **inverse** direction receives the latent
  side: at the deepest scale, that is genuinely close to `N(0, I)`
  for a well-trained flow; at shallower scales, it is the
  intermediate distribution that the next-deeper block produced.

The implication for the table values below:

- "Near-identity on `N(0, I)`" (the V3 finding for rev-KL `f_4`,
  `f_5`) is a **function-property** claim. It is production-relevant
  for `f_5` in the *inverse* direction — `f_5` receives the latent
  `N(0, I)` at sampling time, so "approximately identity on the
  latent" matches what it actually does there.
- It is **not** production-relevant for any block in the *forward*
  direction (slow-mode std ≠ 1), and is only weakly so for
  shallower blocks `f_1..f_4` in the inverse direction (their
  inputs are no longer `N(0, I)` after `f_5` runs).
- "Adjacent-MSE between `f_s` and `f_{s+1}` on the same z" (V1 / V2
  finding) is again a function-property statement: it tells us how
  the two RNVPs compare on a shared 2×2 patch from `N(0, I)`, not
  how the production forward / inverse field behaves under their
  composition.

V4 (data-forward direction on real HS samples, restricted to the
kept-coarse sub-lattice) gives the complementary production-input
view. Read V1–V3 as RNVP-function tests, and V4 / V5 as
production-field tests.

### V1 — global vs per-position z-score

The original probe z-scores each of the 4 cells independently
across the batch. A two-block functional difference of the form
"same shape, different per-position scale" would be invisible.
We re-ran with a **single scalar mean and std** over the full
output tensor. Per-pair MSE values agree to better than 0.1 between
the two normalisations at every pair, every flow:

| pair | flow | global | per-position |
| :--- | :--- | ---: | ---: |
| f_4 → f_5 | T_c hs_dataDriven | 1.95 | 1.92 |
| f_4 → f_5 | T_c hs_bignet     | 1.98 | 1.98 |
| f_4 → f_5 | T_c sym_bignet    | 0.018 | 3 × 10⁻⁵ |
| f_4 → f_5 | T_c STL pathgrad  | 0.019 | 6 × 10⁻⁵ |
| f_4 → f_5 | T = 2.40          | 0.18 | 0.15 |

Verdict: **the per-position normalisation is a non-issue.** The
forward-KL "high deep MSE" and the reverse-KL "near-zero deep MSE"
signatures are both stable under the global-z-score variant.
Plot: `rg_fixed_point_robustness_v1_global.png`.

### V2 — chain-input (production composition)

For each block f_s, feed it the input it would see in production
`h_s = f_{s+1}(...(f_5(z)))` instead of fresh N(0, I). Compare
`f_s(h_s)` vs `f_{s+1}(h_{s+1})` after z-scoring.

| pair | flow | V1 (same z) | V2 (chain h_s) |
| :--- | :--- | ---: | ---: |
| f_4 → f_5 | T_c hs_dataDriven | 1.92 | 1.94 |
| f_4 → f_5 | T_c hs_bignet     | 1.98 | 1.94 |
| f_4 → f_5 | T_c sym_bignet    | 3 × 10⁻⁵ | 0.0000 |
| f_4 → f_5 | T_c STL pathgrad  | 6 × 10⁻⁵ | 0.0001 |
| f_4 → f_5 | T = 2.40          | 0.15 | 0.14 |

Shallow pairs change quite a bit (e.g. T = 2.40 f_1 → f_2 drops
from 2.48 → 0.58 under V2), but the **deep-pair signature is
robust** — chain composition does not change the rev-KL "near
zero" vs fwd-KL "near 2" reading. Plot:
`rg_fixed_point_robustness_v2_chain.png`.

### V2b — chain-input with MERA slot geometry (3 fresh + 1 kept-coarse)

V2 above feeds the **entire 4-element output** of block f_{s+1} as
the 2×2 patch input of block f_s. That does not match how MERA
actually composes scale-blocks. The dispatch pattern at scale s
reads each 2×2 patch at index offsets `{(0, 0), (0, 2^s),
(2^s, 0), (2^s, 2^s)}` relative to a stride-`2^(s+1)` corner. Only
the (0, 0) offset is on the next-deeper scale's stride-`2^(s+1)`
sub-lattice; the other 3 offsets are stride-`2^s` positions that
no deeper block ever touches — they carry fresh N(0, I) latents
(in the inverse direction) or are frozen latents (in the forward
direction). V2's "feed the whole 4-tuple" composition systematically
re-uses output positions that production never re-reads.

**V2b** repairs the slot geometry. For each batch element:

- Patch position `(0, 0)` ← `o_{s+1}[..., 0, 0]`  *(kept-coarse from
  previous block)*
- Patch positions `(0, 1)`, `(1, 0)`, `(1, 1)` ← fresh `N(0, I)`
  draws

Then `f_s` is applied to this geometrically-correct `h_s`. The
comparison metric (MSE between z-scored outputs of `f_s(h_s)` and
`f_{s+1}(h_{s+1})`) is identical to V2.

**Adjacent-pair MSE on z-scored outputs, V2b:**

| pair      | T = 2.15 | T_c hs_dataDriven | T_c hs_bignet | **T_c sym_bignet** | **T_c STL pathgrad** | T = 2.40 |
| :-------- | -------: | ----------------: | ------------: | -----------------: | -------------------: | -------: |
| f_1 → f_2 |    0.984 |             1.361 |         2.129 |              1.950 |                1.891 |    2.210 |
| f_2 → f_3 |    1.723 |             1.307 |         1.537 |              1.533 |                1.563 |    1.452 |
| f_3 → f_4 |    1.510 |             1.936 |         1.539 |              1.510 |                1.511 |    1.426 |
| f_4 → f_5 |    1.645 |             2.611 |         1.774 |          **1.489** |            **1.489** |    1.508 |

Compare directly to V2 at f_4 → f_5:

| flow                        | V2 (full 4-tuple chain) | V2b (one-slot chain) |
| :-------------------------- | ----------------------: | -------------------: |
| T_c sym_bignet (rev-KL)     |                  0.0000 |            **1.489** |
| T_c STL pathgrad (rev-KL)   |                  0.0001 |            **1.489** |
| T_c hs_dataDriven (fwd-KL)  |                  1.9396 |                2.611 |
| T_c hs_bignet     (fwd-KL)  |                  1.9436 |                1.774 |

**The rev-KL deep-pair "plateau" disappears under the correct slot
geometry.** Under V2b the rev-KL `MSE(f_4, f_5) = 1.489` is in the
same band as the off-T_c controls (1.508 / 1.645) and as the
fwd-KL T_c runs (1.774 / 2.611). The V2 reading of "rev-KL
adjacent-MSE → 0 at deep scales" is an artefact of re-injecting the
full output 4-tuple instead of only the kept-coarse slot — once 3
of the 4 patch positions carry fresh N(0, I), the perfectly-flat
"plateau" of rev-KL is gone.

This is independent confirmation of V3's reinterpretation: the
rev-KL deep-pair convergence in V1 and V2 was a consequence of
**(a)** the deep blocks being near-identity on N(0, I) (V3) and
**(b)** the V2 chain re-feeding all 4 output positions of a
near-identity block into another near-identity block (V2b removes
this confound). The original "rev-KL satisfies our hypothesis"
reading therefore relied on two stacked geometric mismatches with
production. Plot: `rg_fixed_point_robustness_v2b_chain_oneslot.png`.

### V3 — per-block identity residual (the decisive check)

For each block f_s, compute the residual
`r_s = E[(f_s(z) − z)^2]` on `z ~ N(0, I)`. If `r_s ≈ 0`, the
block is approximately the identity map on standardised noise — in
which case "two blocks have the same output on the same input" is
a **triviality** (any two identity maps agree), not a learned
scale-invariance.

Relative residuals `r_s / E[z²]` per scale-block:

| flow                       |   f_1   |  f_2   |  f_3   |  f_4   |    f_5    |
| :------------------------- | ------: | -----: | -----: | -----: | --------: |
| T = 2.15                   |    1.99 |   3.65 |   4.65 |   1.15 | **0.025** |
| T_c hs_dataDriven (fwd-KL) |    1.34 |   5.19 |   2.61 |**15.42** |  5.84   |
| T_c hs_bignet (fwd-KL)     |    1.78 |   1.81 |   4.86 |   2.76 |    0.30   |
| T_c sym_bignet (rev-KL)    |   13.42 |   2.66 |   0.91 | **0.26** | **0.08** |
| T_c STL pathgrad           |   13.26 |   4.13 |   0.89 | **0.16** | **0.018** |
| T = 2.40                   |    0.94 |   5.45 |   4.82 |   1.60 |    0.13   |

This table is the decisive piece of evidence. Reading row-by-row:

- **T = 2.15** and **T = 2.40** (off-T_c, forward-KL): f_5 has
  residual ≈ 0.03 / 0.13 — **near identity at the coarsest scale**.
  Consistent with "off-T_c, ξ < L/2, the deepest scale sees
  decorrelated field — no work needed."

- **T_c hs_dataDriven (fwd-KL)**: f_4 residual = **15.4**, the
  largest in the whole sweep. f_5 residual = 5.8. **Both deep
  blocks do substantial, non-identity work.** The MSE = 1.92
  between f_4 and f_5 in the original probe is a *real* functional
  difference between two non-trivial maps.

- **T_c hs_bignet (fwd-KL)**: f_5 has residual 0.30 (near identity),
  f_4 has residual 2.76 (not). They are doing different jobs —
  the high f_4→f_5 MSE = 1.98 captures real functional
  asymmetry.

- **T_c sym_bignet (rev-KL)** and **T_c STL pathgrad**: both have
  f_5 AND f_4 near identity (residuals 0.08, 0.26 for sym; 0.018,
  0.16 for STL). **The "MSE ≈ 0 between f_4 and f_5" finding from
  the original probe is therefore a triviality: two near-identity
  functions on the same Gaussian probe trivially produce matching
  outputs.** This is not a learned scale-invariant fixed-point
  map; it is "the reverse-KL optimiser learned to leave the
  coarsest scale-blocks as approximate identities."

Plot: `rg_fixed_point_robustness_v3_identity.png`.

### V4 — data-forward direction, kept-coarse sub-lattice only

V1–V3 share the inverse-direction, isolated-block setup of the
original probe: each scale-block is fed `z ~ N(0, I)` and
evaluated in `inverse` mode on a (N, 1, 2, 2) patch. That is the
wrong direction for an RG fingerprint — at inference the flow
processes a *sample* in the forward (analysis) direction, scale-
block by scale-block, while the index pattern coarse-grains. V4
reuses the same set of six flows (two off-T_c controls, T_c
hs_dataDriven, T_c hs_bignet, T_c sym_bignet, T_c STL
pathgrad_bignet_long_ext) and runs the diagnostic on the real
path:

```
x ~ p_HS  →  f_1.forward  → y_1  →  f_2.forward  → y_2  → … → y_S
                (finest)                                    (latent)
```

where `f_s` is the s-th scale-block (`layerList[2(s−1)..2s−1]`)
and `y_s` is the field after `s` coarse-graining scale-blocks have
been applied.

**Kept-coarse restriction (V4 correction).** After s scale-blocks
in the MERA forward direction, **only the positions on the
stride-2^s sub-lattice carry the slow mode** that flows into the
next scale-block. The other `(1 − 1/4^s)` positions are
"frozen latents": the optimiser drives them toward N(0, 1) and
they are never re-read by any deeper block. Averaging moments,
G(r), KS, W1 etc. over the full L × L `y_s` dilutes the slow-mode
signal by `4^s` and is dominated by the frozen-latent floor for
deep s (e.g. for L=32 at s=5, only 1 of 1024 positions is slow
mode). V4 therefore restricts to

```
y_s_coarse = y_s[..., ::2^s, ::2^s]      # shape (B, 1, L/2^s, L/2^s)
```

and computes every statistic on this slow-mode-only field.
(A prior version of V4 averaged over all L² positions and is now
retracted — the rev-KL "std ≈ 1.09, kurt ≈ 0" finding it reported
at the deepest scale was the latent-floor of the 1023 frozen
positions, not the kept-coarse signal.)

**What V4 measures, on the kept-coarse sub-lattice.** At a true
RG fixed point, the slow mode should be distributionally
self-similar between adjacent scales (when each is measured in
its own coarse units). For each adjacent pair we compute:

- KS / W1 on the standardised marginal `(y_coarse − μ)/σ`
- RMS deviation of the normalised axial two-point function
  `G_s(r) / G_s(0)`, defined on the L_s × L_s coarse lattice
  for r = 1..min(L_s, L_{s+1})/2

Identity-triviality on the slow mode is not a confound: a
near-identity block on a non-trivial slow-mode distribution
leaves a non-trivial distribution, so adjacent-scale agreement
still requires the flow to actually reproduce the coarse
statistics.

**Slow-mode std cascade (the key V4 measurement).** Std of `y_s`
on the kept-coarse sub-lattice (one of the most informative
single numbers per scale):

| flow                     | y_0  | y_1   | y_2   | y_3   | y_4   | y_5   |
| :----------------------- | ---: | ----: | ----: | ----: | ----: | ----: |
| T = 2.15  (low T)        | 3.81 | 13.86 | 13.08 |  7.12 |  5.95 |  4.90 |
| T_c hs_dataDriven (fwd)  | 3.51 |  8.35 | 14.33 |**28.94**| 24.19 | 11.14 |
| T_c hs_bignet     (fwd)  | 3.51 |  8.38 | 15.92 | 10.03 |  8.71 |  5.50 |
| T_c sym_bignet    (rev)  | 3.51 |  2.12 |  1.91 |  1.49 |  1.17 |**0.94**|
| T_c STL pathgrad  (rev)  | 3.51 |  2.46 |  1.93 |  1.34 |  0.97 |**0.96**|
| T = 2.40  (high T)       | 3.19 |  9.23 |  7.93 |  6.76 |  4.88 |  3.50 |

Two contrasting pathologies emerge:

- **Reverse-KL slow modes shrink monotonically toward N(0, 1).**
  Std cascades 3.51 → 2.12 → 1.91 → 1.49 → 1.17 → 0.94 (sym), and
  kurtosis goes -1.33 → +0.27 → +0.12 → +0.04 → +0.07 → -0.11.
  The block-RG ground truth at the matching coarse resolution
  (V5) gives std 3.51 → 3.18 → 2.90 → 2.69 → 2.54 → 2.46 with
  kurtosis remaining bimodal at -1.33 to -1.84. **The rev-KL slow
  mode is being erased, not coarse-grained.**
- **Forward-KL slow modes balloon then partially relax.** T_c
  hs_dataDriven cascades to std 28.94 at y_3 (8× larger than the
  input) before falling back to 11.14 at y_5. Block-RG ground
  truth predicts std to *decrease gently* (3.51 → 2.46) since
  averaging reduces variance. Forward-KL is moving in the wrong
  direction.

**Adjacent KS on standardised slow-mode marginals.** This isolates
*shape* (not amplitude) self-similarity:

| flow                            | 0→1   | 1→2   | 2→3   | 3→4   | 4→5   |
| :------------------------------ | ----: | ----: | ----: | ----: | ----: |
| T = 2.15                        | 0.114 | 0.129 | 0.039 | 0.025 | 0.029 |
| T_c hs_dataDriven (fwd-KL)      | 0.095 | 0.066 | 0.026 | 0.118 | 0.053 |
| T_c hs_bignet     (fwd-KL)      | 0.059 | 0.052 | 0.120 | 0.085 | 0.056 |
| T_c sym_bignet    (rev-KL)      | 0.110 |**0.006**|**0.004**| 0.011 | 0.014 |
| T_c STL pathgrad  (rev-KL/STL)  | 0.110 |**0.007**|**0.004**| 0.013 | 0.015 |
| T = 2.40                        | 0.084 | 0.036 | 0.053 | 0.026 | 0.019 |

Rev-KL slow modes are KS-self-similar from `y_1` onward to within
0.4 % — i.e. the *shape* (after standardisation) is essentially
the same across scales. This is consistent with the rev-KL flow
sending the slow mode through a single canonical Gaussian shape
at every depth, then progressively shrinking its amplitude.
Fwd-KL slow modes are NOT self-similar in shape — KS stays at
5–12 % at intermediate scales and shows a non-monotone bump at
`y_3`.

**Anti-correlation in the rev-KL slow mode.** G(r)/G(0) on the
kept-coarse lattice at `y_1` and `y_2` for rev-KL flows shows
*negative* nearest-neighbour correlation (−0.107 for sym, −0.097
for STL at the first coarse step) — a Néel-style anti-correlated
slow mode, **not** the ferromagnetic short-range positive
correlation block-RG produces. Fwd-KL G(1)/G(0) stays in the
0.62–0.79 range (physical). This is a new diagnostic: even when
the rev-KL slow mode is small (std ≈ 1), its spatial structure
is wrong.

V4-corrected therefore adds three concrete facts to the V3
picture:

- Rev-KL flows erase the slow mode (std → 1, kurt → 0) instead of
  coarse-graining it (block-RG predicts std → 2.5, kurt → −1.84).
  The slow mode is being collapsed to the latent prior at every
  coarse-grained position, not just at the deepest one.
- Fwd-KL flows do the opposite pathology: slow-mode std blows up
  by 8× at intermediate scales, recovers partially at the deepest
  scale, and never reaches a self-similar shape across scales.
- Rev-KL slow modes have *negative* nearest-neighbour correlations
  on the coarse lattice — a structural pathology that the
  per-block identity-residual probe (V3) cannot see.

Plots: `rg_v4_dataforward_ks.png`, `rg_v4_dataforward_w1.png`,
`rg_v4_dataforward_rmsG.png`, `rg_v4_dataforward_Goverlay.png`.
CSV: `analyzers/rg_v4_dataforward.csv` (overwritten by the
kept-coarse rerun, job 39551162).

### V5 — Wilson–Kadanoff block-RG ground truth

V4 measures the MERA forward cascade on the kept-coarse
sub-lattice but has no external reference for *what the slow-mode
cascade should look like*. V5 supplies that reference by running
the classical Wilson–Kadanoff block-average cascade on the **same**
HS-field input and comparing distribution-by-distribution against
the MERA-subsampled `y_s_coarse` at every scale.

**Block-RG cascade.** For each batch `x ~ p_HS` we compute

```
x_0 = x                                          (B, 1, L,   L)
x_s = AvgPool2d(2)(x_{s−1})                       (B, 1, L/2^s, L/2^s)
```

i.e. each coarse pixel at scale `s` is the unweighted mean of a
`2^s × 2^s` block of the original HS field. This is the textbook
block-spin transformation (in the continuous-field limit appropriate
to the HS dual-variable representation), executed without any
learning. At T_c the block-averaged field flows under the 2D Ising
universality class toward a non-Gaussian fixed-point distribution
characterised by power-law two-point correlations and a bimodal
marginal — *not* toward a Gaussian.

**Cross-comparison.** For every scale `s = 0..5` we evaluate three
distances between the standardised MERA-subsampled slow-mode
`(y_s_coarse − μ)/σ` and the standardised block-RG `(x_s − μ)/σ`,
**on the same input batch** so the two are paired:

- `v5_ks` — KS distance on the standardised marginal
- `v5_w1` — Wasserstein-1 on the standardised marginal
- `v5_rms_g` — RMS deviation of `G_s(r)/G_s(0)` between the two

At `s = 0` (the input HS field), MERA-subsampled and block-RG are
literally the same tensor, so all three distances are ~0; the
informative columns are `s = 1..5`.

**Block-RG ground-truth moments (the numbers V4 referenced).** Std
and kurtosis of `x_s`, identical across all six probed flows since
the input is the same:

| scale s | L_s | std    | kurt   |
| :-----: | :-: | :----: | :----: |
|    0    | 32  |  3.51  | −1.33  |
|    1    | 16  |  3.18  | −1.53  |
|    2    |  8  |  2.90  | −1.65  |
|    3    |  4  |  2.69  | −1.72  |
|    4    |  2  |  2.54  | −1.78  |
|    5    |  1  |  2.46  | −1.84  |

Variance drops gently (averaging-induced) while the marginal stays
strongly bimodal (kurt drifts more negative as the coarse box
contains fewer, more-aligned spins). This is the target a
real RG-fixed-point flow's slow mode should reproduce.

**KS(MERA-subsampled, block-RG) on the standardised slow-mode
marginal.** Lower is better; this is the quantitative read on
"does the trained flow's slow-mode field match Wilson at scale s":

| flow                            | s=1   | s=2   | s=3   | s=4   | s=5   |
| :------------------------------ | ----: | ----: | ----: | ----: | ----: |
| T = 2.15                        | 0.161 | 0.099 | 0.133 | 0.176 | 0.229 |
| T_c hs_dataDriven (fwd-KL)      | 0.101 | 0.085 | 0.097 | 0.097 | 0.140 |
| T_c hs_bignet     (fwd-KL)      | 0.080 | 0.085 | 0.121 | 0.081 | 0.128 |
| T_c sym_bignet    (rev-KL)      | 0.132 | 0.147 | 0.160 | 0.181 | 0.201 |
| T_c STL pathgrad  (rev-KL/STL)  | 0.133 | 0.147 | 0.159 | 0.177 | 0.202 |
| T = 2.40                        | 0.083 | 0.077 | 0.068 | 0.044 | 0.037 |

Two clean signatures:

- **Rev-KL marginals walk steadily away from Wilson with depth.**
  KS grows monotonically 0.13 → 0.20 across `s = 1..5`. Both
  rev-KL flows give nearly identical KS at every scale
  (0.132 vs 0.133 at s=1, 0.201 vs 0.202 at s=5), consistent with
  V3's identity-residual reading that the deep blocks for both
  flows have collapsed to the same near-trivial map on the slow
  mode.
- **Fwd-KL T_c marginals stay within ~0.08–0.14 of Wilson at every
  scale.** The match is not exact (KS ~0.10 means a visible shape
  mismatch — see the `marg_s2`, `marg_s3` overlays), but the
  fwd-KL flows track Wilson far more closely than rev-KL, and the
  off-T_c control at T = 2.40 (also fwd-KL) does even better
  (KS drops to 0.04 at s=5) — disorder is easier to coarse-grain
  than criticality.

**RMS deviation of G(r)/G(0) (spatial-structure mismatch).** The
two-point function on the kept-coarse lattice cuts to the heart of
the "is this a real coarse-graining" question — KS only sees the
marginal, but G(r) tests whether the same neighbours stay
correlated after one block-step:

| flow                       | s=1   | s=2   | s=3   |
| :------------------------- | ----: | ----: | ----: |
| T = 2.15                   | 0.125 | 0.095 | 0.138 |
| T_c hs_dataDriven (fwd-KL) | 0.110 | 0.068 | 0.053 |
| T_c hs_bignet     (fwd-KL) | 0.106 | 0.049 |**0.016**|
| T_c sym_bignet    (rev-KL) |**0.620**|**0.669**|**0.674**|
| T_c STL pathgrad  (rev-KL) |**0.614**|**0.646**|**0.640**|
| T = 2.40                   | 0.100 | 0.065 | 0.070 |

(`s = 4, 5` have L_s < 4 so the cosine grid for G(r) collapses and
the row reports `nan`; reliable G-comparison is `s = 1..3`.)

The rev-KL row is an order of magnitude worse than every other
row, at every scale, and the mismatch *grows* with depth. This is
the quantitative footprint of the anti-correlated nearest-neighbour
pathology that V4's G(r) inset called out — rev-KL has built a
slow-mode field whose marginal shrinks toward `N(0, 1)` **and**
whose spatial structure is the wrong sign at short range. Fwd-KL
hs_bignet, by contrast, gets G(r) on the slow mode right to within
1.6% by scale 3, which is a genuinely encouraging match in the
spatial-structure axis (the marginal axis still fails, per the KS
row above and the V4 std cascade — std 15.92 at y_2 vs block-RG's
2.90).

**Reading V5.** The cross-comparison closes the V4 question
("compared to what?") with a non-learned reference. The verdict:

- **No flow in the sweep produces a Wilson-style coarse-graining
  of the slow mode at every scale.** The fwd-KL flows track G(r)
  well at coarse depth but inflate the marginal variance 4–9×;
  the rev-KL flows keep the marginal small (and slowly
  Gaussianising) but break the spatial structure entirely.
- **The training objective dictates which axis is broken.**
  fwd-KL preserves *correlations* but distorts amplitudes;
  rev-KL preserves *amplitudes* (closer to the latent prior) but
  destroys correlations. Neither is RG.
- **Off-T_c controls succeed differently.** T = 2.40 (disorder)
  reproduces Wilson well in both axes (KS → 0.04, RMS-G → 0.07
  at deep scales), because the target distribution is itself
  weakly correlated and easy to push toward Gaussian. T = 2.15
  (order) has a comparable RMS-G to T = 2.40 but its KS rises
  with depth — the broken-symmetry magnetised mode is *much*
  harder to coarse-grain along the marginal axis.

V5 therefore provides the external benchmark V4 needed: it
confirms with a non-learned reference that neither objective
family does Wilson-style RG at T_c on this architecture, and it
discriminates *how* each family fails (correlation-correct but
amplitude-wrong for fwd-KL; amplitude-near-prior but
correlation-wrong for rev-KL).

Plots: `rg_v5_blockRG_std.png`, `rg_v5_blockRG_kurt.png`,
`rg_v5_blockRG_ks.png`, `rg_v5_blockRG_w1.png`,
`rg_v5_blockRG_rmsG.png`, `rg_v5_blockRG_marg_s2.png`,
`rg_v5_blockRG_marg_s3.png`.
CSV: `analyzers/rg_fixed_point/csv/rg_v5_blockRG_compare.csv`.

### The reinterpretation

The original report concluded that **reverse-KL flows satisfy
our hypothesis** (deep blocks become functionally identical at
T_c). V3 shows this conclusion is wrong on its terms: the deep
blocks are not "the same non-trivial scale-invariant
transformation", they are "near-identity". The MSE-on-z-scored
outputs cannot distinguish those two cases by construction.

What the data does support:

- **Forward-KL flows at T_c do real work at every MERA scale.**
  Their per-block identity residual is large at f_3, f_4, f_5
  (15.4 at f_4 for hs_dataDriven, 4.9 at f_3 for hs_bignet), and
  the adjacent-MSE between f_4 and f_5 captures a real functional
  difference, not a redundancy. This is consistent with the
  physical picture that forward-KL must explain HS-field data
  with structure at all scales.
- **Reverse-KL flows at T_c collapse the coarsest scales toward
  identity.** Optimiser pressure pushes f_4 and f_5 to near-identity
  once the shallower blocks fit the bulk. The "function" the
  coarsest blocks implement is not "the universal RG
  transformation"; it is "do nothing".
- **The forward-KL / reverse-KL split that the cross-method panel
  exposed is therefore a split between "non-trivial deep blocks"
  and "trivial (identity) deep blocks"**, not between two
  different non-trivial RG fixed-point implementations.

The original methodology framing — "if the deep blocks act the
same, this is the RG fixed point" — is necessary but not
sufficient. To distinguish a fixed-point map from a trivial
identity, the V3 identity residual must be checked. Without it,
"low adjacent-MSE" is ambiguous.

### What the original framing would still require to be testable

A bona-fide RG fixed-point reading on this architecture would
require a flow where:

- f_4 and f_5 have **large** identity residual (they do non-trivial
  work), AND
- their adjacent-MSE on the same input is small (their non-trivial
  work is *the same* transformation).

No flow in the sweep satisfies both conditions. The closest is
**T_c hs_bignet** (large f_3, f_4 residuals; moderate f_5; but its
f_4 → f_5 MSE = 1.98, i.e. the non-trivial blocks DISAGREE). So
**none of the current flows is plausibly at an RG fixed point** in
the strict sense the methodology requires.

### Acceptable claims (post-robustness)

- Forward-KL at T_c does substantial work at every MERA scale;
  reverse-KL at T_c does not. This is itself a useful
  characterisation of the two training objectives.
- The cross-method panel still discriminates objectives cleanly
  (rev-KL → near-identity deep blocks; fwd-KL → non-trivial deep
  blocks).
- The MERA scale grouping and the dispatch-bypassing functional
  probe are well-defined and reproducible.
- **(from V4-corrected, kept-coarse)** Reverse-KL flows on the
  slow-mode sub-lattice shrink std monotonically 3.51 → 0.94 with
  kurtosis → 0 and *negative* nearest-neighbour correlations at
  intermediate coarse scales. The slow mode is being erased
  (collapsed to N(0, 1) + anti-correlated noise), not
  coarse-grained. Block-RG ground truth (V5) predicts std → 2.46
  with bimodal kurt ≈ −1.84.
- **(from V4-corrected)** Forward-KL slow modes balloon (T_c
  hs_dataDriven: std 3.51 → 28.94 at y_3) before partial
  relaxation, never reaching shape self-similarity (adjacent
  standardised KS stays ≥ 0.05 at every coarse pair), G(1)/G(0)
  stays positive (physical) but no flow's coarse cascade actually
  matches block-RG. **No flow on this architecture is doing
  Wilson-style RG**, and the original-V4 framing that "rev-KL
  approaches scale invariance trivially at the latent" was the
  full-L² averaging effect, not the slow mode.
- **(from V5, cross-comparison against Wilson–Kadanoff block-RG
  on the same input)** The training objective dictates which axis
  of "real coarse-graining" is broken: fwd-KL preserves G(r) at
  coarse depth (RMS-G ≈ 0.02–0.07 for hs_bignet at s = 2–3) but
  inflates the marginal variance 4–9× above block-RG; rev-KL
  keeps the marginal small but breaks G(r) entirely (RMS-G ≈
  0.62–0.67 at every scale, an order of magnitude worse than any
  fwd-KL row). The two-axis decomposition (KS for marginal,
  RMS-G for spatial structure) makes the failure quantitative,
  not qualitative.

### See also

- `temp_sweep_L32.md` — broader L=32 forward-KL temperature sweep
- `concise_report_L32_T2.269.md` — full L=32 T_c method comparison
- `fss_sweep_report.md` — cross-L KL ∝ L^α criticality witness
  (α = 2.20 at T_c vs ~2.0 off-critical — macroscopic complement to
  this microscopic probe)
- `criticality_analysis.py` / `criticality_flow.py` — data-side and
  flow-side universal-value tests (Binder cumulant crossings, χ FSS
  exponent, ξ_eff/L crossings, P(M) collapse)

## Plots

### Baseline panel (temperature controls)

![RG fixed-point probe — baseline](figures/rg_fixed_point.png)

### Methods panel (concise_report methods at T_c)

![RG probe — training-method comparison](figures/rg_fixed_point_methods.png)

Reverse-KL methods (sym, STL) sit on the bottom, with f_4 → f_5
visually pinned to the x-axis at ~10⁻⁵; forward-KL methods
(hs_bignet, bridge) sit on top at MSE ≈ 2; mixed objectives
(jsLoss, phase2_finetune) lie between. The off-T_c controls
T = 2.15 and T = 2.40 are included to show that the forward-KL
high-deep-MSE plateau is a T_c phenomenon (T = 2.40 drops
sharply).

### V1 — global vs per-position z-score

![V1 robustness — global vs per-position z-score](figures/rg_fixed_point_robustness_v1_global.png)

### V2 — chain-input (production composition)

![V2 robustness — chain-input probe](figures/rg_fixed_point_robustness_v2_chain.png)

### V2b — chain-input with MERA slot geometry (3 fresh + 1 kept-coarse)

![V2b robustness — one-slot chain](figures/rg_fixed_point_robustness_v2b_chain_oneslot.png)

Under the production-correct slot geometry, the rev-KL "plateau"
at deep pairs vanishes — the deepest MSE for both sym_bignet and
STL pathgrad lands at 1.489, indistinguishable from off-T_c
controls.

### V3 — per-block identity residual

![V3 robustness — identity residual per scale-block](figures/rg_fixed_point_robustness_v3_identity.png)

The rightmost cell (f_5) being small for both off-T_c flows AND
both reverse-KL T_c flows is the punchline: f_5 sits near identity
on Gaussian noise in 4 of the 6 probed flows. Only the forward-KL
T_c runs have non-trivial deep blocks at every scale.

### V4 — data-forward direction (HS data, full path)

![V4 — KS between adjacent y_s, y_{s+1}](figures/rg_v4_dataforward_ks.png)

![V4 — Wasserstein-1 between adjacent y_s, y_{s+1}](figures/rg_v4_dataforward_w1.png)

![V4 — G(r)/G(0) shape mismatch between adjacent y_s, y_{s+1}](figures/rg_v4_dataforward_rmsG.png)

![V4 — G_s(r)/G_s(0) per scale, one panel per flow](figures/rg_v4_dataforward_Goverlay.png)

The KS/W1/RMS-G panels all decay toward the right edge for every
flow including both off-T_c controls — the universal "deepest pair
matches" behaviour discussed in §V4. The per-flow G_s(r)/G_s(0)
overlay (last figure) makes the cross-objective difference visual:
rev-KL panels collapse onto a single near-flat curve at deep
scales (consistent with the N(0, 1) latent endpoint), while the
fwd-KL panels keep visible curvature at every scale.

### V5 — Wilson–Kadanoff block-RG ground truth

![V5 — slow-mode std per scale: MERA-subsampled vs block-RG](figures/rg_v5_blockRG_std.png)

![V5 — slow-mode kurtosis per scale: MERA-subsampled vs block-RG](figures/rg_v5_blockRG_kurt.png)

The std panel shows the cleanest single-number summary: the
block-RG curve drops gently from 3.51 to 2.46 (averaging-induced),
the fwd-KL flows balloon above it (peak 28.94 for hs_dataDriven at
s = 3), and the rev-KL flows shrink monotonically below it toward
~1 (slow-mode erasure). Kurtosis shows the block-RG marginal
staying bimodal (−1.33 → −1.84) while rev-KL marginals go through
zero (Gaussianisation).

![V5 — KS(MERA, block-RG) per scale](figures/rg_v5_blockRG_ks.png)

![V5 — W1(MERA, block-RG) per scale](figures/rg_v5_blockRG_w1.png)

![V5 — RMS deviation of G(r)/G(0) per scale](figures/rg_v5_blockRG_rmsG.png)

The RMS-G panel is the punchline: rev-KL traces sit at 0.62–0.67
across every coarse depth, an order of magnitude above every
fwd-KL trace. This is the spatial-structure axis fwd-KL gets right
and rev-KL gets backwards.

![V5 — marginal overlay at s = 2 (MERA-subsampled vs block-RG, per flow)](figures/rg_v5_blockRG_marg_s2.png)

![V5 — marginal overlay at s = 3](figures/rg_v5_blockRG_marg_s3.png)

The marginal overlays make the bimodality story visual: block-RG
keeps two lobes at every depth; rev-KL slow modes have collapsed
to a single near-Gaussian peak; fwd-KL slow modes have wide,
heavy-tailed shapes that bracket the block-RG marginal but at the
wrong scale.
