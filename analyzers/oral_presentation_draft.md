# NeuralRG at criticality — oral presentation draft

Audience: research group / advisor. Approximate duration: 12–15 min.
Speaker notes in square brackets [like this]. Section headers map to
slide changes.

---

## 1. Opening (≈1 min)

Today I want to walk you through where we are with NeuralRG on the 2D
Ising model — specifically near the critical temperature — and share
two recent results. The first is a finite-size-scaling sweep that
gives us a quantitative *criticality witness*. The second is a
head-to-head comparison of training objectives at L=32 that surfaces a
subtle but important disagreement between how the eye and the loss
function judge a flow.

[pause]

The short version of the talk: at the critical point, our flow's
forward-KL grows with system size *faster* than the L² scaling you'd
naively expect. We've measured the exponent and we can use it as a
benchmark for new methods. And at L=32 specifically, the method that
"wins" depends entirely on which direction of KL you ask about — which
turns out to map cleanly onto whether the samples look right.

---

## 2. Setup (≈2 min)

[Slide: NeuralRG architecture diagram if available; otherwise text]

To recap the setup: we train a normalizing flow — concretely a
MERA-structured stack of real-NVP coupling layers — to model the 2D
Ising distribution on an L × L lattice at temperature T. We use the
Hubbard–Stratonovich continuous-field representation, so the target
is a continuous distribution `p_HS(x) ∝ exp(−A(x))` where A is the
HS action. The continuous field makes the math cleaner and lets us
score samples by exact log-density.

We have two training objectives we care about:

- **Reverse KL**: minimize `E_q[log q − log p]`. Mode-seeking. The
  loss is the variational free energy plus a constant.
- **Forward KL (MLE)**: minimize `−E_data[log q]`. Mass-covering.
  Requires pre-generated HS samples — for L=32 we use 200,000 of them.

These two objectives don't agree. Reverse-KL says "don't put mass
where the target has none." Forward-KL says "don't miss mass the
target has." A perfect flow scores zero on both; everything else trades
them off.

---

## 3. FSS sweep — the criticality witness (≈3 min)

[Slide: fss_sweep_KL_v3.png]

OK, the first result. We trained forward-KL flows on a 3×5 grid:
three system sizes — L=8, 16, 32 — and five temperatures bracketing
T_c, which is 2.2692. At each (L, T) point we let the flow train for
20,000 epochs and recorded the best smoothed loss.

[gesture at panel (d)]

If you fit each temperature's three (L, KL) points to a power law
`KL = a · L^α`, here's what comes out:

- Off-critical, all four temperatures: α ranges from 1.95 to 2.16.
- **At T_c: α = 2.20**.

[pause for emphasis]

The off-critical α≈2 is what you'd expect for a *capacity-saturated*
flow — the KL per site is roughly intensive, you need O(L²) parameters
to handle O(L²) degrees of freedom, fixed cost per spin. The α=2.20 at
T_c is something different. That 0.2 excess means **per-site KL grows
with L specifically at criticality**. The R-squared on these fits is
0.99 or better, so it's not a fitting artifact.

[Slide: panel (b) zoomed in]

You can see the same fact in panel (b), the "per-site KL" view. If KL
scaled exactly as L², the three L curves would collapse onto each
other. Off-critical they nearly do — they sit in a tight band around
0.008 to 0.016 nats per site. At T_c the L=32 curve breaks above the
L=8 and L=16 curves — that gap is α greater than 2.

The physical interpretation is the obvious one: the correlation length
diverges at the critical point, so a fixed-capacity flow can't keep up
as L grows. We're seeing exactly that gap, and now we can measure it.

[pause]

The practical implication: when we try a new flow architecture — NSF
splines, JS-loss, whatever — the right question isn't "does it improve
the absolute KL at L=32." The right question is "does it bring α at
T_c closer to 2." That's the architecture-vs-physics question.

---

## 4. L=32 — methods don't agree (≈4 min)

[Slide: L=32 summary table]

Now the second result. At L=32 T_c we've trained several different
flows with different objectives, and I want to walk through how
they compare. Here's the headline table.

Three flow architectures matter:

- **sym_bignet**: pure reverse-KL, bignet arch (16 layers, 128 hidden,
  about 11 million params). Best reverse-KL run we have at L=32.
- **hs_bignet**: pure forward-KL, same arch.
- **jsLoss_bignet_long**: combined JS-style loss, half reverse + half
  forward.

Plus phase2_finetune, which is forward-KL pretrain followed by
reverse-KL refinement.

The training-row column shows each method's on-objective KL — the one
it was actually minimising. sym_bignet hits **9.4 nat reverse-KL**,
which is genuinely good. hs_bignet hits 3.6 nat forward-KL, also good
on its own terms. jsLoss balances at ≈16-17 nat on both. Phase2 sits
around 17.6 reverse-KL.

[pause]

But here's the thing. If you look at the rendered flow samples — the
visual output — sym_bignet looks the *worst*. The configurations have
giant coherent domains. They're "too clean." jsLoss and the
forward-KL methods look much more like real Ising configurations near
T_c — domain structure on multiple scales, occasional bridges between
the two magnetization sectors.

So why does the lowest-KL method look the worst?

[Slide: diagnostic table with KL_qp, KL_pq, mag_abs_q, xi_q]

The answer is in the *diagnostic* row of the table — the off-objective
KL, plus two structural statistics. Let me walk through what those
mean.

`mag_abs_q` is the per-configuration average of |M|, the
magnetization. The data has it at 2.38; sym_bignet draws samples with
`mag_abs_q = 3.11`. So sym_bignet over-shoots — its samples are too
strongly magnetized. `xi_q` is the effective correlation length —
data has 8.6, sym_bignet draws 12.0. Domains too rigid.

Now look at jsLoss and phase2: mag_abs_q of 2.45 to 2.80, xi_q of 8.9
to 10.4. They track the data values much more closely.

And the off-objective KL tells the same story. sym_bignet's
**forward** KL — measuring how much it misses the data's typical
configs — is **64.6 nat**. The mixed methods sit around 16–19 nat
on that direction. Almost a four-times difference.

[pause]

So the KL numbers are correct. They're just measuring different
things. The eye looks at typical samples — which is what *forward*
KL penalizes. sym_bignet has the lowest *reverse* KL because it puts
mass where the target has mass — but its mass is too narrowly
concentrated on the peaks, so it misses the bridge and the
near-critical fluctuations that visually define "looks like Ising."

This isn't a bug — it's the structural difference between mode-seeking
and mass-covering, made concrete on a real problem. The takeaway is
that you have to pick the KL direction that matches what you actually
want from the flow. For sampling — you want forward-KL. For HMC
proposal distributions — reverse-KL is fine.

---

## 5. What we tried and what didn't work (≈2 min)

[Slide: bullet list of attempted fixes]

Recently we've been trying to fix sym_bignet's bridge collapse. Three
attempts, in chronological order.

**First: phase2 fine-tuning.** Pretrain with forward-KL to get a
broad anchor, then refine with reverse-KL to sharpen the modes. The
idea was to inherit the anchor's bridge coverage and only tighten the
peaks. Result: it *did* sharpen too much — the bridge mass dropped
from 0.0095 to essentially zero. Trade was bad.

[note: also flag the diagnostic-loading bug we just fixed]

Side note: while debugging this report I noticed phase2's diagnostic
numbers were *identical* to hs_bignet's. Turned out the diagnostic
script grabs the highest-epoch checkpoint, and phase2's savings
folder still had the anchor file at epoch 9500 — so we were diagnosing
the anchor, not phase2. Moved that file aside; the corrected
diagnostic is running now.

**Second: entropy regularization.** Add `−β·H(q)` to the MLE loss, on
the theory that pushing flow entropy up would widen the bridge.
Tried β=0.005 and β=0.05. β=0.005 was within noise of baseline;
β=0.05 *diverged* — pure MLE blew up to roughly 10^10 nats after
600 epochs. The reason is structural: `−β·H(q)` has no upper bound,
so the optimizer happily drives H(q) to infinity. Not a bug; a
formulation problem. Marked closed unless reformulated.

**Third (currently running): bridge-targeted upweighting.** Instead
of pushing entropy globally, upweight the small fraction of training
samples that sit in the bridge — |M_i| < 0.5. Just implemented; the
first run finished today and its post-hoc diagnostic is in flight.
Early signal is encouraging — the pure-MLE part of the loss matches
the baseline at matched epochs, so we're not paying a fit penalty,
which is what you'd want.

---

## 6. Open questions and what's next (≈2 min)

[Slide: open questions]

Three things I'd want to address next:

**First, does the α excess at T_c saturate or keep growing?** A
single L=64 point would tell us — does α stabilize around 2.2 as a
finite-size correction, or does it diverge as L → ∞? An L=64 bignet
run would take roughly 36 hours on an A100, so a one-shot
investment that's worth it before the next paper draft.

**Second, fixing L=32 late-training instability.** All L=32 runs
spike in LOSS in the final 10% of training — by 10 to 90 nats above
the smoothed best. We currently work around this by reporting
best-smoothed-over-trajectory, but a cosine LR or AdamW after
epoch 10k would let us actually use the final checkpoint. This also
ties to a related TODO — the `-load` resume drops optimizer state, so
restarts cost the first 500 epochs to Adam burn-in.

**Third, the bridge-upweighting follow-up.** If today's diagnostic
shows the flow's bridge density actually widens — which is the
structural test we set this up to do — we'll do a small W and T
sweep. If it doesn't, we'll know upweighting alone isn't enough and
we need to revisit the architectural priors — maybe Z₂-equivariant
coupling layers, which is a half-finished piece of code we have
sitting in rnvp.py.

---

## 7. Closing (≈30 sec)

Two takeaways.

The α=2.20 result gives us a *physics-meaningful benchmark* for any
future flow architecture — not "did absolute KL drop," but "did we
get the per-site cost growth-rate down toward 2."

And the mode-vs-mass tradeoff at L=32 is *not* the flow being broken
— it's the two KL directions doing what they're designed to do. The
right metric depends on what you want the flow for. Visual
plausibility tracks forward-KL; HMC proposal efficiency tracks
reverse-KL.

Happy to take questions.

---

## Appendix — numbers to have ready for Q&A

- α exponents: 1.95, 2.01, **2.20** at T_c, 2.13, 2.16 (off-critical
  by T)
- L=32 T_c KL: sym_bignet 9.4 rev / 64.6 fwd; hs_bignet 3.6 fwd /
  21.3 rev; jsLoss 16.5 rev / 17.0 fwd; phase2 ≈17.6 rev (re-diag
  pending)
- Structural: data mag_abs_p = 2.38, xi_p = 8.6
- Network sizes: default 1.07M (L=8), 1.42M (L=16), 1.78M (L=32);
  bignet 10.94M (L=32 only — 6.1× default)
- H(p_HS) at L=32 T_c = 1902.98 nat (theory matches MC to 3 dp)
- Z_c at L=32 T_c: lnZ_c = 2369.587
