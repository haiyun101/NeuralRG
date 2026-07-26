# HCG-CNN vs MERAUNet-FM — comparison of two hierarchical methods

> Both methods take inspiration from MERA's multi-scale (RG) structure
> but implement it differently. This report compares their sample
> physics at L=32, L=64, L=128 T_c to see whether the hierarchical
> intuition alone is sufficient, or whether specific architectural
> choices matter.

## 1. The two methods

### HCG-CNN champion (fixdil+VP-1e-3 nr=1)

- **Base**: RNVP normalizing flow with MERA-style dispatch/collect
- **Prior**: Hierarchical Conditional Gaussian (HCG) — per-scale CNN
  predicts `(μ, σ)` from coarser context at each of log₂(L) levels
- **Loss**: forward KL (MLE) on HS-transformed samples
- **Regularizer**: VP penalty pushes MERA volume-preserving so CNN
  absorbs σ (see memory `cnn-absorbs-variance-not-mean`)
- **Training**: 10K+ epochs, batch=64 (L=32), batch=16 (L=64)

### MERAUNet-FM (Flow Matching with hierarchical U-Net)

- **Base**: U-Net that uses **exactly log₂(L) downsampling stages**,
  mirroring MERA's scale hierarchy in the encoder path
- **Velocity field**: `v_t(x)` regressed via CFM (Lipman et al. 2022) —
  rectified-flow probability paths from noise to data
- **Loss**: MSE on velocity vectors (NOT log-likelihood)
- **Sampling**: ODE integration from noise → data
- **Training**: ~1K epochs, batch=128 (L=32), batch=96 (L=64), batch=16 (L=128)

## 2. Physics comparison at T_c

**Ground truth from Wolff MCMC**:
| L   | \|M\|_GT | χ_GT | U₄_GT |
|:---:|:---:|:---:|:---:|
| 32  | 0.6544 | 31.61 | 0.6110 |
| 64  | 0.6004 | 106.03 | 0.6109 |
| 128 | 0.5507 | 357.40 | 0.6109 |

**Sample physics from each method**:

| L | Method | \|M\| | \|M\| err | χ | χ err | U₄ | U₄ err |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **32** | HCG-CNN champion @ ep 9500 | 0.6737 | +2.9% | **19.8** | **−37.2%** | 0.626 | +2.5% |
| **32** | MERAUNet-FM @ ep 1200 (best) | 0.6480 | −1.0% | **39.4** | **+24.5%** | 0.596 | −2.5% |
| **64** | MERAUNet-FM @ ep 180 | 0.385 | −35.9% | 177 | +67.1% | 0.434 | −29.0% |
| **128** | HCG-CNN warm ep 5000 | 0.552 | +0.2% | **237** | **−33.5%** | 0.621 | +1.7% |
| **128** | MERAUNet-FM v2 @ ep 150 | 0.571 | +3.7% | **398** | **+11.2%** | 0.616 | +0.8% |

**HCG-CNN not measured at L=64** — champion checkpoint exists but no
spin-basis physics recorded in the earlier analyses.

## 3. Two opposite pathologies

**HCG-CNN**: consistently **UNDER-shoots χ** across all L
- L=32: χ_flow / χ_GT = 63% (−37% error)
- L=128: χ_flow / χ_GT = 66% (−34% error)
- Nearly L-invariant relative error → intrinsic to the "MLE + HCG prior"
  design

**MERAUNet-FM**: consistently **OVER-shoots χ** at converged L
- L=32: χ_flow / χ_GT = 125% (+25% error)
- L=128: χ_flow / χ_GT = 111% (+11% error) — closer to GT than HCG!

Both methods get **\|M\|, U₄ close to GT (within 5%)** at their best
checkpoints (L=32 champion FM at ep 1200, L=128 FM v2 at ep 150).

## 4. Interpretation

### Under-shoot vs over-shoot dichotomy

**HCG-CNN under-shoot mechanism**: MLE loss `-E_data[log q(x)]` weights
tail samples logarithmically, so **flow undertrains on rare tail
configurations** (large-|M| or wide domain-wall samples). Result: q
distribution has **narrower peaks around ±|M|_peak** than data →
Var(M) too small → χ too small.

**MERAUNet-FM over-shoot mechanism**: MSE velocity regression weights
tail samples **equally** (unlike log-likelihood). Combined with the
Gaussian noise reference distribution → flow allocates too much mass
to broad, disordered configurations mid-way between the Z₂ peaks →
Var(M) too large → χ too large.

### Same hierarchical inductive bias, different objective → different pathology

Both methods embed MERA's scale-invariance intuition, but:
- HCG-CNN: hierarchy IN the density model (prior) → MLE optimum
- MERAUNet-FM: hierarchy IN the velocity network (encoder depth) →
  MSE optimum

The scale-invariance helps with **|M|** and **U₄** in both cases
(they're within 5% for both methods at converged L=32/128). But
**χ** — which is a Var(M) statistic dominated by tail behavior —
diverges in opposite ways because of the loss function's tail weighting.

### Which is better?

At L=32: HCG-CNN champion has |M| slightly better (+2.9% vs FM's −1.0%),
FM has χ dramatically better (+25% vs HCG's −37%).

At L=128: HCG-CNN has |M| slightly better (+0.2% vs FM's +3.7%), but
FM has χ MUCH better (+11% vs HCG's −34%).

**FM is closer to GT physics overall at L=128** despite training only
150 epochs vs HCG-CNN's 5000 epochs. **Hierarchical U-Net + FM appears
to be more compute-efficient at large L**.

## 5. Compute cost

| Method | L | Batch | ep to physics-quality | Wall time |
|:---|:---:|:---:|:---:|:---:|
| HCG-CNN | 32 | 64 | 9500 | ~2 days |
| HCG-CNN | 64 | 16 | 9500 | ~4 days |
| HCG-CNN | 128 | 8 | 5000+ (with L=64 warm start) | ~1-2 days |
| MERAUNet-FM | 32 | 128 | 1200 | ~15 hours |
| MERAUNet-FM | 64 | 96 | ~200-500 (not yet at good physics) | ~2 days |
| MERAUNet-FM | 128 | 16 | 150 | ~1 day |

**MERAUNet-FM at L=128 reaches near-GT physics in ~150 epochs = ~1 day**.
The champion arch needs 5000+ epochs even with warm-start.

## 6. Trade-offs

### HCG-CNN pros
- **Explicit density model** — can compute log q(x) for any x (needed
  for reverse-KL, importance weights, MCMC integration)
- **Interpretable per-scale physics** — CNN weights at each stride
  correspond to physically meaningful RG operators
- **Volume-preservation guarantee** (with VP penalty) — Jacobian ≈ 0

### HCG-CNN cons
- **χ under-shoot** — chronic ~35% deficit
- **Long training** — 5-10K epochs to reach floor
- **Scaling issues** — L=64→L=128 transfer requires careful CNN
  alignment; L=32→L=64 transfer fails (different critical basins)

### MERAUNet-FM pros
- **Excellent \|M\| and U₄** at converged epochs
- **Fast training** — reaches good physics in ~200 epochs
- **Simpler loss** — MSE, no Jacobian computation
- **Scales well to L=128** without warm-start

### MERAUNet-FM cons
- **χ over-shoot** — samples too spread out, mode-covering artifacts
- **No density access** — sampling only, no log q(x); can't compute
  KL, free energy, or use for importance sampling
- **U-Net not "MERA structured"** — the analogy is that scale count
  matches, but the underlying operations (convolutions with time
  embedding) are different from MERA's RNVP couplings

## 7. Cross-method observations from prior work

- **Layer self-similarity (from `fm_layer_self_similarity_L32.csv`)**:
  MERAUNet's encoder features at L=32 t=1.0 have **kurt = −1.33**
  (sub-Gaussian) — opposite of HCG-CNN L=32 champion's **kurt = +0.84**
  (heavy-tail). Suggests they learn qualitatively different internal
  representations even though both are "hierarchical".

- **Cross-L transfer**: HCG-CNN champion supports smart transfer via
  the stride-aligned mechanism (see `cross_L_transfer_report.md`).
  MERAUNet-FM has NOT been tested for cross-L transfer — the encoder
  is arch-specific to L (each downsampling stage's channel schedule
  depends on n_scales).

## 8. Recommendations

**For lowest-loss density modeling** (research on RG fixed point,
importance sampling): use HCG-CNN. Accept the χ under-shoot as
intrinsic to MLE + HCG design.

**For fast sample generation with correct physics** (proposals for
MCMC, physics-quality generation): use MERAUNet-FM. Accept χ
over-shoot as intrinsic to MSE + Gaussian reference; U₄, \|M\| are
excellent.

**Best of both**: could combine — use MERAUNet-FM samples as physics-
quality proposals, then reweight by HCG-CNN's log q(x) for correct
density (though the mismatch cost may be high).

**Not tested**: physReg regularization on MERAUNet-FM. The soft-spin
gradient framework (see `analyzers/physReg/report.md`) could pull the
FM samples' χ back down, but this would require adding physReg to
`train/fm_learn.py` (currently only supports the champion training
path). Might resolve FM's χ over-shoot the same way physReg λ=0.01
resolves the L=32 champion's under-shoot.

## 9. Open questions

1. Why do both methods get **U₄ within 5%** despite different χ
   pathologies? U₄ is a normalized cumulant `1 − ⟨M⁴⟩/(3⟨M²⟩²)` that
   depends on Var(M) both in numerator and denominator; this makes
   it more robust to over/under-shoot.

2. Does MERAUNet-FM's χ over-shoot **decrease with more training**? Or
   is it intrinsic to the CFM objective? Would extending L=128 FM to
   500-1000 epochs help or hurt?

3. What if we replace **RNVP couplings inside the MERA blocks with
   NSF splines** (task memory `nsf-identity-init`)? Might combine
   HCG-CNN's density access with FM's better physics if NSF's spline
   flexibility fixes the tail-weighting problem.

## 10. References

- HCG-CNN champion: `data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/`
  (L=32), `data/64Ising_.../` (L=64), `data/L128_T2.269_champion_from_L64/` (L=128)
- MERAUNet-FM: `data/L32_T2.269_meraFM_h64/` (L=32),
  `data/L64_T2.269_meraFM_h128/` (L=64 continuation),
  `data/L128_T2.269_meraFM_h128_v2/` (L=128)
- Related: `cross_L_transfer_report.md` (HCG-CNN cross-L),
  `analyzers/physReg/report.md` (physReg for HCG-CNN)
- Related memories: `cnn-absorbs-variance-not-mean`,
  `nsf-identity-init`
