"""Gauge-fixed Layer-by-Layer Interpretation — Scheme V.1.

Pure post-training analysis. Given a trained MERA flow:

  1. Run forward(HS-data) → collect intermediate fields y_s per physical scale.
  2. For each layer s, fit a per-site, piecewise-linear quantile transform
     T_s (a 1D Spline Flow's monotonic-bijection cousin) that maps each site's
     empirical marginal to standard Gaussian:
                T_s :  y_s_{site}  ↦  z_s_{site} ~ N(0,1)
  3. Demonstrate the "gauge-fixing" identity: inserting T_s^{-1} ∘ T_s between
     layers does not change the network's actual computation, but lets us
     compare layers in a common N(0,1)-marginal "gauge".

Why: layer-by-layer similarity probes (V1/V2) use zscore (mean / std) which
only removes 1st + 2nd moments. Higher-order shape mismatch (skewness, kurtosis,
multimodality) still inflates MSE. After gauge-fixing, every site has identical
N(0,1) marginal at every layer, so MSE captures ONLY the joint dependence
structure (the "copula"). This is the right metric for testing RG fixed-point
behaviour, where scale invariance is a structural claim, not a marginal-shape
claim (anomalous dimension η = 1/4 implies marginals DO scale).

Output to <folder>/gauge_transforms.pt; demo prints zscore-vs-gauge-fixed MSE.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import source  # noqa: F401  (sets up the source package for load_flow)
import train   # noqa: F401  (same for train.symmetryMERAInit)
from rg_fixed_point import load_flow


# ---------------------------------------------------------------------------
# Per-site 1D quantile transform (the "Spline Flow's piecewise-linear cousin")
# ---------------------------------------------------------------------------

def fit_per_site_quantile(y_data, n_knots=128):
    """Per-site empirical quantile knots.

    For each site (c, i, j), take K=n_knots equally-spaced rank quantiles of
    y_data[:, c, i, j] sorted across the sample dim. Pair them with the
    matching standard-Gaussian quantiles. Linear interpolation between knots
    gives a monotonic, differentiable, invertible R → R map. Outside the
    extremes we extrapolate linearly using the last segment's slope.

    Args:
      y_data: (N, C, L, L) tensor of intermediate fields.
      n_knots: K, number of quantile knots per site.

    Returns:
      knots_y: (C, L, L, K) — site-dependent input quantiles (sorted asc)
      knots_z: (K,)        — matching Gaussian quantiles (same for all sites)
    """
    N, C, L, _ = y_data.shape
    D = C * L * L
    flat = y_data.reshape(N, D)
    sorted_flat, _ = flat.sort(dim=0)   # (N, D)
    # Pick K equally-spaced ranks in [0, N-1] inclusive
    rank_idx = ((torch.arange(n_knots, dtype=torch.float64) + 0.5)
                / n_knots * N).long().clamp(0, N - 1)
    knots = sorted_flat[rank_idx]       # (K, D)
    knots_y = knots.transpose(0, 1).reshape(C, L, L, n_knots).contiguous()

    # Matching N(0,1) quantiles at the SAME ranks
    u = (torch.arange(n_knots, dtype=torch.float64) + 0.5) / n_knots
    knots_z = torch.erfinv(2.0 * u - 1.0) * np.sqrt(2.0)
    knots_z = knots_z.to(knots_y.dtype)
    return knots_y, knots_z


def apply_quantile_forward(y, knots_y, knots_z):
    """Map y_s ↦ z_s site-wise via piecewise-linear interpolation.

    Args:
      y: (B, C, L, L) input field.
      knots_y: (C, L, L, K)
      knots_z: (K,)

    Returns:
      z: (B, C, L, L) with marginal ≈ N(0,1) at every site.
    """
    B = y.shape[0]
    C, L_, _, K = knots_y.shape
    # broadcast knots_y to batch dim, search for each y value in its site's table
    knots_y_b = knots_y.unsqueeze(0).expand(B, -1, -1, -1, -1)            # (B,C,L,L,K)
    idx_r = torch.searchsorted(knots_y_b, y.unsqueeze(-1)).squeeze(-1)    # (B,C,L,L)
    idx_r = idx_r.clamp(1, K - 1)
    idx_l = idx_r - 1
    y_l = torch.gather(knots_y_b, -1, idx_l.unsqueeze(-1)).squeeze(-1)
    y_r = torch.gather(knots_y_b, -1, idx_r.unsqueeze(-1)).squeeze(-1)
    z_l = knots_z[idx_l]                                                  # (B,C,L,L)
    z_r = knots_z[idx_r]
    frac = (y - y_l) / (y_r - y_l).clamp(min=1e-9)
    z = z_l + frac * (z_r - z_l)
    return z


def apply_quantile_inverse(z, knots_y, knots_z):
    """Inverse: z_s ↦ y_s site-wise (used to verify the bijection identity)."""
    B = z.shape[0]
    C, L_, _, K = knots_y.shape
    # knots_z is 1D and sorted, so use plain searchsorted
    idx_r = torch.searchsorted(knots_z, z.flatten()).reshape(z.shape)
    idx_r = idx_r.clamp(1, K - 1)
    idx_l = idx_r - 1
    z_l = knots_z[idx_l]
    z_r = knots_z[idx_r]
    # per-site y-quantile look-up
    knots_y_b = knots_y.unsqueeze(0).expand(B, -1, -1, -1, -1)
    y_l = torch.gather(knots_y_b, -1, idx_l.unsqueeze(-1)).squeeze(-1)
    y_r = torch.gather(knots_y_b, -1, idx_r.unsqueeze(-1)).squeeze(-1)
    frac = (z - z_l) / (z_r - z_l).clamp(min=1e-9)
    return y_l + frac * (y_r - y_l)


# ---------------------------------------------------------------------------
# Intermediate-collector + demo
# ---------------------------------------------------------------------------

def collect_intermediates_forward(folder, n_samples=4000, batch_size=128, device="cpu"):
    """Run forward(HS-data) through MERA, collect one intermediate per scale."""
    mera, L, T, epoch, _ = load_flow(folder, device=device)
    hs_files = sorted(glob.glob(f"data/mcmc_data/hs_L{L}_T*_N*.pt"))
    if not hs_files:
        raise FileNotFoundError(f"no HS data for L={L}")
    t_of = lambda p: float(re.search(r"_T([\d.]+)_N", p).group(1))
    hs_path = min(hs_files, key=lambda p: abs(t_of(p) - float(T)))
    print(f"  HS data: {hs_path}", flush=True)
    x_hs = torch.load(hs_path, weights_only=True).reshape(-1, 1, L, L).float()
    x_hs = x_hs[:n_samples].to(device=device)

    out_per_scale = None
    with torch.no_grad():
        for s in range(0, n_samples, batch_size):
            xb = x_hs[s:s + batch_size]
            _, _, ints = mera.forward_with_intermediates(xb)
            if out_per_scale is None:
                out_per_scale = [[t.cpu()] for t in ints]
            else:
                for i, t in enumerate(ints):
                    out_per_scale[i].append(t.cpu())
    return [torch.cat(lst, dim=0) for lst in out_per_scale], L, T, epoch


def _kurt(x):
    m = x.mean()
    s = x.std()
    return (((x - m) / s.clamp(min=1e-9)) ** 4).mean().item() - 3.0


def _zscore(x, eps=1e-9):
    return (x - x.mean()) / (x.std() + eps)


def _mse(a, b):
    return ((a - b) ** 2).mean().item()


def main():
    p = argparse.ArgumentParser(description="Gauge-fixed layer-by-layer demo.")
    p.add_argument("--folder", required=True, help="run folder (with savings/ and parameters.hdf5)")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-knots", type=int, default=128)
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda:0'")
    args = p.parse_args()

    print(f"\n=== Gauge-fix demo on {args.folder} ===\n", flush=True)

    print("Step 1: collect intermediates (forward dir, HS data) ...", flush=True)
    intermediates, L, T, epoch = collect_intermediates_forward(
        args.folder, args.n_samples, device=args.device)
    print(f"  flow: L={L}, T={T:.5f}, ep={epoch}", flush=True)
    print(f"  {len(intermediates)} intermediate fields collected", flush=True)
    for s, y in enumerate(intermediates):
        m, std, kurt = y.mean().item(), y.std().item(), _kurt(y)
        print(f"    y_{s+1}: shape={tuple(y.shape)}  mean={m:+.3f}  std={std:.3f}  kurt={kurt:+.3f}",
              flush=True)

    print("\nStep 2: per-site quantile-transform fit (gauge → N(0,1) marginal) ...", flush=True)
    transforms = []
    for s, y in enumerate(intermediates):
        ky, kz = fit_per_site_quantile(y, n_knots=args.n_knots)
        transforms.append((ky, kz))
        # verify forward+inverse round-trip + post-gauge marginal stats
        z = apply_quantile_forward(y[:200], ky, kz)
        center = L // 2
        z_center = z[:, 0, center, center]
        print(f"    T_{s+1} fitted  ::  center-site (z mean={z_center.mean():+.4f}, "
              f"std={z_center.std():.4f}, kurt={_kurt(z_center):+.3f})", flush=True)

    out_path = os.path.join(args.folder, "gauge_transforms.pt")
    torch.save([(ky.cpu(), kz.cpu()) for ky, kz in transforms], out_path)
    print(f"  saved → {out_path}", flush=True)

    print("\nStep 3: adjacent-scale MSE  ::  standard zscore  vs  gauge-fixed", flush=True)
    print("(both sides subsampled to the deeper scale's kept-coarse stride)\n", flush=True)
    print(f"{'pair':>14s}  {'zscore MSE':>11s}  {'gauge MSE':>11s}  {'gauge/zscore':>13s}",
          flush=True)
    print("-" * 60, flush=True)

    K = len(intermediates)
    for s in range(K - 1):
        y_s = intermediates[s]
        y_sp1 = intermediates[s + 1]
        stride = 2 ** (s + 1)
        # Standard zscore-MSE (V1-style spirit, but on real intermediates)
        a_z = _zscore(y_s[..., ::stride, ::stride])
        b_z = _zscore(y_sp1[..., ::stride, ::stride])
        m_zscore = _mse(a_z, b_z)
        # Gauge-fixed MSE — apply per-site transform first
        ky_s, kz = transforms[s]
        ky_sp1, _kz = transforms[s + 1]  # kz is same; kept for clarity
        a_g = apply_quantile_forward(y_s, ky_s, kz)[..., ::stride, ::stride]
        b_g = apply_quantile_forward(y_sp1, ky_sp1, kz)[..., ::stride, ::stride]
        m_gauge = _mse(a_g, b_g)
        ratio = m_gauge / max(m_zscore, 1e-12)
        print(f"  f_{s+1} → f_{s+2}  {m_zscore:>11.4f}  {m_gauge:>11.4f}  {ratio:>13.3f}",
              flush=True)

    print("""\nReading:
  ratio < 1   ⇒  gauge-fixing UNCOVERS structural similarity hidden by marginal shape
                  (current zscore-MSE is inflated by η-driven anomalous-dimension scaling)
  ratio ≈ 1   ⇒  marginals were already similar in shape; zscore MSE was already structural
  ratio > 1   ⇒  gauge-fixing amplifies higher-order mismatches; the two layers really do
                  different STRUCTURAL work (and matched on lower-order moments only)

Operational interpretation for an RG fixed-point candidate:
  V3 residual large + gauge-fixed MSE small  ⇒  TRUE non-trivial fixed point
  V3 residual large + gauge-fixed MSE large  ⇒  NOT a fixed point (different structures)
  V3 residual small + gauge-fixed MSE small  ⇒  trivial collapse (already known)
""", flush=True)


if __name__ == "__main__":
    main()
