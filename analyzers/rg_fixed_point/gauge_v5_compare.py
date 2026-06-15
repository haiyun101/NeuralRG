"""Gauge-fixed V5: block-RG cross-comparison with per-site Gaussianized fields.

Same skeleton as `rg_v5_blockRG_compare.py`, but every field (MERA y_s
and block-RG x_s) is **per-site Gaussianized** before computing cross
distances. The gauge transform for MERA y_s is loaded from
`<folder>/gauge_transforms.pt` (pre-computed by `gauge_fix.py`); the
block-RG transform is fit on-the-fly from the block-RG output.

After per-site gauge-fix every site has identical N(0,1) marginal, so:
  - KS and W1 collapse to ~0 (no marginal info left — by design)
  - **RMS-G remains informative** — it now measures pure spatial
    structure mismatch (rank-correlation difference), independent of
    marginal shape.

This is the operational test for "did I.1 Student-t actually IMPROVE
structure at L=64, or was the gL signal a marginal-only artefact?".
After gauge-fix:
  - If Student-t's gauge-fixed RMS-G IS still better than baseline → real structural win
  - If it's the same as / worse than baseline → marginal-only artefact

CSV output schema matches rg_v5_blockRG_compare.csv but with metric
prefix `v5_gauge_` and source `cross_gauge`.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
import torch
import scipy.stats as sps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import source  # noqa
import train   # noqa

# Re-use everything possible from V5
from rg_v5_blockRG_compare import (
    FOLDERS, load_flow, block_rg_cascade, mera_forward_intermediates,
    subsample_mera_to_scale,
)
from rg_fixed_point_v4_dataforward import load_hs_data, two_point_axial


# ---------------------------------------------------------------------------
# Per-site gauge-fix helpers (mirroring gauge_fix.py but kept inline for clarity)
# ---------------------------------------------------------------------------

def apply_quantile_forward(y, knots_y, knots_z):
    """y → z_site via piecewise-linear interpolation using pre-fit knots."""
    B = y.shape[0]
    C, L_, _, K = knots_y.shape
    knots_y_b = knots_y.unsqueeze(0).expand(B, -1, -1, -1, -1)
    idx_r = torch.searchsorted(knots_y_b, y.unsqueeze(-1)).squeeze(-1)
    idx_r = idx_r.clamp(1, K - 1)
    idx_l = idx_r - 1
    y_l = torch.gather(knots_y_b, -1, idx_l.unsqueeze(-1)).squeeze(-1)
    y_r = torch.gather(knots_y_b, -1, idx_r.unsqueeze(-1)).squeeze(-1)
    z_l = knots_z[idx_l]
    z_r = knots_z[idx_r]
    frac = (y - y_l) / (y_r - y_l).clamp(min=1e-9)
    return z_l + frac * (z_r - z_l)


def fit_per_site_quantile(y_data, n_knots=128):
    N, C, L_, _ = y_data.shape
    D = C * L_ * L_
    flat = y_data.reshape(N, D)
    sorted_flat, _ = flat.sort(dim=0)
    rank_idx = ((torch.arange(n_knots, dtype=torch.float64) + 0.5) / n_knots * N).long().clamp(0, N - 1)
    knots = sorted_flat[rank_idx]
    knots_y = knots.transpose(0, 1).reshape(C, L_, L_, n_knots).contiguous()
    u = (torch.arange(n_knots, dtype=torch.float64) + 0.5) / n_knots
    knots_z = torch.erfinv(2.0 * u - 1.0) * np.sqrt(2.0)
    knots_z = knots_z.to(knots_y.dtype)
    return knots_y, knots_z


# ---------------------------------------------------------------------------
# Gauge-fixed cross-scale comparison
# ---------------------------------------------------------------------------

def standardise_flat(x_tensor):
    flat = x_tensor.flatten().cpu().numpy()
    mu = flat.mean()
    sd = flat.std() + 1e-12
    return (flat - mu) / sd


def cross_scale_distances_gauge(mera_per_scale, blockrg_per_scale, mera_transforms,
                                L, n_knots=128):
    """Compute KS / W1 / RMS-G on gauge-fixed fields at every scale.

    mera_transforms: list of (knots_y, knots_z) per scale (output of
                     fit_per_site_quantile on the FULL field at that scale,
                     loaded from <folder>/gauge_transforms.pt).
    blockrg fields are fit-and-applied inline at each scale.
    """
    out = []
    keys = sorted(set(mera_per_scale.keys()) & set(blockrg_per_scale.keys()))
    rng_ks_a, rng_ks_b = np.random.default_rng(0), np.random.default_rng(1)
    rng_w_a, rng_w_b = np.random.default_rng(2), np.random.default_rng(3)

    for s in keys:
        # MERA: apply saved gauge transform (covers full L×L) then subsample.
        # transforms list is indexed by scale (0 is finest, s_max coarsest).
        # mera_per_scale[0] is the HS input itself; transforms[0] is for scale-1
        # output. So use transforms[s-1] for y at scale s? Let's check.
        # gauge_fix.py collects transforms in order corresponding to
        # forward_with_intermediates list. y_s is post-scale-s output. The
        # gauge_transforms[s] is for the (s+1)-th intermediate. For s=0
        # (HS input itself), there's no transform — we fit on-the-fly.
        y_full = mera_per_scale[s]
        if s == 0:
            ky, kz = fit_per_site_quantile(y_full, n_knots=n_knots)
        else:
            # Pick transform by scale index. gauge_transforms.pt is a list of
            # (knots_y, knots_z) tuples ordered by scale 0..K-1 from the FORWARD
            # direction (gauge_fix.py collects via forward_with_intermediates).
            # mera_per_scale here uses the same indexing.
            idx = s - 1 if s - 1 < len(mera_transforms) else len(mera_transforms) - 1
            ky, kz = mera_transforms[idx]
            # Spatial dims must match
            if ky.shape[1] != y_full.shape[2]:
                # Fall back to on-the-fly fit if mismatch (e.g. different L)
                ky, kz = fit_per_site_quantile(y_full, n_knots=n_knots)
        y_gauge = apply_quantile_forward(y_full, ky, kz)
        # Subsample to scale s sub-lattice (same as raw V5)
        if s == 0:
            y_sub = y_gauge
        else:
            y_sub = subsample_mera_to_scale(y_gauge, s)

        # block-RG: fit + apply on-the-fly (no saved transform)
        x_blk = blockrg_per_scale[s]
        ky_b, kz_b = fit_per_site_quantile(x_blk, n_knots=n_knots)
        x_blk_gauge = apply_quantile_forward(x_blk, ky_b, kz_b)

        # Match sizes after sub-sampling
        m = min(y_sub.shape[-1], x_blk_gauge.shape[-1])
        y_sub = y_sub[..., :m, :m]
        x_blk_gauge = x_blk_gauge[..., :m, :m]

        # KS / W1 on flat (should be near 0 since both are gauge-fixed)
        z_y = standardise_flat(y_sub)
        z_x = standardise_flat(x_blk_gauge)
        n_keep = min(200_000, z_y.size, z_x.size)
        ridx_a = rng_ks_a.choice(z_y.size, n_keep, replace=False)
        ridx_b = rng_ks_b.choice(z_x.size, n_keep, replace=False)
        ks_stat = float(sps.ks_2samp(z_y[ridx_a], z_x[ridx_b]).statistic)
        n_w = min(20_000, z_y.size, z_x.size)
        ridx_c = rng_w_a.choice(z_y.size, n_w, replace=False)
        ridx_d = rng_w_b.choice(z_x.size, n_w, replace=False)
        w1 = float(sps.wasserstein_distance(z_y[ridx_c], z_x[ridx_d]))

        # G(r)/G(0) on gauge-fixed fields — this is the KEY metric
        if m >= 4:
            Gy = two_point_axial(y_sub)
            Gx = two_point_axial(x_blk_gauge)
            half = max(1, m // 2)
            Gy_n = Gy[:half] / (Gy[0] + 1e-12)
            Gx_n = Gx[:half] / (Gx[0] + 1e-12)
            rms_g = float(np.sqrt(np.mean((Gy_n - Gx_n) ** 2)))
        else:
            rms_g = float('nan')

        out.append(dict(s=s, L_s=m, ks=ks_stat, w1=w1, rms_g=rms_g))
    return out


def run_one_gauge(label, folder, N=2000, device="cpu", n_knots=128):
    print(f"\n=== {label}  folder={folder}  (gauge-fixed) ===", flush=True)
    transforms_path = os.path.join(folder, "gauge_transforms.pt")
    if not os.path.exists(transforms_path):
        print(f"  SKIP — no gauge_transforms.pt at {transforms_path}", flush=True)
        return None
    mera_transforms_raw = torch.load(transforms_path, weights_only=True)
    mera_transforms = [(ky.float(), kz.float()) for ky, kz in mera_transforms_raw]
    print(f"  loaded {len(mera_transforms)} per-scale transforms", flush=True)

    mera, L, T, epoch, wt = load_flow(folder, device=device)
    print(f"  L={L} T={T} ep={epoch}", flush=True)
    samples = load_hs_data(L, T, N, device=device)

    mera_per_scale, n_phys_scales = mera_forward_intermediates(mera, samples, L)
    blockrg_per_scale = block_rg_cascade(samples, n_phys_scales)

    cross = cross_scale_distances_gauge(mera_per_scale, blockrg_per_scale,
                                         mera_transforms, L, n_knots=n_knots)
    for d in cross:
        rg = f"{d['rms_g']:.4f}" if not math.isnan(d['rms_g']) else "n/a"
        print(f"  gauge s={d['s']} (L_s={d['L_s']}):  KS={d['ks']:.4f}  "
              f"W1={d['w1']:.4f}  rmsG={rg}", flush=True)

    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch, cross=cross)


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "scale_s",
                    "L_s", "source", "metric", "value"])
        for r in results:
            if r is None: continue
            for d in r["cross"]:
                for k in ("ks", "w1", "rms_g"):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], d["s"], d["L_s"], "cross_gauge",
                                f"v5_gauge_{k}", d[k]])
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2000, help="HS samples to use")
    p.add_argument("--outdir", default="analyzers")
    p.add_argument("--n-knots", type=int, default=128)
    args = p.parse_args()
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)

    results = []
    for label, folder in FOLDERS.items():
        try:
            r = run_one_gauge(label, folder, N=args.N, n_knots=args.n_knots)
            if r is not None:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  FAILED on {label}: {e}")
            traceback.print_exc()

    if not results:
        raise SystemExit("no successful gauge-fixed runs (any gauge_transforms.pt missing?)")

    write_csv(results, os.path.join(args.outdir, "csv", "rg_v5_gauge_compare.csv"))


if __name__ == "__main__":
    main()
