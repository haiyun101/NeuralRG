"""Matched-pair MSE on gauge-fixed V5 fields.

Same paired samples as gauge_v5_compare.py but reports a sample-by-sample
alignment metric instead of distributional RMS-G:

    MSE_paired(s) = E_i[ ||T_y(y_s^i) - T_x(x_s^i)||^2 ] / L_s^2

Works at every scale s = 0..K (no m < 4 cutoff), in particular at L_s = 2
where RMS-G is n/a. Range [0, 2]: 0 = perfect MERA/Wilson alignment,
2 = totally uncorrelated under N(0,1) marginals.

Restricted to 4 main flows (hs_bignet, sym_bignet, T=2.15, T=2.40) so we
can iterate quickly. Output: analyzers/csv/rg_v5_gauge_matched_mse.csv
"""
import csv
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import source  # noqa
import train   # noqa

from rg_v5_blockRG_compare import (
    FOLDERS, load_flow, block_rg_cascade, mera_forward_intermediates,
    subsample_mera_to_scale,
)
from rg_fixed_point_v4_dataforward import load_hs_data
from gauge_v5_compare import fit_per_site_quantile, apply_quantile_forward

# Only the 4 main focus flows
KEEP = {
    "T = 2.15  (low T, ordered)",
    "T = 2.269 (T_c, hs_bignet)",
    "T_c sym_bignet (rev-KL)",
    "T = 2.40  (high T, disorder)",
}


def matched_mse_per_scale(mera_per_scale, blockrg_per_scale, mera_transforms,
                           n_knots=128):
    """Compute matched-pair MSE on gauge-fixed fields per scale."""
    out = []
    keys = sorted(set(mera_per_scale.keys()) & set(blockrg_per_scale.keys()))

    for s in keys:
        y_full = mera_per_scale[s]
        x_blk = blockrg_per_scale[s]

        # Always re-fit per-site quantile transform on this exact field to
        # guarantee strict N(0,1) marginals (the saved-transform path is
        # fragile under scale/shape mismatches and led to MSE > 2 outliers
        # in an earlier debug run).
        ky, kz = fit_per_site_quantile(y_full, n_knots=n_knots)
        y_gauge = apply_quantile_forward(y_full, ky, kz)

        # Subsample MERA to scale s sub-lattice (matches RMS-G convention)
        if s == 0:
            y_sub = y_gauge
        else:
            y_sub = subsample_mera_to_scale(y_gauge, s)

        # Gauge-fix block-RG inline
        ky_b, kz_b = fit_per_site_quantile(x_blk, n_knots=n_knots)
        x_blk_gauge = apply_quantile_forward(x_blk, ky_b, kz_b)

        # Match sizes after sub-sampling
        m = min(y_sub.shape[-1], x_blk_gauge.shape[-1])
        y_sub = y_sub[..., :m, :m]
        x_blk_gauge = x_blk_gauge[..., :m, :m]

        # Matched-pair MSE: per-sample, per-site, then average
        # shapes: (B, 1, m, m) - take element-wise difference over same i
        diff = y_sub - x_blk_gauge
        mse = float((diff ** 2).mean().item())
        out.append(dict(s=s, L_s=m, matched_mse=mse))
    return out


def run_one(label, folder, N=2000, device="cpu", n_knots=128):
    print(f"\n=== {label}  folder={folder} ===", flush=True)
    transforms_path = os.path.join(folder, "gauge_transforms.pt")
    mera_transforms = []
    if os.path.exists(transforms_path):
        raw = torch.load(transforms_path, weights_only=True)
        mera_transforms = [(ky.float(), kz.float()) for ky, kz in raw]
        print(f"  loaded {len(mera_transforms)} saved transforms", flush=True)
    else:
        print(f"  no saved transforms — will fit on-the-fly", flush=True)

    mera, L, T, epoch, wt = load_flow(folder, device=device)
    print(f"  L={L} T={T} ep={epoch}", flush=True)
    samples = load_hs_data(L, T, N, device=device)

    mera_per_scale, n_phys_scales = mera_forward_intermediates(mera, samples, L)
    blockrg_per_scale = block_rg_cascade(samples, n_phys_scales)

    results = matched_mse_per_scale(mera_per_scale, blockrg_per_scale,
                                     mera_transforms, n_knots=n_knots)
    for r in results:
        print(f"  s={r['s']} (L_s={r['L_s']}):  matched MSE = {r['matched_mse']:.4f}", flush=True)
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch, scales=results)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--out", default="analyzers/csv/rg_v5_gauge_matched_mse.csv")
    args = p.parse_args()

    results = []
    for label, folder in FOLDERS.items():
        if label not in KEEP:
            continue
        try:
            results.append(run_one(label, folder, N=args.N))
        except Exception as e:
            import traceback
            print(f"  FAILED on {label}: {e}")
            traceback.print_exc()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "scale_s", "L_s",
                    "metric", "value"])
        for r in results:
            for d in r["scales"]:
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            d["s"], d["L_s"], "v5_gauge_matched_mse",
                            d["matched_mse"]])
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
