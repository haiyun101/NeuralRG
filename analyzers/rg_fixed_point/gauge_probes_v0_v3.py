"""Gauge-fixed V0-V3 probes (mirror of rg_fixed_point_robustness.py).

For each scale-block f_s, compute the gauge-fixed versions of:
  V1 global / V1 perpos / V2 chain / V2b chain-oneslot / V3 identity

The probe input is still z ~ N(0, I) at shape (N, 1, 2, 2). The "gauge"
is per-site quantile transform T_s fit on f_s(z) outputs so that every
of the 4 positions has standard-normal marginal. Then we compute the
same MSE / residual statistics in the gauge-fixed coordinates.

What this teaches:
- raw V1 deep MSE small + gauge V1 deep MSE still small → blocks really
  do match structurally (or are both identity → V3 tells us which)
- raw V1 deep MSE small + gauge V1 deep MSE small AND V3 r_s near 0 → trivial
- raw V1 deep MSE small + gauge V1 deep MSE LARGE → blocks differ in
  higher-order moments only; zscore was hiding it
- raw V1 deep MSE large + gauge V1 deep MSE small → mismatch was marginal-only
- raw V1 deep MSE large + gauge V1 deep MSE still large → real structural difference

Output CSV: analyzers/csv/rg_v0_v3_gauge.csv with
    label, folder, L, T, epoch, variant, scale_pair_or_block, value
where variant ∈ {gauge_v1_global, gauge_v1_perpos, gauge_v2_chain,
                  gauge_v2b_chain_oneslot, gauge_v3_identity_rel}.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rg_fixed_point import (
    load_flow, extract_scale_blocks, probe_scale_block,
)
from rg_fixed_point_robustness import FOLDERS

# Re-use gauge-fix kernels
from gauge_fix import fit_per_site_quantile, apply_quantile_forward


# ---------------------------------------------------------------------
# Helpers — applied per-output of each scale-block
# ---------------------------------------------------------------------

def gauge_fix_per_position(o, n_knots=128):
    """Per-position quantile transform on a (N, 1, 2, 2) tensor.

    Each of the 4 spatial positions (and the single channel) has its own
    1D quantile transform fit on the N samples, mapping its empirical
    marginal to N(0, 1).
    """
    knots_y, knots_z = fit_per_site_quantile(o, n_knots=n_knots)
    return apply_quantile_forward(o, knots_y, knots_z)


def gauge_fix_global(o, n_knots=128):
    """Single global quantile transform on a (N, 1, 2, 2) tensor.

    Pools all 4 * N samples to fit one transform; applies it to all
    positions. Mirror of zscore_global (global vs per-position).
    """
    N = o.shape[0]
    flat = o.reshape(N, -1)              # (N, 4)
    pooled = flat.reshape(-1, 1)         # (4N, 1)
    knots_y, knots_z = fit_per_site_quantile(
        pooled.reshape(-1, 1, 1, 1), n_knots=n_knots)
    # Broadcast the same transform to all 4 positions
    ky = knots_y.expand(1, 2, 2, knots_y.shape[-1]).contiguous()
    return apply_quantile_forward(o, ky, knots_z)


def mse(a, b):
    return float(((a - b) ** 2).mean().item())


# ---------------------------------------------------------------------
# Probe runners — gauge versions
# ---------------------------------------------------------------------

def run_gauge_v1(folder, label, N, seed):
    """Gauge-fixed V1: compute pairwise MSE of gauge-fixed block outputs.

    Returns global and per-position variants.
    """
    print(f"\n--- gauge V1: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)
    outs_gl, outs_pp = [], []
    with torch.no_grad():
        for s, blocks in enumerate(groups):
            o = probe_scale_block(blocks, z)
            outs_gl.append(gauge_fix_global(o))
            outs_pp.append(gauge_fix_per_position(o))
    g_mse = [mse(outs_gl[s], outs_gl[s + 1]) for s in range(len(outs_gl) - 1)]
    p_mse = [mse(outs_pp[s], outs_pp[s + 1]) for s in range(len(outs_pp) - 1)]
    for s, (mg, mp) in enumerate(zip(g_mse, p_mse)):
        print(f"   gauge MSE(f_{s+1}, f_{s+2})  global={mg:.4f}   per-pos={mp:.4f}")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                global_mse=g_mse, perpos_mse=p_mse)


def run_gauge_v2(folder, label, N, seed):
    """Gauge-fixed V2: chain inputs, gauge-fix outputs, compare adjacent."""
    print(f"\n--- gauge V2 chain: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    S = len(groups)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)
    h = [None] * S
    h[S - 1] = z
    out = z
    with torch.no_grad():
        for s_idx in range(S - 1, 0, -1):
            out = probe_scale_block(groups[s_idx], out)
            h[s_idx - 1] = out
        outs = [gauge_fix_per_position(probe_scale_block(groups[i], h[i]))
                for i in range(S)]
    rec = [mse(outs[s], outs[s + 1]) for s in range(S - 1)]
    for s, m in enumerate(rec):
        print(f"   gauge MSE(f_{s+1}(h_{s+1}), f_{s+2}(h_{s+2})) = {m:.4f}")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                chain_mse=rec)


def run_gauge_v2b(folder, label, N, seed):
    """Gauge-fixed V2b: 1-slot chain + 3 fresh N(0,I), gauge-fix outputs."""
    print(f"\n--- gauge V2b one-slot: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    S = len(groups)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N, 1, 2, 2, generator=g)
    h = [None] * S
    h[S - 1] = z
    out = z
    with torch.no_grad():
        for s_idx in range(S - 1, 0, -1):
            block = groups[s_idx]
            out = probe_scale_block(block, out)
            h_next = torch.randn(N, 1, 2, 2, generator=g)
            h_next[:, :, 0, 0] = out[:, :, 0, 0]
            h[s_idx - 1] = h_next
            out = h_next
        outs = [gauge_fix_per_position(probe_scale_block(groups[i], h[i]))
                for i in range(S)]
    rec = [mse(outs[s], outs[s + 1]) for s in range(S - 1)]
    for s, m in enumerate(rec):
        print(f"   gauge MSE_oneslot(f_{s+1}, f_{s+2}) = {m:.4f}")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                chain_mse_oneslot=rec)


def run_gauge_v3(folder, label, N, seed):
    """Gauge-fixed V3: compare T_s(f_s(z)) to z.

    Since z ~ N(0,I) already, and T_s pushes f_s(z) to N(0,1) marginal,
    we measure the (copula) residual r_s_gauge = E[(T_s(f_s(z)) − z)²].
    """
    print(f"\n--- gauge V3 identity: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)
    z_norm = float((z ** 2).mean().item())
    rec_raw, rec_rel = [], []
    with torch.no_grad():
        for s, blocks in enumerate(groups):
            o = probe_scale_block(blocks, z)
            og = gauge_fix_per_position(o)
            raw = float(((og - z) ** 2).mean().item())
            rec_raw.append(raw)
            rec_rel.append(raw / z_norm)
        for s, (raw, rel) in enumerate(zip(rec_raw, rec_rel)):
            print(f"   gauge r_{s+1} = {raw:.4f}  (rel = {rel:.4f})")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                identity_raw=rec_raw, identity_rel=rec_rel)


def write_csv(v1, v2, v2b, v3, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "variant",
                    "scale_pair_or_block", "value"])
        for r in v1:
            for i, (g, p) in enumerate(zip(r["global_mse"], r["perpos_mse"])):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v1_global",  f"f_{i+1}->f_{i+2}", g])
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v1_perpos",  f"f_{i+1}->f_{i+2}", p])
        for r in v2:
            for i, m in enumerate(r["chain_mse"]):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v2_chain", f"f_{i+1}->f_{i+2}", m])
        for r in v2b:
            for i, m in enumerate(r["chain_mse_oneslot"]):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v2b_chain_oneslot",
                            f"f_{i+1}->f_{i+2}", m])
        for r in v3:
            for i, (raw, rel) in enumerate(zip(r["identity_raw"], r["identity_rel"])):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v3_identity_raw", f"f_{i+1}", raw])
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "gauge_v3_identity_rel", f"f_{i+1}", rel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analyzers/csv/rg_v0_v3_gauge.csv")
    p.add_argument("--six", action="store_true",
                   help="Only run on the original 6 base flows (skip improvement folders)")
    args = p.parse_args()

    items = list(FOLDERS.items())
    if args.six:
        keep = {
            "T = 2.15  (low T, ordered)", "T = 2.269 (T_c, hs_dataDriven)",
            "T = 2.269 (T_c, hs_bignet)", "T_c sym_bignet (rev-KL)",
            "T_c pathgrad_bignet_long_ext (STL)", "T = 2.40  (high T, disorder)",
        }
        items = [(l, f) for (l, f) in items if l in keep]
    print(f"Running on {len(items)} folders", flush=True)

    v1, v2, v2b, v3 = [], [], [], []
    for label, folder in items:
        try:
            v1.append(run_gauge_v1(folder, label, args.N, args.seed))
            v2.append(run_gauge_v2(folder, label, args.N, args.seed))
            v2b.append(run_gauge_v2b(folder, label, args.N, args.seed))
            v3.append(run_gauge_v3(folder, label, args.N, args.seed))
        except Exception as e:
            import traceback
            print(f"  FAILED on {label}: {e}")
            traceback.print_exc()
    write_csv(v1, v2, v2b, v3, args.out)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
