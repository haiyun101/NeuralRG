"""Generate comparison plots for physReg experiments (parameterized).

Produces the same 4 plot types as plot_L128_transfer.py but for
arbitrary cells passed via --cells CLI. Useful for:
  - L=32 physReg sweep: plain vs λ=0.01 vs λ=0.1 vs λ=1.0
  - L=64 physReg: plain vs warm+physReg vs fresh+physReg
  - Any other multi-model physics comparison

Output: <out_dir>/{configurations,two_point_correlation,M_distribution,observables}.png + summary.json
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))                    # analyzers/
sys.path.insert(0, os.path.join(_here, "..", ".."))              # repo root
sys.path.insert(0, os.path.join(_here, "..", "rg_fixed_point"))  # for plot_L128_transfer helpers

# Reuse the sampling / physics / plot helpers from plot_L128_transfer.py
from plot_L128_transfer import (
    sample_from_ckpt, load_hs_data, compute_spin_observables,
    compute_gr_axial, plot_config_grid, plot_gr, plot_M_hist,
    plot_observables_bar,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="label:folder:epoch specs")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[sample] drawing samples from {len(args.cells)} cells...")
    samples = []
    for spec in args.cells:
        parts = spec.split(":")
        label = parts[0]
        folder = parts[1]
        ep = int(parts[2]) if len(parts) > 2 else 999999
        try:
            x, L, T, ep_actual = sample_from_ckpt(folder, ep, N=args.N,
                                                    batch=args.batch,
                                                    device=args.device)
            print(f"  {label}: got {x.shape[0]} samples at ep {ep_actual}")
            samples.append((label, x, ep_actual))
        except Exception as e:
            print(f"  {label}: SKIP ({e})")

    if not samples:
        print("no samples — abort")
        return

    L = samples[0][1].shape[-1]
    T = 2.269185314213022
    print(f"[data] loading L={L} HS reference")
    x_data = load_hs_data(L, T, N=args.N)
    samples.append((f"L={L} HS data", x_data, "GT"))

    # Observables
    print("[compute] observables per cell")
    obs = []
    for label, x, ep in samples:
        absM, chi, U4, M = compute_spin_observables(x)
        obs.append((label, absM, chi, U4, M))
        print(f"  {label}: |M|={absM:.4f}  χ={chi:.3f}  U₄={U4:.4f}")

    gt_absM, gt_chi, gt_U4 = obs[-1][1], obs[-1][2], obs[-1][3]

    # G(r)
    print("[compute] G(r) per cell")
    grs = []
    styles = ["-", "--", "-.", ":", "-"]
    for i, (label, x, ep) in enumerate(samples):
        G = compute_gr_axial(x)
        style = styles[i % len(styles)]
        grs.append((label, G, style))

    # Plots
    print("[plot] generating figures...")
    plot_config_grid([(l, x, ep) for l, x, ep in samples],
                     os.path.join(args.out, "configurations.png"),
                     n_samples=4)
    plot_gr(grs, os.path.join(args.out, "two_point_correlation.png"), L)

    hist_data = []
    palette = ["steelblue", "darkorange", "seagreen", "crimson",
               "purple", "brown", "black"]
    for i, (label, x, ep) in enumerate(samples):
        M = obs[i][4].numpy()
        hist_data.append((label, M, palette[i % len(palette)]))
    plot_M_hist(hist_data, os.path.join(args.out, "M_distribution.png"), gt_absM)

    obs_data = [(l, a, c, u) for (l, a, c, u, _) in obs]
    plot_observables_bar(obs_data, os.path.join(args.out, "observables.png"),
                          (gt_absM, gt_chi, gt_U4))

    # Summary JSON
    summary = {
        "L": int(L), "T": T,
        "gt": {"absM": gt_absM, "chi": gt_chi, "U4": gt_U4},
        "cells": [
            dict(label=l, epoch=ep, absM=a, chi=c, U4=u)
            for (l, x, ep), (l2, a, c, u, _) in zip(samples, obs)
        ],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
