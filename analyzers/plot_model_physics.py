"""Standalone per-model physics diagnostic — routine for any trained
MERA + HCG flow.

For a given run folder:
  1. Loads the requested (or latest) checkpoint
  2. Reads L, T from parameters.hdf5
  3. Draws N samples via flow.sample()
  4. Loads HS data reference (data/mcmc_data/hs_L{L}_T*_N*.pt)
  5. Generates 4 comparison plots (spin-basis: sign(x)):
     - configurations.png : side-by-side lattice grids
     - two_point_correlation.png : |G(r)| for flow + data on log scale
     - M_distribution.png : per-sample M histogram overlaid
     - observables.png : |M|, χ, U₄ bars vs GT reference lines
  6. Writes summary.json with all numerical values

Output goes to <folder>/physics_plots/ by default (self-contained
per-model diagnostic).

Usage (single model):
  python analyzers/plot_model_physics.py <folder> [--epoch N] [--N samples]

Usage (multi-model comparison):
  python analyzers/plot_model_physics.py \\
      --cells label1:folder1[:ep1] label2:folder2[:ep2] ... \\
      --out figures/my_comparison
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_fixed_point"))

# rg_fixed_point/ isn't a package (no __init__.py), so import directly
from plot_L128_transfer import (
    sample_from_ckpt, load_hs_data, compute_spin_observables,
    compute_gr_axial, plot_config_grid, plot_gr, plot_M_hist,
    plot_observables_bar,
)


def run_multi(cells_spec, N, batch, out_dir, device):
    """Multi-model comparison mode."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sample] drawing {N} samples from {len(cells_spec)} cells...")
    samples = []
    for spec in cells_spec:
        parts = spec.split(":")
        label = parts[0]
        folder = parts[1]
        ep = int(parts[2]) if len(parts) > 2 else 999999
        try:
            x, L, T, ep_actual = sample_from_ckpt(folder, ep, N=N, batch=batch, device=device)
            print(f"  {label}: {x.shape[0]} samples @ ep {ep_actual}")
            samples.append((label, x, ep_actual))
        except Exception as e:
            print(f"  {label}: SKIP ({e})")

    if not samples:
        print("no samples — abort"); return

    L = samples[0][1].shape[-1]
    T = 2.269185314213022
    print(f"[data] loading L={L} HS reference")
    x_data = load_hs_data(L, T, N=N)
    samples.append((f"L={L} HS data", x_data, "GT"))

    _generate_plots_and_json(samples, out_dir)


def run_single(folder, ep, N, batch, out_dir, device):
    """Single-model mode: generate plots + JSON for one folder."""
    if out_dir is None:
        out_dir = os.path.join(folder, "physics_plots")
    os.makedirs(out_dir, exist_ok=True)

    label = os.path.basename(folder.rstrip("/"))
    print(f"[sample] {label} — drawing {N} samples...")
    try:
        x, L, T, ep_actual = sample_from_ckpt(folder, ep or 999999,
                                                N=N, batch=batch, device=device)
    except Exception as e:
        print(f"  ERROR: {e}"); return
    print(f"  got {x.shape[0]} samples @ ep {ep_actual}")

    samples = [(f"flow ({label} ep {ep_actual})", x, ep_actual)]
    print(f"[data] loading L={L} HS reference")
    try:
        x_data = load_hs_data(L, T, N=N)
        samples.append((f"L={L} HS data", x_data, "GT"))
    except FileNotFoundError as e:
        print(f"  no HS reference at L={L} — plotting without GT")

    _generate_plots_and_json(samples, out_dir)


def _generate_plots_and_json(samples, out_dir):
    print("[compute] observables per cell")
    obs = []
    for label, x, ep in samples:
        absM, chi, U4, M = compute_spin_observables(x)
        obs.append((label, absM, chi, U4, M))
        print(f"  {label}: |M|={absM:.4f}  χ={chi:.3f}  U₄={U4:.4f}")

    # GT ref = last cell if it's labeled "HS data", else no GT
    has_gt = "HS data" in samples[-1][0]
    if has_gt:
        gt_absM, gt_chi, gt_U4 = obs[-1][1], obs[-1][2], obs[-1][3]
    else:
        gt_absM = gt_chi = gt_U4 = float("nan")

    L = samples[0][1].shape[-1]

    print("[compute] G(r) per cell")
    grs = []
    styles = ["-", "--", "-.", ":", "-"]
    for i, (label, x, ep) in enumerate(samples):
        G = compute_gr_axial(x)
        grs.append((label, G, styles[i % len(styles)]))

    print("[plot] generating figures...")
    plot_config_grid([(l, x, ep) for l, x, ep in samples],
                     os.path.join(out_dir, "configurations.png"), n_samples=4)
    plot_gr(grs, os.path.join(out_dir, "two_point_correlation.png"), L)

    hist_data = []
    palette = ["steelblue", "darkorange", "seagreen", "crimson",
               "purple", "brown", "olive", "black"]
    for i, (label, x, ep) in enumerate(samples):
        M = obs[i][4].numpy()
        hist_data.append((label, M, palette[i % len(palette)]))
    plot_M_hist(hist_data, os.path.join(out_dir, "M_distribution.png"), gt_absM)

    obs_data = [(l, a, c, u) for (l, a, c, u, _) in obs]
    plot_observables_bar(obs_data, os.path.join(out_dir, "observables.png"),
                          (gt_absM, gt_chi, gt_U4))

    summary = {
        "L": int(L),
        "T": 2.269185314213022,
        "gt": {"absM": gt_absM, "chi": gt_chi, "U4": gt_U4},
        "cells": [
            dict(label=l, epoch=ep, absM=a, chi=c, U4=u)
            for (l, x, ep), (l2, a, c, u, _) in zip(samples, obs)
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[out] wrote {out_dir}/{{configurations,two_point_correlation,M_distribution,observables}}.png + summary.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default=None,
                    help="Single run folder — outputs to <folder>/physics_plots/")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Checkpoint epoch (nearest saved); default = latest")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Multi-model mode: label:folder[:epoch] specs")
    ap.add_argument("--out", default=None,
                    help="Output dir. Default: <folder>/physics_plots (single) or REQUIRED (multi)")
    ap.add_argument("--N", type=int, default=1000, help="Samples per cell")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="'cpu' or 'cuda' (cuda faster if flow is big)")
    args = ap.parse_args()

    if args.cells:
        if not args.out:
            print("ERROR: --out required for multi-model mode"); sys.exit(1)
        run_multi(args.cells, args.N, args.batch, args.out, args.device)
    elif args.folder:
        run_single(args.folder, args.epoch, args.N, args.batch,
                    args.out, args.device)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
