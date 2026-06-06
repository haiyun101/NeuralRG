"""RG fixed-point probe of the trained MERA flow.

Methodology (advisor's framing):
  At a critical phase transition the system is scale-invariant, so a
  flow that has actually learned the physics should have its deeper
  scale-blocks become functionally identical. We extract each
  scale-block as a standalone map on a small (2x2) Gaussian probe
  patch and measure how its action evolves with depth.

For L=32 the MERA has depth = log2(L)*2 = 10 RNVP modules in
total -- one offset-0 + one offset-1 mask per scale, for 5 scales.
A 'scale-level' transformation f_s is the composition of those two
masks: f_s(z) = inverse(layer[2s+1], inverse(layer[2s], z)).

Pipeline:
  1. Load a trained checkpoint.
  2. Unwrap Symmetrized -> MERA -> layerList. Group into 5 scale-blocks.
  3. Draw a B x 1 x 2 x 2 probe batch of N(0, 1) noise.
  4. Pass the same batch through each scale-block independently.
  5. Z-score each output (per output, over the batch).
  6. Compute MSE between consecutive normalised outputs and plot.

If the flow is at an RG fixed point at T_c, MSE drops to a low
plateau at deeper layers. Off-T_c the curve should look different
(no scale-invariance, MSE shouldn't plateau or should land at trivial-
identity values).

Usage:
  python analyzers/rg_fixed_point.py [--N 10000]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import build_flow

# All flows the probe knows about. Each flow has a folder and a set of
# group memberships, since one flow can serve as both a baseline-panel
# entry and a methods-panel reference.
#
# The "methods" panel uses the five entries from the concise_report_L32
# cross-method comparison table at T_c (bignet architecture throughout):
#   - sym_bignet            reverse-KL
#   - hs_bignet             forward-KL (HS field, MLE)
#   - jsLoss_bignet_long    mixed: JS divergence (lam=0.5)
#   - phase2_finetune       mixed: 2-stage (rev-KL warmup then forward-KL)
#   - bridge_w5.0t0.5       bridge-reweighted forward-KL
FOLDERS = {
    # temperature controls (baseline panel)
    "T = 2.15  (low T, ordered)":      "data/32Ising_T2.15_hs_dataDriven",
    "T = 2.269 (T_c, hs_dataDriven)":  "data/32Ising_T2.269185314213022_hs_dataDriven",
    "T = 2.269 (T_c, hs_bignet)":      "data/32Ising_T2.269_hs_bignet",
    "T = 2.40  (high T, disorder)":    "data/32Ising_T2.4_hs_dataDriven",
    # methods from the concise_report_L32 comparison table
    "T_c sym_bignet (rev-KL)":         "data/32Ising_T2.269_sym_bignet",
    "T_c pathgrad_bignet_long_ext (STL)": "data/32Ising_T2.269_pathgrad_bignet_long_ext",
    "T_c jsLoss_bignet_long (mixed JS)": "data/32Ising_T2.269_jsLoss_bignet_long_lam0.5",
    "T_c phase2_finetune (mixed 2-stage)": "data/32Ising_T2.269_phase2_finetune",
    "T_c bridge_w5.0t0.5 (bridge-reweighted)": "data/32Ising_T2.269_hsBignet_bridge_w5.0t0.5",
}

# Panel groupings for output plots. hs_bignet (forward-KL T_c) anchors
# the methods panel as the forward-KL reference. T=2.15 and T=2.40 stay
# in the methods panel as the off-T_c controls — they make the deep-MSE
# T_c signature visible (T_c methods cluster vs the off-T_c sharp drop).
PANELS = {
    "rg_fixed_point.png": [
        "T = 2.15  (low T, ordered)",
        "T = 2.269 (T_c, hs_dataDriven)",
        "T = 2.269 (T_c, hs_bignet)",
        "T = 2.40  (high T, disorder)",
    ],
    "rg_fixed_point_methods.png": [
        "T = 2.15  (low T, ordered)",
        "T_c sym_bignet (rev-KL)",
        "T_c pathgrad_bignet_long_ext (STL)",
        "T = 2.269 (T_c, hs_bignet)",
        "T_c jsLoss_bignet_long (mixed JS)",
        "T_c phase2_finetune (mixed 2-stage)",
        "T_c bridge_w5.0t0.5 (bridge-reweighted)",
        "T = 2.40  (high T, disorder)",
    ],
}

# Linestyle / color per flow. T_c emphasised with magenta family;
# methods panel uses distinct colors per objective.
STYLE = {
    # temperature controls
    "T = 2.15  (low T, ordered)":          dict(color="#2c6cb0", linestyle="-",  marker="o"),
    "T = 2.269 (T_c, hs_dataDriven)":      dict(color="#9b1f8e", linestyle="-",  marker="s",
                                                linewidth=2.6, markersize=9),
    "T = 2.269 (T_c, hs_bignet)":          dict(color="#d36ac7", linestyle="--", marker="D",
                                                linewidth=2.2, markersize=8),
    "T = 2.40  (high T, disorder)":        dict(color="#c1311b", linestyle="-",  marker="^"),
    # methods (T_c, bignet, different training objectives)
    "T_c sym_bignet (rev-KL)":             dict(color="#1f7a3a", linestyle="-",  marker="v",
                                                linewidth=2.0, markersize=8),
    "T_c pathgrad_bignet_long_ext (STL)":  dict(color="#444444", linestyle="-",  marker="*",
                                                linewidth=2.0, markersize=11),
    "T_c jsLoss_bignet_long (mixed JS)":   dict(color="#6a3d9a", linestyle="-",  marker="P",
                                                linewidth=2.0, markersize=8),
    "T_c phase2_finetune (mixed 2-stage)": dict(color="#e08a1e", linestyle="-",  marker="X",
                                                linewidth=2.0, markersize=9),
    "T_c bridge_w5.0t0.5 (bridge-reweighted)": dict(color="#0a8aa6", linestyle="-",  marker="h",
                                                linewidth=2.0, markersize=9),
}


def latest_saving(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no .saving files in {folder}")
    return files[-1]


def load_flow(folder, device="cpu"):
    ckpt = latest_saving(folder)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state)
    fw.eval()
    # Unwrap Symmetrized -> MERA
    mera = fw.flow if hasattr(fw, "flow") else fw
    return mera, L, T, epoch, wt


def extract_scale_blocks(mera, L, kernelSize=2):
    """Group the MERA's layerList into 5 scale-blocks for L=32.

    MERA construction (train/learn.py:59): depth = log2(L) * nrepeat * 2
    -> 10 RNVP modules for L=32, nrepeat=1. Each pair (offset-0,
    offset-1) lives at one physical scale.
    """
    n_phys_scales = int(math.log(L, kernelSize))           # 5 for L=32
    layers = list(mera.layerList)
    blocks_per_scale = len(layers) // n_phys_scales
    if blocks_per_scale * n_phys_scales != len(layers):
        raise RuntimeError(
            f"layerList length {len(layers)} not divisible by "
            f"{n_phys_scales} physical scales")
    groups = []
    for s in range(n_phys_scales):
        groups.append(layers[s * blocks_per_scale:(s + 1) * blocks_per_scale])
    return groups, blocks_per_scale


def probe_scale_block(block_list, z):
    """Apply a scale-block (list of RNVP modules) to z in the
    inverse (generative) direction, in MERA's order:
        f(z) = inverse(layer[last], inverse(..., inverse(layer[0], z)))
    -- MERA.inverse iterates `for no in reversed(range(...))` and
    calls `layerList[no].inverse(...)`, so the composition order is
    reversed-index in the latent->physical pass.
    """
    out = z
    for layer in reversed(block_list):
        out, _ = layer.inverse(out)
    return out


def zscore(x, eps=1e-8):
    """Per-element z-score across the batch dimension."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def mse(a, b):
    return float(((a - b) ** 2).mean().item())


def run_one(folder, label, N=10000, seed=0, device="cpu"):
    print(f"\n=== {label}  folder={folder} ===", flush=True)
    mera, L, T, epoch, wt = load_flow(folder, device=device)
    print(f"  L={L} T={T} ep={epoch} weightTying={wt}")
    groups, bps = extract_scale_blocks(mera, L)
    print(f"  {len(groups)} scale-blocks, {bps} RNVP modules per scale")

    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2, device=device)

    # Pass z through each scale-block, z-score, store
    out_norms = []
    with torch.no_grad():
        for s, block_list in enumerate(groups):
            o = probe_scale_block(block_list, z)        # (N, 1, 2, 2)
            on = zscore(o)
            out_norms.append(on)
            print(f"  scale {s}: out_norm mean={on.mean():.3e}  "
                  f"std={on.std():.4f}", flush=True)

    # Adjacent MSE
    adj_mse = []
    for s in range(len(out_norms) - 1):
        m = mse(out_norms[s], out_norms[s + 1])
        adj_mse.append(m)
        print(f"  MSE(f_{s+1}, f_{s+2}) = {m:.4f}")

    return dict(
        label=label, folder=folder, L=L, T=T, epoch=epoch,
        weightTying=wt,
        n_scales=len(groups), blocks_per_scale=bps,
        adj_mse=adj_mse,
    )


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch",
                    "weightTying", "n_scales", "blocks_per_scale",
                    "scale_pair", "mse_norm"])
        for r in results:
            for i, m in enumerate(r["adj_mse"]):
                w.writerow([r["label"], r["folder"], r["L"], r["T"],
                            r["epoch"], r["weightTying"], r["n_scales"],
                            r["blocks_per_scale"],
                            f"f_{i+1}->f_{i+2}", m])
    print(f"wrote {path}")


def plot_panel(results_subset, savepath, title, annotate=True):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for r in results_subset:
        st = dict(STYLE[r["label"]])         # copy so we can mutate
        st.setdefault("markersize", 8)
        xs = np.arange(1, len(r["adj_mse"]) + 1)
        ax.plot(xs, r["adj_mse"], label=r["label"], **st)
    ax.set_xlabel(r"adjacent scale pair  (f_s $\to$ f_{s+1})")
    ax.set_ylabel(r"MSE on z-scored outputs")
    ax.set_title(title)
    if results_subset:
        ax.set_xticks(np.arange(1, max(len(r["adj_mse"]) for r in results_subset) + 1))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)
    if annotate:
        ax.annotate("if T_c flow is at an RG fixed point,\n"
                    "MSE should drop at deeper layers (right side)",
                    xy=(0.98, 0.95), xycoords="axes fraction",
                    fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              alpha=0.7))
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_results(results, outdir):
    by_label = {r["label"]: r for r in results}
    titles = {
        "rg_fixed_point.png":
            "RG fixed-point probe at L=32 — adjacent scale-block dissimilarity",
        "rg_fixed_point_methods.png":
            "RG probe at T_c — training-method comparison (concise_report methods, bignet arch)",
    }
    for fname, labels in PANELS.items():
        subset = [by_label[L] for L in labels if L in by_label]
        if not subset:
            print(f"  skip {fname} (no results)")
            continue
        plot_panel(subset, os.path.join(outdir, "figures", fname),
                   title=titles.get(fname, fname),
                   annotate=(fname == "rg_fixed_point.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10000, help="probe batch size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="analyzers")
    args = p.parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)

    results = []
    for label, folder in FOLDERS.items():
        try:
            r = run_one(folder, label, N=args.N, seed=args.seed)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")

    if not results:
        raise SystemExit("no successful runs")

    write_csv(results, os.path.join(args.outdir, "csv", "rg_fixed_point_summary.csv"))
    plot_results(results, args.outdir)


if __name__ == "__main__":
    main()
