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
    # ----- L=64 T_c comparison (2026-07-13, champion added) -----
    "L=64 baseline_b16 (A nr=1)":                 "data/64Ising_T2.269_hsBignet_baseline_b16",
    "L=64 i2_stride8h32_nr2_b16 (D nr=2)":        "data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16",
    "L=64 fixdil+VP-1e-3 nr=1 (champion ★)":      "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16",
    # ----- L=64 cross-T comparison (2026-07-14) -----
    # nr=2 models trained across the transition. All three at the same
    # architecture; only T changes. Answers "is V0/V1 deep-MSE a T_c
    # fingerprint at L=64 like it is at L=32?"
    "L=64 T2.15 baseline_nr2 (C, ordered)":         "data/64Ising_T2.15_hsBignet_baseline_nr2_b16",
    "L=64 T2.22 baseline_nr2 (C, near T_c)":         "data/64Ising_T2.22_hsBignet_baseline_nr2_b16",
    "L=64 T2.32 baseline_nr2 (C, near T_c)":         "data/64Ising_T2.32_hsBignet_baseline_nr2_b16",
    "L=64 T2.4  baseline_nr2 (C, disordered)":       "data/64Ising_T2.4_hsBignet_baseline_nr2_b16",
    "L=64 T2.15 i2_stride8h32_nr2 (D, ordered)":     "data/64Ising_T2.15_hsBignet_i2_stride8h32_nr2_b16",
    "L=64 T2.22 i2_stride8h32_nr2 (D, near T_c)":     "data/64Ising_T2.22_hsBignet_i2_stride8h32_nr2_b16",
    "L=64 T2.32 i2_stride8h32_nr2 (D, near T_c)":     "data/64Ising_T2.32_hsBignet_i2_stride8h32_nr2_b16",
    "L=64 T2.4  i2_stride8h32_nr2 (D, disordered)":   "data/64Ising_T2.4_hsBignet_i2_stride8h32_nr2_b16",
    "L=64 T2.15 fixdil+VP-1e-3 nr=2 (VP, ordered)":     "data/64Ising_T2.15_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16",
    "L=64 T2.22 fixdil+VP-1e-3 nr=2 (VP, near T_c)":     "data/64Ising_T2.22_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16",
    "L=64 T2.32 fixdil+VP-1e-3 nr=2 (VP, near T_c)":     "data/64Ising_T2.32_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16",
    "L=64 T2.4  fixdil+VP-1e-3 nr=2 (VP, disordered)":   "data/64Ising_T2.4_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16",
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
    # L=64 champion vs L=64 baselines (2026-07-13). Answers: does the
    # VP-regularized fixdil+HCG champion show a different depth-MSE
    # signature than baseline A or the Phase-2 D reference?
    "rg_fixed_point_L64_champion.png": [
        "L=64 baseline_b16 (A nr=1)",
        "L=64 i2_stride8h32_nr2_b16 (D nr=2)",
        "L=64 fixdil+VP-1e-3 nr=1 (champion ★)",
    ],
    # L=64 cross-T for D (Phase-2 reference architecture — same config,
    # only T varies). Answers: is V0/V1 deep-MSE a T_c fingerprint at
    # L=64 like it is at L=32 (per the T=2.15 / T=2.269 / T=2.4 baseline
    # panel)?
    "rg_fixed_point_L64_across_T_D.png": [
        "L=64 T2.15 i2_stride8h32_nr2 (D, ordered)",
        "L=64 T2.22 i2_stride8h32_nr2 (D, near T_c)",
        "L=64 i2_stride8h32_nr2_b16 (D nr=2)",   # T_c row
        "L=64 T2.32 i2_stride8h32_nr2 (D, near T_c)",
        "L=64 T2.4  i2_stride8h32_nr2 (D, disordered)",
    ],
    # Same for the VP-regularized champion-analog (fixdil+VP-1e-3 nr=2).
    "rg_fixed_point_L64_across_T_VP.png": [
        "L=64 T2.15 fixdil+VP-1e-3 nr=2 (VP, ordered)",
        "L=64 T2.22 fixdil+VP-1e-3 nr=2 (VP, near T_c)",
        "L=64 T2.32 fixdil+VP-1e-3 nr=2 (VP, near T_c)",
        "L=64 T2.4  fixdil+VP-1e-3 nr=2 (VP, disordered)",
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
    # ----- Phase-1 improvement ablation (L=32 b=64) -----
    "L=32 baseline_b64":                        dict(color="#888888", linestyle="-",  marker="o"),
    "L=32 iii1_lam1.0_b64 (+III.1)":            dict(color="#1565c0", linestyle="-",  marker="^"),
    "L=32 i2_stride8h32_b64 (+I.2 cond)":       dict(color="#2e7d32", linestyle="-",  marker="s"),
    "L=32 combined_lam1.0_b64 (I.2+III.1)":     dict(color="#6a1b9a", linestyle="-",  marker="D"),
    # L=32 sweeps
    "L=32 iii1_lam0.1_b64":                     dict(color="#90caf9", linestyle="--", marker="^"),
    "L=32 iii1_lam10.0_b64":                    dict(color="#0d47a1", linestyle=":",  marker="^"),
    "L=32 i2_stride4h32 (b128)":                dict(color="#1b5e20", linestyle="--", marker="s"),
    "L=32 i2_stride16h32 (b128)":               dict(color="#a5d6a7", linestyle=":",  marker="s"),
    "L=32 i1_df4.0 (Student-t b128)":           dict(color="#ef6c00", linestyle="-",  marker="v"),
    # ----- Phase-1 improvement ablation (L=64 b=16) -----
    "L=64 baseline_b16":                        dict(color="#bdbdbd", linestyle="-",  marker="o"),
    "L=64 iii1_lam1.0_b16 (+III.1)":            dict(color="#42a5f5", linestyle="-",  marker="^"),
    "L=64 i2_stride16h32_b16 (+I.2)":           dict(color="#66bb6a", linestyle="-",  marker="s"),
    "L=64 i1_df4.0_b16 (Student-t)":            dict(color="#ffa726", linestyle="-",  marker="v"),
    # ----- Phase-2 I.2 capacity scan -----
    "L=32 i2_stride4h32_b64 (Phase-2)":          dict(color="#388e3c", linestyle="--", marker="s"),
    "L=32 i2_stride8h64_b64 (Phase-2)":          dict(color="#1b5e20", linestyle=":",  marker="s"),
    "L=64 i2_stride8h32_b16 (Phase-2)":          dict(color="#81c784", linestyle="-",  marker="s"),
    "L=64 i2_stride4h32_b16 (Phase-2)":          dict(color="#43a047", linestyle="--", marker="s"),
    "L=64 i2_stride8h64_b16 (Phase-2)":          dict(color="#2e7d32", linestyle="-",  marker="D"),
    "L=64 i2_stride4h64_b16 (Phase-2)":          dict(color="#1b5e20", linestyle="--", marker="D"),
    # ----- Phase-2 P2.x: i2 + nrepeat=2 (2026-06-25, winner D64) -----
    "L=32 i2_stride8h32_nr2_b64 (P2.x D32)":     dict(color="#c2185b", linestyle="-",  marker="*",
                                                     linewidth=2.4, markersize=12),
    "L=64 baseline_nr2_b16 (P2.x C64)":           dict(color="#ad1457", linestyle=":",  marker="x",
                                                     linewidth=2.0, markersize=10),
    "L=64 i2_stride8h32_nr2_b16 (P2.x D64 ★)":   dict(color="#880e4f", linestyle="-",  marker="*",
                                                     linewidth=2.8, markersize=14),
    # ----- L=64 champion panel (2026-07-13) -----
    "L=64 baseline_b16 (A nr=1)":                 dict(color="#bdbdbd", linestyle="-",  marker="o",
                                                     linewidth=2.0, markersize=9),
    "L=64 i2_stride8h32_nr2_b16 (D nr=2)":        dict(color="#880e4f", linestyle="--", marker="s",
                                                     linewidth=2.2, markersize=9),
    "L=64 fixdil+VP-1e-3 nr=1 (champion ★)":      dict(color="#0a8aa6", linestyle="-",  marker="*",
                                                     linewidth=2.8, markersize=14),
    # ----- L=64 cross-T (2026-07-14): shared marker per model type, color = T -----
    # D nr=2 across T (marker "s")
    "L=64 T2.15 i2_stride8h32_nr2 (D, ordered)":     dict(color="#2c6cb0", linestyle="-",  marker="s"),
    "L=64 T2.22 i2_stride8h32_nr2 (D, near T_c)":     dict(color="#7d5bcc", linestyle="-",  marker="s"),
    "L=64 T2.32 i2_stride8h32_nr2 (D, near T_c)":     dict(color="#c05a95", linestyle="-",  marker="s"),
    "L=64 T2.4  i2_stride8h32_nr2 (D, disordered)":   dict(color="#c1311b", linestyle="-",  marker="s"),
    # champion-analog VP nr=2 across T (marker "*")
    "L=64 T2.15 fixdil+VP-1e-3 nr=2 (VP, ordered)":     dict(color="#2c6cb0", linestyle="-",  marker="*", linewidth=2.0, markersize=11),
    "L=64 T2.22 fixdil+VP-1e-3 nr=2 (VP, near T_c)":     dict(color="#7d5bcc", linestyle="-",  marker="*", linewidth=2.0, markersize=11),
    "L=64 T2.32 fixdil+VP-1e-3 nr=2 (VP, near T_c)":     dict(color="#c05a95", linestyle="-",  marker="*", linewidth=2.0, markersize=11),
    "L=64 T2.4  fixdil+VP-1e-3 nr=2 (VP, disordered)":   dict(color="#c1311b", linestyle="-",  marker="*", linewidth=2.0, markersize=11),
    # C = baseline_nr2 across T (marker "o"). T_c already covered as "L=64 baseline_nr2_b16 (P2.x C64)".
    "L=64 T2.15 baseline_nr2 (C, ordered)":         dict(color="#2c6cb0", linestyle=":",  marker="o"),
    "L=64 T2.22 baseline_nr2 (C, near T_c)":         dict(color="#7d5bcc", linestyle=":",  marker="o"),
    "L=64 T2.32 baseline_nr2 (C, near T_c)":         dict(color="#c05a95", linestyle=":",  marker="o"),
    "L=64 T2.4  baseline_nr2 (C, disordered)":       dict(color="#c1311b", linestyle=":",  marker="o"),
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
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
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
    p.add_argument("--filter", default=None,
                   help="Regex; keep only FOLDERS labels matching. Speeds "
                        "up scoped reruns (e.g. --filter 'champion|A nr=1' "
                        "or --filter 'L=64'). Case-insensitive.")
    args = p.parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)

    folders = FOLDERS
    if args.filter:
        pat = re.compile(args.filter, re.IGNORECASE)
        folders = {k: v for k, v in FOLDERS.items() if pat.search(k)}
        print(f"[filter] '{args.filter}' matched {len(folders)}/{len(FOLDERS)} labels:")
        for k in folders:
            print(f"    {k}")

    results = []
    for label, folder in folders.items():
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
