"""Generate plots for the L=64→L=128 transfer report.

Produces:
  1. Configuration grid — sample lattices from warm/fresh at multiple epochs
  2. Two-point correlation G(r) — warm/fresh vs L=128 HS data
  3. Magnetization distribution — histogram of per-sample |M| for each
  4. Physical observables bar chart — χ, U4, |M| vs GT

Output goes to figures/L128_transfer/*.png
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_sample_diagnostic import build_flow


def sample_from_ckpt(folder, target_ep, N=1000, batch=32, device="cpu"):
    """Sample from a flow at nearest checkpoint to target_ep."""
    ckpts = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in ckpts]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - target_ep))
    ckpt = ckpts[idx]
    ep_actual = eps[idx]

    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, _, L, T, sym, wt, hp = build_flow(folder, state, device=device)
    fw.eval()

    # Sigma for un-standardization
    sigma = 1.0
    ps = os.path.join(folder, "flow_input_sigma.json")
    if os.path.exists(ps):
        with open(ps) as f:
            sigma = float(json.load(f).get("sigma", 1.0))

    all_x = []
    n_done = 0
    while n_done < N:
        b = min(batch, N - n_done)
        with torch.no_grad():
            u, _ = fw.sample(b)
            x = sigma * u  # physical HS field
        all_x.append(x.cpu())
        n_done += b
    x = torch.cat(all_x, 0)
    return x, L, T, ep_actual


def load_hs_data(L, T, N=1000):
    pat = f"./data/mcmc_data/hs_L{L}_T{T:.15f}_N*.pt"
    cand = sorted(glob.glob(pat),
                  key=lambda s: int(s.split("_N")[-1].split(".")[0]),
                  reverse=True)
    if not cand:
        raise FileNotFoundError(f"no HS at L={L}, T={T}")
    x = torch.load(cand[0], weights_only=False, map_location="cpu").float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x[:N]


def compute_spin_observables(x):
    """|M|, χ, U4 from HS field via sign()."""
    N = x.shape[-1] * x.shape[-2]
    s = torch.sign(x)
    M = s.reshape(x.shape[0], -1).mean(dim=-1)
    absM = M.abs().mean().item()
    M2 = M.pow(2).mean().item()
    M4 = M.pow(4).mean().item()
    chi = N * (M2 - absM ** 2)
    U4 = 1.0 - M4 / (3 * M2 ** 2)
    return absM, chi, U4, M


def compute_gr_axial(x):
    """Axial two-point correlation G(r) = <s_0 s_r> averaged over samples."""
    B, _, L, _ = x.shape
    s = torch.sign(x).squeeze(1)   # (B, L, L), ±1
    # G(r) = <s_i s_{i+r}> averaged over i (position) and B (samples)
    # Compute horizontal + vertical then average
    r_max = L // 2
    G = np.zeros(r_max + 1)
    for r in range(r_max + 1):
        Gh = (s * torch.roll(s, r, dims=-1)).mean().item()
        Gv = (s * torch.roll(s, r, dims=-2)).mean().item()
        G[r] = 0.5 * (Gh + Gv)
    return G


def plot_config_grid(cells, savepath, n_samples=3, spin_binarize=True):
    """3xN grid: rows=different cells, cols=different sample indices."""
    n_cells = len(cells)
    fig, axes = plt.subplots(n_cells, n_samples,
                             figsize=(2.5 * n_samples, 2.7 * n_cells),
                             squeeze=False)
    for i, (label, x, ep) in enumerate(cells):
        for j in range(n_samples):
            if spin_binarize:
                config = torch.sign(x[j, 0]).numpy()
                vmin, vmax = -1, 1
                cmap = "gray"
            else:
                config = x[j, 0].numpy()
                lim = float(np.abs(config).max())
                vmin, vmax = -lim, lim
                cmap = "RdBu_r"
            axes[i, j].imshow(config, cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation="nearest")
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=10)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"sample #{j}", fontsize=9)
    fig.suptitle("Sample configurations (spins via sign(x))",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {savepath}")


def plot_gr(cells_with_gr, savepath, L):
    """G(r) curves — one line per cell."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, G, style in cells_with_gr:
        r = np.arange(len(G))
        ax.semilogy(r, np.abs(G), style, label=label, linewidth=1.5)
    ax.set_xlabel("r (axial)")
    ax.set_ylabel("|G(r)| = |<s_0 s_r>|")
    ax.set_title(f"Two-point correlation at L={L} T_c (log scale)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, L // 2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {savepath}")


def plot_M_hist(cells_with_M, savepath, gt_absM):
    """Histogram of per-sample M for each cell + GT indicator."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, M_arr, color in cells_with_M:
        ax.hist(M_arr, bins=60, alpha=0.4, label=label, color=color, density=True)
    ax.axvline(gt_absM, color="red", linestyle="--", linewidth=1.5,
               label=f"GT |M|={gt_absM:.3f}")
    ax.axvline(-gt_absM, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("M (per-sample magnetization = mean sign(x))")
    ax.set_ylabel("density")
    ax.set_title("Magnetization distribution (Z2 bimodal for critical Ising)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {savepath}")


def plot_observables_bar(cells_with_obs, savepath, gt):
    """Bar chart of |M|, χ, U4 for each cell + GT reference line."""
    labels = [c[0] for c in cells_with_obs]
    absM = [c[1] for c in cells_with_obs]
    chi  = [c[2] for c in cells_with_obs]
    U4   = [c[3] for c in cells_with_obs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))

    for ax, (values, name, gt_val) in zip(axes,
        [(absM, "|M|", gt[0]), (chi, "χ", gt[1]), (U4, "U₄", gt[2])]):
        ax.bar(x, values, color=colors)
        ax.axhline(gt_val, color="red", linestyle="--", linewidth=1.5, label=f"GT={gt_val:.3f}")
        ax.set_ylabel(name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {savepath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000, help="samples per cell")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="figures/L128_transfer")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------- gather cells ----------
    print("[sample] drawing samples from each cell...")

    cells = [
        ("L=128 warm ep 200",  "data/L128_T2.269_champion_from_L64", 200),
        ("L=128 warm ep 5000", "data/L128_T2.269_champion_from_L64", 5000),
        ("L=128 fresh ep 200", "data/L128_T2.269_champion_freshInit", 200),
        ("L=128 fresh ep 8000","data/L128_T2.269_champion_freshInit", 8000),
    ]
    samples = []
    for label, folder, ep in cells:
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
    # HS data reference
    print(f"[data] loading L={L} HS reference")
    x_data = load_hs_data(L, T, N=args.N)
    samples.append((f"L={L} HS data (reference)", x_data, "GT"))

    # ---------- compute observables ----------
    print("[compute] observables per cell")
    obs = []
    for label, x, ep in samples:
        absM, chi, U4, M = compute_spin_observables(x)
        obs.append((label, absM, chi, U4, M))
        print(f"  {label}: |M|={absM:.4f}  χ={chi:.3f}  U₄={U4:.4f}")

    # GT reference (compute from actual HS data)
    gt_absM = obs[-1][1]; gt_chi = obs[-1][2]; gt_U4 = obs[-1][3]
    print(f"[GT ref] |M|={gt_absM:.4f}  χ={gt_chi:.3f}  U₄={gt_U4:.4f}")

    # ---------- G(r) ----------
    print("[compute] G(r) per cell")
    grs = []
    styles = ["-", "--", "-.", ":", "-"]
    for i, (label, x, ep) in enumerate(samples):
        G = compute_gr_axial(x)
        style = styles[i % len(styles)]
        grs.append((label, G, style))

    # ---------- plots ----------
    print("[plot] generating figures...")
    plot_config_grid([(l, x, ep) for l, x, ep in samples],
                     os.path.join(args.out, "configurations.png"),
                     n_samples=4)
    plot_gr(grs, os.path.join(args.out, "two_point_correlation.png"), L)

    # M histogram — sample colors
    hist_data = []
    palette = ["steelblue", "darkorange", "seagreen", "crimson", "black"]
    for i, (label, x, ep) in enumerate(samples):
        M = obs[i][4].numpy()
        hist_data.append((label, M, palette[i % len(palette)]))
    plot_M_hist(hist_data, os.path.join(args.out, "M_distribution.png"), gt_absM)

    # Observables bar chart
    obs_data = [(l, a, c, u) for (l, a, c, u, _) in obs]
    plot_observables_bar(obs_data, os.path.join(args.out, "observables.png"),
                          (gt_absM, gt_chi, gt_U4))

    # ---------- save numeric summary ----------
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

    print("Done.")


if __name__ == "__main__":
    main()
