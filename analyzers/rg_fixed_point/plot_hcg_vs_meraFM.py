"""Physics comparison plots for HCG-CNN champion vs MERAUNet-FM at all L.

Samples from both methods at each L, plus HS data reference, and
generates the 4 physics plots (configurations, G(r), M distribution,
observables) at figures/hcg_vs_meraFM/{L32,L64,L128}/.

HCG-CNN loading via build_flow (from flow_sample_diagnostic).
MERAUNet-FM loading via flow.flow_matching + sample_euler.
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)

from plot_L128_transfer import (
    sample_from_ckpt as sample_from_ckpt_mera,
    load_hs_data, compute_spin_observables,
    compute_gr_axial, plot_config_grid, plot_gr, plot_M_hist,
    plot_observables_bar,
)
from flow.flow_matching import MERAUNet, sample_euler


def sample_from_meraFM(folder, target_ep, L, nhidden, N=1000, batch=32,
                        n_steps=50, device="cpu"):
    """Sample from a MERAUNet-FM checkpoint (folder + epoch).

    fm_learn.py saves as `fm_L{L}_T{T}_epoch{N}.pt` containing
    {model, optimizer, epoch, config}.
    """
    ckpts = sorted(glob.glob(os.path.join(folder, "savings/fm_*_epoch*.pt")),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no FM checkpoints at {folder}/savings/")
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in ckpts]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - target_ep))
    ckpt = ckpts[idx]
    ep_actual = eps[idx]

    state = torch.load(ckpt, weights_only=False, map_location=device)
    cfg = state.get("config", {})
    if not isinstance(cfg, dict): cfg = vars(cfg) if hasattr(cfg, "__dict__") else {}
    mc = cfg.get("maxChannelMult", 4)

    net = MERAUNet(L=L, nhidden=nhidden, temb_dim=128, max_channel_mult=mc).to(device)
    net.load_state_dict(state["model"])
    net.eval()

    all_x = []
    n_done = 0
    while n_done < N:
        b = min(batch, N - n_done)
        with torch.no_grad():
            x = sample_euler(net, (b, 1, L, L), n_steps=n_steps, device=device)
        all_x.append(x.cpu())
        n_done += b
    x = torch.cat(all_x, 0)
    return x, L, 2.269185314213022, ep_actual


def run_comparison(L, hcg_folder, hcg_ep, fm_folder, fm_ep, fm_nhidden,
                    out_dir, N, batch, device):
    """Generate 4 plots for HCG-CNN vs FM comparison at given L."""
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    print(f"\n[L={L}] sampling from HCG-CNN...")
    try:
        x, _, _, ep = sample_from_ckpt_mera(hcg_folder, hcg_ep,
                                              N=N, batch=batch, device=device)
        print(f"  HCG-CNN ep {ep}: {x.shape[0]} samples")
        samples.append((f"HCG-CNN L={L} ep{ep}", x, ep))
    except Exception as e:
        print(f"  HCG-CNN failed: {e}")

    print(f"[L={L}] sampling from MERAUNet-FM...")
    try:
        x, _, _, ep = sample_from_meraFM(fm_folder, fm_ep, L=L,
                                           nhidden=fm_nhidden,
                                           N=N, batch=batch, device=device)
        print(f"  FM ep {ep}: {x.shape[0]} samples")
        samples.append((f"MERAUNet-FM L={L} ep{ep}", x, ep))
    except Exception as e:
        print(f"  FM failed: {e}")

    print(f"[L={L}] loading HS data reference...")
    try:
        x_data = load_hs_data(L, 2.269185314213022, N=N)
        samples.append((f"HS data (GT) L={L}", x_data, "GT"))
    except Exception as e:
        print(f"  HS data failed: {e}")

    if len(samples) < 2:
        print(f"[L={L}] too few samples ({len(samples)}) — skipping plots")
        return

    print(f"[L={L}] computing observables + G(r)...")
    obs = []
    for label, x, ep in samples:
        absM, chi, U4, M = compute_spin_observables(x)
        obs.append((label, absM, chi, U4, M))
        print(f"  {label}: |M|={absM:.4f}  χ={chi:.3f}  U4={U4:.4f}")

    grs = []
    styles = ["-", "--", "-.", ":"]
    for i, (label, x, ep) in enumerate(samples):
        G = compute_gr_axial(x)
        grs.append((label, G, styles[i % len(styles)]))

    # Extract gt from last cell if HS data present
    has_gt = "HS data" in samples[-1][0]
    if has_gt:
        gt_absM, gt_chi, gt_U4 = obs[-1][1], obs[-1][2], obs[-1][3]
    else:
        gt_absM = gt_chi = gt_U4 = float("nan")

    print(f"[L={L}] plotting...")
    plot_config_grid([(l, x, ep) for l, x, ep in samples],
                     os.path.join(out_dir, "configurations.png"), n_samples=4)
    plot_gr(grs, os.path.join(out_dir, "two_point_correlation.png"), L)

    palette = ["steelblue", "darkorange", "black", "seagreen"]
    hist_data = []
    for i, (label, x, ep) in enumerate(samples):
        M = obs[i][4].numpy()
        hist_data.append((label, M, palette[i % len(palette)]))
    plot_M_hist(hist_data, os.path.join(out_dir, "M_distribution.png"), gt_absM)

    obs_data = [(l, a, c, u) for (l, a, c, u, _) in obs]
    plot_observables_bar(obs_data, os.path.join(out_dir, "observables.png"),
                          (gt_absM, gt_chi, gt_U4))

    # Summary JSON
    summary = {
        "L": int(L),
        "gt": {"absM": gt_absM, "chi": gt_chi, "U4": gt_U4},
        "cells": [dict(label=l, epoch=ep, absM=a, chi=c, U4=u)
                  for (l, x, ep), (l2, a, c, u, _) in zip(samples, obs)],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[L={L}] wrote {out_dir}/*.png + summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="figures/hcg_vs_meraFM")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # L=32
    run_comparison(
        L=32,
        hcg_folder="data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64",
        hcg_ep=9500,
        fm_folder="data/L32_T2.269_meraFM_h64",
        fm_ep=1200,
        fm_nhidden=64,
        out_dir=os.path.join(args.out, "L32"),
        N=args.N, batch=args.batch, device=args.device,
    )

    # L=64
    run_comparison(
        L=64,
        hcg_folder="data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16",
        hcg_ep=9500,
        fm_folder="data/L64_T2.269_meraFM_h128",
        fm_ep=999999,  # latest
        fm_nhidden=128,
        out_dir=os.path.join(args.out, "L64"),
        N=args.N, batch=args.batch, device=args.device,
    )

    # L=128
    run_comparison(
        L=128,
        hcg_folder="data/L128_T2.269_champion_from_L64",
        hcg_ep=5000,
        fm_folder="data/L128_T2.269_meraFM_h128_v2",
        fm_ep=999999,  # latest
        fm_nhidden=128,
        out_dir=os.path.join(args.out, "L128"),
        N=args.N, batch=args.batch, device=args.device,
    )


if __name__ == "__main__":
    main()
