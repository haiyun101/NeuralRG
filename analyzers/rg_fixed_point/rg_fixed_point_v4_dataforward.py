"""V4 — RG self-similarity probe using HS data and the forward
(analysis) direction, restricted to the kept-coarse sub-lattice.

Methodology
-----------
Original probe (rg_fixed_point.py) feeds N(0, I) noise into each
scale-block IN ISOLATION on a (N, 1, 2, 2) patch and goes
generatively (z -> x via layer.inverse). Two issues followed:

  - Probe input distribution does not match the production slow-mode
    distribution that each block actually receives in the forward
    pass.
  - Identity-triviality on the standard-Gaussian patch: two
    near-identity-on-N(0, I) blocks trivially produce matching
    outputs on any z ~ N(0, I), but this says nothing about what
    they do on the wider slow-mode statistics they see in
    production.

V4 fixes both by using real HS data and the forward (analysis)
direction -- the way the flow processes a sample during training:

    x ~ p_HS  ->  f_1.forward  -> y_1  ->  f_2.forward  -> y_2 -> ... -> y_S
                  (finest)                                      (latent prior)

where f_s is the s-th scale-block in the forward direction
(layerList[2(s-1)..2s-1]) and y_s is the field after s
coarse-graining scale-blocks have been applied to x. Each y_s
lives on the L x L tensor (MERA is a bijection on the lattice).

KEPT-COARSE RESTRICTION (this is the v4-corrected core).
---------------------------------------------------------
In the MERA forward direction, scale-block s reads patches at
stride 2^s positions and writes back to the SAME L x L tensor. The
positions NOT read by any later scale-block (i.e., those outside
the stride-2^(s+1) sub-lattice) become "frozen latents" after
scale s: the optimiser drives them toward N(0, 1) and they never
get touched again. Only the (L/2^s)^2 positions on the stride-2^s
sub-lattice carry the slow-mode signal that flows into the next
scale-block.

If we compute moments / G(r) / KS / W1 over all L^2 positions of
y_s, the (1 - 1/4^s) frozen-latent component dominates the
average. For a rev-KL flow whose frozen latents are well-trained
to N(0, 1), naive V4 reports "std(y_5) ~ 1.1, kurt ~ 0" -- which
is the latent prior, not the slow mode.

At each scale s, V4-corrected extracts only the stride-2^s
sub-lattice (offset (0, 0)) before computing any statistic:

    y_s_coarse = y_s[..., ::2^s, ::2^s]      # shape (B, 1, L/2^s, L/2^s)

All moments, G_s(r), KS, W1, and rmsG are then on this coarse
field.

What we compare
---------------
At each scale s:

  - Marginal moments of the slow mode (mean / std / kurt / skew)
  - Axial two-point G_s(r) for r = 1..L_s/2  (L_s = L/2^s)
  - Adjacent-pair distance between y_s_coarse and y_{s+1}_coarse:
      * KS / W1 on standardised marginals
      * RMS deviation of G(r)/G(0), restricted to
        r = 1..min(L_s, L_{s+1})/2

At a true RG fixed point at T_c, the slow mode should be
distributionally self-similar across adjacent scales (when each is
measured in its own coarse units).

Identity-triviality is not a confound on the coarse field: the
slow-mode distribution is unconstrained by N(0, I), so coincidence
across scales requires the flow to actually reproduce the coarse
field's statistics.

We probe the same set of flows as the V1/V2/V3 robustness script.

Usage:
  python analyzers/rg_fixed_point_v4_dataforward.py [--N 2000]

Note: this script overwrites the previous V4 outputs. The original
V4 (averaging over all L^2 positions) is retracted -- see the
report's V4 section for the correction note.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import build_flow
from rg_fixed_point import (
    STYLE as ORIG_STYLE,
    latest_saving,
)
from flow.hierarchy.im2col import dispatch, collect


# Reuse the same six flows from rg_fixed_point_robustness.py
FOLDERS = {
    "T = 2.15  (low T, ordered)":       "data/32Ising_T2.15_hs_dataDriven",
    "T = 2.269 (T_c, hs_dataDriven)":   "data/32Ising_T2.269185314213022_hs_dataDriven",
    "T = 2.269 (T_c, hs_bignet)":       "data/32Ising_T2.269_hs_bignet",
    "T_c sym_bignet (rev-KL)":          "data/32Ising_T2.269_sym_bignet",
    "T_c pathgrad_bignet_long_ext (STL)": "data/32Ising_T2.269_pathgrad_bignet_long_ext",
    "T = 2.40  (high T, disorder)":     "data/32Ising_T2.4_hs_dataDriven",
    # ----- Phase-1 improvement ablation (L=32 b=64) -----
    "L=32 baseline_b64":                 "data/32Ising_T2.269_hsBignet_baseline_b64",
    "L=32 iii1_lam1.0_b64 (+III.1)":     "data/32Ising_T2.269_hsBignet_iii1_lam1.0_b64",
    "L=32 i2_stride8h32_b64 (+I.2 cond)":"data/32Ising_T2.269_hsBignet_i2_stride8h32_b64",
    "L=32 combined_lam1.0_b64 (I.2+III.1)": "data/32Ising_T2.269_hsBignet_combined_lam1.0_stride8h32_b64",
    # L=32 sweeps (exclude broken i2_stride8h32 b=128 -- late-training divergence)
    "L=32 iii1_lam0.1_b64":              "data/32Ising_T2.269_hsBignet_iii1_lam0.1_b64",
    "L=32 iii1_lam10.0_b64":             "data/32Ising_T2.269_hsBignet_iii1_lam10.0_b64",
    "L=32 i2_stride4h32 (b128)":         "data/32Ising_T2.269_hsBignet_i2_stride4h32",
    "L=32 i2_stride16h32 (b128)":        "data/32Ising_T2.269_hsBignet_i2_stride16h32",
    "L=32 i1_df4.0 (Student-t b128)":    "data/32Ising_T2.269_hsBignet_i1_df4.0",
    # ----- Phase-1 improvement ablation (L=64 b=16) -----
    "L=64 baseline_b16":                 "data/64Ising_T2.269_hsBignet_baseline_b16",
    "L=64 iii1_lam1.0_b16 (+III.1)":     "data/64Ising_T2.269_hsBignet_iii1_lam1.0_b16",
    "L=64 i2_stride16h32_b16 (+I.2)":    "data/64Ising_T2.269_hsBignet_i2_stride16h32_b16",
    "L=64 i1_df4.0_b16 (Student-t)":     "data/64Ising_T2.269_hsBignet_i1_df4.0_b16",
    # ----- Phase-2 I.2 capacity scan -----
    "L=32 i2_stride4h32_b64 (Phase-2)":  "data/32Ising_T2.269_hsBignet_i2_stride4h32_b64",
    "L=32 i2_stride8h64_b64 (Phase-2)":  "data/32Ising_T2.269_hsBignet_i2_stride8h64_b64",
    "L=64 i2_stride8h32_b16 (Phase-2)":  "data/64Ising_T2.269_hsBignet_i2_stride8h32_b16",
    "L=64 i2_stride4h32_b16 (Phase-2)":  "data/64Ising_T2.269_hsBignet_i2_stride4h32_b16",
    "L=64 i2_stride8h64_b16 (Phase-2)":  "data/64Ising_T2.269_hsBignet_i2_stride8h64_b16",
    "L=64 i2_stride4h64_b16 (Phase-2)":  "data/64Ising_T2.269_hsBignet_i2_stride4h64_b16",
}
STYLE = {k: ORIG_STYLE[k] for k in FOLDERS}


def load_flow(folder, device="cpu"):
    ckpt = latest_saving(folder)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    # Unwrap Symmetrized -> MERA
    mera = fw.flow if hasattr(fw, "flow") else fw
    return mera, L, T, epoch, wt


def load_hs_data(L, T, N, device="cpu"):
    """Pick the HS dataset whose T value is closest to the flow's T.

    Parameter T can be a rounded float (2.269) while the data file
    uses higher precision (2.269185314213022). A lexicographic glob
    fallback picks the wrong T (e.g. T=2.4 alphabetically sorts after
    T=2.269...). Match by numerical distance instead.
    """
    candidates = sorted(glob.glob(f"data/mcmc_data/hs_L{L}_T*_N*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no HS dataset matching L={L} under data/mcmc_data/")
    def t_of(path):
        m = re.search(r"_T([0-9.]+)_N", path)
        return float(m.group(1)) if m else float("inf")
    target_T = float(T)
    path = min(candidates, key=lambda p: abs(t_of(p) - target_T))
    actual_T = t_of(path)
    if abs(actual_T - target_T) > 0.01:
        raise FileNotFoundError(
            f"no HS dataset within 0.01 of T={target_T} for L={L}; "
            f"closest is T={actual_T} at {path}")
    print(f"  loading HS data from {path}  (file T={actual_T}, flow T={target_T})",
          flush=True)
    samples = torch.load(path, map_location=device)
    if isinstance(samples, dict):
        samples = samples.get("data", samples.get("x", samples))
    samples = samples[:N]
    return samples.to(device=device, dtype=torch.float32)


def keep_coarse_subsample(y_s, s):
    """Extract the kept-coarse sub-lattice from y_s.

    After s MERA forward scale-blocks, the positions on the
    stride-2^s sub-lattice (offset (0, 0)) are the slow modes that
    feed the next scale-block. The other (1 - 1/4^s) positions are
    frozen latents already pushed toward the N(0, 1) prior. Only
    the slow modes carry the coarse-grained physics.

    y_s: (B, 1, L, L)
    Returns (B, 1, L/2^s, L/2^s) for s >= 0.
    """
    if s == 0:
        return y_s
    stride = 2 ** s
    return y_s[..., ::stride, ::stride].contiguous()


def forward_through_scale_block(x, indI_list, indJ_list, layer_list,
                                kernelShape, channelSize):
    """Apply a scale-block's layers to x using MERA's dispatch/collect.

    Each call advances the field by one physical scale -- the analogue
    of running the analysis (forward) direction one coarse-graining
    step.
    """
    for i in range(len(layer_list)):
        x, x_ = dispatch(indI_list[i], indJ_list[i], x)
        x_, _ = layer_list[i].forward(
            x_.reshape(-1, channelSize, *kernelShape))
        x = collect(indI_list[i], indJ_list[i], x, x_)
    return x


def two_point_axial(x):
    """G(r) along x-axis, averaged over rows and channel.

    x: (B, 1, L, L). Returns G shape (L,) where G[r] = <x_{ij} x_{i,j+r}>
    averaged over (B, i, j).
    """
    B, C, L, _ = x.shape
    xx = x.squeeze(1)             # (B, L, L)
    G = torch.zeros(L)
    for r in range(L):
        if r == 0:
            G[r] = (xx * xx).mean()
        else:
            G[r] = (xx[:, :, :L - r] * xx[:, :, r:]).mean()
    return G.numpy()


def field_stats(x):
    """Marginal moments of a flattened 2D-field batch."""
    flat = x.flatten().numpy()
    return dict(
        mean=float(flat.mean()),
        std=float(flat.std()),
        var=float(flat.var()),
        kurtosis=float(sps.kurtosis(flat)),
        skew=float(sps.skew(flat)),
    )


def adjacent_distances(coarse_per_scale):
    """Compute KS, W1, and G(r)/G(0) RMS distance for each adjacent
    pair (y_s_coarse, y_{s+1}_coarse).

    Inputs are ALREADY restricted to the kept-coarse sub-lattice --
    y_s_coarse has shape (B, 1, L/2^s, L/2^s). Field marginals are
    standardised before comparison so KS / W1 measure shape.

    For G(r): each y_s_coarse has its own coarse-lattice spacing,
    so G_s(r=1) is at physical distance 2^s. The cleanest RG
    invariance check is "do the *coarse-unit* correlation shapes
    match?" -- compare G_s(r)/G_s(0) and G_{s+1}(r)/G_{s+1}(0) at
    matching coarse-unit r values, using r = 1..min(L_s, L_{s+1})/2.
    Where the deeper field has L_{s+1} < 4 (no usable r), report
    NaN.
    """
    keys = sorted(coarse_per_scale.keys())
    out = []
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    rng_c = np.random.default_rng(2)
    rng_d = np.random.default_rng(3)
    for k1, k2 in zip(keys[:-1], keys[1:]):
        y1 = coarse_per_scale[k1].flatten().numpy()
        y2 = coarse_per_scale[k2].flatten().numpy()
        z1 = (y1 - y1.mean()) / (y1.std() + 1e-12)
        z2 = (y2 - y2.mean()) / (y2.std() + 1e-12)
        # KS / W1: sub-sample to common, bounded size
        n_keep = min(200_000, z1.size, z2.size)
        ridx1 = rng_a.choice(z1.size, n_keep, replace=False)
        ridx2 = rng_b.choice(z2.size, n_keep, replace=False)
        ks_stat = float(sps.ks_2samp(z1[ridx1], z2[ridx2]).statistic)
        n_w = min(20_000, z1.size, z2.size)
        ridx3 = rng_c.choice(z1.size, n_w, replace=False)
        ridx4 = rng_d.choice(z2.size, n_w, replace=False)
        w1 = float(sps.wasserstein_distance(z1[ridx3], z2[ridx4]))
        # G(r) on the coarse field of each scale. Match on r values
        # available in BOTH coarse lattices.
        L1 = coarse_per_scale[k1].shape[-1]
        L2 = coarse_per_scale[k2].shape[-1]
        L_min = min(L1, L2)
        if L_min >= 4:
            G1 = two_point_axial(coarse_per_scale[k1])
            G2 = two_point_axial(coarse_per_scale[k2])
            r_max = L_min // 2
            g1n = G1[: r_max + 1] / (G1[0] + 1e-12)
            g2n = G2[: r_max + 1] / (G2[0] + 1e-12)
            rms_g = float(np.sqrt(((g1n - g2n) ** 2).mean()))
        else:
            rms_g = float("nan")
        out.append(dict(
            s=k1, s_next=k2,
            ks=ks_stat, w1=w1, rms_g=rms_g,
            L_s=L1, L_s_next=L2,
        ))
    return out


def run_one(folder, label, N=2000, device="cpu"):
    print(f"\n=== {label}  folder={folder} ===", flush=True)
    mera, L, T, epoch, wt = load_flow(folder, device=device)
    print(f"  L={L} T={T} ep={epoch} weightTying={wt}")
    # Load HS data
    samples = load_hs_data(L, T, N, device=device)
    print(f"  loaded {samples.shape} from HS dataset", flush=True)

    # Group layers by physical scale (2 layers per scale)
    layers = list(mera.layerList)
    indexI = mera.indexI
    indexJ = mera.indexJ
    n_phys_scales = int(math.log(L, 2))
    blocks_per_scale = len(layers) // n_phys_scales
    kernelShape = mera.kernelShape
    channelSize = samples.shape[1]

    field_per_scale = {0: samples.clone()}    # y_0 = input
    x = samples
    with torch.no_grad():
        for s in range(n_phys_scales):
            beg = s * blocks_per_scale
            end = (s + 1) * blocks_per_scale
            block_layers = layers[beg:end]
            block_I = indexI[beg:end]
            block_J = indexJ[beg:end]
            x = forward_through_scale_block(
                x, block_I, block_J, block_layers, kernelShape, channelSize)
            field_per_scale[s + 1] = x.clone()

    # Kept-coarse restriction: at scale s, only the stride-2^s
    # sub-lattice carries the slow mode. Everything else is frozen
    # latent. Restrict here before computing any statistic.
    coarse_per_scale = {s: keep_coarse_subsample(y, s)
                        for s, y in field_per_scale.items()}

    # Per-scale slow-mode moments + G(r)/G(0) on the coarse lattice
    stats_per_scale = {}
    G_per_scale = {}
    for s, y_c in coarse_per_scale.items():
        L_s = y_c.shape[-1]
        st = field_stats(y_c)
        st["L_s"] = L_s
        if L_s >= 2:
            G = two_point_axial(y_c)
        else:
            G = np.array([float(y_c.pow(2).mean().item())])
        stats_per_scale[s] = st
        G_per_scale[s] = G
        if L_s >= 4:
            g_1 = G[1] / G[0]
            g_half = G[L_s // 2] / G[0]
            G_str = (f"G(0)={G[0]:.3f}  G(1)/G(0)={g_1:.3f}  "
                     f"G(L_s/2)/G(0)={g_half:.3f}")
        elif L_s == 2:
            g_1 = G[1] / G[0]
            G_str = f"G(0)={G[0]:.3f}  G(1)/G(0)={g_1:.3f}  (L_s=2, no deeper r)"
        else:
            G_str = f"G(0)={G[0]:.3f}  (L_s=1, no spatial extent)"
        print(f"  y_{s}_coarse  L_s={L_s}:  mean={st['mean']:+.3f}  "
              f"std={st['std']:.3f}  kurt={st['kurtosis']:+.3f}  {G_str}",
              flush=True)

    dists = adjacent_distances(coarse_per_scale)
    for d in dists:
        rg = f"{d['rms_g']:.4f}" if not math.isnan(d['rms_g']) else "n/a"
        print(f"  (y_{d['s']}_c L={d['L_s']}, y_{d['s_next']}_c L={d['L_s_next']}): "
              f"KS={d['ks']:.4f}  W1={d['w1']:.4f}  rmsG={rg}", flush=True)

    return dict(
        label=label, folder=folder, L=L, T=T, epoch=epoch,
        weightTying=wt,
        stats_per_scale=stats_per_scale,
        G_per_scale=G_per_scale,
        adjacent=dists,
        n_scales=n_phys_scales,
    )


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "scale_s",
                    "metric", "value"])
        for r in results:
            for s, st in r["stats_per_scale"].items():
                for k, v in st.items():
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], s, f"moment_{k}", v])
            for d in r["adjacent"]:
                for k in ("ks", "w1", "rms_g"):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], f"{d['s']}_to_{d['s_next']}",
                                f"adj_{k}", d[k]])
    print(f"wrote {path}")


def plot_adjacent_metric(results, metric, ylabel, title, savepath):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for r in results:
        st = dict(STYLE[r["label"]])
        st.setdefault("markersize", 8)
        pairs = [(d["s"] + 1, d[metric]) for d in r["adjacent"]
                 if not (isinstance(d[metric], float) and math.isnan(d[metric]))]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, label=r["label"], **st)
    ax.set_xlabel(r"adjacent scale pair  $(y_s, y_{s+1})$  on the kept-coarse "
                  "sub-lattice  (s = 0 is HS data)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, max(len(r["adjacent"]) for r in results) + 1))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_G_overlay(results, savepath):
    """For each flow, overlay G_s(r)/G_s(0) for every scale s on one
    panel, where G_s is computed on the kept-coarse sub-lattice of
    size L_s = L/2^s. r is in coarse-lattice units; the physical
    distance for r is r * 2^s.
    """
    n = len(results)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        Gs = r["G_per_scale"]
        cmap = plt.cm.viridis
        keys = sorted(Gs.keys())
        for j, s in enumerate(keys):
            G = Gs[s]
            L_s = len(G)
            if L_s < 4:
                continue
            r_max = L_s // 2
            ax.plot(np.arange(1, r_max + 1),
                    G[1:r_max + 1] / (G[0] + 1e-12),
                    color=cmap(j / max(1, len(keys) - 1)),
                    label=f"y_{s} (L_s={L_s})", lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel("r (coarse-lattice units)")
        ax.set_ylabel("G(r) / G(0)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("V4 probe (kept-coarse) — G_s(r)/G_s(0) on the\n"
                 "stride-2^s sub-lattice of y_s (slow modes only)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2000, help="HS samples to use")
    p.add_argument("--outdir", default="analyzers")
    args = p.parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)

    results = []
    for label, folder in FOLDERS.items():
        try:
            r = run_one(folder, label, N=args.N)
            results.append(r)
        except Exception as e:
            print(f"  FAILED on {label}: {e}")

    if not results:
        raise SystemExit("no successful runs")

    write_csv(results, os.path.join(args.outdir, "csv", "rg_v4_dataforward.csv"))
    plot_adjacent_metric(results, "ks", "KS statistic on standardised marginal",
                         "V4 probe — KS distance between adjacent y_s, y_{s+1}",
                         os.path.join(args.outdir, "figures", "rg_v4_dataforward_ks.png"))
    plot_adjacent_metric(results, "w1", "Wasserstein-1 on standardised marginal",
                         "V4 probe — W1 distance between adjacent y_s, y_{s+1}",
                         os.path.join(args.outdir, "figures", "rg_v4_dataforward_w1.png"))
    plot_adjacent_metric(results, "rms_g", r"RMS$\,[\,G_s(r)/G_s(0) - G_{s+1}(r)/G_{s+1}(0)\,]$",
                         "V4 probe — G(r) shape mismatch between adjacent scales",
                         os.path.join(args.outdir, "figures", "rg_v4_dataforward_rmsG.png"))
    plot_G_overlay(results, os.path.join(args.outdir, "figures", "rg_v4_dataforward_Goverlay.png"))


if __name__ == "__main__":
    main()
