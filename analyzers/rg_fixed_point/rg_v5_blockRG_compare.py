"""V5 — Block-RG diagnostic ground truth.

Methodology
-----------
V4 measured whether MERA's forward intermediates y_s reach distributional
self-similarity scale-to-scale ("did anything change between y_s and
y_{s+1}?"). Self-similarity is necessary for an RG fixed point, but it
is not sufficient — a flow that pushes everything to N(0, I) latent
prior also looks self-similar at deep scales without doing anything
RG-like.

V5 adds a Wilson-Kadanoff ground truth. For each HS configuration x at
T_c we build the block-average cascade

    x_0 = x                                        (L     x L)
    x_1 = AvgPool2d(2) (x_0)                       (L/2   x L/2)
    x_s = AvgPool2d(2) (x_{s-1})                   (L/2^s x L/2^s)

This is the canonical real-space coarse-graining used in block-spin RG.
The distribution of x_s (after standardisation) tells us what the
coarse-grained field of Ising at T_c "should" look like at scale s.

We then take MERA's forward intermediate y_s (full L x L lattice; same
as V4) and sub-sample it at stride 2^s starting at offset 0, giving a
field at resolution L/2^s x L/2^s. (MERA's scale-block leaves all L^2
indices populated, but the slow / coarse modes live on a sub-lattice of
size L/2^s.) We compare this sub-sampled MERA field to the block-RG
field at the same scale.

What we compare at each scale s
-------------------------------
  - Standardised marginal distribution (KS, Wasserstein-1)
  - Two-point correlation shape G_s(r)/G_s(0) for r = 1..L_s/2
    where L_s = L/2^s
  - Raw moments (mean / std / kurtosis / skew)

Interpretation
--------------
A flow whose forward direction implements a coarse-graining consistent
with block-RG will have small KS / W1 / RMS G(r) between y_s and x_s at
ALL scales (not just the deepest, where both have already collapsed to
their respective asymptotes). Mismatch at intermediate scales tells us
that the MERA path through scales is different from real-space RG,
even if both endpoints (data and latent) look similar.

We probe the same six flows as V4.

Usage
-----
  python analyzers/rg_v5_blockRG_compare.py [--N 2000]
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
import torch.nn.functional as F
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
from rg_fixed_point_v4_dataforward import (
    FOLDERS,
    load_hs_data,
    forward_through_scale_block,
    two_point_axial,
    field_stats,
)

STYLE = {k: ORIG_STYLE[k] for k in FOLDERS}


def load_flow(folder, device="cpu"):
    ckpt = latest_saving(folder)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    mera = fw.flow if hasattr(fw, "flow") else fw
    return mera, L, T, epoch, wt


def block_rg_cascade(x, n_scales):
    """Wilson-Kadanoff block-average cascade.

    x: (B, 1, L, L) input HS field.
    Returns dict {s: (B, 1, L_s, L_s)} for s = 0..n_scales, with L_s = L/2^s.
    """
    out = {0: x.clone()}
    cur = x
    for s in range(1, n_scales + 1):
        if cur.shape[-1] < 2:
            break
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
        out[s] = cur.clone()
    return out


def mera_forward_intermediates(mera, samples, L):
    """Same probe as V4: feed HS samples through MERA forward direction
    and record y_s after each scale-block. Returns dict {s: (B, 1, L, L)}.
    """
    layers = list(mera.layerList)
    indexI = mera.indexI
    indexJ = mera.indexJ
    n_phys_scales = int(math.log(L, 2))
    blocks_per_scale = len(layers) // n_phys_scales
    kernelShape = mera.kernelShape
    channelSize = samples.shape[1]

    field_per_scale = {0: samples.clone()}
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
    return field_per_scale, n_phys_scales


def subsample_mera_to_scale(y_s, s):
    """Sub-sample y_s (shape (B,1,L,L)) at stride 2^s starting from
    (0,0). Returns (B, 1, L/2^s, L/2^s).

    The slow / coarse modes after s MERA scale-blocks live on the
    sub-lattice {(2^s i, 2^s j)}.
    """
    stride = 2 ** s
    return y_s[..., ::stride, ::stride].contiguous()


def standardise_flat(x_tensor):
    flat = x_tensor.flatten().cpu().numpy()
    mu = flat.mean()
    sd = flat.std() + 1e-12
    return (flat - mu) / sd


def cross_scale_distances(mera_per_scale, blockrg_per_scale, L):
    """For each scale s where both fields exist, compute KS, W1, and
    G(r)/G(0) RMS distance between the standardised MERA field
    (sub-sampled to L/2^s) and the block-RG field."""
    out = []
    keys = sorted(set(mera_per_scale.keys()) & set(blockrg_per_scale.keys()))
    rng_ks_a = np.random.default_rng(0)
    rng_ks_b = np.random.default_rng(1)
    rng_w_a = np.random.default_rng(2)
    rng_w_b = np.random.default_rng(3)
    for s in keys:
        if s == 0:
            # Sanity: y_0 = x_0 = HS data; should give exact match.
            y_sub = mera_per_scale[s]
        else:
            y_sub = subsample_mera_to_scale(mera_per_scale[s], s)
        x_blk = blockrg_per_scale[s]
        # Match sizes after sub-sampling (drop edge if off-by-one)
        m = min(y_sub.shape[-1], x_blk.shape[-1])
        y_sub = y_sub[..., :m, :m]
        x_blk = x_blk[..., :m, :m]

        z_y = standardise_flat(y_sub)
        z_x = standardise_flat(x_blk)

        n_keep = min(200_000, z_y.size, z_x.size)
        ridx_a = rng_ks_a.choice(z_y.size, n_keep, replace=False)
        ridx_b = rng_ks_b.choice(z_x.size, n_keep, replace=False)
        ks_stat = float(sps.ks_2samp(z_y[ridx_a], z_x[ridx_b]).statistic)

        n_w = min(20_000, z_y.size, z_x.size)
        ridx_c = rng_w_a.choice(z_y.size, n_w, replace=False)
        ridx_d = rng_w_b.choice(z_x.size, n_w, replace=False)
        w1 = float(sps.wasserstein_distance(z_y[ridx_c], z_x[ridx_d]))

        # G(r)/G(0): only meaningful if L_s >= 4
        if m >= 4:
            Gy = two_point_axial(y_sub)
            Gx = two_point_axial(x_blk)
            half = max(1, m // 2)
            gyn = Gy[: half + 1] / (Gy[0] + 1e-12)
            gxn = Gx[: half + 1] / (Gx[0] + 1e-12)
            rms_g = float(np.sqrt(((gyn - gxn) ** 2).mean()))
        else:
            rms_g = float("nan")

        out.append(dict(
            s=s, L_s=m,
            ks=ks_stat, w1=w1, rms_g=rms_g,
        ))
    return out


def per_scale_moments(field_per_scale, sub_sample_mera=False):
    out = {}
    for s, y in field_per_scale.items():
        if sub_sample_mera and s > 0:
            y = subsample_mera_to_scale(y, s)
        out[s] = field_stats(y)
        out[s]["L_s"] = y.shape[-1]
    return out


def run_one(folder, label, N=2000, device="cpu"):
    print(f"\n=== {label}  folder={folder} ===", flush=True)
    mera, L, T, epoch, wt = load_flow(folder, device=device)
    print(f"  L={L} T={T} ep={epoch} weightTying={wt}")

    samples = load_hs_data(L, T, N, device=device)
    print(f"  loaded {samples.shape} from HS dataset", flush=True)

    mera_per_scale, n_phys_scales = mera_forward_intermediates(mera, samples, L)
    blockrg_per_scale = block_rg_cascade(samples, n_phys_scales)
    print(f"  MERA scales: {sorted(mera_per_scale.keys())}", flush=True)
    print(f"  block-RG scales: {sorted(blockrg_per_scale.keys())}  "
          f"(sizes: "
          + ", ".join(f"{s}->{blockrg_per_scale[s].shape[-1]}"
                      for s in sorted(blockrg_per_scale.keys()))
          + ")", flush=True)

    mera_moments = per_scale_moments(mera_per_scale, sub_sample_mera=True)
    blk_moments = per_scale_moments(blockrg_per_scale, sub_sample_mera=False)
    for s in sorted(mera_moments.keys()):
        m = mera_moments[s]
        if s in blk_moments:
            b = blk_moments[s]
            print(f"  scale {s} (L_s={m['L_s']}):  "
                  f"MERA  mu={m['mean']:+.3f} std={m['std']:.3f} kurt={m['kurtosis']:+.2f}  |  "
                  f"blkRG mu={b['mean']:+.3f} std={b['std']:.3f} kurt={b['kurtosis']:+.2f}",
                  flush=True)
        else:
            print(f"  scale {s}: MERA only  mu={m['mean']:+.3f} std={m['std']:.3f} kurt={m['kurtosis']:+.2f}",
                  flush=True)

    cross = cross_scale_distances(mera_per_scale, blockrg_per_scale, L)
    for d in cross:
        rg = f"{d['rms_g']:.4f}" if not math.isnan(d['rms_g']) else "n/a"
        print(f"  cross s={d['s']} (L_s={d['L_s']}):  KS={d['ks']:.4f}  "
              f"W1={d['w1']:.4f}  rmsG={rg}", flush=True)

    return dict(
        label=label, folder=folder, L=L, T=T, epoch=epoch,
        weightTying=wt,
        mera_moments=mera_moments,
        blk_moments=blk_moments,
        cross=cross,
        mera_per_scale={s: y.cpu() for s, y in mera_per_scale.items()},
        blk_per_scale={s: y.cpu() for s, y in blockrg_per_scale.items()},
        n_scales=n_phys_scales,
    )


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "scale_s",
                    "L_s", "source", "metric", "value"])
        for r in results:
            for s, m in r["mera_moments"].items():
                for k in ("mean", "std", "var", "kurtosis", "skew"):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], s, m["L_s"], "mera_subsampled",
                                f"moment_{k}", m[k]])
            for s, b in r["blk_moments"].items():
                for k in ("mean", "std", "var", "kurtosis", "skew"):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], s, b["L_s"], "blockRG",
                                f"moment_{k}", b[k]])
            for d in r["cross"]:
                for k in ("ks", "w1", "rms_g"):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"],
                                r["epoch"], d["s"], d["L_s"], "cross",
                                f"v5_{k}", d[k]])
    print(f"wrote {path}")


def plot_cross_metric(results, metric, ylabel, title, savepath):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for r in results:
        st = dict(STYLE[r["label"]])
        st.setdefault("markersize", 8)
        xs = [d["s"] for d in r["cross"]]
        ys = [d[metric] for d in r["cross"]]
        ax.plot(xs, ys, label=r["label"], **st)
    ax.set_xlabel(r"scale $s$  (0 = HS data, $L_s = L/2^s$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, max(len(r["cross"]) for r in results) + 1))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_moment_compare(results, key, ylabel, title, savepath):
    """Side-by-side: each flow's MERA-subsampled vs block-RG moment as a
    function of scale, in stacked subplots."""
    n = len(results)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        scales = sorted(r["blk_moments"].keys())
        blk_vals = [r["blk_moments"][s][key] for s in scales]
        mera_vals = [r["mera_moments"][s][key]
                     if s in r["mera_moments"] else float("nan")
                     for s in scales]
        ax.plot(scales, blk_vals, "k-o", label="block-RG (ground truth)",
                lw=1.6, markersize=5)
        ax.plot(scales, mera_vals, color="C3", marker="s",
                label="MERA (sub-sampled y_s)", lw=1.4, markersize=5)
        ax.set_xlabel("scale s")
        ax.set_ylabel(ylabel)
        ax.set_title(r["label"], fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_marginal_overlay(results, savepath, scale_to_show=2):
    """At a single intermediate scale, overlay the standardised marginal
    histograms of MERA and block-RG for each flow. This is the most
    direct visual diagnostic for "does the path through scales match?"."""
    n = len(results)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        s = scale_to_show
        if s not in r["mera_per_scale"] or s not in r["blk_per_scale"]:
            ax.axis("off")
            continue
        y_sub = subsample_mera_to_scale(r["mera_per_scale"][s], s)
        z_y = standardise_flat(y_sub)
        z_x = standardise_flat(r["blk_per_scale"][s])
        bins = np.linspace(-4, 4, 60)
        ax.hist(z_x, bins=bins, density=True, alpha=0.5,
                color="0.4", label="block-RG x_s")
        ax.hist(z_y, bins=bins, density=True, histtype="step",
                color="C3", lw=1.8, label="MERA y_s (subsampled)")
        ax.set_xlim(-4, 4)
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel(f"standardised field, scale s={s}")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"V5 — marginal at scale s={scale_to_show}: "
                 f"MERA sub-sampled (red) vs block-RG (grey)", fontsize=11)
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

    write_csv(results, os.path.join(args.outdir, "csv", "rg_v5_blockRG_compare.csv"))
    plot_cross_metric(
        results, "ks",
        "KS statistic on standardised marginal (MERA vs block-RG)",
        "V5 probe — KS distance between MERA y_s (sub-sampled) and block-RG x_s",
        os.path.join(args.outdir, "figures", "rg_v5_blockRG_ks.png"))
    plot_cross_metric(
        results, "w1",
        "Wasserstein-1 on standardised marginal (MERA vs block-RG)",
        "V5 probe — W1 distance between MERA y_s (sub-sampled) and block-RG x_s",
        os.path.join(args.outdir, "figures", "rg_v5_blockRG_w1.png"))
    plot_cross_metric(
        results, "rms_g",
        r"RMS$\,[G_s(r)/G_s(0)]$  (MERA vs block-RG)",
        "V5 probe — G(r) shape mismatch (MERA y_s sub-sampled vs block-RG x_s)",
        os.path.join(args.outdir, "figures", "rg_v5_blockRG_rmsG.png"))
    plot_moment_compare(
        results, "std", "std of field",
        "V5 — std of field by scale (block-RG vs MERA sub-sampled)",
        os.path.join(args.outdir, "figures", "rg_v5_blockRG_std.png"))
    plot_moment_compare(
        results, "kurtosis", "excess kurtosis",
        "V5 — kurtosis of field by scale (block-RG vs MERA sub-sampled)",
        os.path.join(args.outdir, "figures", "rg_v5_blockRG_kurt.png"))
    plot_marginal_overlay(
        results, os.path.join(args.outdir, "figures", "rg_v5_blockRG_marg_s2.png"),
        scale_to_show=2)
    plot_marginal_overlay(
        results, os.path.join(args.outdir, "figures", "rg_v5_blockRG_marg_s3.png"),
        scale_to_show=3)


if __name__ == "__main__":
    main()
