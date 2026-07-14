"""Visualize what conditional law σ(z_slow) the CNN learned per level.

Two configuration-independent probes:

Option B — σ vs. scalar-summary scatter
  For each fast site (across many HS samples), plot σ_i against several
  scalar summaries of the local z_slow neighborhood:
    - mean(z_slow window)
    - |mean(z_slow window)|
    - std(z_slow window)      (local disorder / roughness)
    - Laplacian(z_slow center) (curvature)
  If σ collapses onto a curve for one summary → that scalar is what the
  CNN keys on. If none collapses → CNN uses higher-order info.

Plot 3 — Predicted vs. empirical σ² binning
  Bin the samples by the mean-window statistic. In each bin:
    - CNN's mean predicted σ²
    - Empirical Var(z_fast | bin)  (variance of actual z_fast in that bin)
  Overlay curves. Matching curves ⇒ CNN's law is calibrated to data.

Usage:
  python analyzers/rg_fixed_point/hcg_sigma_law.py \\
      --folder DATAFOLDER --N 500 --epoch BEST_EPOCH --out figures/sigma_law/
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import build_flow
from rg_fixed_point_v4_dataforward import load_hs_data


def latest_saving(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no .saving files in {folder}")
    return files[-1]


def pick_saving(folder, epoch):
    if epoch is None:
        return latest_saving(folder)
    savs = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    return savs[min(range(len(eps)), key=lambda i: abs(eps[i] - epoch))]


def get_sigma_at_level(prior, z, k):
    ctx_mask = prior._buffers[f"context_mask_up_to_{k-1}"].to(z.dtype)
    z_ctx = z * ctx_mask
    # Shared HCG: single CNN reused at every level. Per-scale: cnns[k-1].
    if getattr(prior, "scale_shared", False) or prior.cnns is None:
        cnn = prior.cnn_shared
    else:
        cnn = prior.cnns[k - 1]
    with torch.no_grad():
        out = cnn(z_ctx)
    mu, log_sigma = out.chunk(2, dim=1)
    log_sigma = log_sigma.clamp(min=-5.0, max=5.0)
    sigma = torch.exp(log_sigma)
    return sigma, mu


def local_summaries(z_ctx, ctx_mask, window=7):
    """Per-site local statistics of z_ctx over CONTEXT POSITIONS ONLY.

    Original version averaged the raw 7×7 window with a fixed 1/49 kernel.
    At coarse HCG levels the context mask is sparse — most window positions
    are fast-site zeros — so raw mean/std/Laplacian all collapsed to ~0,
    causing R² = NaN and hiding the CNN's actual dependence.

    Fix: normalize by the *count* of context positions inside each window,
    not by the full window area. Same for the Laplacian stencil, which
    now uses only its context neighbors.

    Returns four (B, 1, L, L) tensors:
      mean_ctx, absmean_ctx, std_ctx, laplacian_ctx
    """
    ones_kernel = torch.ones(1, 1, window, window, device=z_ctx.device, dtype=z_ctx.dtype)
    pad = window // 2

    # count of context sites inside the window at each center site
    m_pad = F.pad(ctx_mask, (pad, pad, pad, pad), mode="circular")
    n_ctx = F.conv2d(m_pad, ones_kernel).clamp(min=1.0)

    # sum of z (which is z * ctx_mask because non-ctx zeros are already there)
    z_pad = F.pad(z_ctx, (pad, pad, pad, pad), mode="circular")
    z_sum = F.conv2d(z_pad, ones_kernel)
    mean_ctx = z_sum / n_ctx

    zsq_pad = F.pad(z_ctx.pow(2), (pad, pad, pad, pad), mode="circular")
    zsq_sum = F.conv2d(zsq_pad, ones_kernel)
    meansq_ctx = zsq_sum / n_ctx
    var_ctx = (meansq_ctx - mean_ctx.pow(2)).clamp(min=0)
    std_ctx = var_ctx.sqrt()

    # Context-aware Laplacian: (Σ_i ctx_i · z_i − n_i · z_center) / n_i
    # where n_i counts context neighbors. At coarse levels this uses the
    # actual reachable context sites, not the (mostly-masked) 4-neighbourhood.
    lap_kernel_neighbors = torch.tensor([[0., 1., 0.],
                                          [1., 0., 1.],
                                          [0., 1., 0.]],
                                         device=z_ctx.device, dtype=z_ctx.dtype).view(1, 1, 3, 3)
    z_pad3 = F.pad(z_ctx, (1, 1, 1, 1), mode="circular")
    m_pad3 = F.pad(ctx_mask, (1, 1, 1, 1), mode="circular")
    z_nb = F.conv2d(z_pad3, lap_kernel_neighbors)
    n_nb = F.conv2d(m_pad3, lap_kernel_neighbors).clamp(min=1.0)
    # Only compute Laplacian at context center sites; at fast sites the
    # center's z is masked to 0 which is fine.
    laplacian_ctx = z_nb / n_nb - z_ctx

    return mean_ctx, mean_ctx.abs(), std_ctx, laplacian_ctx


# Per-run cross-level table populated by analyze_level()
_MARG_TABLE = []


def bin_stats(x, y, n_bins=20):
    """Bin y against x, return bin centers and per-bin mean/std of y."""
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, stds, ns = [], [], []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 20:
            means.append(np.nan); stds.append(np.nan); ns.append(int(mask.sum()))
            continue
        means.append(float(y[mask].mean()))
        stds.append(float(y[mask].std()))
        ns.append(int(mask.sum()))
    return centers, np.array(means), np.array(stds), np.array(ns)


def analyze_level(prior, z, k, out_dir, folder_tag, window=7):
    """Plot σ-vs-scalar scatter (Option B) + predicted-vs-empirical σ² (Plot 3)."""
    sigma, mu = get_sigma_at_level(prior, z, k)  # (B, 1, L, L)
    level_mask = prior._buffers[f"level_mask_{k}"].to(z.dtype).bool()  # (1,1,L,L)
    ctx_mask = prior._buffers[f"context_mask_up_to_{k-1}"].to(z.dtype)
    z_ctx = z * ctx_mask

    # Adaptive window: the nearest context sites live at least strides[k-1]
    # away (that's the coarsest reachable context). A 7×7 window works for
    # fine levels, but at Level 1 (stride L/2) it doesn't reach ANY context
    # site, giving n_ctx = 0 → all-zero summaries → R² = NaN.
    # Rule: window ≥ 2·strides[k-1] + 1 so every fast-site window reaches
    # its immediate coarse context.
    stride_prev = prior.strides[k - 1]  # coarser-level stride (larger number)
    L = z.shape[-1]
    win_needed = int(min(L - 1, max(window, 2 * stride_prev + 1)))
    if win_needed % 2 == 0:
        win_needed += 1
    print(f"    (level {k}: stride_prev={stride_prev}, using window={win_needed})")

    mean_win, absmean_win, std_win, lap = local_summaries(z_ctx, ctx_mask, window=win_needed)

    # Broadcast level_mask
    mb = level_mask.expand_as(sigma)

    sig_np = sigma[mb].cpu().numpy()
    mu_np = mu[mb].cpu().numpy()
    zfast_np = z[mb].cpu().numpy()          # actual z at fast sites
    mean_np = mean_win[mb].cpu().numpy()
    abs_np = absmean_win[mb].cpu().numpy()
    std_np = std_win[mb].cpu().numpy()
    lap_np = lap[mb].cpu().numpy()

    # ── Marginal-Gaussianize normalization ───────────────────────────────
    # emp_var  = Var(z_fast − μ)  = the actual data variance CNN's σ² should
    #                               match. Comparing raw σ across variants
    #                               conflates model calibration with
    #                               marginal scale of MERA output.
    # pred_var = mean(σ²)         = CNN's average variance prediction
    # ratio    = pred_var / emp_var — 1 = perfect, <1 = under-confident,
    #                                     >1 = over-confident
    residuals = zfast_np - mu_np
    emp_var = float(residuals.var())
    pred_var = float((sig_np ** 2).mean())
    calib_ratio = pred_var / emp_var if emp_var > 1e-12 else float("nan")

    print(f"\n  Level {k}: n_sites × B = {sig_np.size}, "
          f"σ mean={sig_np.mean():.4f} std={sig_np.std():.4f}  "
          f"μ mean={mu_np.mean():+.4f} std={mu_np.std():.4f}")
    print(f"    marginal: emp_var(z-μ) = {emp_var:.4f}  "
          f"pred_var = {pred_var:.4f}  "
          f"calib_ratio = {calib_ratio:+.4f}")
    # Store for cross-level table at end of run
    _MARG_TABLE.append(dict(level=k, emp_var=emp_var,
                            pred_var=pred_var, calib=calib_ratio,
                            sigma_ref=float(emp_var ** 0.5)))

    # ── Option B: σ vs. four scalar summaries ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    scalars = [
        ("mean(z_slow)", mean_np),
        ("|mean(z_slow)|", abs_np),
        ("std(z_slow)", std_np),
        ("Laplacian(z_slow)", lap_np),
    ]
    n_show = min(30000, sig_np.size)  # subsample for readability
    idx = np.random.default_rng(0).choice(sig_np.size, n_show, replace=False)
    for ax, (name, xs) in zip(axes.flat, scalars):
        ax.scatter(xs[idx], sig_np[idx], s=1, alpha=0.15, color="steelblue")
        # overlay binned mean
        c, m, s, n = bin_stats(xs, sig_np, n_bins=25)
        ax.plot(c, m, color="crimson", lw=2, label="binned mean σ")
        ax.fill_between(c, m - s, m + s, color="crimson", alpha=0.2, label="±1 std")
        ax.set_xlabel(name)
        ax.set_ylabel("σ_i (CNN prediction)")
        ax.set_title(f"L{k}: σ vs {name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{folder_tag} — Level {k}  (σ vs local z_slow summaries)")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{folder_tag}_L{k}_optionB_scatter.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"    saved {fname}")

    # ── Plot 3: predicted σ² vs empirical Var(z_fast|bin) — binned by mean_win ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # bin by mean_win
    c, m_sig, _, ns = bin_stats(mean_np, sig_np, n_bins=25)
    _,  m_pred_sq, _, _  = bin_stats(mean_np, sig_np ** 2, n_bins=25)
    # empirical variance in each bin (of z_fast)
    lo, hi = np.percentile(mean_np, 2), np.percentile(mean_np, 98)
    bins = np.linspace(lo, hi, 26)
    emp_var = []
    for i in range(25):
        mask = (mean_np >= bins[i]) & (mean_np < bins[i+1])
        if mask.sum() < 20:
            emp_var.append(np.nan); continue
        # subtract the CNN's μ prediction so we compare like-for-like
        residuals = zfast_np[mask] - mu_np[mask]
        emp_var.append(float(residuals.var()))
    emp_var = np.array(emp_var)

    ax = axes[0]
    ax.plot(c, m_pred_sq, color="crimson", lw=2, label="CNN predicted σ² (mean per bin)")
    ax.plot(c, emp_var,   color="black", lw=2, ls="--", label="empirical Var(z_fast − μ | bin)")
    ax.set_xlabel("mean(z_slow) in bin")
    ax.set_ylabel("variance")
    ax.set_title(f"L{k}: predicted σ² vs empirical σ²")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    # ratio panel
    ratio = np.where(np.isfinite(emp_var) & (emp_var > 0), m_pred_sq / emp_var, np.nan)
    ax.plot(c, ratio, color="steelblue", lw=2)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5, label="perfect calibration")
    ax.set_xlabel("mean(z_slow) in bin")
    ax.set_ylabel("predicted / empirical  σ²")
    ax.set_title(f"L{k}: calibration ratio")
    ax.set_ylim(0, min(3, np.nanmax(ratio) * 1.1 if np.isfinite(ratio).any() else 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"{folder_tag} — Level {k}  (Plot 3: predicted vs empirical σ²)")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{folder_tag}_L{k}_plot3_calibration.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"    saved {fname}")

    # Print numerical summary — how much of σ variance is explained by each scalar?
    print(f"    R² of σ against each scalar (higher = tighter collapse):")
    for name, xs in scalars:
        # simple R² of linear regression of σ on xs
        if xs.std() < 1e-9:
            r2 = float("nan")
        else:
            r2 = float(np.corrcoef(xs, sig_np)[0, 1] ** 2)
        print(f"      {name:>22s}: R² = {r2:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--epoch", type=int, default=None,
                    help="checkpoint epoch (default: latest)")
    ap.add_argument("--N", type=int, default=500, help="HS samples")
    ap.add_argument("--out", default="figures/sigma_law/", help="output dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=int, default=7,
                    help="local-neighborhood window (odd int)")
    ap.add_argument("--levels", type=str, default=None,
                    help="comma-separated levels to analyze (default: all)")
    args = ap.parse_args()

    folder = args.folder.rstrip("/")
    ckpt = pick_saving(folder, args.epoch)
    ep = int(re.search(r"epoch(\d+)", ckpt).group(1))
    print(f"[sigma-law] {folder}  epoch {ep}")

    state = torch.load(ckpt, weights_only=False, map_location=args.device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=args.device)
    fw.eval()

    prior = fw.flow.prior if hasattr(fw, "flow") else fw.prior
    if prior.__class__.__name__ != "HierarchicalConditionalGaussian":
        print("not HCG, abort"); return

    # Shared HCG: 1 CNN reused at all levels. K derived from strides.
    # Interesting to analyze: same CNN weights, different level context →
    # is the σ-law scale-invariant, or does the CNN express distinct laws
    # at different levels via the changing context density?
    if getattr(prior, "scale_shared", False) or prior.cnns is None:
        K = len(prior.strides) - 1
        print(f"  shared HCG: 1 CNN, K={K} levels")
    else:
        K = len(prior.cnns)
    if args.levels:
        levels = [int(x) for x in args.levels.split(",")]
    else:
        levels = list(range(1, K + 1))

    samples = load_hs_data(L, T, args.N, device=args.device)
    mera = fw.flow if hasattr(fw, "flow") else fw
    with torch.no_grad():
        z, _ = mera.forward(samples)
    print(f"  loaded {samples.shape[0]} HS, z shape {tuple(z.shape)}")

    out_dir = args.out.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    # Use full basename (no truncation) — earlier 60-char cap caused
    # nr=1 and nr=2 nodilate folders to share the same prefix and
    # overwrite each other's PNGs.
    tag = os.path.basename(folder).replace("/", "_")

    global _MARG_TABLE
    _MARG_TABLE = []
    for k in levels:
        try:
            analyze_level(prior, z, k, out_dir, folder_tag=tag, window=args.window)
        except Exception as e:
            print(f"  level {k}: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-level summary: marginal Gaussianize table ──────────────────
    if _MARG_TABLE:
        print(f"\n[MARGINAL_TABLE] {tag}")
        print(f"  {'level':>5}  {'emp_var':>10}  {'σ_ref':>10}  {'pred_var':>10}  "
              f"{'calib_ratio':>12}  {'log10_ratio':>12}")
        for row in _MARG_TABLE:
            print(f"  {row['level']:>5d}  {row['emp_var']:>10.4f}  "
                  f"{row['sigma_ref']:>10.4f}  {row['pred_var']:>10.4f}  "
                  f"{row['calib']:>+12.4f}  "
                  f"{np.log10(max(row['calib'], 1e-12)):>+12.3f}")
        print(f"  (calib_ratio = pred_var / emp_var; 1 = perfect, <1 = under-confident)")

    print("\nDone.")


if __name__ == "__main__":
    main()
