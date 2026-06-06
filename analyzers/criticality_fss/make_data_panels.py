"""Render HS-dataset overview panels for each L's temp-sweep report.

Three plots per L:
  1. data_configs_L{L}.png   --  configurations row across T
                                  (one column per T, sigmoid(2x) rendering)
  2. data_mag_overlay_L{L}.png  -- P(M) histograms, one curve per T
                                    (different colors, all on the same axes)
  3. data_corr_overlay_L{L}.png -- |G(r)|/G(0) on log-log, one curve per T,
                                    plus a dashed theoretical reference per T:
                                      * T_c: G ~ r^(-1/4)  (Onsager eta = 1/4)
                                      * off-T_c: fitted A * exp(-r/xi) on
                                        r in [1, L/2] (effective exponential
                                        decay; ξ extracted from the data
                                        itself, so the line shows the
                                        characteristic decay length not a
                                        Onsager exact value).

Reads HS data from data/mcmc_data/hs_L{L}_T{T_str}_N200000.pt.
Writes PNGs into analyzers/.

Usage (one L at a time so the data load fits a 24G CPU job):
    python analyzers/make_data_panels.py --L 8
    python analyzers/make_data_panels.py --L 16
    python analyzers/make_data_panels.py --L 32
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import _batch_corr, _radial_G  # noqa: E402

MCMC_DIR = "data/mcmc_data"
TS = ["2.15", "2.22", "2.269185314213022", "2.32", "2.4"]
TS_LABEL = ["2.15", "2.22", "2.269", "2.32", "2.40"]
T_C = 2.269185314213022

# Warm-to-cool palette with a prominent middle (T_c). Default
# coolwarm puts T_c at washed-out grey-white; instead we use a
# saturated dark magenta for T_c so it sits visually inside the
# blue->red gradient but stands out.
COLOR_BY_T = {
    "2.15":              "#2c6cb0",   # deep blue (well below T_c)
    "2.22":              "#8fc1e3",   # light blue (just below T_c)
    "2.269185314213022": "#9b1f8e",   # saturated dark magenta (T_c)
    "2.32":              "#f0a08c",   # light salmon (just above T_c)
    "2.4":               "#c1311b",   # deep red (well above T_c)
}
TC_LINEWIDTH = 3.0
OFF_LINEWIDTH = 1.8


def onsager_xi(T, J=1.0):
    """Onsager exact axial correlation length for the infinite 2D square
    Ising lattice. Returns None at exactly T_c (divergence).

    Formula:  xi^{-1}(T) = | 2K - ln coth K |    with K = J/T.
    At K = K_c (T = T_c), 2K = ln coth K = ln(1+sqrt(2)) -> xi = inf.
    """
    K = J / T
    log_coth = -math.log(math.tanh(K))    # ln coth K = -ln tanh K
    inv_xi = abs(2.0 * K - log_coth)
    if inv_xi < 1e-9:
        return None
    return 1.0 / inv_xi


def mean_M_sq(x):
    """<M^2> with M = (1/N) sum_i x_i.  Plateau of G(r) below T_c."""
    L = x.shape[-1]
    M = x.numpy().reshape(-1, L * L).mean(axis=1)
    return float((M ** 2).mean())


def find_hs(L, T_str, tol=1e-3):
    T = float(T_str)
    for fp in sorted(os.listdir(MCMC_DIR)):
        m = re.match(rf"hs_L{L}_T([\d.]+)_N\d+\.pt$", fp)
        if not m:
            continue
        Tfile = float(m.group(1))
        if abs(Tfile - T) < tol:
            return os.path.join(MCMC_DIR, fp)
    raise FileNotFoundError(f"no HS file at L={L}, T={T_str}")


def load(L, T_str, N):
    fp = find_hs(L, T_str)
    x = torch.load(fp, weights_only=True)
    return x[:N].reshape(-1, 1, L, L).to(torch.float32)


def panel_configs(L, savepath, n_per=16, nrow=4):
    n = len(TS)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    for i, (T, lab) in enumerate(zip(TS, TS_LABEL)):
        x = load(L, T, n_per)
        img = torch.sigmoid(2.0 * x)
        grid = make_grid(img, nrow=nrow, padding=1, pad_value=0.5)
        axes[i].imshow(grid[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[i].set_title(f"T = {lab}", fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f"L = {L}: HS dataset configurations (sigmoid(2x))", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {savepath}")


def panel_mag(L, savepath, N=8000):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for T, lab in zip(TS, TS_LABEL):
        c = COLOR_BY_T[T]
        is_Tc = (T == "2.269185314213022")
        lw = TC_LINEWIDTH if is_Tc else OFF_LINEWIDTH
        x = load(L, T, N)
        M = x.numpy().reshape(-1, L * L).mean(axis=1)
        mmax = float(np.abs(M).max())
        bins = np.linspace(-mmax * 1.05, mmax * 1.05, 71)
        ax.hist(M, bins=bins, density=True, histtype="step",
                color=c, linewidth=lw,
                label=f"T = {lab}" + ("  (T_c)" if is_Tc else ""))
    ax.set_xlabel("per-config magnetisation  M = (1/N) Σ x_i")
    ax.set_ylabel("density")
    ax.set_title(f"L = {L}: HS dataset P(M) across temperature")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {savepath}")


def panel_corr(L, savepath, N=4000, batch=128):
    """Two-point correlation panel, log-log. Data curves are raw
    |G(r)|/G(0). The only dashed theoretical line is the T_c
    r^(-1/4) Onsager universal power law (eta = 1/4) anchored at the
    r = 1 datapoint. Off-T_c theory lines are omitted because the
    HS-field correlator does not match Onsager spin-correlator
    exp(-r/xi) directly; see the discussion above
    `theoretical-line-for-hs` -- the asymptotic spin xi is the
    correct long-distance value for x too, but the finite-r OZ
    correction makes the simple exp(-r/xi) misleading and the
    convolution with K is too heavy for an inline reference.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    r_all = np.arange(1, L // 2 + 1)
    for T, lab in zip(TS, TS_LABEL):
        c = COLOR_BY_T[T]
        is_Tc = abs(float(T) - T_C) < 1e-2
        lw = TC_LINEWIDTH if is_Tc else OFF_LINEWIDTH
        ms = 8 if is_Tc else 5
        x = load(L, T, N)
        csum = np.zeros((L, L))
        ncfg = 0
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            _, cb = _batch_corr(xb)
            csum += cb
            ncfg += xb.shape[0]
        G = _radial_G(csum / ncfg)
        G0 = float(G[0])
        Gnorm = np.abs(G[1:] / G0)

        ax.loglog(r_all, Gnorm, "o-", color=c, linewidth=lw,
                  markersize=ms,
                  label=f"T = {lab}" + ("  (T_c)" if is_Tc else "") + "  data")

        if is_Tc:
            y1 = float(Gnorm[0])
            ref = y1 * (r_all / 1.0) ** (-0.25)
            ax.loglog(r_all, ref, "--", color=c, linewidth=2.0, alpha=0.9,
                      label=r"  theory $T_c$: $r^{-1/4}$  (Onsager $\eta=1/4$)")

    ax.set_xlabel("lattice distance  r")
    ax.set_ylabel("|G(r)| / G(0)")
    ax.set_title(f"L = {L}: HS dataset two-point correlation across T "
                 f"(log-log; dashed = $T_c$ Onsager $r^{{-1/4}}$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {savepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, required=True, choices=(8, 16, 32))
    p.add_argument("--outdir", default="analyzers")
    args = p.parse_args()
    L = args.L
    print(f"=== L = {L} ===")
    panel_configs(L, f"{args.outdir}/data_configs_L{L}.png")
    panel_mag(L,     f"{args.outdir}/data_mag_overlay_L{L}.png")
    panel_corr(L,    f"{args.outdir}/data_corr_overlay_L{L}.png")


if __name__ == "__main__":
    main()
