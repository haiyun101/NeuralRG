"""Criticality witnesses on the HS continuous-field dataset.

Five plots, all computed directly from data/mcmc_data/hs_L{L}_T*_N200000.pt:

  1. criticality_binder.png      Binder cumulant U_4(L, T); curves for L=8/16/32
                                 should cross at T_c with universal value
                                 U_4(T_c) ~ 0.6107 (2D Ising).

  2. criticality_chi.png         Susceptibility chi(L, T) and FSS scaling
                                 chi_max(L) ~ L^(gamma/nu) = L^(7/4).
                                 Two panels: chi(T) per L, then log-log chi_max
                                 vs L with both a fit slope and the Onsager
                                 prediction 7/4.

  3. criticality_xi_over_L.png   Second-moment correlation length ratio
                                 xi_eff(T) / L. Curves cross at T_c with
                                 universal value ~ 0.905 (2D Ising, torus,
                                 second-moment xi).

  4. criticality_mag_collapse.png  <|M|>(T) * L^(beta/nu) = <|M|> * L^(1/8).
                                   Curves should collapse near T_c.

  5. criticality_PM_collapse.png   Rescaled P(M * L^(beta/nu)) at T_c only;
                                   the three L histograms should collapse
                                   onto one universal scaling curve.

Also writes criticality_summary.csv with the raw per-(L, T) numbers.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MCMC_DIR = "data/mcmc_data"
LS = [8, 16, 32]
TS = ["2.15", "2.22", "2.269185314213022", "2.32", "2.4"]
TS_FLOAT = [2.15, 2.22, 2.269185314213022, 2.32, 2.40]
TS_LABEL = ["2.15", "2.22", "2.269", "2.32", "2.40"]
T_C = 2.269185314213022

# Universal 2D-Ising reference values
U4_TC = 0.6107            # Binder cumulant at T_c on the torus
XI_OVER_L_TC = 0.905      # second-moment xi/L universal at T_c (torus)
GAMMA_OVER_NU = 7.0 / 4.0
BETA_OVER_NU = 1.0 / 8.0

COLOR_BY_L = {8: "#1f4e9d", 16: "#9b1f8e", 32: "#c1311b"}
MARKER_BY_L = {8: "o", 16: "s", 32: "D"}


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
    return x[:N].numpy().reshape(-1, L, L).astype(np.float64)


def per_config_M(x):
    """M_i = (1/N) sum_a x_{i,a}, per-config magnetisation."""
    return x.reshape(x.shape[0], -1).mean(axis=1)


def binder(M):
    M2 = (M ** 2).mean()
    M4 = (M ** 4).mean()
    return 1.0 - M4 / (3.0 * M2 ** 2)


def susceptibility(M, L):
    """chi = N * (<M^2> - <|M|>^2). N = L^2.

    The <|M|>^2 subtraction is the conventional Z2-symmetric finite-L
    convention -- the sample mean <M> is identically 0 by Z2 averaging,
    so subtracting <|M|>^2 gives the connected susceptibility.
    """
    N = L * L
    return N * ((M ** 2).mean() - (np.abs(M).mean()) ** 2)


def xi_second_moment(x, L):
    """Second-moment correlation length from the Fourier transform.

        xi^2 = (1 / [4 sin^2(pi/L)]) * [chi(k=0) / chi(k_min) - 1]

    chi(k) = (1/Nconf) sum_conf |m(k)|^2, m(k) = sum_a x_{i,a} exp(-i k . r_a).
    k_min = (2 pi / L, 0); we average axial directions.
    """
    f = np.fft.fft2(x, axes=(1, 2))                     # (Nconf, L, L)
    chi_k = (np.abs(f) ** 2).mean(axis=0)               # (L, L)
    chi0 = float(chi_k[0, 0])
    chi_kmin = 0.5 * (float(chi_k[0, 1]) + float(chi_k[1, 0]))
    if chi_kmin <= 0:
        return None
    xi_sq = (chi0 / chi_kmin - 1.0) / (4.0 * np.sin(np.pi / L) ** 2)
    return float(np.sqrt(max(xi_sq, 0.0)))


def gather(N=8000):
    """Compute all per-(L, T) quantities. Returns dict[L][T_float] = {...}."""
    out = {L: {} for L in LS}
    for L in LS:
        for T_str, T in zip(TS, TS_FLOAT):
            x = load(L, T_str, N)
            M = per_config_M(x)
            row = dict(
                T=T,
                L=L,
                N=N,
                absM=float(np.abs(M).mean()),
                M2=float((M ** 2).mean()),
                M4=float((M ** 4).mean()),
                U4=float(binder(M)),
                chi=float(susceptibility(M, L)),
                xi_eff=xi_second_moment(x, L),
            )
            out[L][T] = row
            print(f"L={L:3d} T={T:.4f}  U4={row['U4']:.4f}  "
                  f"chi={row['chi']:.3f}  xi/L={row['xi_eff']/L:.4f}  "
                  f"|M|={row['absM']:.4f}")
    return out


def write_summary_csv(out, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "T", "N", "absM", "M2", "M4",
                    "U4", "chi", "xi_eff", "xi_over_L"])
        for L in LS:
            for T in sorted(out[L].keys()):
                r = out[L][T]
                w.writerow([L, T, r["N"], r["absM"], r["M2"], r["M4"],
                            r["U4"], r["chi"], r["xi_eff"],
                            r["xi_eff"] / L if r["xi_eff"] else None])
    print(f"wrote {path}")


def _plot_per_L(ax, out, key, ylabel, title):
    for L in LS:
        Ts = sorted(out[L].keys())
        ys = [out[L][T][key] for T in Ts]
        ax.plot(Ts, ys, marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=2, markersize=7, label=f"L = {L}")
    ax.axvline(T_C, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"$T_c = {T_C:.4f}$")
    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)


def panel_binder(out, savepath):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    _plot_per_L(ax, out, "U4", r"$U_4 = 1 - \langle M^4\rangle/(3\langle M^2\rangle^2)$",
                "Binder cumulant — crossings localise $T_c$")
    ax.axhline(U4_TC, color="black", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"2D Ising $U_4(T_c) \\approx {U4_TC}$")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_chi(out, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))
    _plot_per_L(axes[0], out, "chi",
                r"$\chi = L^2 (\langle M^2\rangle - \langle |M|\rangle^2)$",
                r"Susceptibility $\chi(T)$ per L")
    axes[0].set_yscale("log")

    # FSS log-log of peak chi vs L
    ax = axes[1]
    Ls = np.array(LS, dtype=float)
    chi_at_Tc = np.array([out[L][T_C]["chi"] for L in LS])
    log_L = np.log(Ls)
    log_chi = np.log(chi_at_Tc)
    slope, intercept = np.polyfit(log_L, log_chi, 1)
    ax.loglog(Ls, chi_at_Tc, "o", markersize=10, color="black",
              label=f"data $\\chi(T_c, L)$  (slope = {slope:.3f})")
    Ls_fine = np.linspace(Ls[0] * 0.9, Ls[-1] * 1.1, 100)
    ax.loglog(Ls_fine, np.exp(intercept) * Ls_fine ** slope,
              "-", color="black", alpha=0.6)
    anchor = chi_at_Tc[0]
    ax.loglog(Ls_fine, anchor * (Ls_fine / Ls[0]) ** GAMMA_OVER_NU,
              "--", color="red", linewidth=1.5,
              label=f"Onsager $\\gamma/\\nu = {GAMMA_OVER_NU}$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi(T_c, L)$")
    ax.set_title(r"FSS at $T_c$: $\chi \sim L^{\gamma/\nu}$ "
                 r"(Onsager $\gamma/\nu = 7/4$)")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_xi_over_L(out, savepath):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for L in LS:
        Ts = sorted(out[L].keys())
        ys = [out[L][T]["xi_eff"] / L for T in Ts]
        ax.plot(Ts, ys, marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=2, markersize=7, label=f"L = {L}")
    ax.axvline(T_C, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"$T_c$")
    ax.axhline(XI_OVER_L_TC, color="black", linestyle=":", linewidth=1.0,
               alpha=0.6,
               label=f"2D Ising $\\xi_{{eff}}/L \\approx {XI_OVER_L_TC}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\xi_{eff} / L$")
    ax.set_title(r"Second-moment $\xi_{eff}/L$ — crossings localise $T_c$")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_mag_collapse(out, savepath):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for L in LS:
        Ts = sorted(out[L].keys())
        ys = [out[L][T]["absM"] * (L ** BETA_OVER_NU) for T in Ts]
        ax.plot(Ts, ys, marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=2, markersize=7, label=f"L = {L}")
    ax.axvline(T_C, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"$T_c$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle |M| \rangle \cdot L^{\beta/\nu}$  "
                  r"($\beta/\nu = 1/8$)")
    ax.set_title("Magnetisation FSS collapse "
                 r"($\langle|M|\rangle \cdot L^{1/8}$ should fan-collapse at $T_c$)")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_PM_collapse(savepath, N=8000):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for L in LS:
        x = load(L, "2.269185314213022", N)
        M = per_config_M(x)
        M_scaled = M * (L ** BETA_OVER_NU)
        rng = 1.05 * float(np.abs(M_scaled).max())
        bins = np.linspace(-rng, rng, 81)
        ax.hist(M_scaled, bins=bins, density=True, histtype="step",
                color=COLOR_BY_L[L], linewidth=2.2, label=f"L = {L}")
    ax.set_xlabel(r"$M \cdot L^{\beta/\nu}$")
    ax.set_ylabel("density")
    ax.set_title(r"Rescaled $P(M \cdot L^{\beta/\nu})$ at $T_c$ — "
                 r"collapse $\Leftrightarrow$ universality")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8000,
                   help="samples per (L, T) (HS files have 200000)")
    p.add_argument("--outdir", default="analyzers")
    args = p.parse_args()

    out = gather(args.N)
    write_summary_csv(out, f"{args.outdir}/criticality_summary.csv")
    panel_binder(out,        f"{args.outdir}/criticality_binder.png")
    panel_chi(out,           f"{args.outdir}/criticality_chi.png")
    panel_xi_over_L(out,     f"{args.outdir}/criticality_xi_over_L.png")
    panel_mag_collapse(out,  f"{args.outdir}/criticality_mag_collapse.png")
    panel_PM_collapse(       f"{args.outdir}/criticality_PM_collapse.png",
                             N=args.N)


if __name__ == "__main__":
    main()
