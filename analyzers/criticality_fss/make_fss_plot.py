"""Render the FSS forward-KL sweep plot from fss_sweep_KL_v2.csv.

Writes analyzers/fss_sweep_KL_v3.png with:
- T_c row drawn in red (everywhere)
- Larger markers/lines, fewer overlapping labels
- Panel (c) cleaned up: per-T log-log with markers labeled by T only on legend
- Panel (d) uses dashed fit lines through markers + α=2 and α=1 references in gray

Usage:  python analyzers/make_fss_plot.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T_C = 2.269185314213022
CSV = os.path.join(os.path.dirname(__file__), "fss_sweep_KL_v2.csv")
OUT = os.path.join(os.path.dirname(__file__), "fss_sweep_KL_v3.png")


def load():
    by_L = defaultdict(list)  # L -> [(T, KL, N)]
    by_T = defaultdict(list)  # T(rounded) -> [(L, KL, N)]
    with open(CSV) as f:
        for row in csv.DictReader(f):
            L = int(float(row["L"]))
            T = float(row["T"])
            KL = float(row["KL_best"])
            N = int(float(row["N"]))
            by_L[L].append((T, KL, N))
            by_T[round(T, 4)].append((L, KL, N))
    for k in by_L:
        by_L[k].sort()
    for k in by_T:
        by_T[k].sort()
    return by_L, by_T


def fit_alpha(Ls, KLs):
    x = np.log(Ls)
    y = np.log(KLs)
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    b = ym - a * xm
    return a, np.exp(b)


def main():
    by_L, by_T = load()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axA, axB), (axC, axD) = axes

    # Colors: L curves use cool palette; T curves use perceptual gradient w/ red at T_c
    L_colors = {8: "#1f77b4", 16: "#2ca02c", 32: "#9467bd"}
    L_markers = {8: "o", 16: "s", 32: "D"}
    Ts_sorted = sorted(by_T)

    def color_for_T(T):
        if abs(T - T_C) < 1e-3:
            return "#d62728"  # red for T_c
        # cool->warm without using red
        idx = Ts_sorted.index(round(T, 4))
        cmap = plt.get_cmap("viridis")
        return cmap(idx / max(1, len(Ts_sorted) - 1))

    # ---------------- (a) Absolute KL vs T ----------------
    for L in sorted(by_L):
        Ts = [t for t, _, _ in by_L[L]]
        KLs = [kl for _, kl, _ in by_L[L]]
        axA.plot(
            Ts, KLs,
            marker=L_markers[L], color=L_colors[L], linewidth=2, markersize=9,
            label=f"L={L}  (N={L*L})",
        )
    axA.axvline(T_C, color="#d62728", linestyle="--", alpha=0.6, label="T_c=2.2692")
    axA.set_xlabel("Temperature T")
    axA.set_ylabel("KL_fwd  [nat]")
    axA.set_yscale("log")
    axA.set_title("(a) Absolute KL_fwd vs T (log y)")
    axA.legend(loc="best", fontsize=10)
    axA.grid(True, alpha=0.3)

    # ---------------- (b) Per-site KL vs T (intensive check) ----------------
    for L in sorted(by_L):
        Ts = [t for t, _, _ in by_L[L]]
        per = [kl / N for _, kl, N in by_L[L]]
        axB.plot(
            Ts, per,
            marker=L_markers[L], color=L_colors[L], linewidth=2, markersize=9,
            label=f"L={L}",
        )
    axB.axvline(T_C, color="#d62728", linestyle="--", alpha=0.6)
    axB.set_xlabel("Temperature T")
    axB.set_ylabel("KL_fwd / N  [nat/site]")
    axB.set_title("(b) Per-site KL  (collapse ⇔ intensive ⇔ α=2)")
    axB.legend(loc="best", fontsize=10)
    axB.grid(True, alpha=0.3)

    # ---------------- (c) KL vs L per T (log-log) ----------------
    for T in Ts_sorted:
        pts = by_T[T]
        Ls = np.array([p[0] for p in pts], dtype=float)
        KLs = np.array([p[1] for p in pts], dtype=float)
        is_Tc = abs(T - T_C) < 1e-3
        c = color_for_T(T)
        lw = 3 if is_Tc else 1.8
        ms = 12 if is_Tc else 8
        label = f"T={T:.3f}" + ("  (T_c)" if is_Tc else "")
        axC.plot(Ls, KLs, marker="o", color=c, linewidth=lw, markersize=ms,
                 label=label, zorder=5 if is_Tc else 3)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xticks([8, 16, 32])
    axC.set_xticklabels(["8", "16", "32"])
    axC.set_xlabel("Linear size L")
    axC.set_ylabel("KL_fwd  [nat]")
    axC.set_title("(c) KL_fwd vs L, per T  (log-log)")
    axC.legend(loc="upper left", fontsize=10)
    axC.grid(True, which="both", alpha=0.3)

    # ---------------- (d) Power-law fits + reference slopes ----------------
    for T in Ts_sorted:
        pts = by_T[T]
        Ls = np.array([p[0] for p in pts], dtype=float)
        KLs = np.array([p[1] for p in pts], dtype=float)
        alpha, a = fit_alpha(Ls, KLs)
        is_Tc = abs(T - T_C) < 1e-3
        c = color_for_T(T)
        lw = 3 if is_Tc else 1.8
        ms = 12 if is_Tc else 8
        label = f"T={T:.3f}  α={alpha:.2f}" + ("  ← T_c" if is_Tc else "")
        Ls_fit = np.array([6, 40])
        axD.plot(Ls_fit, a * Ls_fit ** alpha, color=c, linewidth=lw, alpha=0.85)
        axD.plot(Ls, KLs, marker="o", linestyle="None", color=c, markersize=ms,
                 label=label, zorder=5 if is_Tc else 3)

    # reference slopes anchored to mid of data
    L_ref = np.array([6, 40])
    a_ext = 0.011  # roughly the off-critical prefactor
    axD.plot(L_ref, a_ext * L_ref ** 2.0, color="black", linestyle=":",
             linewidth=1.5, alpha=0.6, label="α=2 (extensive)")
    axD.plot(L_ref, a_ext * 4 * L_ref ** 1.0, color="black", linestyle="-.",
             linewidth=1.5, alpha=0.4, label="α=1 (perimeter)")

    axD.set_xscale("log")
    axD.set_yscale("log")
    axD.set_xticks([8, 16, 32])
    axD.set_xticklabels(["8", "16", "32"])
    axD.set_xlim(6, 40)
    axD.set_xlabel("Linear size L")
    axD.set_ylabel("KL_fwd  [nat]")
    axD.set_title("(d) Power-law fits  KL_fwd = a·L^α")
    axD.legend(loc="upper left", fontsize=9, ncol=1)
    axD.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Forward-KL FSS sweep — HS continuous-field training "
        "(L=8/16 default arch, L=32 bignet arch)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
