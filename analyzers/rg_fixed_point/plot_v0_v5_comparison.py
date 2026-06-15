"""V0-V5 + raw/gauge comparison matrix figure (6 flows × 6 probes).

Visualizes how each probe lights up differently across the 6 main flows,
and how gauge-fixing changes the signal. Highlights fwd-KL hs_bignet
(currently best performer) and reveals which signals are "real structure"
vs "marginal-shape artefacts".

Output: analyzers/rg_fixed_point/figures/v0_v5_probe_panel.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----- Data extraction ---------------------------------------------------

FLOWS = [
    "T = 2.15",
    "T_c hs_dataDriven",
    "T_c hs_bignet",      # ← FOCUS
    "T_c sym_bignet",
    "T_c STL pathgrad",
    "T = 2.40",
]

PROBES = [
    "V0/V1 per-pos\nMSE f_4→f_5",
    "V2 chain\nMSE f_4→f_5",
    "V2b one-slot\nMSE f_4→f_5",
    "V3 rel r_5",
    "V4 zscore\nMSE f_3→f_4",
    "V5 RMS-G s=2",
]

# Raw values (from existing reports + CSVs)
RAW = np.array([
    # V0V1     V2       V2b      V3      V4     V5
    [1.92,    1.620,   1.508,   0.025,  1.039, 0.095],   # T=2.15
    [1.92,    1.940,   2.611,   5.840,  2.257, 0.068],   # hs_dataDriven (fwd-KL)
    [1.98,    1.940,   1.774,   0.300,  2.066, 0.049],   # hs_bignet (fwd-KL) ★
    [3e-5,    0.000,   1.489,   0.080,  0.011, 0.669],   # sym_bignet (rev-KL)
    [6e-5,    0.000,   1.489,   0.018,  0.002, 0.646],   # STL pathgrad
    [0.15,    0.140,   1.508,   0.130,  0.144, 0.065],   # T=2.40
])

# Gauge-fixed values
GAUGE = np.array([
    [0.789,   0.746,   1.636,   0.007,  0.781, 0.046],   # T=2.15
    [1.911,   1.927,   2.624,   0.079,  1.527, 0.043],   # hs_dataDriven
    [1.991,   1.937,   1.750,   0.022,  1.787, 0.029],   # hs_bignet ★
    [0.000,   0.000,   1.517,   0.001,  0.000, 0.539],   # sym_bignet
    [0.000,   0.000,   1.517,   0.001,  0.000, 0.522],   # STL pathgrad
    [0.152,   0.140,   1.557,   0.011,  0.087, 0.068],   # T=2.40
])

FOCUS_ROW = 2   # hs_bignet


# ----- Plot helpers ------------------------------------------------------

def add_text(ax, data, fmt="{:.3f}"):
    """Annotate each cell with its value."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = fmt.format(v) if v >= 1e-3 else f"{v:.1e}"
            color = "white" if v > 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color,
                    fontsize=8)


def make_panel():
    # Single-panel gauge-fixed heatmap (raw values dropped — only gauge is canonical).
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    eps = 1e-5
    log_gauge = np.log10(np.clip(GAUGE, eps, None))
    cmap = plt.cm.viridis
    im = ax.imshow(log_gauge, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(PROBES)))
    ax.set_xticklabels(PROBES, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(FLOWS)))
    ax.set_yticklabels(FLOWS, fontsize=11)
    ax.set_title("V0–V5 gauge-fixed values × 6 flows\n"
                 "(per-site quantile transform → N(0,1) marginal; only copula matters)\n"
                 "red highlight = fwd-KL hs_bignet (focus, currently best on V5)",
                 fontsize=12)
    add_text(ax, GAUGE)
    ax.add_patch(plt.Rectangle((-0.5, FOCUS_ROW - 0.5),
                                len(PROBES), 1, fill=False,
                                edgecolor="red", lw=3))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\log_{10}(\mathrm{gauge\;value})$", fontsize=10)

    plt.tight_layout()
    out_path = "analyzers/rg_fixed_point/figures/v0_v5_probe_panel.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    make_panel()
