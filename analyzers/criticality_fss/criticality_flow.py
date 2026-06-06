"""Criticality witnesses computed on samples drawn from the trained
forward-KL flow (`hs_dataDriven`), compared against the same
quantities on the HS dataset (criticality_summary.csv).

Outputs:
  criticality_flow_summary.csv  -- per-(L, T) flow numbers
  criticality_binder_flow.png
  criticality_chi_flow.png
  criticality_xi_over_L_flow.png
  criticality_PM_collapse_flow.png   (flow samples at T_c, rescaled)

Each plot overlays data (dashed/marker) and flow (solid/marker) on
the same axes so the universality test is visual: if the flow
reproduces the universality class, its curves should also cross at
T_c and converge to the same universal value.
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
from criticality_analysis import (
    LS, TS, TS_FLOAT, TS_LABEL, T_C,
    U4_TC, XI_OVER_L_TC, GAMMA_OVER_NU, BETA_OVER_NU,
    COLOR_BY_L, MARKER_BY_L,
    per_config_M, binder, susceptibility, xi_second_moment,
)
from flow_sample_diagnostic import build_flow, read_sigma


def folder_for(L, T_str):
    """Resolve the canonical forward-KL run folder for (L, T)."""
    candidates = [
        f"data/{L}Ising_T{T_str}_hs_dataDriven_default",
        f"data/{L}Ising_T{T_str}_hs_dataDriven",
    ]
    # try T_c numeric form too
    if T_str == "2.269185314213022":
        candidates += [
            f"data/{L}Ising_T2.269_hs_dataDriven",
        ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "savings")):
            return c
    raise FileNotFoundError(f"no flow folder for L={L} T={T_str}")


def sample_flow(folder, N, batch_size=128, seed=0, device="cpu"):
    """Draw N HS-field samples (physical scale) from the trained flow."""
    saving_files = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    ckpt_path = saving_files[-1]
    epoch = int(re.search(r"epoch(\d+)", ckpt_path).group(1))
    state = torch.load(ckpt_path, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state)
    sigma, _ = read_sigma(folder)

    torch.manual_seed(seed)
    xs = []
    n_done = 0
    with torch.no_grad():
        while n_done < N:
            b = min(batch_size, N - n_done)
            u, _ = fw.sample(b)
            x = (sigma * u).cpu().numpy().astype(np.float64).reshape(b, L, L)
            xs.append(x)
            n_done += b
    return np.concatenate(xs, axis=0), L, T, epoch


def gather_flow(N=8000, batch_size=128, seed=0):
    """Compute per-(L, T) statistics on flow samples."""
    out = {L: {} for L in LS}
    for L in LS:
        for T_str, T in zip(TS, TS_FLOAT):
            folder = folder_for(L, T_str)
            print(f"=== L={L} T={T_str}  folder={folder} ===", flush=True)
            x, Lf, Tf, ep = sample_flow(folder, N, batch_size, seed)
            assert Lf == L
            M = per_config_M(x)
            row = dict(
                T=T, L=L, N=int(N),
                epoch=ep,
                absM=float(np.abs(M).mean()),
                M2=float((M ** 2).mean()),
                M4=float((M ** 4).mean()),
                U4=float(binder(M)),
                chi=float(susceptibility(M, L)),
                xi_eff=xi_second_moment(x, L),
            )
            out[L][T] = row
            print(f"   ep {ep}  U4={row['U4']:.4f}  chi={row['chi']:.3f}  "
                  f"xi/L={row['xi_eff']/L:.4f}  |M|={row['absM']:.4f}",
                  flush=True)
    return out


def load_data_csv(path):
    """Load the HS-dataset baseline CSV produced by criticality_analysis."""
    out = {L: {} for L in LS}
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            L = int(r["L"]); T = float(r["T"])
            if L not in out: continue
            out[L][T] = {k: (float(v) if v else None)
                         for k, v in r.items() if k not in ("L", "N")}
            out[L][T]["L"] = L
            out[L][T]["T"] = T
    return out


def write_summary_csv(out, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "T", "epoch", "N", "absM", "M2", "M4",
                    "U4", "chi", "xi_eff", "xi_over_L"])
        for L in LS:
            for T in sorted(out[L].keys()):
                r = out[L][T]
                w.writerow([L, T, r["epoch"], r["N"], r["absM"],
                            r["M2"], r["M4"], r["U4"], r["chi"],
                            r["xi_eff"],
                            r["xi_eff"]/L if r["xi_eff"] else None])
    print(f"wrote {path}")


def _overlay(ax, data, flow, key, ylabel, title):
    for L in LS:
        Ts = sorted(flow[L].keys())
        ax.plot(Ts, [data[L][T][key] for T in Ts],
                marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=1.5, markersize=6, linestyle=":",
                alpha=0.7, label=f"L={L} data")
        ax.plot(Ts, [flow[L][T][key] for T in Ts],
                marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=2.2, markersize=8, linestyle="-",
                label=f"L={L} flow")
    ax.axvline(T_C, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"$T_c = {T_C:.4f}$")
    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)


def panel_binder_flow(data, flow, savepath):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _overlay(ax, data, flow, "U4",
             r"$U_4$",
             "Binder cumulant — flow (solid) vs HS data (dotted)")
    ax.axhline(U4_TC, color="black", linestyle=":", linewidth=1.0,
               alpha=0.5, label=f"2D Ising $U_4(T_c) \\approx {U4_TC}$")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_chi_flow(data, flow, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    _overlay(axes[0], data, flow, "chi",
             r"$\chi$",
             r"Susceptibility — flow vs data")
    axes[0].set_yscale("log")

    ax = axes[1]
    Ls = np.array(LS, dtype=float)
    chi_data = np.array([data[L][T_C]["chi"] for L in LS])
    chi_flow = np.array([flow[L][T_C]["chi"] for L in LS])
    s_d, i_d = np.polyfit(np.log(Ls), np.log(chi_data), 1)
    s_f, i_f = np.polyfit(np.log(Ls), np.log(chi_flow), 1)
    ax.loglog(Ls, chi_data, "o", color="black", markersize=10,
              label=f"data slope = {s_d:.3f}")
    ax.loglog(Ls, chi_flow, "D", color="red", markersize=10,
              label=f"flow slope = {s_f:.3f}")
    Lf = np.linspace(0.9*Ls[0], 1.1*Ls[-1], 100)
    ax.loglog(Lf, np.exp(i_d)*Lf**s_d, "-", color="black", alpha=0.5)
    ax.loglog(Lf, np.exp(i_f)*Lf**s_f, "-", color="red", alpha=0.5)
    ax.loglog(Lf, chi_data[0]*(Lf/Ls[0])**GAMMA_OVER_NU,
              "--", color="green",
              label=f"Onsager $\\gamma/\\nu = {GAMMA_OVER_NU}$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi(T_c, L)$")
    ax.set_title("FSS at $T_c$: data vs flow vs Onsager")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_xi_over_L_flow(data, flow, savepath):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for L in LS:
        Ts = sorted(flow[L].keys())
        ax.plot(Ts, [data[L][T]["xi_eff"]/L for T in Ts],
                marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=1.5, markersize=6, linestyle=":",
                alpha=0.7, label=f"L={L} data")
        ax.plot(Ts, [flow[L][T]["xi_eff"]/L for T in Ts],
                marker=MARKER_BY_L[L], color=COLOR_BY_L[L],
                linewidth=2.2, markersize=8, linestyle="-",
                label=f"L={L} flow")
    ax.axvline(T_C, color="grey", linestyle="--", alpha=0.7,
               label=f"$T_c$")
    ax.axhline(XI_OVER_L_TC, color="black", linestyle=":",
               alpha=0.5,
               label=f"2D Ising $\\xi_{{eff}}/L \\approx {XI_OVER_L_TC}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\xi_{eff}/L$")
    ax.set_title(r"$\xi_{eff}/L$ crossing — flow vs data")
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def panel_PM_collapse_flow(savepath, N=8000):
    """P(M * L^(beta/nu)) at T_c, flow side."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for L in LS:
        folder = folder_for(L, "2.269185314213022")
        x, _, _, ep = sample_flow(folder, N)
        M = per_config_M(x)
        M_scaled = M * L ** BETA_OVER_NU
        rng = 1.05 * float(np.abs(M_scaled).max())
        bins = np.linspace(-rng, rng, 81)
        ax.hist(M_scaled, bins=bins, density=True, histtype="step",
                color=COLOR_BY_L[L], linewidth=2.2, label=f"L = {L} flow")
    ax.set_xlabel(r"$M \cdot L^{\beta/\nu}$")
    ax.set_ylabel("density")
    ax.set_title(r"Flow $P(M \cdot L^{\beta/\nu})$ at $T_c$ — "
                 r"flow-side collapse test")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="analyzers")
    args = p.parse_args()

    flow = gather_flow(args.N, args.batch, args.seed)
    write_summary_csv(flow, f"{args.outdir}/criticality_flow_summary.csv")

    data_csv = os.path.join(args.outdir, "criticality_summary.csv")
    if not os.path.exists(data_csv):
        raise SystemExit(f"missing {data_csv} -- run criticality_analysis.py first")
    data = load_data_csv(data_csv)

    panel_binder_flow(data, flow, f"{args.outdir}/criticality_binder_flow.png")
    panel_chi_flow(data, flow,    f"{args.outdir}/criticality_chi_flow.png")
    panel_xi_over_L_flow(data, flow, f"{args.outdir}/criticality_xi_over_L_flow.png")
    panel_PM_collapse_flow(       f"{args.outdir}/criticality_PM_collapse_flow.png",
                                  N=args.N)


if __name__ == "__main__":
    main()
