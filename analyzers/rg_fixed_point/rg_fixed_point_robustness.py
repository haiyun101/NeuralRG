"""Robustness checks for the RG fixed-point probe.

Addresses three concerns raised in the methodology critique of
`rg_fixed_point.py`:

  V1. The original probe uses per-position z-score (per-cell mean/std).
      A two-layer functional difference of the form 'same shape, scaled
      differently per position' would be invisible. Re-run with a
      GLOBAL z-score (one scalar mean and one scalar std for the whole
      4-element output) and check whether the rev-KL deep-MSE ~ 0
      survives.

  V2. The original probe feeds every scale-block the SAME N(0, I)
      input, which is not what blocks see in production. Re-run with
      a CHAIN-input probe: for each block f_s, give it the input it
      would see in the production composition --
          h_s := f_{s+1}(f_{s+2}(...(f_5(z))...)),  h_5 := z.
      Then compare f_s(h_s) vs f_{s+1}(h_{s+1}) on z-scored outputs.
      This matches what each block actually does in the generative
      direction.

  V3. Sanity check: is each scale-block f_s individually near-identity?
      If yes, the original 'deep MSE ~ 0' rev-KL signature is a
      triviality (two near-identity functions trivially agree on any
      input), not an RG fixed-point statement. Test by drawing
      z ~ N(0, I), passing through f_s, and measuring the residual
      r_s = mean(|f_s(z) - z|^2). For an exact identity, r_s = 0.

We rerun only the flows that are most informative:
  - T = 2.15  (low T baseline)
  - T_c hs_dataDriven  (forward-KL baseline)
  - T_c hs_bignet      (forward-KL, robustness)
  - T_c sym_bignet     (reverse-KL, the deep-MSE-zero method)
  - T_c pathgrad_bignet_long_ext (STL, the other deep-MSE-zero method)
  - T = 2.40  (high T baseline)

Outputs:
  analyzers/rg_fixed_point_robustness.csv
  analyzers/rg_fixed_point_robustness_v1_global.png  (variant 1)
  analyzers/rg_fixed_point_robustness_v2_chain.png   (variant 2)
  analyzers/rg_fixed_point_robustness_v3_identity.png (variant 3)
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
from flow_sample_diagnostic import build_flow
# Reuse helpers from the original probe
from rg_fixed_point import (
    FOLDERS as ORIG_FOLDERS,
    STYLE as ORIG_STYLE,
    latest_saving,
    load_flow,
    extract_scale_blocks,
    probe_scale_block,
)


# Subset of folders to re-probe -- the most informative ones.
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
    # L=32 sweeps
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
    # ----- Phase-2 P2.x: i2 + nrepeat=2 (D winner / C reference) -----
    "L=32 i2_stride8h32_nr2_b64 (P2.x D32)":     "data/32Ising_T2.269_hsBignet_i2_stride8h32_nr2_b64",
    "L=64 baseline_nr2_b16 (P2.x C64)":           "data/64Ising_T2.269_hsBignet_baseline_nr2_b16",
    "L=64 i2_stride8h32_nr2_b16 (P2.x D64 ★)":   "data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16",
    # ----- L=64 champion (2026-07-14) -----
    "L=64 fixdil+VP-1e-3 nr=1 (champion ★)":      "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16",
}

STYLE = {k: ORIG_STYLE[k] for k in FOLDERS}


# ---------------------------------------------------------------------
# Z-score variants
# ---------------------------------------------------------------------

def zscore_per_position(x, eps=1e-8):
    """Original probe's z-score: each of the 4 cells, independently
    across batch."""
    mean = x.mean(dim=0, keepdim=True)
    std  = x.std(dim=0,  keepdim=True).clamp_min(eps)
    return (x - mean) / std


def zscore_global(x, eps=1e-8):
    """V1 robustness: a single scalar mean and std across the entire
    (batch, channel, H, W) tensor. Preserves any per-position scaling
    asymmetry that per-position z-score would erase."""
    mean = x.mean()
    std  = x.std().clamp_min(eps)
    return (x - mean) / std


def mse(a, b):
    return float(((a - b) ** 2).mean().item())


# ---------------------------------------------------------------------
# V1: global z-score, same probe input
# ---------------------------------------------------------------------

def run_v1_global_zscore(folder, label, N, seed):
    print(f"\n--- V1 global z-score: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)
    outs_gl = []
    outs_pp = []
    with torch.no_grad():
        for s, block_list in enumerate(groups):
            o = probe_scale_block(block_list, z)
            outs_gl.append(zscore_global(o))
            outs_pp.append(zscore_per_position(o))
    rec_global = [mse(outs_gl[s], outs_gl[s + 1]) for s in range(len(outs_gl) - 1)]
    rec_perpos = [mse(outs_pp[s], outs_pp[s + 1]) for s in range(len(outs_pp) - 1)]
    for s, (mg, mp) in enumerate(zip(rec_global, rec_perpos)):
        print(f"   MSE(f_{s+1}, f_{s+2})  global={mg:.4f}   per-position={mp:.4f}")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                global_mse=rec_global, perpos_mse=rec_perpos)


# ---------------------------------------------------------------------
# V2: chain-input probe -- production composition input
# ---------------------------------------------------------------------

def run_v2_chain_input(folder, label, N, seed, zscore_fn=zscore_per_position):
    """For each block f_s, build the input it would see in production
    h_s = f_{s+1}(...(f_5(z))) and compare f_s(h_s) vs f_{s+1}(h_{s+1}).

    Production direction is generative (latent -> physical), so we
    apply scale-blocks in MERA's reversed order: f_5 first (top of the
    stack, indices 8-9) then ... then f_1 last (indices 0-1).

    With groups[0] = f_1 (finest, last in generative) and
    groups[-1] = f_5 (coarsest, first in generative), the production
    composition is groups[-1] -> groups[-2] -> ... -> groups[0].
    """
    print(f"\n--- V2 chain-input: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    S = len(groups)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)

    # Compute h_s = production input to f_s, for s = 1..S.
    # h_S is z (top of the generative stack).
    # h_{s} = f_{s+1}(h_{s+1}) for s < S.
    # Indexing in the groups list: groups[s-1] is f_s (1-indexed).
    h = [None] * S       # h[s-1] is h_s
    h[S - 1] = z         # h_S = z
    out = z
    with torch.no_grad():
        for s_idx in range(S - 1, 0, -1):    # s_idx = S-1, S-2, ..., 1
            # Apply f_{s_idx+1} (i.e. groups[s_idx]) to h_{s_idx+1}
            #   getting h_{s_idx} = f_{s_idx+1}(h_{s_idx+1})
            block = groups[s_idx]
            out = probe_scale_block(block, out)
            h[s_idx - 1] = out

    # Compute outputs in production: o_s = f_s(h_s) for s=1..S
    with torch.no_grad():
        outs = [zscore_fn(probe_scale_block(groups[i], h[i])) for i in range(S)]
    # Compare f_s(h_s) vs f_{s+1}(h_{s+1})  for s=1..S-1
    rec = [mse(outs[s], outs[s + 1]) for s in range(S - 1)]
    for s, m in enumerate(rec):
        print(f"   MSE(f_{s+1}(h_{s+1}), f_{s+2}(h_{s+2})) = {m:.4f}")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                chain_mse=rec)


# ---------------------------------------------------------------------
# V2b: chain-input probe with MERA-correct slot geometry
# ---------------------------------------------------------------------
#
# V2 above feeds the entire 4-element output of block f_{s+1} as the
# 2x2 patch input of block f_s. That does not match the production
# inverse pass: in MERA the dispatch index pattern at scale s reads
# patches whose 4 elements are at (0,0), (0, 2^s), (2^s, 0),
# (2^s, 2^s) offsets within a stride-2^(s+1) cell. The (0,0) offset
# is the *only* one on the next-deeper scale's stride-2^(s+1)
# sub-lattice -- i.e. the only patch position fed by the kept-coarse
# output of f_{s+1}. The other 3 positions are still on the
# stride-2^s sub-lattice, untouched by any deeper block, so they
# carry fresh N(0, I) latent in the inverse direction.
#
# V2b reproduces this by:
#   h_s[:, :, 0, 0]  <-  o_{s+1}[:, :, 0, 0]   (kept-coarse slot)
#   h_s[:, :, 0, 1]  <-  fresh N(0, I)
#   h_s[:, :, 1, 0]  <-  fresh N(0, I)
#   h_s[:, :, 1, 1]  <-  fresh N(0, I)
#
# Then runs probe_scale_block on h_s, compares f_s(h_s) vs
# f_{s+1}(h_{s+1}) on z-scored outputs.

def run_v2b_chain_input_one_slot(folder, label, N, seed,
                                 zscore_fn=zscore_per_position):
    """Like V2, but only the (0, 0) patch slot is fed by the previous
    block; the remaining 3 slots get fresh N(0, I). This matches the
    production inverse-direction composition for the kept-coarse
    sub-lattice of each scale.
    """
    print(f"\n--- V2b one-slot chain-input: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    S = len(groups)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N, 1, 2, 2, generator=g)

    h = [None] * S       # h[s-1] is h_s
    h[S - 1] = z         # h_S = top-of-stack latent
    out = z
    with torch.no_grad():
        for s_idx in range(S - 1, 0, -1):
            block = groups[s_idx]               # = f_{s_idx + 1}
            out = probe_scale_block(block, out)  # o_{s_idx + 1}
            # Build h_{s_idx} (which is h[s_idx - 1] in 0-index):
            # (0,0) <- kept-coarse from o_{s_idx + 1}, others fresh.
            h_next = torch.randn(N, 1, 2, 2, generator=g)
            h_next[:, :, 0, 0] = out[:, :, 0, 0]
            h[s_idx - 1] = h_next
            # The next iteration of `out = probe_scale_block(block, out)`
            # in V2's loop would use `out` as the full input to the
            # SHALLOWER block; in V2b we instead recompute out from
            # h_next on the next loop iteration:
            out = h_next

    # Compute outputs in production order: o_s = f_s(h_s) for s=1..S
    with torch.no_grad():
        outs = [zscore_fn(probe_scale_block(groups[i], h[i])) for i in range(S)]
    rec = [mse(outs[s], outs[s + 1]) for s in range(S - 1)]
    for s, m in enumerate(rec):
        print(f"   MSE(f_{s+1}(h_{s+1}), f_{s+2}(h_{s+2})) = {m:.4f}  [V2b one-slot]")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                chain_mse_oneslot=rec)


# ---------------------------------------------------------------------
# V3: identity-residual sanity check
# ---------------------------------------------------------------------

def run_v3_identity_check(folder, label, N, seed):
    """For each block f_s, compute the residual ||f_s(z) - z||^2 / ||z||^2
    on z ~ N(0, I). If f_s is near-identity, residual ~ 0.
    Also record raw mean-squared residual.

    We compute two normalisations:
      r_raw_s = E[ (f_s(z) - z)^2 ]
      r_rel_s = E[ (f_s(z) - z)^2 ] / E[ z^2 ]    (relative)
    """
    print(f"\n--- V3 identity-residual: {label} ---", flush=True)
    mera, L, T, epoch, _ = load_flow(folder)
    groups, _ = extract_scale_blocks(mera, L)
    torch.manual_seed(seed)
    z = torch.randn(N, 1, 2, 2)
    z_norm = float((z ** 2).mean().item())   # ~ 1.0 for N(0,I)
    rec_raw, rec_rel = [], []
    with torch.no_grad():
        for s, block_list in enumerate(groups):
            o = probe_scale_block(block_list, z)
            raw = float(((o - z) ** 2).mean().item())
            rel = raw / z_norm
            rec_raw.append(raw)
            rec_rel.append(rel)
            print(f"   f_{s+1}: residual raw={raw:.4f}   "
                  f"relative={rel:.4f}   (output std={o.std():.4f})")
    return dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                identity_raw=rec_raw, identity_rel=rec_rel)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def write_csv(v1, v2, v3, path, v2b=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch", "variant",
                    "scale_pair_or_block", "value"])
        for r in v1:
            for i, (g, p) in enumerate(zip(r["global_mse"], r["perpos_mse"])):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "v1_global", f"f_{i+1}->f_{i+2}", g])
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "v1_perpos", f"f_{i+1}->f_{i+2}", p])
        for r in v2:
            for i, m in enumerate(r["chain_mse"]):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "v2_chain", f"f_{i+1}->f_{i+2}", m])
        if v2b is not None:
            for r in v2b:
                for i, m in enumerate(r["chain_mse_oneslot"]):
                    w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                                "v2b_chain_oneslot", f"f_{i+1}->f_{i+2}", m])
        for r in v3:
            for i, (raw, rel) in enumerate(zip(r["identity_raw"], r["identity_rel"])):
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "v3_identity_raw", f"f_{i+1}", raw])
                w.writerow([r["label"], r["folder"], r["L"], r["T"], r["epoch"],
                            "v3_identity_rel", f"f_{i+1}", rel])
    print(f"wrote {path}")


def plot_v2b(v2b, savepath):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for r in v2b:
        st = dict(STYLE[r["label"]])
        st.setdefault("markersize", 8)
        xs = np.arange(1, len(r["chain_mse_oneslot"]) + 1)
        ax.plot(xs, r["chain_mse_oneslot"], label=r["label"], **st)
    ax.set_xlabel(r"scale pair  $f_s(h_s)$ vs $f_{s+1}(h_{s+1})$"
                  "  [V2b one-slot]")
    ax.set_ylabel("MSE on z-scored outputs")
    ax.set_title("RG probe — V2b: chain-input with MERA slot geometry\n"
                 "(patch (0,0) = kept-coarse from previous block; "
                 "patches (0,1), (1,0), (1,1) = fresh N(0, I))")
    ax.set_xticks(np.arange(1, 1 + max(len(r["chain_mse_oneslot"]) for r in v2b)))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_v1(v1, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    titles = ["V1: global z-score (single scalar per block)",
              "Reference: original per-position z-score"]
    keys = ["global_mse", "perpos_mse"]
    for ax, key, title in zip(axes, keys, titles):
        for r in v1:
            st = dict(STYLE[r["label"]])
            st.setdefault("markersize", 8)
            xs = np.arange(1, len(r[key]) + 1)
            ax.plot(xs, r[key], label=r["label"], **st)
        ax.set_xlabel("scale pair f_s -> f_{s+1}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(1, 1 + max(len(r[key]) for r in v1)))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MSE on z-scored outputs")
    axes[0].legend(framealpha=0.9, fontsize=8, loc="upper right")
    fig.suptitle("RG probe — V1 robustness: global vs per-position z-score",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_v2(v2, savepath):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for r in v2:
        st = dict(STYLE[r["label"]])
        st.setdefault("markersize", 8)
        xs = np.arange(1, len(r["chain_mse"]) + 1)
        ax.plot(xs, r["chain_mse"], label=r["label"], **st)
    ax.set_xlabel(r"scale pair  $f_s(h_s)$ vs $f_{s+1}(h_{s+1})$")
    ax.set_ylabel("MSE on z-scored outputs")
    ax.set_title("RG probe — V2 robustness: chain-input (production composition)")
    ax.set_xticks(np.arange(1, 1 + max(len(r["chain_mse"]) for r in v2)))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.annotate("each f_s receives its production input\n"
                "h_s = f_{s+1}(...(f_5(z))), h_5 = z\n"
                "instead of the same N(0,I) probe",
                xy=(0.98, 0.95), xycoords="axes fraction",
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          alpha=0.7))
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def plot_v3(v3, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    titles = ["raw MSE residual  E[(f_s(z) - z)^2]",
              "relative residual  E[(f_s(z) - z)^2] / E[z^2]"]
    keys = ["identity_raw", "identity_rel"]
    for ax, key, title in zip(axes, keys, titles):
        for r in v3:
            st = dict(STYLE[r["label"]])
            st.setdefault("markersize", 8)
            xs = np.arange(1, len(r[key]) + 1)
            ax.plot(xs, r[key], label=r["label"], **st)
        ax.set_xlabel(r"scale-block index s  (f_s, finest -> coarsest)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(1, 1 + max(len(r[key]) for r in v3)))
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", lw=0.6, alpha=0.3)
    axes[0].set_ylabel("residual MSE")
    axes[0].legend(framealpha=0.9, fontsize=8, loc="upper right")
    fig.suptitle("RG probe — V3: identity residual per scale-block "
                 "(small = block is near-identity)", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="analyzers")
    p.add_argument("--filter", default=None,
                   help="Regex; keep only FOLDERS labels matching (case-insensitive)")
    args = p.parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)

    folders = FOLDERS
    if args.filter:
        pat = re.compile(args.filter, re.IGNORECASE)
        folders = {k: v for k, v in FOLDERS.items() if pat.search(k)}
        print(f"[filter] '{args.filter}' matched {len(folders)}/{len(FOLDERS)} labels")

    v1, v2, v2b, v3 = [], [], [], []
    for label, folder in folders.items():
        try:
            v1.append(run_v1_global_zscore(folder, label, args.N, args.seed))
            v2.append(run_v2_chain_input(folder, label, args.N, args.seed))
            v2b.append(run_v2b_chain_input_one_slot(folder, label, args.N, args.seed))
            v3.append(run_v3_identity_check(folder, label, args.N, args.seed))
        except Exception as e:
            print(f"  FAILED on {label}: {e}")

    if not v1:
        raise SystemExit("no successful runs")

    write_csv(v1, v2, v3,
              os.path.join(args.outdir, "csv", "rg_fixed_point_robustness.csv"),
              v2b=v2b)
    plot_v1(v1, os.path.join(args.outdir, "figures", "rg_fixed_point_robustness_v1_global.png"))
    plot_v2(v2, os.path.join(args.outdir, "figures", "rg_fixed_point_robustness_v2_chain.png"))
    plot_v2b(v2b, os.path.join(args.outdir, "figures", "rg_fixed_point_robustness_v2b_chain_oneslot.png"))
    plot_v3(v3, os.path.join(args.outdir, "figures", "rg_fixed_point_robustness_v3_identity.png"))


if __name__ == "__main__":
    main()
