"""Data-driven cross-L layer alignment analysis.

For each pair (s_32, s_64) of scale indices in L=32 and L=64 champion captures,
compute cross-L similarity in gauge (per-site z-scored). Also compute
within-model adjacent-pair similarity to identify each model's "internally
self-similar" layer set.

Key question: does the internally self-similar layer set at L=32 align with
the internally self-similar layer set at L=64 UNDER SOME SHIFT? If yes, the
self-similar core is a scale-invariant fixed point that can be transferred
to larger L.

Alignment considered:
  - Same scale index (s_32 = s_64): requires shape match, only works for s ≤ 4
    since L=32 has scales 1..5 and L=64 scales 1..6
  - Same output lattice size (s_64 = s_32 + 1): after L=64 does one extra
    coarse-graining, its output lattice matches L=32's at one earlier scale
  - Same relative position from coarsest (s_32 = 5 - k, s_64 = 6 - k)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade_layer_analysis import rbf_mmd2, w1_marginal, rank_corr_summary


def load_cap(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def get_y_gaussianized(cap, s):
    return cap["forward"].get(f"y_{s}_gaussianized")


def within_model_adjacent(cap, name):
    """MMD² + W1 between y_s and y_{s+1} at each s (gauge-fixed)."""
    n_scales = cap["n_scales"]
    print(f"\n[{name}] within-model adjacent-scale similarity (MSE_adj):")
    print(f"  {'scale pair':<12} {'lat size a→b':<14} {'MMD²':>10} {'W1':>8} {'rank_corr':>10}")
    rows = []
    for s in range(1, n_scales):
        y_a = get_y_gaussianized(cap, s)
        y_b = get_y_gaussianized(cap, s + 1)
        if y_a is None or y_b is None:
            continue
        # Since y_{s+1} is at coarser lattice, upsample smaller to compare?
        # Actually for adjacent within-model, MERA's y_s is at (L/2^s) and
        # y_{s+1} is at (L/2^{s+1}) — different sizes. cascade does
        # DISTRIBUTIONAL comparison (KS, MMD², W1) on flattened marginals,
        # NOT pointwise. So we compare marginal distributions.
        y_a_flat = y_a.reshape(-1, 1)
        y_b_flat = y_b.reshape(-1, 1)
        nb = min(y_a_flat.shape[0], y_b_flat.shape[0])
        y_a_flat = y_a_flat[:nb]
        y_b_flat = y_b_flat[:nb]
        try:
            mmd = rbf_mmd2(y_a_flat.numpy(), y_b_flat.numpy())
        except Exception:
            mmd = float("nan")
        w1 = w1_marginal(y_a_flat, y_b_flat)
        print(f"  {s}→{s+1}         {y_a.shape[-1]:<3}→{y_b.shape[-1]:<3}         "
              f"{mmd:>10.4g} {w1:>8.3f}")
        rows.append((s, s + 1, y_a.shape[-1], y_b.shape[-1], mmd, w1))
    return rows


def cross_L_all_pairs(cap_32, cap_64):
    """Compute cross-L MMD² and W1 for all (s_32, s_64) scale pairs where
    output lattice sizes match."""
    n_32 = cap_32["n_scales"]
    n_64 = cap_64["n_scales"]
    print(f"\n[cross-L] all scale pairs with matching lattice size:")
    print(f"  {'s_32':<4} {'s_64':<4} {'lat size':<9} {'MMD²':>10} {'W1':>8} {'rank_corr':>10}")
    rows = []
    for s32 in range(1, n_32 + 1):
        y32 = get_y_gaussianized(cap_32, s32)
        if y32 is None:
            continue
        size32 = y32.shape[-1]
        for s64 in range(1, n_64 + 1):
            y64 = get_y_gaussianized(cap_64, s64)
            if y64 is None:
                continue
            size64 = y64.shape[-1]
            if size32 != size64:
                continue
            # Same lattice size — can do proper spatial + marginal comparison
            nb = min(y32.shape[0], y64.shape[0])
            y32_ = y32[:nb]
            y64_ = y64[:nb]
            try:
                mmd = rbf_mmd2(y32_.numpy(), y64_.numpy())
            except Exception:
                mmd = float("nan")
            w1 = w1_marginal(y32_, y64_)
            try:
                rc = rank_corr_summary(y32_, y64_)
            except Exception:
                rc = float("nan")
            print(f"  {s32:<4} {s64:<4} {size32:<9} {mmd:>10.4g} {w1:>8.3f} {rc:>+10.3f}")
            rows.append((s32, s64, size32, mmd, w1, rc))
    return rows


def main():
    L32_path = "data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/mera_layer_flow_capture.pt"
    L64_path = "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/mera_layer_flow_capture.pt"

    print(f"[load] {L32_path}")
    cap_32 = load_cap(L32_path)
    print(f"  L=32 capture: {cap_32['n_scales']} scales, "
          f"N_used={cap_32.get('N_used', '?')}")
    print(f"  y_kept lattice sizes: "
          f"{[get_y_gaussianized(cap_32, s).shape[-1] if get_y_gaussianized(cap_32, s) is not None else '?' for s in range(1, cap_32['n_scales']+1)]}")

    print(f"\n[load] {L64_path}")
    cap_64 = load_cap(L64_path)
    print(f"  L=64 capture: {cap_64['n_scales']} scales, "
          f"N_used={cap_64.get('N_used', '?')}")
    print(f"  y_kept lattice sizes: "
          f"{[get_y_gaussianized(cap_64, s).shape[-1] if get_y_gaussianized(cap_64, s) is not None else '?' for s in range(1, cap_64['n_scales']+1)]}")

    # 1. Within-model adjacent-scale similarity
    within_32 = within_model_adjacent(cap_32, "L=32 champion")
    within_64 = within_model_adjacent(cap_64, "L=64 champion")

    # 2. Cross-L: all pairs (s_32, s_64) with matched lattice size
    cross = cross_L_all_pairs(cap_32, cap_64)

    # 3. Summary — align by matched-size
    print(f"\n{'='*72}")
    print(f"SUMMARY — matched-lattice-size alignment (s_64 = s_32 + 1)")
    print(f"{'='*72}")
    print(f"{'lat size':<8} {'s_32 within-model MSE':<24} {'s_64 within-model MSE':<24} {'cross-L MSE':<12}")

    # Within-model MSE indexed by lattice size at the START of the pair
    # within_32: (s_a, s_b, size_a, size_b, mmd, w1). Use size_a as key.
    w32_by_size = {r[2]: r[5] for r in within_32}   # size_a → W1
    w64_by_size = {r[2]: r[5] for r in within_64}
    cross_by_size = {r[2]: r[4] for r in cross}     # size → W1

    # Show for each lattice size (from L=32 perspective)
    for r in within_32:
        size = r[2]
        w32 = f"{w32_by_size.get(size, float('nan')):.3f}" if size in w32_by_size else "—"
        w64 = f"{w64_by_size.get(size, float('nan')):.3f}" if size in w64_by_size else "—"
        cx = f"{cross_by_size.get(size, float('nan')):.3f}" if size in cross_by_size else "—"
        print(f"{size:<8} {w32:<24} {w64:<24} {cx:<12}")


if __name__ == "__main__":
    main()
