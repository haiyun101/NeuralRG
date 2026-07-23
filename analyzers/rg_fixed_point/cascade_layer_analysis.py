"""Per-scale statistical analysis of the flow's *actual generation cascade*.

Replaces the V0/V1/V2/V2b probes' fresh-N(0,I) inputs with the real chained
activations stored in mera_layer_flow_capture.pt.

Five metric sections at every MERA scale:
  A. per-scale marginal characterization (skew, kurtosis, KS to N(0,1))
  B. cross-scale self-similarity within a single model  ← real V0/V1
  C. G(r) axial + xi_s from raw kept-coarse fields
  D. forward-inverse consistency (y_s vs w_s at same lattice size)
  E. cross-model at same scale (champion vs A, etc.)

All A/B/D/E use `_gaussianized` (per-site z-scored) fields — gauge-fixed.
C uses `_kept` (raw) fields — G(r) needs the physical scale, and per-site
z-scoring destroys the amplitude information.

Usage:
  python analyzers/rg_fixed_point/cascade_layer_analysis.py \
      --captures  champion=<path_to_champion.pt>  A=<path_to_A.pt>  D=<path_to_D.pt> \
      --out       analyzers/rg_fixed_point/csv/cascade_layer_analysis.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import kstest, spearmanr, wasserstein_distance


# ─────────────────────── helpers ───────────────────────


def load_capture(path):
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d


def flatten_pool(t, cap=None):
    """(B, 1, L, L) → 1-D numpy of B*L² per-site scalars. Optionally subsample."""
    x = t.reshape(-1).float().numpy()
    if cap is not None and x.size > cap:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=cap, replace=False)
    return x


def marginal_stats(y_gz):
    """Skew, excess kurtosis, KS-to-N(0,1) on the per-site pool."""
    x = flatten_pool(y_gz, cap=50000)
    s = float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12))
    k = float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12) - 3.0)
    ks = float(kstest(x, "norm", args=(0.0, 1.0)).statistic)
    return dict(mean=float(x.mean()), std=float(x.std()),
                skew=s, kurt_excess=k, ks_to_N01=ks, n_sites=int(x.size))


def rbf_mmd2(x, y, cap=5000):
    """Unbiased MMD² with RBF kernel; median-heuristic bandwidth. x, y need not
    have the same dimensionality per sample if we flatten to (N, 1) marginals
    — for full-vector comparison the caller should ensure same shape."""
    x = torch.as_tensor(x, dtype=torch.float32).reshape(x.shape[0], -1)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(y.shape[0], -1)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"vector-MMD requires matching dims: {x.shape[1]} vs {y.shape[1]}")
    if x.shape[0] > cap:
        x = x[torch.randperm(x.shape[0])[:cap]]
    if y.shape[0] > cap:
        y = y[torch.randperm(y.shape[0])[:cap]]
    xy = torch.cat([x, y], 0)
    d2 = torch.cdist(xy, xy).pow(2)
    med = torch.median(d2[d2 > 0]).sqrt().clamp(min=1e-6)
    K = torch.exp(-d2 / (2 * med.item() ** 2))
    n, m = x.shape[0], y.shape[0]
    Kxx = (K[:n, :n].sum() - K[:n, :n].diag().sum()) / (n * (n - 1))
    Kyy = (K[n:, n:].sum() - K[n:, n:].diag().sum()) / (m * (m - 1))
    Kxy = K[:n, n:].mean()
    return float(Kxx + Kyy - 2 * Kxy)


def mmd2_marginal(y1, y2, cap=5000):
    """MMD² between pooled per-site marginals (works across differing lattice sizes)."""
    x = flatten_pool(y1, cap=cap * 5).reshape(-1, 1)
    y = flatten_pool(y2, cap=cap * 5).reshape(-1, 1)
    # subsample down to cap for the pairwise kernel
    rng = np.random.default_rng(0)
    if x.shape[0] > cap:
        x = rng.choice(x.ravel(), cap, replace=False).reshape(-1, 1)
    if y.shape[0] > cap:
        y = rng.choice(y.ravel(), cap, replace=False).reshape(-1, 1)
    return rbf_mmd2(x, y, cap=cap)


def w1_marginal(y1, y2):
    return float(wasserstein_distance(flatten_pool(y1, cap=50000),
                                       flatten_pool(y2, cap=50000)))


def ks_marginal(y1, y2):
    from scipy.stats import ks_2samp
    return float(ks_2samp(flatten_pool(y1, cap=50000),
                          flatten_pool(y2, cap=50000)).statistic)


def gr_axial(y_kept, r_max=None):
    """G(r) axial on raw kept-coarse field.
    y_kept: (B, 1, L, L) → return G[0..r_max], per-batch averaged."""
    B, _, L, _ = y_kept.shape
    y = y_kept.reshape(B, L, L).float().numpy()
    if r_max is None:
        r_max = L // 2 if L >= 4 else L - 1
    Gs = []
    for r in range(r_max + 1):
        Gx = (y * np.roll(y, -r, axis=2)).mean()
        Gy = (y * np.roll(y, -r, axis=1)).mean()
        Gs.append(0.5 * (Gx + Gy))
    return np.array(Gs)


def fit_xi(Gs):
    """Correlation length from exponential fit on log|G(r)/G(0)|."""
    Gs = np.asarray(Gs)
    G0 = Gs[0]
    if abs(G0) < 1e-12 or len(Gs) < 4:
        return float("nan")
    Gn = np.abs(Gs / G0)
    rs = np.arange(1, len(Gs))
    logG = np.log(Gn[1:] + 1e-12)
    mask = np.isfinite(logG) & (Gn[1:] > 1e-6)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(rs[mask], logG[mask], 1)
    return float(-1.0 / slope) if slope < 0 else float("inf")


def rank_corr_summary(y1, y2):
    """Spearman rank correlation of per-sample summary mean(|y|). y1, y2 same B."""
    s1 = y1.reshape(y1.shape[0], -1).abs().mean(-1).float().numpy()
    s2 = y2.reshape(y2.shape[0], -1).abs().mean(-1).float().numpy()
    return float(spearmanr(s1, s2).correlation)


# ─────────────────────── main analysis ───────────────────────


def analyze_one_model(name, cap, out_rows):
    """Sections A + B + C + D for a single model."""
    n_scales = cap["n_scales"]
    fwd = cap["forward"]
    inv = cap["inverse"]

    print(f"\n===================================================================")
    print(f"==== {name}  (L={cap['L']}, T={cap['T']:.4f}, ep={cap['checkpoint_epoch']}, "
          f"N_used={cap['N_used']}, {n_scales} scales)")
    print(f"===================================================================")

    # ── A. per-scale marginal ──
    print("\n[A] per-scale marginal characterization  (fwd, gaussianized)")
    print(f"  {'scale':<6} {'sites':>8} {'skew':>8} {'kurt(exc)':>10} {'KS→N(0,1)':>10}")
    for s in range(1, n_scales + 1):
        y = fwd.get(f"y_{s}_gaussianized")
        if y is None:
            continue
        st = marginal_stats(y)
        print(f"  y_{s:<4} {st['n_sites']:>8} {st['skew']:>+8.3f} "
              f"{st['kurt_excess']:>+10.3f} {st['ks_to_N01']:>10.3f}")
        out_rows.append(dict(model=name, section="A_marginal", scale=s,
                             metric="skew",       value=st["skew"]))
        out_rows.append(dict(model=name, section="A_marginal", scale=s,
                             metric="kurt_excess", value=st["kurt_excess"]))
        out_rows.append(dict(model=name, section="A_marginal", scale=s,
                             metric="ks_to_N01",  value=st["ks_to_N01"]))

    # ── B. cross-scale self-similarity ──
    print("\n[B] cross-scale self-similarity WITHIN model (real V0/V1)")
    print(f"  {'pair':<10} {'MMD²':>10} {'W1':>8} {'KS':>8}")
    for s in range(1, n_scales):
        y_a = fwd.get(f"y_{s}_gaussianized")
        y_b = fwd.get(f"y_{s+1}_gaussianized")
        if y_a is None or y_b is None:
            continue
        mmd = mmd2_marginal(y_a, y_b)
        w1  = w1_marginal(y_a, y_b)
        ks  = ks_marginal(y_a, y_b)
        print(f"  y_{s}→y_{s+1}   {mmd:>10.4g} {w1:>8.3f} {ks:>8.3f}")
        out_rows.append(dict(model=name, section="B_cross_scale", scale=s,
                             metric="mmd2_marginal", value=mmd))
        out_rows.append(dict(model=name, section="B_cross_scale", scale=s,
                             metric="w1_marginal",   value=w1))
        out_rows.append(dict(model=name, section="B_cross_scale", scale=s,
                             metric="ks_marginal",   value=ks))

    # ── C. G(r) physics ──
    print("\n[C] G(r) axial (raw kept), G(0) + correlation length xi_s")
    print(f"  {'scale':<6} {'L_s':>4} {'G(0)':>10} {'xi_s':>8}")
    for s in range(1, n_scales + 1):
        y_kept = fwd.get(f"y_{s}_kept")
        if y_kept is None:
            continue
        L_s = y_kept.shape[-1]
        if L_s < 2:
            print(f"  y_{s:<4} {L_s:>4} {'—':>10} {'—':>8}  (single site)")
            continue
        Gs = gr_axial(y_kept)
        xi = fit_xi(Gs)
        print(f"  y_{s:<4} {L_s:>4} {Gs[0]:>10.4f} {xi:>8.3f}")
        out_rows.append(dict(model=name, section="C_Gr", scale=s,
                             metric="G0", value=float(Gs[0])))
        out_rows.append(dict(model=name, section="C_Gr", scale=s,
                             metric="xi_s", value=xi))

    # ── D. forward-inverse consistency (y_s vs w_s at same lattice size) ──
    print("\n[D] forward-inverse consistency (y_s vs w_s)")
    print(f"  {'scale':<6} {'L_s':>4} {'MMD² marg':>10}")
    for s in range(1, n_scales):  # y_s ∈ 1..S-1 has matching w_s
        y = fwd.get(f"y_{s}_gaussianized")
        w = inv.get(f"w_{s}_gaussianized")
        if y is None or w is None:
            continue
        if y.shape[-1] != w.shape[-1]:
            print(f"  y_{s}/w_{s}: shape mismatch {tuple(y.shape[-2:])} vs {tuple(w.shape[-2:])}, skip")
            continue
        mmd = mmd2_marginal(y, w)
        print(f"  s={s:<4} {y.shape[-1]:>4} {mmd:>10.4g}")
        out_rows.append(dict(model=name, section="D_fwd_inv", scale=s,
                             metric="mmd2_marginal", value=mmd))


def cross_model(name_a, cap_a, name_b, cap_b, out_rows):
    """Section E: cross-model comparison at each scale."""
    print(f"\n===================================================================")
    print(f"==== [E] cross-model: {name_a}  vs  {name_b}")
    print(f"===================================================================")
    print(f"  {'scale':<6} {'L_s':>4}  {'MMD² vec':>10} {'W1':>8} {'ρ_sum':>8}")
    n = min(cap_a["n_scales"], cap_b["n_scales"])
    for s in range(1, n + 1):
        y_a = cap_a["forward"].get(f"y_{s}_gaussianized")
        y_b = cap_b["forward"].get(f"y_{s}_gaussianized")
        if y_a is None or y_b is None or y_a.shape[-1] != y_b.shape[-1]:
            continue
        nb = min(y_a.shape[0], y_b.shape[0])
        y_a = y_a[:nb]; y_b = y_b[:nb]
        try:
            mmd_v = rbf_mmd2(y_a.numpy(), y_b.numpy())
        except ValueError as e:
            mmd_v = float("nan")
        w1 = w1_marginal(y_a, y_b)
        rc = rank_corr_summary(y_a, y_b)
        print(f"  s={s:<4} {y_a.shape[-1]:>4}  {mmd_v:>10.4g} {w1:>8.3f} {rc:>+8.3f}")
        out_rows.append(dict(model=f"{name_a}_vs_{name_b}", section="E_cross_model",
                             scale=s, metric="mmd2_vec", value=mmd_v))
        out_rows.append(dict(model=f"{name_a}_vs_{name_b}", section="E_cross_model",
                             scale=s, metric="w1_marginal", value=w1))
        out_rows.append(dict(model=f"{name_a}_vs_{name_b}", section="E_cross_model",
                             scale=s, metric="rank_corr_sum", value=rc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="+", required=True,
                    help="name=<path> pairs, e.g. champion=/path/to/x.pt A=/path/to/y.pt")
    ap.add_argument("--out", required=True, help="CSV output")
    args = ap.parse_args()

    caps = {}
    for spec in args.captures:
        assert "=" in spec, f"bad --captures spec: {spec}"
        name, path = spec.split("=", 1)
        caps[name] = load_capture(path)
        print(f"[load] {name}: {path}  "
              f"L={caps[name]['L']}, ep={caps[name]['checkpoint_epoch']}")

    rows = []
    for name, cap in caps.items():
        analyze_one_model(name, cap, rows)

    names = list(caps.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross_model(names[i], caps[names[i]], names[j], caps[names[j]], rows)

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "section", "scale", "metric", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[csv] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
