"""V6-style CNN offload metrics for HierarchicalConditionalGaussian (HCG).

Companion to rg_v6_cnn_offload.py, which only handles single-CNN
ConditionalGaussian (i2 cells). This script produces the same
per-CNN offload scalars for HCG's multi-CNN prior, one row per level:

  mean_sigma, std_sigma, mean_abs_log_sigma
  cnn_mu_rms / z_at_level_rms           (μ share of level-k signal)
  ks_raw, w1_raw, kl_gauss_raw          (level-k marginal vs N(0,1) — no CNN)
  ks_whit, w1_whit, kl_gauss_whit       (level-k marginal after CNN whitening)
  improvement columns                    (raw − whit; positive = CNN cleans up)

Usage:
  python analyzers/rg_fixed_point/hcg_cnn_offload.py \\
      --cells L32_champion:data/32Ising_.../:9500 \\
              L64_champion:data/64Ising_.../:13500 \\
      --N 2000 --device cpu \\
      --csv-out analyzers/csv/rg_v6_hcg_champion_offload.csv
"""
import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import scipy.stats as sps
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_sample_diagnostic import build_flow
from rg_fixed_point_v4_dataforward import load_hs_data


def latest_saving(folder, prefer_epoch=None):
    import glob
    savs = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                  key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if prefer_epoch is None:
        return savs[-1]
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - prefer_epoch))
    return savs[idx]


def get_flow_and_prior(folder, prefer_epoch=None, device="cpu"):
    ckpt = latest_saving(folder, prefer_epoch)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    prior = fw.flow.prior if hasattr(fw, "flow") else fw.prior
    mera = fw.flow if hasattr(fw, "flow") else fw
    return fw, mera, prior, L, T, epoch


def kl_gauss_scalar(samples):
    """MC-friendly ½ (mu² + σ² − 1 − log σ²) approximation to KL(q || N(0,1))."""
    s = samples.flatten().double().cpu().numpy()
    if s.size == 0:
        return float("nan")
    mu = s.mean()
    var = s.var()
    if var <= 0:
        return float("nan")
    return 0.5 * (mu * mu + var - 1.0 - math.log(var))


def standardised_KS_W1(s, n_max=200_000, seed=0):
    arr = s.flatten().double().cpu().numpy()
    if arr.size == 0:
        return float("nan"), float("nan")
    mu = arr.mean()
    sd = arr.std() + 1e-12
    arr = (arr - mu) / sd
    if arr.size > n_max:
        rng = np.random.default_rng(seed)
        arr = rng.choice(arr, n_max, replace=False)
    ks = float(sps.kstest(arr, "norm").statistic)
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal(arr.size)
    arr_s = np.sort(arr)
    ref_s = np.sort(ref)
    w1 = float(np.mean(np.abs(arr_s - ref_s)))
    return ks, w1


def run_one(label, folder, prefer_epoch=None, N=2000, device="cpu"):
    print(f"\n=== {label}  folder={folder}", flush=True)
    fw, mera, prior, L, T, epoch = get_flow_and_prior(folder, prefer_epoch, device)
    print(f"  L={L} T={T} ep={epoch}  prior={prior.__class__.__name__}", flush=True)

    if prior.__class__.__name__ != "HierarchicalConditionalGaussian":
        print(f"  ⚠ not HCG (got {prior.__class__.__name__}) — skipping")
        return []

    # Push HS data through MERA to get latent z
    samples = load_hs_data(L, T, N, device=device)
    print(f"  loaded {tuple(samples.shape)} HS samples", flush=True)
    with torch.no_grad():
        z, _ = mera.forward(samples)
    print(f"  latent z: shape={tuple(z.shape)}  mean={z.mean():+.3f}  std={z.std():.3f}", flush=True)

    strides = prior.strides
    K = prior.K
    print(f"  {K} levels (strides {strides}), scale_shared={prior.scale_shared}", flush=True)

    rows = []
    for k in range(1, K):                                              # levels 1..K-1
        mask_k = prior._buffers[f"level_mask_{k}"].to(z.dtype)         # (1, 1, L, L)
        mask_b = mask_k.bool().expand_as(z)                            # boolean

        # CNN output at ALL positions; caller masks to level-k sites
        with torch.no_grad():
            mu_k, log_sigma_k = prior._mu_logsig_level_k(z, k)
        sigma_k = torch.exp(log_sigma_k)

        z_lvl   = z[mask_b]
        mu_lvl  = mu_k[mask_b]
        sig_lvl = sigma_k[mask_b]
        log_sig_lvl = log_sigma_k[mask_b]

        z_rms   = float(z_lvl.pow(2).mean().sqrt().item())
        mu_rms  = float(mu_lvl.pow(2).mean().sqrt().item())
        mean_sig = float(sig_lvl.mean().item())
        std_sig  = float(sig_lvl.std().item())
        mean_abs_log_sig = float(log_sig_lvl.abs().mean().item())

        # Raw: level-k z's marginal
        ks_raw, w1_raw = standardised_KS_W1(z_lvl)
        kl_raw = kl_gauss_scalar(z_lvl)

        # Whitened: (z − μ) / σ
        z_whit = (z_lvl - mu_lvl) / (sig_lvl + 1e-12)
        ks_whit, w1_whit = standardised_KS_W1(z_whit)
        kl_whit = kl_gauss_scalar(z_whit)

        row = dict(
            label=label, folder=folder, L=L, T=T, epoch=epoch,
            level=k, stride=strides[k], sites=prior.sites_per_level[k],
            z_at_level_rms=z_rms, cnn_mu_rms=mu_rms,
            cnn_mu_rms_over_z_rms=mu_rms / (z_rms + 1e-12),
            mean_sigma=mean_sig, std_sigma=std_sig, mean_abs_log_sigma=mean_abs_log_sig,
            ks_raw=ks_raw, ks_whit=ks_whit, ks_improvement=ks_raw - ks_whit,
            w1_raw=w1_raw, w1_whit=w1_whit, w1_improvement=w1_raw - w1_whit,
            kl_gauss_raw=kl_raw, kl_gauss_whit=kl_whit, kl_improvement=kl_raw - kl_whit,
        )
        rows.append(row)
        print(f"    L{k} (stride={strides[k]:2d}, {prior.sites_per_level[k]:>5} sites): "
              f"||μ||/||z||={row['cnn_mu_rms_over_z_rms']:.3f}  "
              f"⟨σ⟩={mean_sig:.3f}  "
              f"KS: {ks_raw:.4f}→{ks_whit:.4f} (Δ={ks_raw-ks_whit:+.4f})  "
              f"KL: {kl_raw:.3f}→{kl_whit:.3f} (Δ={kl_raw-kl_whit:+.3f})",
              flush=True)
    return rows


def write_csv(rows, path):
    cols = ["label", "folder", "L", "T", "epoch", "level", "stride", "sites",
            "z_at_level_rms", "cnn_mu_rms", "cnn_mu_rms_over_z_rms",
            "mean_sigma", "std_sigma", "mean_abs_log_sigma",
            "ks_raw", "ks_whit", "ks_improvement",
            "w1_raw", "w1_whit", "w1_improvement",
            "kl_gauss_raw", "kl_gauss_whit", "kl_improvement"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nwrote {path}  ({len(rows)} rows)", flush=True)


def parse_cell(spec):
    """Parse "label:folder[:epoch]" spec."""
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    raise ValueError(f"bad cell spec: {spec} (expected label:folder[:epoch])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="one or more 'label:folder[:epoch]' specs")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv-out", default="analyzers/csv/rg_v6_hcg_champion_offload.csv")
    args = ap.parse_args()

    all_rows = []
    for spec in args.cells:
        label, folder, prefer_epoch = parse_cell(spec)
        rows = run_one(label, folder, prefer_epoch, args.N, args.device)
        all_rows.extend(rows)
    if all_rows:
        write_csv(all_rows, args.csv_out)
        # Compact verdict
        print("\n" + "=" * 78)
        print("HCG champion CNN offload — per-level verdict")
        print("=" * 78)
        print(f"{'cell':<20}{'L':>4}{'stride':>7}{'||μ||/||z||':>13}{'⟨σ⟩':>7}{'KS_raw':>9}{'KS_whit':>9}{'ΔKS':>8}{'ΔKL':>8}")
        for r in all_rows:
            print(f"{r['label']:<20}{r['level']:>4}{r['stride']:>7}"
                  f"{r['cnn_mu_rms_over_z_rms']:>13.3f}{r['mean_sigma']:>7.3f}"
                  f"{r['ks_raw']:>9.4f}{r['ks_whit']:>9.4f}"
                  f"{r['ks_improvement']:>+8.4f}{r['kl_improvement']:>+8.3f}")


if __name__ == "__main__":
    main()
