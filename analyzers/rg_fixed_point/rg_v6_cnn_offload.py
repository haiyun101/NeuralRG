"""V6 — CNN offload measurement (does the conditional-Gaussian prior absorb
physics that V5 attributes to "MERA deviating from Wilson"?).

Hypothesis
----------
For i2 cells, the conditional prior P(z_fast | z_slow; CNN) can absorb part
of the Ising short-range structure. MERA forward then does NOT need to fully
Gaussianize z_fast — it can leave residual coupling to z_slow, because the
CNN-prior accepts that. V5 only probes MERA, so MERA "looks non-Wilson" not
because MERA learned a different fixed point, but because part of the work
was off-loaded to the prior.

What V6 measures
----------------
For each cell (a) the magnitude of the CNN's μ, σ output on real latents
and (b) whether CNN whitening of z makes the latent look truly isotropic.

Per cell:
  A. CNN strength:
     ||μ_fast||_RMS / ||z_fast||_RMS  — fraction of fast variance explained
                                         by the CNN-mean
     mean(σ_fast)                     — average CNN-std at fast positions
     std(σ_fast)                      — spread of CNN-std
     mean(|log σ_fast|)               — magnitude of log-σ excursion (0 if CNN
                                         degenerates to identity)
  B. Latent distance from N(0, I):
     raw  — KS / W1 between standardized z_fast and N(0,1)
     z̃   — KS / W1 between CNN-whitened (z_fast - μ)/σ and N(0,1)
     improvement = raw - z̃  — how much structure CNN cleans up

  C. KL divergence vs N(0,1) (Monte Carlo estimate from latent samples):
     KL_raw  = mean( log q(z_fast) − log N(z_fast; 0,1) )    [coarse]
     KL_whit = mean( log q(z̃)     − log N(z̃; 0,1) )         [coarse]

Cells that *are* offloading physics to CNN should show large A values, large
improvement in B, and KL_whit << KL_raw.

For Gaussian-prior cells (A baseline, C nrepeat-only) the CNN is absent;
report a single row with ``i2=False`` and skip CNN columns.

Output
------
CSV: analyzers/csv/rg_v6_cnn_offload.csv
Verdict block printed to stdout summarising whether CNN offload correlates
with V5 RMS-G deviation.
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import build_flow
from rg_fixed_point import latest_saving
from rg_fixed_point_v4_dataforward import FOLDERS, load_hs_data


def get_mera_and_prior(folder, device="cpu"):
    ckpt = latest_saving(folder)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    inner = fw.flow if hasattr(fw, "flow") else fw   # peel Symmetrized
    mera = inner
    prior = inner.prior
    return mera, prior, L, T, epoch


def is_cond_prior(prior):
    return prior.__class__.__name__ == "ConditionalGaussian"


def mera_forward_full(mera, x):
    """Run MERA.forward(x) and return the final latent z = MERA.forward(x).

    Uses the Flow.forward method (preserves all coupling masks etc.).
    """
    with torch.no_grad():
        z, _ = mera.forward(x)
    return z


def kl_to_standard_normal_mc(samples):
    """Coarse MC estimate of KL[q(z) || N(0,1)] from samples z.

    KL = E_q[log q - log p]. We don't have q(z) closed-form, so we use the
    Gaussian-approximation gap:  KL ≈ ½(mu² + sig² - 1 - log sig²) where
    (mu, sig²) are sample mean / variance. This is exact iff q is also
    Gaussian; for our diagnostic purpose it tracks the right direction
    (larger when q deviates from N(0,1) in mean / variance).
    """
    s = samples.flatten().double().cpu().numpy()
    mu = s.mean()
    var = s.var()
    if var <= 0:
        return float("nan")
    # ½ (mu² + sig² - 1 - log sig²)
    return 0.5 * (mu * mu + var - 1.0 - math.log(var))


def standardised_KS_W1(s, ref="normal", n_max=200_000):
    """KS, W1 distance of standardised s vs N(0,1)."""
    arr = s.flatten().double().cpu().numpy()
    mu = arr.mean()
    sd = arr.std() + 1e-12
    arr = (arr - mu) / sd
    if arr.size > n_max:
        rng = np.random.default_rng(0)
        arr = rng.choice(arr, n_max, replace=False)
    ks = float(sps.kstest(arr, "norm").statistic)
    # W1 vs N(0,1) sampled at the same size
    rng = np.random.default_rng(0)
    ref_samples = rng.standard_normal(arr.size)
    arr_s = np.sort(arr)
    ref_s = np.sort(ref_samples)
    w1 = float(np.mean(np.abs(arr_s - ref_s)))
    return ks, w1


def run_one(folder, label, N=2000, device="cpu"):
    print(f"\n=== {label}  folder={folder} ===", flush=True)
    mera, prior, L, T, epoch = get_mera_and_prior(folder, device=device)
    print(f"  L={L} T={T} ep={epoch}  prior={prior.__class__.__name__}")

    samples = load_hs_data(L, T, N, device=device)
    print(f"  loaded {samples.shape} from HS dataset", flush=True)

    z = mera_forward_full(mera, samples)            # (B, 1, L, L)
    print(f"  latent z shape={tuple(z.shape)}  mean={z.mean().item():+.3f}  std={z.std().item():.3f}", flush=True)

    out = dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
               cond_prior=is_cond_prior(prior))

    if is_cond_prior(prior):
        with torch.no_grad():
            mu, log_sigma = prior._mu_logsig(z)
            sigma = torch.exp(log_sigma)
        slow = prior.slow_mask.to(z.dtype)
        fast = prior.fast_mask.to(z.dtype)
        fast_b = fast.bool().expand_as(z)

        z_fast = z[fast_b]
        mu_fast = mu[fast_b]
        sg_fast = sigma[fast_b]
        ls_fast = log_sigma[fast_b]

        rms_z = float(torch.sqrt((z_fast ** 2).mean()).item())
        rms_mu = float(torch.sqrt((mu_fast ** 2).mean()).item())
        out["cnn_mu_rms_over_z_rms"] = rms_mu / (rms_z + 1e-12)
        out["cnn_mu_rms"] = rms_mu
        out["z_fast_rms"] = rms_z
        out["mean_sigma"] = float(sg_fast.mean().item())
        out["std_sigma"] = float(sg_fast.std().item())
        out["mean_abs_log_sigma"] = float(ls_fast.abs().mean().item())

        # raw vs whitened latent distance to N(0,1)
        z_tilde = (z_fast - mu_fast) / sg_fast
        ks_raw, w1_raw = standardised_KS_W1(z_fast)
        ks_wh,  w1_wh  = standardised_KS_W1(z_tilde)
        out["ks_raw"] = ks_raw
        out["ks_whit"] = ks_wh
        out["w1_raw"] = w1_raw
        out["w1_whit"] = w1_wh
        out["ks_improvement"] = ks_raw - ks_wh
        out["w1_improvement"] = w1_raw - w1_wh

        # KL via Gaussian-gap on raw vs whitened
        out["kl_gauss_raw"] = kl_to_standard_normal_mc(z_fast)
        out["kl_gauss_whit"] = kl_to_standard_normal_mc(z_tilde)
        out["kl_improvement"] = out["kl_gauss_raw"] - out["kl_gauss_whit"]

        # Slow side: report distance vs N(0,1) (no whitening — slow is iid)
        slow_b = slow.bool().expand_as(z)
        z_slow = z[slow_b]
        ks_slow, w1_slow = standardised_KS_W1(z_slow)
        out["ks_slow"] = ks_slow
        out["w1_slow"] = w1_slow

        print(f"  CNN ||mu||/||z|| = {out['cnn_mu_rms_over_z_rms']:.3f}  "
              f"mean(sigma) = {out['mean_sigma']:.3f}  std(sigma) = {out['std_sigma']:.3f}  "
              f"mean|log sigma| = {out['mean_abs_log_sigma']:.3f}", flush=True)
        print(f"  raw   z_fast  KS={ks_raw:.4f}  W1={w1_raw:.4f}  KL≈{out['kl_gauss_raw']:.3f}", flush=True)
        print(f"  whit  z̃_fast KS={ks_wh:.4f}  W1={w1_wh:.4f}  KL≈{out['kl_gauss_whit']:.3f}", flush=True)
        print(f"  improvement   ΔKS={out['ks_improvement']:+.4f}  ΔW1={out['w1_improvement']:+.4f}  "
              f"ΔKL≈{out['kl_improvement']:+.3f}", flush=True)
    else:
        # Gaussian prior: report z-marginal stats only (no CNN to query)
        flat = z.flatten()
        ks_z, w1_z = standardised_KS_W1(flat)
        out["ks_raw"] = ks_z
        out["w1_raw"] = w1_z
        out["kl_gauss_raw"] = kl_to_standard_normal_mc(flat)
        for k in ("cnn_mu_rms_over_z_rms", "cnn_mu_rms", "z_fast_rms",
                  "mean_sigma", "std_sigma", "mean_abs_log_sigma",
                  "ks_whit", "w1_whit", "ks_improvement",
                  "w1_improvement", "kl_gauss_whit", "kl_improvement",
                  "ks_slow", "w1_slow"):
            out[k] = float("nan")
        print(f"  Gaussian prior; z KS={ks_z:.4f}  W1={w1_z:.4f}  KL≈{out['kl_gauss_raw']:.3f}",
              flush=True)

    return out


def write_csv(rows, path):
    cols = [
        "label", "folder", "L", "T", "epoch", "cond_prior",
        "cnn_mu_rms_over_z_rms", "cnn_mu_rms", "z_fast_rms",
        "mean_sigma", "std_sigma", "mean_abs_log_sigma",
        "ks_raw", "ks_whit", "ks_improvement",
        "w1_raw", "w1_whit", "w1_improvement",
        "kl_gauss_raw", "kl_gauss_whit", "kl_improvement",
        "ks_slow", "w1_slow",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nwrote {path}", flush=True)


def verdict_print(rows):
    """Print a human-readable verdict block."""
    print("\n" + "=" * 78)
    print("V6 verdict block")
    print("=" * 78)
    print(f"{'cell':<35}{'i2?':>5}{'||μ||/||z||':>14}{'mean σ':>10}"
          f"{'KS raw':>10}{'KS whit':>10}{'KL raw':>9}{'KL whit':>10}")
    for r in rows:
        if r["cond_prior"]:
            print(f"{r['label']:<35}{'yes':>5}"
                  f"{r['cnn_mu_rms_over_z_rms']:>14.3f}"
                  f"{r['mean_sigma']:>10.3f}"
                  f"{r['ks_raw']:>10.4f}{r['ks_whit']:>10.4f}"
                  f"{r['kl_gauss_raw']:>9.3f}{r['kl_gauss_whit']:>10.3f}")
        else:
            print(f"{r['label']:<35}{'no':>5}"
                  f"{'—':>14}{'—':>10}"
                  f"{r['ks_raw']:>10.4f}{'—':>10}"
                  f"{r['kl_gauss_raw']:>9.3f}{'—':>10}")

    print()
    print("Reading the table:")
    print("  - i2 cells (B, D) should show non-trivial ||μ||/||z|| and σ ≠ 1 if CNN")
    print("    is genuinely off-loading structure.")
    print("  - 'KS raw' is how non-Gaussian the bare MERA latent looks; 'KS whit'")
    print("    is what's left after CNN cleanup. Big drop (raw → whit) means CNN")
    print("    really is doing the off-loading.")
    print("  - Compare D bignet vs D mega: if CNN strength rises with capacity")
    print("    and V5 RMS-G also rises with capacity, that's direct evidence")
    print("    the 'data-driven fixed point' is the same Wilson fixed point but")
    print("    with the decoupling work split MERA ↔ CNN-prior.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000,
                        help="HS samples drawn per cell")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="optional subset of cell labels from rg_fixed_point.STYLE / FOLDERS")
    parser.add_argument("--csv-out", default="analyzers/csv/rg_v6_cnn_offload.csv")
    args = parser.parse_args()

    cells = args.cells or list(FOLDERS.keys())
    rows = []
    for label in cells:
        folder = FOLDERS[label]
        if not os.path.exists(folder):
            print(f"[skip] {label}: folder missing -> {folder}")
            continue
        try:
            rows.append(run_one(folder, label, N=args.N, device=args.device))
        except Exception as e:
            print(f"[error] {label}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if not rows:
        print("no cells succeeded; aborting CSV.", flush=True)
        return
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    write_csv(rows, args.csv_out)
    verdict_print(rows)


if __name__ == "__main__":
    main()
