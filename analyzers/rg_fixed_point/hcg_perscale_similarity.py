"""Test 2 (refined): per-scale HCG cross-level analysis.

Two probes now:

A. Raw weight similarity (kept from original Test 2):
   - Cosine similarity of flattened CNN weights across levels
   - Weight L2 norm per level (detects dead CNN levels)

B. Output-space similarity (new — better probe):
   Feed real HS data through MERA to get z; for each level k, extract
   CNN_k's sigma(z_slow) output at the sites CNN_k is supposed to score.

     - mean(sigma_k) per level  → is the "typical conditional std" the
                                  same across scales?
     - std(sigma_k)  per level  → does CNN_k respond to z_slow with the
                                  same variability at every scale?
     - histogram(sigma_k) match → do the distributions of predicted sigma
                                  agree across levels?

   The comparison mu is skipped since mu ~ 0 across all cells (V6 finding).

   If per-scale HCG learned scale invariance, the sigma distributions
   should be similar across levels. If per-scale learned non-scale-
   invariant physics, the sigma distributions differ.

C. Cross-application swap test:
   For each level k, apply CNN_k to level k' context (as if CNN_k were
   the CNN for level k'). Compare its output to CNN_{k'}'s native output.
   If per-scale is scale-invariant, swap should give similar sigma.

Usage:
  python analyzers/rg_fixed_point/hcg_perscale_similarity.py \\
      --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nr2_b64 \\
      --N 500
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flow_sample_diagnostic import build_flow
from rg_fixed_point import latest_saving
from rg_fixed_point_v4_dataforward import load_hs_data


def flatten_cnn_weights(cnn):
    pieces = []
    for m in cnn.modules():
        if isinstance(m, torch.nn.Conv2d):
            pieces.append(m.weight.data.flatten())
    return torch.cat(pieces).double().cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analyze_weights(prior):
    """A. Raw weight similarity (old Test 2)."""
    cnns = list(prior.cnns)
    K = len(cnns)
    W = [flatten_cnn_weights(c) for c in cnns]
    L2 = [float(np.linalg.norm(w)) for w in W]

    print("\n  --- A. Raw weight L2 + cosine similarity ---")
    for k in range(K):
        stride_k = prior.strides[k + 1]
        marker = "  ⚠dead" if L2[k] < 0.1 else ""
        print(f"    Level {k+1} (stride {stride_k}): weight L2 = {L2[k]:.4f}{marker}")

    print()
    print(f"    Pairwise cosine (weights only, off-diagonal):")
    print(f"       {'  '.join(f'L{i+1}   ' for i in range(K))}")
    for k in range(K):
        row = [f"1.000 " if k == kk else f"{cosine(W[k], W[kk]):+.3f}" for kk in range(K)]
        print(f"    L{k+1}   {'  '.join(row)}")
    off = [cosine(W[k], W[kk]) for k in range(K) for kk in range(K) if k < kk]
    print(f"    off-diag mean cosine = {np.mean(off):+.3f}")


def get_sigma_at_level(prior, z, k):
    """Apply CNN_k to context z (with mask up to level k-1), return sigma at ALL positions."""
    context_mask = prior._buffers[f"context_mask_up_to_{k-1}"].to(z.dtype)
    z_ctx = z * context_mask
    cnn = prior.cnns[k - 1]   # cnns[0] scores level 1, cnns[k-1] scores level k
    with torch.no_grad():
        out = cnn(z_ctx)
    mu, log_sigma = out.chunk(2, dim=1)
    log_sigma = log_sigma.clamp(min=-5.0, max=5.0)
    sigma = torch.exp(log_sigma)
    return sigma


def analyze_sigma_outputs(fw, prior, samples):
    """B. Output-space analysis — the "compare σ directly" probe.

    For each level k, extract CNN_k's sigma prediction at level_k positions
    on the real MERA(HS) latents.
    """
    # Run MERA forward to get z on real data
    mera = fw.flow if hasattr(fw, "flow") else fw
    with torch.no_grad():
        z, _ = mera.forward(samples)     # (B, 1, L, L)

    K = len(prior.cnns)

    print("\n  --- B. σ predictions at each level (on real HS data via MERA) ---")
    per_level_stats = []
    per_level_all_sigma = []
    for k in range(1, K + 1):
        sigma = get_sigma_at_level(prior, z, k)   # (B, 1, L, L)
        mask_k = prior._buffers[f"level_mask_{k}"].to(z.dtype).bool()
        mask_k_b = mask_k.expand_as(sigma)
        sig_here = sigma[mask_k_b].cpu().numpy()   # values only at level-k positions
        per_level_all_sigma.append(sig_here)
        s = dict(
            level=k,
            stride=prior.strides[k],
            n_sites=int(mask_k.sum().item()),
            mean=float(sig_here.mean()),
            std=float(sig_here.std()),
            median=float(np.median(sig_here)),
            q10=float(np.percentile(sig_here, 10)),
            q90=float(np.percentile(sig_here, 90)),
        )
        per_level_stats.append(s)

    print(f"    {'level':>6}{'stride':>8}{'n_sites':>9}"
          f"{'mean σ':>10}{'std σ':>10}{'median':>10}{'q10':>8}{'q90':>8}")
    for s in per_level_stats:
        print(f"    {s['level']:>6}{s['stride']:>8}{s['n_sites']:>9}"
              f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['median']:>10.4f}"
              f"{s['q10']:>8.3f}{s['q90']:>8.3f}")

    print("\n    Pairwise |Δmean σ| across levels (small = scale-invariant):")
    print(f"       {'  '.join(f'L{i+1}   ' for i in range(K))}")
    means = [s['mean'] for s in per_level_stats]
    for k in range(K):
        row = ['0.000' if k == kk else f"{abs(means[k] - means[kk]):.3f}" for kk in range(K)]
        print(f"    L{k+1}   {'  '.join(row)}")

    return per_level_all_sigma, per_level_stats


def analyze_swap(fw, prior, samples):
    """C. Swap test.

    For each pair (k, k'), apply CNN_k to level k' context and see if its
    sigma at level_k' positions matches CNN_{k'}'s native sigma.

    If CNNs are the same universal function (learned scale invariance),
    swap gives ~identical sigma → sigma matrix is symmetric-ish and small
    off-diag differences.
    """
    mera = fw.flow if hasattr(fw, "flow") else fw
    with torch.no_grad():
        z, _ = mera.forward(samples)

    K = len(prior.cnns)

    def cnn_out_at_level_positions(cnn_idx, level_idx):
        # Apply CNN[cnn_idx] to context appropriate for level_idx, sample sigma at level_idx positions.
        context_mask = prior._buffers[f"context_mask_up_to_{level_idx-1}"].to(z.dtype)
        z_ctx = z * context_mask
        cnn = prior.cnns[cnn_idx - 1]
        with torch.no_grad():
            out = cnn(z_ctx)
        _, log_sigma = out.chunk(2, dim=1)
        log_sigma = log_sigma.clamp(min=-5.0, max=5.0)
        sigma = torch.exp(log_sigma)
        mask = prior._buffers[f"level_mask_{level_idx}"].to(z.dtype).bool()
        return sigma[mask.expand_as(sigma)].cpu().numpy()

    print("\n  --- C. Swap test: mean(σ) when applying CNN_k at level k' positions ---")
    print("        (rows = CNN used; cols = level being scored; diagonal = native use)")
    print(f"       {'  '.join(f'L{kp}    ' for kp in range(1, K+1))}")
    mean_matrix = np.zeros((K, K))
    for k in range(1, K + 1):
        row = []
        for kp in range(1, K + 1):
            m = cnn_out_at_level_positions(k, kp).mean()
            mean_matrix[k-1, kp-1] = m
            row.append(f"{m:.4f}")
        print(f"    CNN_{k}   {'  '.join(row)}")

    print("\n    |off-diag − native|/native (0 = swap gives native output; 1 = totally different):")
    for k in range(K):
        row = []
        for kp in range(K):
            native = mean_matrix[kp, kp]
            other  = mean_matrix[k, kp]
            row.append('0.000' if k == kp else f"{abs(other - native) / (native + 1e-8):.3f}")
        print(f"    CNN_{k+1}   {'  '.join(row)}")


def run_one(folder, N=500, device="cpu"):
    print(f"\n===== {folder} =====", flush=True)
    ckpt = latest_saving(folder)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    prior = fw.flow.prior if hasattr(fw, "flow") else fw.prior

    print(f"  L={L} T={T} epoch={epoch} prior={prior.__class__.__name__}")

    if prior.__class__.__name__ != "HierarchicalConditionalGaussian":
        print("  ⚠ not HCG, skip")
        return
    if getattr(prior, "scale_shared", True):
        print("  ⚠ scale_shared, only 1 CNN, nothing to compare")
        return

    strides = prior.strides
    K = len(prior.cnns)
    print(f"  {K} per-scale CNNs (strides: {strides})")

    # A. weight similarity
    analyze_weights(prior)

    # B. sigma output similarity
    samples = load_hs_data(L, T, N, device=device)
    print(f"\n  Loaded {samples.shape[0]} HS samples for output analysis")
    analyze_sigma_outputs(fw, prior, samples)

    # C. swap test
    analyze_swap(fw, prior, samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", nargs="+", required=True)
    parser.add_argument("--N", type=int, default=500,
                        help="HS samples for sigma-output analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for folder in args.folder:
        try:
            run_one(folder, N=args.N, device=args.device)
        except Exception as e:
            print(f"[error] {folder}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
