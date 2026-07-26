"""MERAUNet FM per-layer self-similarity analysis.

Analog to cascade_layer_analysis.py for the MERA champion. Extracts encoder
feature maps at each U-Net downsampling stage, reduces per-site (channel
mean), applies per-site z-score gauge, and compares adjacent-scale
distributions.

Question addressed: does MERAUNet's internal encoder learn multi-scale
self-similar representations (like the RG cascade of MERA champion), or
just generic multi-scale features?

Since MERAUNet is called at every ODE integration step with different `t`,
the internal features depend on t. We analyze at multiple t values
(default: 0.5, 0.9, 1.0) to see how the encoder representation evolves
along the trajectory.

Output CSV format mirrors cascade sections A (marginal shape) and B
(adjacent-pair similarity) so results can be directly compared to
`cascade_layer_L32vsL64_champions.csv`.

Usage:
  python analyzers/fm_layer_self_similarity.py \\
      --ckpts meraFM_L32:data/L32_T2.269_meraFM_h64/savings/fm_L32_T2.269185314213022_epoch1200.pt \\
      --L 32 --N 500 --nhidden 64 \\
      --out analyzers/csv/fm_layer_self_similarity_L32.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow.flow_matching import MERAUNet
from analyzers.rg_fixed_point.cascade_layer_analysis import rbf_mmd2, w1_marginal


def load_meraunet(ckpt_path, L, nhidden=64, temb_dim=128, max_channel_mult=4, device="cpu"):
    net = MERAUNet(L=L, nhidden=nhidden, temb_dim=temb_dim,
                    max_channel_mult=max_channel_mult).to(device)
    state = torch.load(ckpt_path, weights_only=False, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net.load_state_dict(state)
    net.eval()
    return net


def extract_encoder_skips(net, x, t):
    """Walk MERAUNet's encoder path; return channel-mean skips at each scale."""
    with torch.no_grad():
        temb = net.time_embed(t)
        h = net.in_conv(x)
        skips = []
        for s in range(net.n_scales):
            h = net.enc[s](h, temb)
            # Reduce channels → per-site scalar via channel mean
            skips.append(h.mean(dim=1, keepdim=True).cpu())
            h = net.down[s](h)
    return skips


def gauge_fix(y, eps=1e-8):
    """Per-site z-score across batch (matches cascade section A/B gauge)."""
    mean = y.mean(dim=0, keepdim=True)
    std = y.std(dim=0, keepdim=True).clamp_min(eps)
    return (y - mean) / std


def load_hs_data(L, T, N, device="cpu"):
    """Load HS-transformed Ising samples at (L, T)."""
    pat1 = f"./data/mcmc_data/hs_L{L}_T{T:.15f}_N*.pt"
    pat2 = f"./data/mcmc_data/hs_L{L}_T{T}_N*.pt"
    cand = []
    for p in (pat1, pat2):
        cand.extend(sorted(glob.glob(p),
                           key=lambda s: int(s.split("_N")[-1].split(".")[0]),
                           reverse=True))
    if not cand:
        raise FileNotFoundError(f"no HS data at L={L}, T={T}")
    print(f"[data] loading {cand[0]}")
    x = torch.load(cand[0], weights_only=False, map_location=device)
    if isinstance(x, list):
        x = torch.stack(x)
    while x.dim() < 4:
        x = x.unsqueeze(1) if x.dim() == 3 else x.unsqueeze(0)
    if x.shape[1] != 1:
        x = x[:, :1]
    return x[:N].float()


def analyze_meraunet(net, x, t_values, label, out_rows):
    """Run marginal + adjacent-pair analysis at each requested t."""
    from scipy.stats import kstest, skew as sp_skew, kurtosis
    for t_val in t_values:
        B = x.shape[0]
        t_tensor = torch.full((B,), t_val, dtype=x.dtype)
        skips = extract_encoder_skips(net, x, t_tensor)
        n_scales = len(skips)

        # --- Section A: marginal shape per gauge-fixed scale ---
        print(f"\n[{label}, t={t_val}] section A — marginal (gauge-fixed):")
        print(f"  {'scale':<6} {'size':<6} {'skew':>8} {'kurt':>8} {'ks':>8}")
        for s in range(n_scales):
            y_gz = gauge_fix(skips[s])
            arr = y_gz.reshape(-1).double().numpy()
            skv = float(sp_skew(arr))
            kur = float(kurtosis(arr))
            ks = float(kstest(arr, "norm").statistic)
            size = int(skips[s].shape[-1])
            print(f"  s={s+1:<3} {size:>3}    {skv:>+8.3f} {kur:>+8.3f} {ks:>8.4f}")
            for name, val in [("skew", skv), ("kurt", kur), ("ks", ks)]:
                out_rows.append(dict(model=label, t=t_val, section="A_marginal",
                                      scale=s + 1, size=size, metric=name, value=val))

        # --- Section B: adjacent-scale similarity (gauge-fixed) ---
        print(f"\n[{label}, t={t_val}] section B — adjacent-scale similarity:")
        print(f"  {'pair':<10} {'size a→b':<11} {'MMD²':>10} {'W1':>8}")
        for s in range(n_scales - 1):
            y_a = gauge_fix(skips[s])
            y_b = gauge_fix(skips[s + 1])
            y_a_flat = y_a.reshape(-1, 1)
            y_b_flat = y_b.reshape(-1, 1)
            nb = min(y_a_flat.shape[0], y_b_flat.shape[0])
            y_a_flat = y_a_flat[:nb]
            y_b_flat = y_b_flat[:nb]
            try:
                mmd = float(rbf_mmd2(y_a_flat.numpy(), y_b_flat.numpy()))
            except Exception:
                mmd = float("nan")
            w1 = float(w1_marginal(y_a_flat, y_b_flat))
            size_a = int(skips[s].shape[-1])
            size_b = int(skips[s + 1].shape[-1])
            print(f"  s{s+1}→s{s+2}    {size_a:>3}→{size_b:<3}       {mmd:>10.4g} {w1:>8.3f}")
            for name, val in [("mmd2", mmd), ("w1", w1)]:
                out_rows.append(dict(model=label, t=t_val, section="B_adjacent",
                                      scale=f"{s+1}->{s+2}", size=size_a,
                                      metric=name, value=val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="label:path pairs, e.g. meraFM_L32:path/to/ckpt.pt")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--T", type=float, default=2.269185314213022)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--nhidden", type=int, default=64)
    ap.add_argument("--tembDim", type=int, default=128)
    ap.add_argument("--maxChannelMult", type=int, default=4)
    ap.add_argument("--tValues", type=str, default="0.5,0.9,1.0",
                    help="Comma-separated ODE times to analyze")
    ap.add_argument("--out", default="analyzers/csv/fm_layer_self_similarity.csv")
    args = ap.parse_args()

    t_values = [float(t) for t in args.tValues.split(",")]

    x = load_hs_data(args.L, args.T, args.N)
    print(f"[data] shape={tuple(x.shape)}, mean={x.mean():.3f}, std={x.std():.3f}")

    rows = []
    for spec in args.ckpts:
        label, path = spec.split(":", 1)
        print(f"\n[load] {label}: {path}")
        net = load_meraunet(path, args.L, args.nhidden, args.tembDim,
                            args.maxChannelMult)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  MERAUNet L={args.L} nhidden={args.nhidden}, "
              f"{net.n_scales} scales, {n_params/1e6:.2f}M params")
        analyze_meraunet(net, x, t_values, label, rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "t", "section", "scale",
                                          "size", "metric", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[csv] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
