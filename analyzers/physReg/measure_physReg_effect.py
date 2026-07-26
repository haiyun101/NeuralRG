"""Measure |M|, χ, U₄ of samples from physReg-trained checkpoints vs baseline.

Answers: does physReg actually improve the physics observables it targets?
"""
import argparse
import os
import re
import sys
import glob

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))             # analyzers/
sys.path.insert(0, os.path.join(_here, "..", ".."))       # repo root

from flow_sample_diagnostic import build_flow


def sample_and_measure(folder, ep, N=1000, batch=64, device="cpu"):
    ckpts = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in ckpts]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - ep))
    ckpt = ckpts[idx]
    ep_actual = eps[idx]
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym, wt, hp = build_flow(folder, state, device=device)
    fw.eval()

    # Get sigma for un-standardization
    import json
    sigma = 1.0
    p_sigma = os.path.join(folder, "flow_input_sigma.json")
    if os.path.exists(p_sigma):
        with open(p_sigma) as f:
            sigma = float(json.load(f).get("sigma", 1.0))

    # Sample and compute spin observables
    N_sites = L * L
    n_done = 0
    all_M = []
    while n_done < N:
        b = min(batch, N - n_done)
        with torch.no_grad():
            u, _ = fw.sample(b)   # standardized samples
            x = sigma * u          # physical HS field
            # Convert to spin via sign (approximation)
            s = torch.sign(x)      # ±1 per site
            M_per = s.reshape(b, -1).mean(dim=-1)  # (b,)
        all_M.append(M_per)
        n_done += b
    M = torch.cat(all_M)
    absM = M.abs().mean().item()
    M2 = M.pow(2).mean().item()
    M4 = M.pow(4).mean().item()
    chi = N_sites * (M2 - absM ** 2)
    U4 = 1.0 - M4 / (3.0 * M2 ** 2)
    return dict(L=L, epoch=ep_actual, absM=absM, chi=chi, U4=U4, N_samples=len(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="label:folder:epoch triples")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"  {'cell':<40} {'L':>3} {'ep':>6} {'|M|':>8} {'χ':>8} {'U4':>8}")
    print("-" * 80)
    for spec in args.cells:
        parts = spec.split(":")
        label, folder = parts[0], parts[1]
        ep = int(parts[2]) if len(parts) > 2 else None
        try:
            r = sample_and_measure(folder, ep or 999999, args.N, args.batch, args.device)
            print(f"  {label:<40} {r['L']:>3} {r['epoch']:>6}   "
                  f"{r['absM']:>6.4f}  {r['chi']:>7.3f}  {r['U4']:>6.4f}")
        except Exception as e:
            print(f"  {label:<40} ERROR: {e}")


if __name__ == "__main__":
    main()
