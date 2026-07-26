"""Per-block Jacobian analysis: how does the VP-penalty champion compare
to the Gaussian baseline on a block-by-block basis?

The champion (fixdil+VP-1e-3 nr=1) is trained with a soft VP penalty:

    loss += lambda * (log|det J_MERA|)²      with lambda = 1e-3

which pushes the AGGREGATE log|det J| across all 10 MERA blocks toward 0.
This report checks:
  (a) whether the per-block log|det J| shrinks vs baseline
  (b) whether the sign pattern cancels across blocks
  (c) whether the shrinkage is uniform or scale-selective

Runs both champion and baseline on the same HS data batch (identical inputs)
so any difference in block outputs is pure model difference, not sampling
noise.

Uses the `forward_with_per_block_logjac` method added to
`flow/hierarchy/template.py` — returns per-block log|det J| tensors.

Usage:
  python analyzers/rg_fixed_point/per_block_jacobian.py \\
      --cells champion:data/32Ising_..._vp1e-3_b64:9500 \\
              baseline:data/32Ising_..._baseline_b64:19800 \\
      --N 1000 --out analyzers/csv/per_block_jacobian.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_sample_diagnostic import build_flow
from rg_fixed_point_v4_dataforward import load_hs_data


def latest_saving(folder, prefer_epoch=None):
    savs = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                  key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if prefer_epoch is None:
        return savs[-1]
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - prefer_epoch))
    return savs[idx]


def get_inner_mera(fw):
    """Unwrap Symmetrized wrapper if present. Returns the object that has
    `forward_with_per_block_logjac` (i.e., the MERA / HierarchyBijector)."""
    obj = fw
    for _ in range(3):
        if hasattr(obj, "forward_with_per_block_logjac"):
            return obj
        # Try to descend into common wrapper attributes
        for attr in ("flow", "prior"):
            if hasattr(obj, attr):
                candidate = getattr(obj, attr)
                if hasattr(candidate, "forward_with_per_block_logjac"):
                    return candidate
                obj = candidate
                break
        else:
            break
    raise AttributeError(
        f"could not locate forward_with_per_block_logjac on {type(fw)} — "
        "check flow/hierarchy/template.py")


def run_one(label, folder, prefer_epoch, x_data, device="cpu"):
    print(f"\n=== {label}  folder={folder}", flush=True)
    ckpt = latest_saving(folder, prefer_epoch)
    epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    inner_mera = get_inner_mera(fw)
    print(f"  L={L} T={T} ep={epoch}", flush=True)

    with torch.no_grad():
        z, per_block_logjac = inner_mera.forward_with_per_block_logjac(x_data)
    # per_block_logjac: list of tensors, each shape (B,) — log|det J| per sample
    n_blocks = len(per_block_logjac)
    print(f"  {n_blocks} MERA blocks; input x std={x_data.std():.3f}, output z std={z.std():.3f}", flush=True)

    rows = []
    per_block_mean_arr = []
    per_block_std_arr = []
    for b, lj in enumerate(per_block_logjac):
        arr = lj.detach().cpu().float().numpy()
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        abs_mean = float(np.mean(np.abs(arr)))
        per_block_mean_arr.append(mean)
        per_block_std_arr.append(std)
        row = dict(label=label, folder=folder, L=L, T=T, epoch=epoch,
                   block=b, mean_logjac=mean, std_logjac=std,
                   mean_abs_logjac=abs_mean)
        rows.append(row)

    # Cumulative sum trajectory (running total across blocks, per sample)
    cum = np.zeros(x_data.shape[0])
    print(f"\n  Block-by-block log|det J| (over N={x_data.shape[0]} samples):")
    print(f"  {'block':>6} {'mean':>10} {'|mean|':>8} {'std':>8} {'cum mean':>10}")
    for b, lj in enumerate(per_block_logjac):
        arr = lj.detach().cpu().float().numpy()
        cum = cum + arr
        cum_mean = float(np.mean(cum))
        print(f"  {b:>6d} {per_block_mean_arr[b]:>+10.3f} "
              f"{np.mean(np.abs(arr)):>8.3f} {per_block_std_arr[b]:>8.3f} "
              f"{cum_mean:>+10.3f}")

    total_mean = sum(per_block_mean_arr)
    print(f"  {'-'*46}")
    print(f"  {'TOTAL':>6} {total_mean:>+10.3f}")
    print(f"  (mean cumulative log|det J|)² [∝ VP penalty] = {(cum.mean())**2:.4f}")

    return rows


def parse_cell(spec):
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    raise ValueError(f"bad cell spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="analyzers/csv/per_block_jacobian.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Load HS data ONCE, same across all cells (identical inputs).
    # We infer L, T from the first cell.
    first_label, first_folder, first_ep = parse_cell(args.cells[0])
    # Peek at first ckpt to get L, T
    ckpt0 = latest_saving(first_folder, first_ep)
    state0 = torch.load(ckpt0, weights_only=False, map_location=args.device)
    fw0, target0, L, T, *_ = build_flow(first_folder, state0, device=args.device)
    print(f"[data] loading N={args.N} HS samples at L={L}, T={T}")
    x_data = load_hs_data(L, T, args.N, device=args.device)

    all_rows = []
    for spec in args.cells:
        label, folder, prefer_epoch = parse_cell(spec)
        rows = run_one(label, folder, prefer_epoch, x_data, device=args.device)
        all_rows.extend(rows)

    # Write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["label", "folder", "L", "T", "epoch", "block",
            "mean_logjac", "std_logjac", "mean_abs_logjac"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nwrote {args.out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
