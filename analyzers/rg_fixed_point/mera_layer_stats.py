"""Layer-by-layer analysis of the MERA flow (companion to hcg_perscale_similarity.py).

The HCG probe reveals per-scale CNN structure. This probe does the same for the
MERA RNVP stack: are the 10 RNVP blocks all doing similar work, or does the
best HCG per-scale run also structure the MERA layers into "coarse" vs "fine"
regimes matching what the CNN did?

For each RNVP layer L in flow.layerList:
  - L2 norm of every named parameter (weight vs bias, per sub-module)
  - Cosine similarity of concatenated weight vectors between all pairs of layers
  - Group by scale index (MERA repeats layers per scale, so we can bin them)

Usage:
  python analyzers/rg_fixed_point/mera_layer_stats.py \\
      --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64 \\
      --epoch 7707
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


def latest_saving(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no .saving files in {folder}")
    return files[-1]


def pick_saving(folder, epoch):
    if epoch is None:
        return latest_saving(folder)
    savs = sorted(
        glob.glob(os.path.join(folder, "savings/*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not savs:
        raise FileNotFoundError(f"no .saving in {folder}")
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    return savs[min(range(len(eps)), key=lambda i: abs(eps[i] - epoch))]


def layer_weight_vec(layer):
    """Flatten every trainable parameter of one RNVP into a single vector."""
    parts = []
    for _, p in layer.named_parameters():
        if p.requires_grad:
            parts.append(p.detach().flatten().cpu().numpy().astype(np.float64))
    return np.concatenate(parts) if parts else np.array([])


def per_submodule_stats(layer):
    """Break down L2 norm by named submodule (top-level bucket)."""
    buckets = {}
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".", 1)[0]
        buckets.setdefault(top, []).append(p.detach().flatten().cpu().numpy().astype(np.float64))
    stats = {}
    for k, vs in buckets.items():
        v = np.concatenate(vs)
        stats[k] = (float(np.linalg.norm(v)), int(v.size))
    return stats


def run_one(folder, epoch=None, device="cpu"):
    print(f"\n===== {folder} =====", flush=True)
    ckpt = pick_saving(folder, epoch)
    ep = int(re.search(r"epoch(\d+)", ckpt).group(1))
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=device)
    fw.eval()
    inner = fw.flow if hasattr(fw, "flow") else fw
    print(f"  L={L} T={T} epoch={ep} flow class={type(inner).__name__}")
    if not hasattr(inner, "layerList"):
        print("  flow has no layerList — abort")
        return
    layers = inner.layerList
    K = len(layers)
    print(f"  MERA has {K} RNVP blocks")

    # ── A. per-layer overall L2 + per-submodule breakdown ─────────────────
    print("\n  ── A. per-layer overall L2 norm ──")
    print(f"    {'layer':<8}  {'L2(all)':>10}  {'nparam':>10}  submodule breakdown")
    vecs = []
    for i, layer in enumerate(layers):
        v = layer_weight_vec(layer)
        vecs.append(v)
        total_l2 = float(np.linalg.norm(v))
        subs = per_submodule_stats(layer)
        sub_str = "  ".join(f"{name}: {l2:.3f}({n})" for name, (l2, n) in subs.items())
        print(f"    L{i:>2}       {total_l2:>10.3f}  {len(v):>10d}  {sub_str}")

    # ── B. pairwise cosine similarity ─────────────────────────────────────
    print("\n  ── B. pairwise cosine similarity (per-layer weight vectors) ──")
    n = len(vecs)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = vecs[i], vecs[j]
            if a.size == 0 or b.size == 0:
                C[i, j] = 0.0
            else:
                C[i, j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    header = "         " + "  ".join(f"L{i:>2}" for i in range(n))
    print(header)
    for i in range(n):
        row = "  ".join(f"{C[i, j]:+.3f}" for j in range(n))
        print(f"    L{i:>2}    {row}")
    off = C[np.triu_indices(n, k=1)]
    print(f"    off-diag: mean={off.mean():+.3f}  min={off.min():+.3f}  max={off.max():+.3f}")

    # ── C. adjacent-layer cosine (are neighboring MERA scales similar?) ──
    print("\n  ── C. adjacent-layer cosine ──")
    for i in range(n - 1):
        a, b = vecs[i], vecs[i + 1]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"    L{i} ↔ L{i+1}:  cosine = {c:+.3f}")

    # ── D. L2 norm heatmap-like table ────────────────────────────────────
    L2s = [float(np.linalg.norm(v)) for v in vecs]
    L2max = max(L2s)
    print("\n  ── D. L2-norm profile (bar = fraction of max) ──")
    for i, l2 in enumerate(L2s):
        bar = "█" * int(round(40 * l2 / (L2max + 1e-12)))
        print(f"    L{i:>2}  {l2:>8.3f}  |{bar:<40}|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", nargs="+", required=True)
    ap.add_argument("--epoch", type=int, default=None,
                    help="Pick saving nearest to this epoch (default: latest)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for f in args.folder:
        try:
            run_one(f, epoch=args.epoch, device=args.device)
        except Exception as e:
            print(f"[error] {f}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
