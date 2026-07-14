"""Capture MERA per-layer activations in BOTH directions with per-site
Gaussianization, ready for cross-layer similarity comparison.

Data flow direction (analysis / forward):
    x ~ p_HS       →  scale_block[0].forward → y_1
    y_1            →  scale_block[1].forward → y_2
    ...
    y_{S−1}        →  scale_block[S−1].forward → y_S   (latent prior side)

Latent → data direction (generation / inverse):
    z ~ N(0, I)    →  scale_block[S−1].inverse → w_{S−1}
    w_{S−1}        →  scale_block[S−2].inverse → w_{S−2}
    ...
    w_1            →  scale_block[0].inverse → w_0   (x-like samples)

At each scale s we:
  1) extract the kept-coarse sub-lattice (offset 0, stride 2^s) — the
     only positions that carry the physical slow-mode signal after
     scale s (positions not read by any later block are frozen to the
     latent prior and dominate un-restricted averages).
  2) per-site z-score (Gaussianize marginal) so the output at each
     scale has mean 0 std 1 per dimension.
  3) save both the forward and inverse activations to
     `mera_layer_flow_capture.pt` inside the run folder.

The resulting file can be loaded by a follow-up cross-layer similarity
probe (per-layer KS / W1 / rank-corr / MMD across scales, or against
the prior samples).

Usage:
  python analyzers/rg_fixed_point/mera_layer_flow_capture.py \\
      --folder DATAFOLDER --N 4000 --epoch BEST_200_EPOCH \\
      [--out mera_layer_flow_capture.pt]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flow_sample_diagnostic import build_flow
from rg_fixed_point_v4_dataforward import load_hs_data


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
    savs = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                  key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    return savs[min(range(len(eps)), key=lambda i: abs(eps[i] - epoch))]


def get_scale_blocks(mera, repeats_per_scale=2):
    """Group layerList into scale-blocks.

    MERA has depth = 2·S (offset-0 + offset-1 masks per scale) when
    nrepeat=1. With nrepeat=k there are 2·k blocks per scale, so
    len(layerList) = 2·k·S. Returns S scale-blocks, each a list of
    consecutive layers.
    """
    layers = list(mera.layerList)
    n = len(layers)
    if n % repeats_per_scale != 0:
        raise ValueError(f"layerList len {n} not divisible by {repeats_per_scale}")
    n_scales = n // repeats_per_scale
    blocks = [layers[s * repeats_per_scale:(s + 1) * repeats_per_scale]
              for s in range(n_scales)]
    return blocks


def apply_block_forward(block, x, kernel_shape):
    """Apply forward passes of a scale-block (list of layers). Mirrors
    template.forward's per-layer reshape → apply → sum-log-jac.

    Returns the transformed x (same shape as input) — we discard log-jac
    because we only need activations.
    """
    B = x.shape[0]
    channel = 1
    for layer in block:
        x_r = x.reshape(-1, channel, *kernel_shape)
        x_r, _ = layer.forward(x_r)
        x = x_r.reshape(B, channel, *x.shape[-2:])
    return x


def apply_block_inverse(block, z, kernel_shape):
    """Apply inverse passes of a scale-block IN REVERSE ORDER."""
    B = z.shape[0]
    channel = 1
    for layer in reversed(block):
        z_r = z.reshape(-1, channel, *kernel_shape)
        z_r, _ = layer.inverse(z_r)
        z = z_r.reshape(B, channel, *z.shape[-2:])
    return z


def kept_coarse(x, s):
    """Extract stride 2^s sub-lattice starting at offset 0.

    Args:
      x: (B, 1, L, L) tensor
      s: scale index (0 = finest → whole lattice; 1 = every-other, etc.)
    """
    stride = 2 ** s
    return x[..., ::stride, ::stride]


def gaussianize(x, method="zscore"):
    """Per-site marginal Gaussianize.

    zscore: subtract per-site mean, divide by per-site std over batch dim.
            Gives mean=0 std=1 per lattice site.
    """
    if method == "zscore":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    else:
        raise ValueError(f"unknown gaussianize method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--N", type=int, default=4000, help="samples per direction")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="mera_layer_flow_capture.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder = args.folder.rstrip("/")
    ckpt = pick_saving(folder, args.epoch)
    ep = int(re.search(r"epoch(\d+)", ckpt).group(1))
    print(f"[capture] folder={folder}")
    print(f"[capture] checkpoint = {os.path.basename(ckpt)}  (epoch {ep})")

    state = torch.load(ckpt, weights_only=False, map_location=args.device)
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state, device=args.device)
    fw.eval()

    # Unwrap: Symmetrized wraps a MERA in .flow ; MERA has .layerList.
    mera = fw.flow if hasattr(fw, "flow") else fw
    kernel_shape = tuple(mera.kernelShape) if hasattr(mera, "kernelShape") else (2, 2)

    # Determine repeats_per_scale by matching layerList size to expected depth.
    # For nrepeat=k, len(layerList) = 2·k·S where S = log2(L).
    S = int(np.log2(L))
    n_layers = len(mera.layerList)
    repeats_per_scale = n_layers // S
    print(f"[capture] L={L}, S={S} scales, layerList={n_layers}, "
          f"repeats/scale={repeats_per_scale}")

    scale_blocks = get_scale_blocks(mera, repeats_per_scale=repeats_per_scale)

    # ── FORWARD DIRECTION (data → latent) ───────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    samples = load_hs_data(L, T, args.N, device=args.device)
    samples = samples.to(dtype=torch.float32)
    print(f"[capture] loaded {samples.shape[0]} HS samples")

    forward_activations = {"raw_input": samples[:min(args.N, 500)].clone().cpu()}
    with torch.no_grad():
        y = samples
        for s, block in enumerate(scale_blocks):
            y = apply_block_forward(block, y, kernel_shape)
            y_kc = kept_coarse(y, s + 1)                     # (B, 1, L/2^{s+1}, ...)
            y_gz = gaussianize(y_kc, method="zscore")        # per-site standardize
            forward_activations[f"y_{s+1}_kept"] = y_kc[:min(args.N, 500)].cpu()
            forward_activations[f"y_{s+1}_gaussianized"] = y_gz[:min(args.N, 500)].cpu()
            print(f"  fwd scale {s+1}: kept shape {tuple(y_kc.shape)}, "
                  f"post-gz mean={y_gz.mean().item():+.4f} std={y_gz.std().item():.4f}")

    # ── INVERSE DIRECTION (latent → data) ───────────────────────────────
    inverse_activations = {}
    with torch.no_grad():
        z = torch.randn_like(samples)
        inverse_activations["raw_prior"] = z[:min(args.N, 500)].clone().cpu()
        w = z
        # inverse direction: apply blocks in reverse
        for si, block in enumerate(reversed(scale_blocks)):
            s = len(scale_blocks) - 1 - si
            w = apply_block_inverse(block, w, kernel_shape)
            w_kc = kept_coarse(w, s)   # after this inverse, we're at scale s
            w_gz = gaussianize(w_kc, method="zscore")
            inverse_activations[f"w_{s}_kept"] = w_kc[:min(args.N, 500)].cpu()
            inverse_activations[f"w_{s}_gaussianized"] = w_gz[:min(args.N, 500)].cpu()
            print(f"  inv → scale {s}: kept shape {tuple(w_kc.shape)}, "
                  f"post-gz mean={w_gz.mean().item():+.4f} std={w_gz.std().item():.4f}")

    # ── SAVE ────────────────────────────────────────────────────────────
    out_path = os.path.join(folder, args.out)
    payload = {
        "folder": folder,
        "checkpoint_epoch": ep,
        "L": L,
        "T": T,
        "N_used": min(args.N, 500),   # subsampled saved copy
        "n_scales": len(scale_blocks),
        "repeats_per_scale": repeats_per_scale,
        "forward": forward_activations,
        "inverse": inverse_activations,
    }
    torch.save(payload, out_path)
    print(f"[capture] saved → {out_path}")

    # Also write a small JSON summary
    summary = {
        "folder": folder,
        "epoch": ep,
        "L": L,
        "T": T,
        "n_scales": len(scale_blocks),
        "forward_keys": list(forward_activations.keys()),
        "inverse_keys": list(inverse_activations.keys()),
        "note": ("Per-scale slots contain both the kept-coarse raw activation "
                 "and its per-site Gaussianized (zscore) form. Use the "
                 "gaussianized tensors for cross-layer similarity (MMD, "
                 "moment matching, KS) — same marginal scale everywhere."),
    }
    with open(os.path.join(folder, "mera_layer_flow_capture.json"), "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"[capture] wrote summary JSON")


if __name__ == "__main__":
    main()
