"""Cross-L CNN weight similarity: compare HCG per-scale CNN weights across
L=64 champion, L=128 warm-start, L=128 fresh init. Answers:
  Q1 (within-model): does the L=128 model learn the same "SS pattern" of
      CNN weights as L=64 champion?
  Q2 (cross-model at same stride): do L=128 warm/fresh converge to
      similar CNN weights as L=64 at each stride?
  Q3 (transfer preservation): how much did L=128 warm-start CNN weights
      drift from their L=64 initialization?
"""
import argparse
import glob
import os
import re
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flow_sample_diagnostic import build_flow


def flatten_cnn_weights(cnn):
    """Flatten all cnn weights to a single 1D vector for cosine similarity."""
    ws = []
    for m in cnn.modules():
        if isinstance(m, torch.nn.Conv2d):
            ws.append(m.weight.data.flatten().cpu().numpy())
            if m.bias is not None:
                ws.append(m.bias.data.flatten().cpu().numpy())
    return np.concatenate(ws)


def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: return 0.0
    return float((a @ b) / (na * nb))


def load_prior(folder, target_ep, device="cpu"):
    ckpts = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in ckpts]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - target_ep))
    ckpt = ckpts[idx]
    ep_actual = eps[idx]
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, _, L, T, sym, wt, hp = build_flow(folder, state, device=device)
    prior = fw.flow.prior if hasattr(fw, "flow") else fw.prior
    if prior.__class__.__name__ != "HierarchicalConditionalGaussian":
        raise ValueError(f"{folder}: prior is {prior.__class__.__name__}, not HCG")
    return prior, L, ep_actual


def within_model_cnn_cos(prior, label):
    """Return {level_pair: cos} for all CNN levels within one model."""
    cnns = list(prior.cnns)
    K = len(cnns)
    W = [flatten_cnn_weights(c) for c in cnns]
    L2 = [float(np.linalg.norm(w)) for w in W]
    strides = list(prior.strides)
    print(f"\n=== {label} ({prior.L}×{prior.L}, {K} CNN levels, strides {strides[1:]}) ===")
    print(f"  weight L2 per level: {[f'{v:.2f}' for v in L2]}")
    print(f"  Pairwise cos (within-model):")
    for i in range(K):
        row = []
        for j in range(K):
            c = 1.0 if i == j else cosine(W[i], W[j])
            row.append(f"{c:+.3f}")
        print(f"    L{i+1}(s{strides[i+1]:>2}):  " + "  ".join(row))
    return W, L2, strides[1:]


def cross_model_stride_cos(models):
    """models: list of (label, W_list, L2_list, strides_list). Print same-stride
    cross-model cos matrix."""
    all_strides = sorted(set().union(*[set(m[3]) for m in models]), reverse=True)
    print(f"\n=== CROSS-MODEL cos(same-stride CNN weights) ===")
    print(f"  {'stride':<8} {'  '.join(f'{m[0]:<20}' for m in models)}")
    # Show per-stride cos vs first model as reference
    ref_label = models[0][0]
    for s in all_strides:
        row_out = [f"  {s:<8} "]
        # Get W for each model at this stride (if it has one)
        Ws = {}
        for label, W_list, L2_list, strides in models:
            if s in strides:
                idx = strides.index(s)
                Ws[label] = W_list[idx]
        # Print cos vs first model, or "n/a"
        for label, W_list, L2_list, strides in models:
            if label == ref_label:
                # self cos = 1 (or n/a if this model doesn't have that stride)
                if s in strides: row_out.append(f"{'1.000':<20}")
                else: row_out.append(f"{'—':<20}")
            elif s in strides and ref_label in Ws:
                c = cosine(Ws[ref_label], Ws[label])
                row_out.append(f"{c:+.3f}{'':<15}")
            else:
                row_out.append(f"{'—':<20}")
        print("".join(row_out))
    print(f"  (values are cos vs {ref_label})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="label:folder:epoch specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    models = []
    for spec in args.cells:
        parts = spec.split(":")
        label = parts[0]
        folder = parts[1]
        ep = int(parts[2]) if len(parts) > 2 else 999999
        try:
            prior, L, ep_actual = load_prior(folder, ep, device=args.device)
            W, L2, strides = within_model_cnn_cos(prior, f"{label} @ ep {ep_actual}")
            models.append((label, W, L2, strides))
        except Exception as e:
            print(f"  {label}: SKIP ({e})")

    if len(models) < 2:
        print("need ≥2 models for cross-model comparison"); return

    cross_model_stride_cos(models)

    # Also compare Conv2 (output layer) — the (mu, log_sigma) predictor —
    # more physics-meaningful than full-weight cosine
    print("\n=== CROSS-MODEL cos(Conv2 output-layer weights) ===")
    print("(Conv2 = last conv layer = (mu, log_sigma) predictor)")
    conv2_models = []
    for label_spec, folder, ep in [(spec.split(":")[0], spec.split(":")[1],
                                    int(spec.split(":")[2]) if len(spec.split(":")) > 2 else 999999) for spec in args.cells]:
        try:
            prior, L, ep_actual = load_prior(folder, ep, device=args.device)
            conv2_ws = []
            for cnn in prior.cnns:
                # Find last Conv2d
                convs = [m for m in cnn.modules() if isinstance(m, torch.nn.Conv2d)]
                last_conv = convs[-1]
                w = last_conv.weight.data.flatten().cpu().numpy()
                if last_conv.bias is not None:
                    w = np.concatenate([w, last_conv.bias.data.flatten().cpu().numpy()])
                conv2_ws.append(w)
            conv2_models.append((label_spec, conv2_ws, [float(np.linalg.norm(w)) for w in conv2_ws], list(prior.strides)[1:]))
        except Exception as e:
            pass
    cross_model_stride_cos(conv2_models)


if __name__ == "__main__":
    main()
