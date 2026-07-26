"""Stride-aligned smaller-L → larger-L weight transfer utilities.

Test hypothesis: a well-trained smaller-L champion's per-scale weights
(both MERA RNVP blocks and HCG per-scale CNNs) can initialise a larger-L
run. Rooted in RG universality — at the critical point, operators at
matched physical strides should coincide.

Alignment rules (see project memory `champion-transfer-goal`):
  MERA:  block k in smaller → block k in larger  (both use the same
         2x2 coarse-graining; scale index = physical operation index).
         Larger's extra COARSEST blocks (index >= 2*n_scales_small)
         stay at fresh init.
  HCG:   per-scale CNN aligned by STRIDE, not by index. Larger's
         coarsest CNN levels (with strides not present in smaller)
         stay at fresh init. See
         `HierarchicalConditionalGaussian.init_perscale_from_smaller_L_state`.
"""
import os
import re

import torch


def _find_key_prefix(state_dict, candidates):
    """Return the first prefix in `candidates` that matches any key in
    state_dict; None otherwise."""
    for pref in candidates:
        for k in state_dict.keys():
            if k.startswith(pref):
                return pref
    return None


def load_smaller_L_state(ckpt_path, device="cpu"):
    """Load a checkpoint saved by main.py, return a clean state_dict."""
    obj = torch.load(ckpt_path, weights_only=False, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"unrecognised checkpoint format at {ckpt_path}: {type(obj)}")


def transfer_mera_blocks(target_flow, src_state_dict, verbose=True):
    """Copy MERA layerList[k] weights from src into target_flow, for k in
    0..min(N_src, N_tgt)-1.

    target_flow is expected to be the inner (unwrapped) MERA-Flow object
    or Symmetrized wrapper — we probe for `flow.layerList.*` first, then
    fall back to `layerList.*`.

    Blocks beyond min(N_src, N_tgt) stay at target's fresh init.
    """
    # Detect key prefix used in src state
    src_pref = _find_key_prefix(src_state_dict, ["flow.layerList.", "layerList."])
    if src_pref is None:
        print("[transfer_mera] no layerList keys in source; skipping MERA transfer")
        return 0
    # Detect target module path
    if hasattr(target_flow, "flow") and hasattr(target_flow.flow, "layerList"):
        layer_list = target_flow.flow.layerList
        tgt_pref = "flow.layerList."
    elif hasattr(target_flow, "layerList"):
        layer_list = target_flow.layerList
        tgt_pref = "layerList."
    else:
        print("[transfer_mera] target has no layerList attribute; skipping")
        return 0

    # Find all src block indices
    src_indices = set()
    src_pattern = re.compile(re.escape(src_pref) + r"(\d+)\.")
    for k in src_state_dict.keys():
        m = src_pattern.match(k)
        if m:
            src_indices.add(int(m.group(1)))
    if not src_indices:
        print("[transfer_mera] no block indices found in src; skipping")
        return 0

    N_src = max(src_indices) + 1
    N_tgt = len(layer_list)
    N_copy = min(N_src, N_tgt)

    n_copied = 0
    with torch.no_grad():
        for block_idx in range(N_copy):
            src_prefix = f"{src_pref}{block_idx}."
            block_params = {
                k[len(src_prefix):]: v
                for k, v in src_state_dict.items()
                if k.startswith(src_prefix)
            }
            if not block_params:
                continue
            block_sd = layer_list[block_idx].state_dict()
            for name, param in block_params.items():
                if name in block_sd and block_sd[name].shape == param.shape:
                    block_sd[name].copy_(param)
                    n_copied += 1

    if verbose:
        print(f"[transfer_mera] copied {N_copy}/{N_tgt} block weight sets from source "
              f"({N_src} src blocks, {N_tgt - N_copy} target blocks left fresh at the "
              f"coarsest end)")
    return n_copied


def transfer_hcg_perscale(target_flow, src_state_dict, src_strides, verbose=True):
    """Delegate to HCG.init_perscale_from_smaller_L_state."""
    # Unwrap Symmetrized → inner flow → prior
    prior = None
    if hasattr(target_flow, "flow") and hasattr(target_flow.flow, "prior"):
        prior = target_flow.flow.prior
    elif hasattr(target_flow, "prior"):
        prior = target_flow.prior
    if prior is None:
        print("[transfer_hcg] target has no prior; skipping")
        return
    if prior.__class__.__name__ != "HierarchicalConditionalGaussian":
        print(f"[transfer_hcg] target prior is {prior.__class__.__name__}, not HCG; skipping")
        return

    if verbose:
        print(f"[transfer_hcg] src strides = {src_strides}, "
              f"target strides = {list(prior.strides)}")
    prior.init_perscale_from_smaller_L_state(src_state_dict, src_strides)


def transfer_from_smaller_L(target_flow, ckpt_path, src_strides, device="cpu",
                              components="both"):
    """Top-level entry: load a smaller-L champion's state and transfer
    MERA blocks (scale-index aligned) and/or HCG per-scale CNNs (stride
    aligned) into target_flow.

    target_flow: constructed larger-L flow (Symmetrized or bare MERA).
    ckpt_path:   path to a .saving file from a smaller-L run.
    src_strides: list of HCG strides used in the smaller-L run
                 (coarsest → finest, e.g. [16,8,4,2,1] for L=32).
    components:  "both" (default) — transfer MERA + CNN
                 "mera" — MERA only (CNN stays fresh init)
                 "cnn"  — CNN only (MERA stays fresh init)
                 Ablation experiments to isolate which component
                 contains more transferable physics.
    """
    print(f"[transfer] loading source state from {ckpt_path}")
    print(f"[transfer] components = {components}")
    src_state = load_smaller_L_state(ckpt_path, device)
    print(f"[transfer] source state has {len(src_state)} tensors")
    if components in ("both", "mera"):
        transfer_mera_blocks(target_flow, src_state)
    else:
        print("[transfer_mera] SKIPPED (components=cnn only)")
    if components in ("both", "cnn"):
        transfer_hcg_perscale(target_flow, src_state, src_strides)
    else:
        print("[transfer_hcg] SKIPPED (components=mera only)")
    print("[transfer] done")
