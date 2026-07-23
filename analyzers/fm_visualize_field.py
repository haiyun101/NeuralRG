"""Visualize the learned velocity field v_t^θ(x, t) from a flow-matching model.

Four kinds of plot:
  1. Sample-trajectory frames — snapshots of x_t at t = 0, 0.2, 0.4, 0.6, 0.8, 1.0
     for several samples. Shows the "noise → Ising domain" evolution.
  2. Velocity-magnitude heatmap — |v_t(x)| as a 2D heatmap on the lattice at
     a fixed t and fixed x. Shows where the field wants to push hardest.
  3. Norm-vs-time — ||v_t(x_t)||_2 along a trajectory as function of t. Shows
     when the flow does its work.
  4. Divergence field — ∇·v_t(x) as a heatmap at a fixed x, t. Where does
     volume expand vs contract?

Usage:
  python analyzers/fm_visualize_field.py \\
      --folder data/L32_T2.269_flowmatching_h64/ --epoch 1000 \\
      --n_samples 6 --n_frames 6 --steps 100
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow.flow_matching import VelocityUNet


# ─────────────────── loading ───────────────────


def find_ckpt(folder, epoch_str):
    all_ck = sorted(glob.glob(os.path.join(folder, "savings", "fm_*_epoch*.pt")),
                    key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if not all_ck:
        raise FileNotFoundError(f"no FM checkpoints in {folder}/savings/")
    if epoch_str == "latest":
        return all_ck[-1]
    ep_target = int(epoch_str)
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in all_ck]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - ep_target))
    return all_ck[idx]


def load_fm(ckpt_path, device):
    state = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = state.get("config", {})
    L = int(cfg.get("L", 32))
    nhidden = int(cfg.get("nhidden", 64))
    temb_dim = int(cfg.get("tembDim", 128))
    v_net = VelocityUNet(L=L, nhidden=nhidden, temb_dim=temb_dim).to(device)
    v_net.load_state_dict(state["model"])
    v_net.eval()
    return v_net, L, state.get("epoch", -1)


# ─────────────────── trajectory sampling ───────────────────


@torch.no_grad()
def euler_trajectory(v_net, shape, n_steps, device, frames_at=None):
    """Euler ODE from x_0~N(0,I) to x_1. Returns:
        - final x_1 (B, 1, L, L)
        - stored frames at requested t-values: dict{t: (B, 1, L, L)}
        - per-step ||v||_2 per sample: (n_steps, B)
        - per-step t-value: (n_steps,)
    """
    frames_at = frames_at if frames_at is not None else []
    v_net.eval()
    B = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / n_steps
    v_norms = torch.zeros(n_steps, B, device=device)
    ts = torch.zeros(n_steps, device=device)
    stored = {}
    frame_t_set = set(round(x_ * 1000) for x_ in frames_at)
    # save the t=0 frame (before any step)
    if 0 in frame_t_set:
        stored[0.0] = x.detach().cpu().clone()
    for i in range(n_steps):
        t = i * dt
        ts[i] = t
        t_val = torch.full((B,), t, device=device)
        v = v_net(x, t_val)
        v_norms[i] = v.reshape(B, -1).norm(dim=-1)
        x = x + v * dt
        t_next = (i + 1) * dt
        t_next_key = round(t_next * 1000)
        if t_next_key in frame_t_set:
            stored[t_next] = x.detach().cpu().clone()
    return x, stored, v_norms.cpu(), ts.cpu()


# ─────────────────── visualization ───────────────────


def plot_trajectories(stored, n_show, out_path, title):
    """Grid: rows = samples, cols = time snapshots.
    Renders sign(x) for physical interpretability."""
    times = sorted(stored.keys())
    nt = len(times)
    fig, axes = plt.subplots(n_show, nt, figsize=(nt * 1.5, n_show * 1.5))
    if n_show == 1:
        axes = axes[None, :]
    if nt == 1:
        axes = axes[:, None]
    for i in range(n_show):
        for j, t in enumerate(times):
            ax = axes[i, j]
            x_ij = stored[t][i, 0].numpy()
            ax.imshow(np.sign(x_ij), cmap="binary", vmin=-1, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={t:.2f}", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.92)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_v_norm(v_norms, ts, out_path, title):
    """||v_t(x_t)|| as function of t, one line per sample.
    Also mean±std envelope."""
    v_norms = v_norms.numpy()
    ts = ts.numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for b in range(v_norms.shape[1]):
        ax.plot(ts, v_norms[:, b], color="C0", alpha=0.35, linewidth=0.8)
    mean = v_norms.mean(axis=1)
    std = v_norms.std(axis=1)
    ax.plot(ts, mean, color="black", linewidth=2, label="mean over samples")
    ax.fill_between(ts, mean - std, mean + std, color="black", alpha=0.15,
                     label="mean ± std")
    ax.set_xlabel("t  (ODE time)")
    ax.set_ylabel(r"$\|v_t^\theta(x_t)\|_2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_v_heatmap(v_net, x, t_vals, out_path, title, device):
    """Show |v_t^θ(x, t)| as heatmap for the first sample, at several t values.
    Uses a *fixed* input x — visualizes v as a function of t only."""
    B = x.shape[0]
    fig, axes = plt.subplots(2, len(t_vals), figsize=(2 * len(t_vals), 4))
    for j, t in enumerate(t_vals):
        with torch.no_grad():
            t_val = torch.full((B,), float(t), device=device)
            v = v_net(x.to(device), t_val).cpu()
        v_map = v[0, 0].numpy()                   # first sample's velocity field
        # top row: signed velocity (blue = negative, red = positive)
        ax_top = axes[0, j]
        vmax = max(abs(v_map.min()), abs(v_map.max()))
        im = ax_top.imshow(v_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_top.axis("off")
        ax_top.set_title(f"$v_t(x)$  t={t:.2f}", fontsize=9)
        # bottom row: magnitude
        ax_bot = axes[1, j]
        ax_bot.imshow(np.abs(v_map), cmap="viridis")
        ax_bot.axis("off")
        ax_bot.set_title(f"$|v_t(x)|$", fontsize=9)
    # also show x itself as a reference at top-left
    fig.suptitle(title + "  (single fixed x, both signed and abs magnitude)", fontsize=11)
    fig.subplots_adjust(wspace=0.1, hspace=0.15, top=0.88)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_divergence(v_net, x, t_val, out_path, title, device, n_eps=8):
    """Estimate divergence field via Hutchinson: ε · ∂v/∂x ε averaged over
    multiple ε samples, per lattice site (approximation)."""
    v_net.eval()
    B = x.shape[0]
    x = x.to(device).requires_grad_(True)
    t = torch.full((B,), float(t_val), device=device)
    v = v_net(x, t)
    div_site_accum = torch.zeros_like(x)
    for _ in range(n_eps):
        eps = (torch.rand_like(x) > 0.5).float() * 2 - 1        # Rademacher
        dot = (v * eps).sum()
        grad = torch.autograd.grad(dot, x, create_graph=False, retain_graph=True)[0]
        div_site_accum = div_site_accum + grad * eps / n_eps
    div_map = div_site_accum[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    vmax = max(abs(div_map.min()), abs(div_map.max()))
    im = ax.imshow(div_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axis("off")
    ax.set_title(title + f"  (Hutchinson, n_eps={n_eps})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ─────────────────── main ───────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--epoch", default="latest")
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--n_frames", type=int, default=6,
                   help="how many trajectory frames to save (at evenly spaced t)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = find_ckpt(args.folder, args.epoch)
    v_net, L, epoch = load_fm(ckpt, args.device)
    epoch_num = int(re.search(r"epoch(\d+)", ckpt).group(1))
    print(f"[load] {ckpt}  L={L}  epoch={epoch_num}")

    # ── 1. sample trajectories + collect frames + v-norm ──
    frames_at = list(np.linspace(0, 1, args.n_frames))
    torch.manual_seed(1)
    x_final, stored, v_norms, ts = euler_trajectory(
        v_net, (args.n_samples, 1, L, L),
        n_steps=args.steps, device=args.device, frames_at=frames_at,
    )

    tag = f"fm_L{L}_ep{epoch_num}"
    plot_trajectories(
        stored, n_show=args.n_samples,
        out_path=os.path.join(args.folder, f"{tag}_trajectories.png"),
        title=f"{tag}  ODE trajectory snapshots (sign(x_t))",
    )

    plot_v_norm(
        v_norms, ts,
        out_path=os.path.join(args.folder, f"{tag}_vnorm_vs_t.png"),
        title=f"{tag}  velocity magnitude ||v_t^θ(x_t)|| along ODE trajectory",
    )

    # ── 2. velocity heatmap at fixed x, varying t ──
    # Use the FINAL trajectory endpoint (a real Ising sample) as fixed x.
    x_for_heat = x_final[:1].detach()
    plot_v_heatmap(
        v_net, x_for_heat,
        t_vals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        out_path=os.path.join(args.folder, f"{tag}_v_field_at_x1.png"),
        title=f"{tag}  velocity field v_t^θ(x, t) at fixed x = final sample",
        device=args.device,
    )

    # ── 3. divergence field at fixed t (mid-flow) ──
    plot_divergence(
        v_net, x_for_heat, t_val=0.5,
        out_path=os.path.join(args.folder, f"{tag}_divergence_t0.5.png"),
        title=f"{tag}  divergence ∇·v_t(x) at t=0.5",
        device=args.device, n_eps=16,
    )

    print("\n[done]")
    print(f"  outputs saved under {args.folder}/{tag}_*.png")


if __name__ == "__main__":
    main()
