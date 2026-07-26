"""Flow-matching training loop — parallel to train/learn.py but MUCH simpler
(no Jacobian, no importance weighting, no HMC eval).

Loads HS-transformed MCMC data from data/mcmc_data/hs_L{L}_T{T}_N*.pt, trains
`VelocityUNet` with the CFM loss, and periodically:
  - logs training loss to stdout,
  - saves the model to <folder>/savings/fm_L{L}_T{T}_epoch{N}.pt,
  - samples N configs via Euler ODE and appends physics summary to log.

Usage:
  python train/fm_learn.py -L 32 -T 2.269 -epochs 20000 -batch 128 \\
      -nhidden 64 -lr 1e-3 -sampleSteps 50 \\
      -folder ./data/L32_T2.269_flowmatching_h64/
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow.flow_matching import VelocityUNet, MERAUNet, TrueMERAVelocityField, cfm_loss, sample_euler


def load_hs_data(L, T, dataPath=None):
    if dataPath and os.path.exists(dataPath):
        p = dataPath
    else:
        pattern = f"./data/mcmc_data/hs_L{L}_T{T}_N*.pt"
        cand = sorted(glob.glob(pattern), key=lambda s: int(s.split("_N")[-1].split(".")[0]), reverse=True)
        if not cand:
            # Also allow the full-precision T variant
            pattern = f"./data/mcmc_data/hs_L{L}_T{T:.15f}_N*.pt"
            cand = sorted(glob.glob(pattern), key=lambda s: int(s.split("_N")[-1].split(".")[0]), reverse=True)
        if not cand:
            raise FileNotFoundError(f"no HS data matching {pattern}")
        p = cand[0]
    print(f"[data] loading {p}")
    x = torch.load(p, weights_only=False)
    if isinstance(x, list):
        x = torch.stack(x)
    while x.dim() < 4:
        x = x.unsqueeze(1) if x.dim() == 3 else x.unsqueeze(0)
    if x.shape[1] != 1:
        x = x[:, :1]
    x = x.float()
    print(f"[data] shape={tuple(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}")
    return x


def quick_physics(spins):
    """Compute quick per-batch physics summary. spins: (B, 1, L, L) continuous
    field; we discretize via sign(x) as the Ising configuration."""
    sig = torch.sign(spins).squeeze(1)                 # (B, L, L), values in {-1, 0, +1}
    L = sig.shape[-1]
    N = L * L
    M = sig.mean(dim=(-1, -2))                         # (B,) mean magnetization per sample
    abs_M = M.abs().mean().item()
    M2 = (M ** 2).mean().item()
    M4 = (M ** 4).mean().item()
    chi = N * (M2 - M.abs().mean().item() ** 2)
    U4 = 1.0 - M4 / (3 * M2 ** 2 + 1e-12)
    return dict(absM=abs_M, chi=float(chi), U4=float(U4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-L", type=int, required=True)
    p.add_argument("-T", type=float, required=True)
    p.add_argument("-folder", required=True)
    p.add_argument("-dataPath", default=None)
    p.add_argument("-epochs", type=int, default=20000)
    p.add_argument("-batch", type=int, default=128)
    p.add_argument("-nhidden", type=int, default=64)
    p.add_argument("-tembDim", type=int, default=128)
    p.add_argument("-arch", choices=["unet", "meraunet", "truemera"], default="unet",
                   help="unet = 2-stage U-Net (naive FM); meraunet = "
                        "log2(L)-stage U-Net with MERA-style scale hierarchy; "
                        "truemera = full MERA structure (im2col dispatch/collect "
                        "on 2×2 patches at every scale, time-modulated MLP blocks, "
                        "sequential delta accumulation)")
    p.add_argument("-maxChannelMult", type=int, default=4,
                   help="max channel multiplier for MERA U-Net; caps channel "
                        "doubling at nhidden × max_mult (default 4)")
    p.add_argument("-meraNrepeat", type=int, default=1,
                   help="[truemera only] blocks per scale (nrepeat), matches MERA convention")
    p.add_argument("-meraHiddenLayers", type=int, default=2,
                   help="[truemera only] internal hidden MLP layers per patch block")
    p.add_argument("-lr", type=float, default=1e-3)
    p.add_argument("-savePeriod", type=int, default=1000)
    p.add_argument("-samplePeriod", type=int, default=500)
    p.add_argument("-sampleSteps", type=int, default=50, help="Euler ODE steps for sampling")
    p.add_argument("-sampleN", type=int, default=200, help="samples per periodic eval")
    p.add_argument("-cuda", type=int, default=0)
    p.add_argument("-load", action="store_true", help="resume from latest ckpt in folder")
    p.add_argument("-gradClip", type=float, default=0.0)
    p.add_argument("-seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.join(args.folder, "savings"), exist_ok=True)
    with open(os.path.join(args.folder, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── data ──
    data = load_hs_data(args.L, args.T, args.dataPath)
    n_data = data.shape[0]
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # ── model ──
    if args.arch == "meraunet":
        v_net = MERAUNet(L=args.L, nhidden=args.nhidden, temb_dim=args.tembDim,
                          max_channel_mult=args.maxChannelMult).to(device)
        arch_name = "MERAUNet"
    elif args.arch == "truemera":
        v_net = TrueMERAVelocityField(L=args.L, nrepeat=args.meraNrepeat,
                                       hidden=args.nhidden, temb_dim=args.tembDim,
                                       n_hidden_layers=args.meraHiddenLayers).to(device)
        arch_name = "TrueMERAVelocityField"
    else:
        v_net = VelocityUNet(L=args.L, nhidden=args.nhidden, temb_dim=args.tembDim).to(device)
        arch_name = "VelocityUNet"
    n_params = sum(param.numel() for param in v_net.parameters())
    print(f"[model] {arch_name} L={args.L} nhidden={args.nhidden}  → {n_params/1e6:.2f} M params")

    opt = torch.optim.Adam(v_net.parameters(), lr=args.lr)
    start_epoch = 0

    if args.load:
        ckpts = sorted(glob.glob(os.path.join(args.folder, "savings", "fm_*_epoch*.pt")),
                       key=lambda p: int(p.split("epoch")[-1].split(".")[0]))
        if ckpts:
            state = torch.load(ckpts[-1], weights_only=False, map_location=device)
            v_net.load_state_dict(state["model"])
            if "optimizer" in state:
                opt.load_state_dict(state["optimizer"])
            start_epoch = state.get("epoch", 0)
            print(f"[load] resumed from {ckpts[-1]} @ epoch {start_epoch}")

    # ── train ──
    print(f"[train] {n_data} samples,  batch={args.batch},  epochs={args.epochs}")
    v_net.train()
    step = 0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        loss_epoch = 0.0
        n_batches = 0
        for (x1,) in dl:
            x1 = x1.to(device, non_blocking=True)
            opt.zero_grad()
            loss = cfm_loss(v_net, x1)
            loss.backward()
            if args.gradClip > 0:
                torch.nn.utils.clip_grad_norm_(v_net.parameters(), args.gradClip)
            opt.step()
            loss_epoch += float(loss)
            n_batches += 1
            step += 1
        loss_epoch /= max(1, n_batches)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            wall = time.time() - t0
            print(f"[epoch {epoch:>6}]  loss={loss_epoch:.4f}   wall={wall:.1f}s")

        if (epoch + 1) % args.samplePeriod == 0 or epoch == args.epochs - 1:
            v_net.eval()
            x_gen = sample_euler(v_net, (args.sampleN, 1, args.L, args.L),
                                  n_steps=args.sampleSteps, device=device)
            phys = quick_physics(x_gen.cpu())
            print(f"  [phys sampled N={args.sampleN} steps={args.sampleSteps}] "
                  f"|M|={phys['absM']:.3f}  χ={phys['chi']:.2f}  U₄={phys['U4']:.3f}")
            v_net.train()

        if (epoch + 1) % args.savePeriod == 0 or epoch == args.epochs - 1:
            ck = os.path.join(args.folder, "savings", f"fm_L{args.L}_T{args.T}_epoch{epoch + 1}.pt")
            torch.save({"model": v_net.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch + 1, "config": vars(args)}, ck)
            print(f"  [save] wrote {ck}")

    print("[done]")


if __name__ == "__main__":
    main()
