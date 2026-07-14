"""Tier 1: physical-observables comparison across trained flow variants.

Samples N configs from each trained flow, computes:
  ⟨E⟩       per-site energy   (Ising Hamiltonian)
  ⟨|M|⟩     mean absolute magnetization
  ⟨M²⟩      second moment
  χ = ⟨M²⟩ − ⟨|M|⟩²   susceptibility proxy
  U₄ = 1 − ⟨M⁴⟩ / (3⟨M²⟩²)   Binder cumulant (universal at T_c)
  G(r)      axial two-point correlation for r=1..L/2

Ground-truth reference is the HS-data file (Wolff MCMC samples).

Usage:
  python analyzers/tier1_observables.py \\
      --folder DATAFOLDER --N 10000 --epoch BEST_EPOCH

Output: single-row summary line prefixed [TIER1_ROW] parseable by the
compare script.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    savs = sorted(glob.glob(os.path.join(folder, "savings/*.saving")),
                  key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in savs]
    return savs[min(range(len(eps)), key=lambda i: abs(eps[i] - epoch))]


def spins_from_x(x):
    """x is a continuous flow output. Ising spins are sign(x) → ±1."""
    return torch.sign(x).clamp(min=-1.0, max=1.0)
    # (clamp guards the 0-value edge, which is measure-zero for continuous x)


def ising_energy(spins):
    """Energy per site H/N for 2D Ising with J=1, periodic BC.
    E = -(1/N) Σ_<i,j> s_i s_j     (nearest neighbors, 2 pairs per site: right + down)
    Returns (B,) per-config energies.
    """
    right = spins * torch.roll(spins, shifts=-1, dims=-1)
    down  = spins * torch.roll(spins, shifts=-1, dims=-2)
    B = spins.shape[0]
    pair_sum = (right + down).reshape(B, -1).sum(dim=1)
    N = spins.reshape(B, -1).shape[1]
    return -pair_sum / N   # per site (each bond counted once)


def observables(spins):
    """Compute per-config observables from a batch of spin configurations.

    spins: (B, 1, L, L) ±1 tensor.
    Returns dict with M, absM, M², M⁴, E, and axial G(r) for r=1..L//2.
    """
    B = spins.shape[0]
    L = spins.shape[-1]
    M = spins.reshape(B, -1).float().mean(dim=1)         # (B,)
    absM = M.abs()
    M2 = M.pow(2)
    M4 = M.pow(4)
    E = ising_energy(spins)                              # (B,)

    # Axial two-point correlation G(r) — averaged over both axes and origin
    G = torch.zeros(L // 2 + 1, device=spins.device)
    for r in range(L // 2 + 1):
        gh = (spins * torch.roll(spins, shifts=-r, dims=-1)).reshape(B, -1).mean(dim=1)
        gv = (spins * torch.roll(spins, shifts=-r, dims=-2)).reshape(B, -1).mean(dim=1)
        G[r] = 0.5 * (gh.mean() + gv.mean())

    return {
        "M": M.cpu().numpy(),
        "absM": absM.cpu().numpy(),
        "M2": M2.cpu().numpy(),
        "M4": M4.cpu().numpy(),
        "E":  E.cpu().numpy(),
        "G":  G.cpu().numpy(),
        "L":  L,
    }


def summary(obs):
    m_absM = obs["absM"].mean()
    m_M2 = obs["M2"].mean()
    m_M4 = obs["M4"].mean()
    m_E = obs["E"].mean()
    L = obs["L"]

    # Susceptibility proxy: χ = N * (⟨M²⟩ − ⟨|M|⟩²)  (finite-size volume prefactor)
    chi = (L * L) * (m_M2 - m_absM ** 2)
    # Binder cumulant: U₄ = 1 − ⟨M⁴⟩ / (3⟨M²⟩²)   universal at T_c
    U4 = 1.0 - m_M4 / (3.0 * m_M2 ** 2 + 1e-12)

    return dict(
        E_mean=float(m_E),
        absM_mean=float(m_absM),
        M2_mean=float(m_M2),
        chi=float(chi),
        U4=float(U4),
        E_std=float(obs["E"].std()),
        G=obs["G"].tolist(),
    )


def ground_truth(L, T, N):
    """Load Wolff MCMC ground-truth samples and compute observables.

    Uses data/mcmc_data/hs_L{L}_T{T}_N*.pt (same file the training used).
    """
    pat = f"data/mcmc_data/hs_L{L}_T{T}_N*.pt"
    files = glob.glob(pat)
    if not files:
        # try with truncated T
        Ttrunc = f"{T:.15f}".rstrip('0').rstrip('.')
        pat2 = f"data/mcmc_data/hs_L{L}_T{Ttrunc}_N*.pt"
        files = glob.glob(pat2)
    if not files:
        # try 2.269 or 2.269185314213022
        for cand in ["2.269185314213022", "2.269"]:
            pat3 = f"data/mcmc_data/hs_L{L}_T{cand}_N*.pt"
            files = glob.glob(pat3)
            if files:
                break
    if not files:
        raise FileNotFoundError(f"no HS ground truth for L={L}, T={T}")
    hs_path = sorted(files)[-1]  # largest N
    print(f"[gt] loading {hs_path}")
    data = torch.load(hs_path, map_location="cpu")
    if isinstance(data, dict):
        data = data.get("samples", data.get("data", data))
    # data is (M, 1, L, L) already ±1 (Ising spins)
    idx = np.random.default_rng(0).choice(data.shape[0], size=min(N, data.shape[0]), replace=False)
    spins = data[idx].float()
    if spins.max() > 1.5:  # in case it's ±L
        spins = spins.sign()
    return spins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="trained run folder (or 'GT' for ground truth)")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--N", type=int, default=10000, help="samples to draw from flow")
    ap.add_argument("--batch", type=int, default=500, help="sampling batch size")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default=None, help="label for TIER1_ROW output")
    ap.add_argument("--L", type=int, default=None, help="required if folder==GT")
    ap.add_argument("--T", type=float, default=2.269185314213022,
                    help="required if folder==GT")
    args = ap.parse_args()

    label = args.label if args.label else os.path.basename(args.folder.rstrip("/"))

    if args.folder == "GT":
        L = args.L
        assert L is not None, "for GT, pass --L"
        spins = ground_truth(L, args.T, args.N)
        # move to CPU/GPU consistent
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        spins = spins.to(device)
        obs = observables(spins)
        s = summary(obs)
        print(f"[GT] L={L} T={args.T} N={spins.shape[0]}")
        print(f"[GT] E={s['E_mean']:+.4f} ± {s['E_std']:.4f}  "
              f"|M|={s['absM_mean']:.4f}  M²={s['M2_mean']:.4f}  "
              f"χ={s['chi']:.2f}  U₄={s['U4']:.4f}")
        print(f"[TIER1_ROW] {label}\tN={spins.shape[0]}\tE={s['E_mean']:+.4f}\t"
              f"absM={s['absM_mean']:.4f}\tM2={s['M2_mean']:.4f}\t"
              f"chi={s['chi']:.2f}\tU4={s['U4']:.4f}\t"
              f"G={','.join(f'{g:.4f}' for g in s['G'])}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = pick_saving(args.folder, args.epoch)
    ep = int(re.search(r"epoch(\d+)", ckpt).group(1))
    print(f"[flow] {args.folder}  epoch {ep}")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    fw, target, L, T, sym_used, wt, hp = build_flow(args.folder, state, device=str(device))
    fw.eval()

    # Sample from flow
    all_spins = []
    remaining = args.N
    with torch.no_grad():
        while remaining > 0:
            n = min(args.batch, remaining)
            x, _ = fw.sample(n)
            spins = spins_from_x(x)
            all_spins.append(spins)
            remaining -= n
    spins = torch.cat(all_spins, dim=0)
    print(f"[flow] sampled {spins.shape[0]} configs, shape={tuple(spins.shape)}")

    obs = observables(spins)
    s = summary(obs)
    print(f"[flow] E={s['E_mean']:+.4f} ± {s['E_std']:.4f}  "
          f"|M|={s['absM_mean']:.4f}  M²={s['M2_mean']:.4f}  "
          f"χ={s['chi']:.2f}  U₄={s['U4']:.4f}")
    print(f"[TIER1_ROW] {label}\tN={spins.shape[0]}\tE={s['E_mean']:+.4f}\t"
          f"absM={s['absM_mean']:.4f}\tM2={s['M2_mean']:.4f}\t"
          f"chi={s['chi']:.2f}\tU4={s['U4']:.4f}\t"
          f"G={','.join(f'{g:.4f}' for g in s['G'])}")


if __name__ == "__main__":
    main()
