"""Evaluate a flow-matching checkpoint on the same Tier-1 physics observables
we use for the RNVP+MERA models.

Emits a `[TIER1_ROW]` line so the flow matching results can be dropped into
the same cross-model comparison plots and tables.

Usage:
  python analyzers/fm_evaluate.py --folder data/L32_T2.269_flowmatching_h64/ \\
      --epoch latest --N 4000 --steps 100 --T 2.269

Also computes G(r) and writes an optional flow_correlations.png in the folder
(2-panel, matching the existing save_corr_png convention).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow.flow_matching import VelocityUNet, MERAUNet, sample_euler, sample_rk4


# ─────────────────── observables ───────────────────


def compute_physics(spins):
    """spins: (B, 1, L, L). Compute per-config M = mean(sign(x)), then
    aggregate |M|, χ, U₄, E (per-site Ising energy from nearest-neighbor)."""
    x = spins.squeeze(1)                                     # (B, L, L)
    sig = torch.sign(x)                                      # ±1 or 0
    L = x.shape[-1]
    N = L * L
    M = sig.mean(dim=(-1, -2))                               # (B,) mean magnetization
    abs_M = M.abs().mean().item()
    M2 = (M ** 2).mean().item()
    M4 = (M ** 4).mean().item()
    chi = N * (M2 - abs_M ** 2)                              # standard χ
    U4 = 1.0 - M4 / (3 * M2 * M2 + 1e-12)
    # Ising energy per site: E = -1/N Σ_<ij> σ_i σ_j; here NN sums both axes.
    NN = (sig * torch.roll(sig, -1, dims=-1)).mean(dim=(-1, -2)) \
       + (sig * torch.roll(sig, -1, dims=-2)).mean(dim=(-1, -2))
    E_per_site = -NN.mean().item()
    return dict(absM=abs_M, chi=float(chi), U4=float(U4), E=E_per_site,
                M2=M2, N_used=int(spins.shape[0]))


# ─────────────────── log-density via ODE + Hutchinson trace ───────────────────


def hutchinson_divergence(v_net, x, t, eps):
    """Stochastic estimate of tr(∂v/∂x) via ε^T (∂v/∂x) ε, for one t value.
    Uses one backward pass per (batch element, ε sample).

    x, eps: (B, 1, L, L). t: (B,). Returns scalar divergence per sample (B,).
    """
    x = x.detach().requires_grad_(True)
    v = v_net(x, t)
    dot = (v * eps).sum()
    grad = torch.autograd.grad(dot, x, create_graph=False)[0]
    return (grad * eps).sum(dim=(1, 2, 3))                    # (B,)


@torch.enable_grad()
def logq_via_ode(v_net, x1, n_steps=50, n_eps=1, device="cuda:0", direction="backward"):
    """Compute log q(x_1) for the flow-matching model via ODE integration
    with Hutchinson trace estimator.

    direction:
      "backward" — start from x_1 (data), integrate ODE backward (t: 1 → 0)
                   to get x_0, accumulate ∫ ∇·v dt. Use for KL(p‖q).
      "forward"  — start from x_0 (noise), integrate forward (t: 0 → 1),
                   accumulate divergence. Use during sampling to get log q of samples.

    Returns log_q  (B,).
    """
    B = x1.shape[0]
    x1 = x1.to(device)
    dt = 1.0 / n_steps

    if direction == "backward":
        x = x1.clone()
        div_accum = torch.zeros(B, device=device)
        for i in range(n_steps):
            t_val = torch.full((B,), 1.0 - i * dt, device=device)
            div_batch = torch.zeros(B, device=device)
            for _ in range(n_eps):
                eps = (torch.rand_like(x) > 0.5).float() * 2 - 1     # Rademacher
                div_batch = div_batch + hutchinson_divergence(v_net, x, t_val, eps) / n_eps
            div_accum = div_accum + div_batch * dt
            # ODE backward: dx/dt = v_t, so backward step subtracts v * dt
            with torch.no_grad():
                x = x - v_net(x, t_val) * dt
        # x is now x_0 (in noise space)
        log_q0 = -0.5 * (x ** 2).sum(dim=(1, 2, 3)) - 0.5 * x[0].numel() * math.log(2 * math.pi)
        # Continuity equation: log q_1(x_1) = log q_0(x_0) - ∫_0^1 (∇·v_t) dt
        # (correct sign is MINUS — see Chen et al. 2018 NeurIPS §2, Lipman et al. 2022)
        log_q = log_q0 - div_accum
        return log_q
    else:
        raise NotImplementedError("forward direction: TODO if needed")


def compute_kl_pq(v_net, x_data, n_steps=50, n_eps=1, batch=200, device="cuda:0"):
    """KL(p‖q) = CE − H(p) = -E_p[log q] − H(p).
    Here we return only E_p[log q] (via ODE + Hutchinson trace). Caller supplies
    or computes H(p_HS) via `compute_hp_mc` below.
    """
    N_total = x_data.shape[0]
    log_q_all = []
    for i in range(0, N_total, batch):
        x_b = x_data[i:i + batch]
        log_q = logq_via_ode(v_net, x_b, n_steps=n_steps, n_eps=n_eps, device=device)
        log_q_all.append(log_q.detach().cpu())
    log_q_all = torch.cat(log_q_all)
    return dict(
        E_p_log_q=float(log_q_all.mean()),
        sem_E_p_log_q=float(log_q_all.std() / math.sqrt(len(log_q_all))),
    )


def compute_hp_mc(x_data, L, T, device="cuda:0"):
    """Estimate H(p_HS) = E_p[A(x)] + lnZ_c from x ~ p_HS data samples.
    Uses the analytical HS-continuous log-density term A(x) from source/ising.py
    plus the exact partition function ln Z_c (discrete Z + HS-continuous fix)
    from etc/exactz.md.

    Returns (H_p_mc, ln_Z_c) or (None, None) if exactz has no entry for (L, T).
    """
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import source
    from analyzers.flow_sample_diagnostic import parse_exactz, EXACT_FILE

    target = source.Ising(L, 2, T).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        A_vals = target.energy(x_data.to(device)).cpu().numpy()
    lnZ_d, fix = parse_exactz(os.path.join(root, EXACT_FILE), L, T)
    if lnZ_d is None:
        return None, None
    ln_Z_c = float(lnZ_d + fix)
    H_p_mc = float(A_vals.mean() + ln_Z_c)
    return H_p_mc, ln_Z_c


# ─────────────────── plotting (matches flow_sample_diagnostic conventions) ───────────────────


def save_config_grid(x, out_path, nrow=8, ncol=8, title=None):
    """Grid of nrow × ncol sampled Ising configurations.
    x: (B, 1, L, L). Render via sign(x) so ±1 pixels are black/white."""
    x = x.detach().cpu().numpy().squeeze(1)
    B, L, _ = x.shape
    n = min(B, nrow * ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.4))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(np.sign(x[i]), cmap="binary", vmin=-1, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_M_and_Gr(out_path, M_q, M_p, G_q, G_p, T, L, epoch, label):
    """Two panels: signed M distribution (bimodal at T_c) + normalized G(r)/G(0)
    log-log with Onsager reference. Matches the current save_corr_png format
    from analyzers/flow_sample_diagnostic.py."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # panel 1: signed magnetisation
    mmax = float(np.abs(M_q).max())
    if M_p is not None:
        mmax = max(mmax, float(np.abs(M_p).max()))
    bins = np.linspace(-mmax * 1.05, mmax * 1.05, 51)
    ax1.hist(M_q, bins=bins, density=True, alpha=0.55, color="C0",
             label=f"flow  x ~ q  (⟨|M|⟩={np.abs(M_q).mean():.3f})")
    if M_p is not None:
        ax1.hist(M_p, bins=bins, density=True, alpha=0.55, color="C1",
                 label=f"HS data  x ~ p  (⟨|M|⟩={np.abs(M_p).mean():.3f})")
    ax1.set_xlabel("magnetisation  M = (1/N) sum sign(x_i)")
    ax1.set_ylabel("density")
    ax1.set_title("per-config signed magnetisation")
    ax1.legend(fontsize=9)

    # panel 2: log-log normalized G(r)/G(0), r=0 dropped
    r_all = np.arange(len(G_q))
    r = r_all[r_all >= 1]
    Gq_n = np.abs(np.array(G_q)[r_all >= 1] / G_q[0])
    ax2.loglog(r, Gq_n, "o-", color="C0", label="flow  x ~ q")
    if G_p is not None:
        Gp_n = np.abs(np.array(G_p)[r_all >= 1] / G_p[0])
        ax2.loglog(r, Gp_n, "s-", color="C1", label="HS data  x ~ p")
    if T is not None and abs(T - 2.269185314213022) < 0.01:
        anchor = float(np.abs(G_p[1] / G_p[0])) if G_p is not None else float(np.abs(G_q[1] / G_q[0]))
        eta = 0.25
        r_ref = np.array([1.0, float(r[-1])])
        ax2.loglog(r_ref, anchor * (r_ref / 1.0) ** (-eta),
                   "k--", linewidth=1.2, alpha=0.75,
                   label=r"$T_c$ theory: $G\propto r^{-1/4}$")
    ax2.set_xlabel("lattice distance  r")
    ax2.set_ylabel("|G(r)| / G(0)")
    ax2.set_title("normalised G(r)/G(0) (log-log)" +
                  (f"  (T={T:.3f}{', T_c' if T and abs(T-2.269185314213022)<0.01 else ''})" if T else ""))
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    fig.suptitle(f"{label}    (checkpoint epoch {epoch})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_signed_M(spins):
    """Per-config signed M = mean(sign(x_i)). Returns (B,) numpy array."""
    return torch.sign(spins.squeeze(1)).float().mean(dim=(-1, -2)).cpu().numpy()


def two_point_axial(x, r_max=None):
    """G(r) axial two-point correlation, averaged over batch + sites + axes.
    x: (B, 1, L, L) — continuous field."""
    y = x.squeeze(1).float().cpu().numpy()
    B, L, _ = y.shape
    if r_max is None:
        r_max = L // 2
    Gs = []
    for r in range(r_max + 1):
        gx = (y * np.roll(y, -r, axis=2)).mean()
        gy = (y * np.roll(y, -r, axis=1)).mean()
        Gs.append(0.5 * (gx + gy))
    return np.array(Gs)


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
    arch = cfg.get("arch", "unet")
    if arch == "meraunet":
        max_mult = int(cfg.get("maxChannelMult", 4))
        v_net = MERAUNet(L=L, nhidden=nhidden, temb_dim=temb_dim,
                          max_channel_mult=max_mult).to(device)
        arch_name = "MERAUNet"
    else:
        v_net = VelocityUNet(L=L, nhidden=nhidden, temb_dim=temb_dim).to(device)
        arch_name = "VelocityUNet"
    v_net.load_state_dict(state["model"])
    v_net.eval()
    print(f"[load] {ckpt_path}  arch={arch_name} L={L} nhidden={nhidden}  epoch={state.get('epoch', '?')}")
    return v_net, L, cfg


# ─────────────────── comparison utilities ───────────────────


def compute_gt_physics(L, T, N=4000):
    """Load HS data (Wolff MCMC samples) and compute the same observables
    as compute_physics — gives GT reference on continuous field."""
    pattern = f"./data/mcmc_data/hs_L{L}_T{T}_N*.pt"
    cand = sorted(glob.glob(pattern), key=lambda s: int(s.split("_N")[-1].split(".")[0]), reverse=True)
    if not cand:
        pattern = f"./data/mcmc_data/hs_L{L}_T{T:.15f}_N*.pt"
        cand = sorted(glob.glob(pattern), key=lambda s: int(s.split("_N")[-1].split(".")[0]), reverse=True)
    if not cand:
        raise FileNotFoundError(f"no HS data at {pattern}")
    x_all = torch.load(cand[0], weights_only=False)
    if isinstance(x_all, list):
        x_all = torch.stack(x_all)
    while x_all.dim() < 4:
        x_all = x_all.unsqueeze(1) if x_all.dim() == 3 else x_all.unsqueeze(0)
    idx = torch.randperm(x_all.shape[0])[:N]
    return x_all[idx].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--epoch", default="latest", help="epoch number or 'latest'")
    p.add_argument("--N", type=int, default=4000, help="samples to draw")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    p.add_argument("--solver", choices=["euler", "rk4"], default="rk4")
    p.add_argument("--T", type=float, default=2.269185314213022)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label", default=None)
    p.add_argument("--gt-compare", action="store_true",
                   help="also compute GT-side physics for direct comparison")
    p.add_argument("--compute-kl", action="store_true",
                   help="Also compute KL(p‖q) via Hutchinson-trace ODE. "
                        "Expensive (~8-15 min at L=32, N=4000, 100 steps).")
    p.add_argument("--kl-samples", type=int, default=1000,
                   help="Number of data samples for KL(p‖q) computation "
                        "(smaller than --N since it's the expensive path)")
    p.add_argument("--kl-eps", type=int, default=1,
                   help="Hutchinson estimator samples per step (higher = "
                        "less variance, linearly more expensive)")
    args = p.parse_args()

    ckpt = find_ckpt(args.folder, args.epoch)
    v_net, L, cfg = load_fm(ckpt, args.device)
    label = args.label or f"fm_L{L}_{os.path.basename(args.folder.rstrip('/'))}"

    # ── sample ──
    sample_fn = sample_rk4 if args.solver == "rk4" else sample_euler
    print(f"[sample] N={args.N}  batch={args.batch}  steps={args.steps}  solver={args.solver}")
    t0 = time.time()
    all_samples = []
    n_left = args.N
    while n_left > 0:
        b = min(args.batch, n_left)
        x = sample_fn(v_net, (b, 1, L, L), n_steps=args.steps, device=args.device)
        all_samples.append(x.detach().cpu())
        n_left -= b
    x_gen = torch.cat(all_samples, dim=0)
    t_sample = time.time() - t0
    print(f"[sample] done in {t_sample:.1f}s  → x shape {tuple(x_gen.shape)}, "
          f"mean={x_gen.mean():.4f}, std={x_gen.std():.4f}")

    # ── physics on flow samples ──
    phys_q = compute_physics(x_gen)
    G_q = two_point_axial(x_gen)
    G0_q = float(G_q[0])
    print(f"\n[flow q] N={phys_q['N_used']}  |M|={phys_q['absM']:.4f}  "
          f"M²={phys_q['M2']:.4f}  χ={phys_q['chi']:.2f}  U₄={phys_q['U4']:.4f}  "
          f"E={phys_q['E']:.4f}  G(0)={G0_q:.3f}")

    # ── (optional) GT-side comparison ──
    if args.gt_compare:
        x_gt = compute_gt_physics(L, args.T, N=args.N)
        phys_p = compute_physics(x_gt)
        G_p = two_point_axial(x_gt)
        G0_p = float(G_p[0])
        print(f"[GT   p] N={phys_p['N_used']}  |M|={phys_p['absM']:.4f}  "
              f"M²={phys_p['M2']:.4f}  χ={phys_p['chi']:.2f}  U₄={phys_p['U4']:.4f}  "
              f"E={phys_p['E']:.4f}  G(0)={G0_p:.3f}")
        # Ratios
        print(f"\n[ratio q/p]  |M|={phys_q['absM']/phys_p['absM']:.3f}  "
              f"χ={phys_q['chi']/max(1e-6,phys_p['chi']):.3f}  "
              f"U₄={phys_q['U4']/phys_p['U4']:.3f}  "
              f"G(0)={G0_q/G0_p:.3f}")

    # ── TIER1_ROW line (parseable, matches analyzers/tier1_observables.py output) ──
    Gs_str = ",".join(f"{v/G0_q:.4f}" for v in G_q)
    print(f"\n[TIER1_ROW] {label}\tN={phys_q['N_used']}\tE={phys_q['E']:+.4f}\t"
          f"absM={phys_q['absM']:.4f}\tM2={phys_q['M2']:.4f}\t"
          f"chi={phys_q['chi']:.2f}\tU4={phys_q['U4']:.4f}\tG={Gs_str}")

    # ── (optional) KL(p‖q) via ODE + Hutchinson trace ──
    if args.compute_kl:
        print(f"\n[KL] computing E_p[log q] via ODE (backward, "
              f"N={args.kl_samples}, steps={args.steps}, n_eps={args.kl_eps})…")
        x_data = compute_gt_physics(L, args.T, N=args.kl_samples).to(args.device)
        t0k = time.time()
        kl_result = compute_kl_pq(v_net, x_data,
                                   n_steps=args.steps, n_eps=args.kl_eps,
                                   batch=200, device=args.device)
        t_kl = time.time() - t0k
        print(f"[KL] E_p[log q] = {kl_result['E_p_log_q']:.3f} "
              f"± {kl_result['sem_E_p_log_q']:.3f}   (took {t_kl:.1f}s)")
        # KL(p‖q) = -H(p) - E_p[log q] .  H(p) here is the HS-continuous
        # differential entropy from MCMC on the HS data. For L=32 T_c that's
        # roughly ~800 nat total (per-config); for a proper KL number we'd
        # subtract H(p_HS) from -E_p[log q] with sign convention:
        #   KL(p‖q) = E_p[log p - log q]  =  -H(p_MC) - E_p[log q]
        # We report -E_p[log q] as the "cross-entropy" and note H(p) needs
        # to be filled in from the diagnostic script's Hp_mc.
        CE = -kl_result["E_p_log_q"]
        print(f"[KL] cross-entropy CE_pq = -E_p[log q] = {CE:.3f}")

        # Compute H(p_HS) directly from data + exact ln Z (analytic term A + ln Z_c)
        H_p_mc, ln_Z_c = compute_hp_mc(x_data, L, args.T, device=args.device)
        if H_p_mc is not None:
            KL_pq = CE - H_p_mc
            print(f"[KL] H(p_HS) [MC + exactz L=32 T=T_c] = {H_p_mc:.3f}")
            print(f"[KL] ln Z_c = {ln_Z_c:.3f}")
            print(f"[KL] KL(p‖q) = CE − H(p_HS) = {KL_pq:.3f}")
        else:
            KL_pq = None
            print(f"[KL] H(p_HS): no exactz entry for L={L} T={args.T}, "
                  f"skipping KL_pq computation.")

        result_kl = dict(
            n_samples_kl=args.kl_samples,
            ode_steps_kl=args.steps,
            n_eps_hutchinson=args.kl_eps,
            E_p_log_q=kl_result["E_p_log_q"],
            sem_E_p_log_q=kl_result["sem_E_p_log_q"],
            CE_pq=CE,
            Hp_mc=H_p_mc,
            ln_Z_c=ln_Z_c,
            KL_pq=KL_pq,
            kl_wall_s=t_kl,
        )
    else:
        result_kl = None

    # ── save numeric result to JSON ──
    result = dict(
        checkpoint=os.path.basename(ckpt),
        epoch=int(re.search(r"epoch(\d+)", ckpt).group(1)),
        L=L, T=args.T,
        n_samples=phys_q["N_used"],
        ode_solver=args.solver, ode_steps=args.steps,
        sample_wall_s=t_sample,
        physics_q=phys_q,
        G_q=[float(v) for v in G_q],
    )
    if args.gt_compare:
        result["physics_p"] = phys_p
        result["G_p"] = [float(v) for v in G_p]
    if result_kl is not None:
        result["kl_diagnostic"] = result_kl

    # ── plots ──
    epoch_num = int(re.search(r"epoch(\d+)", ckpt).group(1))
    cfg_path = os.path.join(args.folder, f"fm_samples_ep{epoch_num}.png")
    corr_path = os.path.join(args.folder, f"fm_correlations_ep{epoch_num}.png")

    save_config_grid(x_gen[:64], cfg_path, nrow=8, ncol=8,
                     title=f"{label}  ep{epoch_num}  (sign(x) render, 64 samples)")
    print(f"[plot] wrote {cfg_path}")

    M_q = compute_signed_M(x_gen)
    if args.gt_compare:
        M_p = compute_signed_M(x_gt)
        save_M_and_Gr(corr_path, M_q, M_p,
                       list(map(float, G_q)), list(map(float, G_p)),
                       args.T, L, epoch_num, label)
    else:
        save_M_and_Gr(corr_path, M_q, None,
                       list(map(float, G_q)), None,
                       args.T, L, epoch_num, label)
    print(f"[plot] wrote {corr_path}")

    out_json = os.path.join(args.folder, f"fm_eval_ep{result['epoch']}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[wrote] {out_json}")


if __name__ == "__main__":
    main()
