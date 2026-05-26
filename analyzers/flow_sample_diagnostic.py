"""Post-hoc flow diagnostic for a trained NeuralRG run.

Loads the latest checkpoint under <folder>/savings/ and reports:

  Model-side (samples x ~ q from the flow):
    <A>_q  = E_q[A(x)]                action expectation under the model
    H(q)   = -E_q[log q(x)]           model entropy
    F_c^q  = <A>_q - H(q)             variational free energy  (>= -lnZ_c)
    KL(q||p) = F_c^q + lnZ_c          reverse KL (mode-seeking)

  Target-side (samples x ~ p_HS from data/mcmc_data/hs_L*.pt):
    CE     = -E_p[log q(x)]           forward cross-entropy  (= MLE training loss)
    H(p)   = E_p[A(x)] + lnZ_c        MC estimate of the HS differential entropy
    KL(p||q) = CE - H(p)              forward KL (mass-covering); catches mode-dropping

Standardization handling
------------------------
Data-driven runs may train the flow on a rescaled input u = x / sigma (see
train/learn.py).  In that case flow.sample() returns u and flow.logProbability
scores u, so physical-space quantities need the change-of-variables shift:

    x = sigma * u
    log q_X(x) = log q_U(x / sigma) - N * log(sigma)
    H(q_X)     = H(q_U) + N * log(sigma)

sigma is read from <folder>/flow_input_sigma.json (written by learn.py;
sigma=1.0 for reverse-KL).  If absent, sigma=1.0 is assumed with a warning.

Results are written to <folder>/flow_diagnostic.json for loss_analyzer_fixT.py.

Two PNGs are also written (unless --no-png):

  flow_samples.png      side-by-side spin grids: the *actual* trained-flow
                        output (x ~ q) next to the HS target data (x ~ p),
                        same sigmoid(2x) rendering as learn.py proposals_*.png.
                        (proposals_*.png in a data-driven run shows the input
                        data, NOT the model -- this is the one that shows what
                        the flow learned.)
  flow_correlations.png two structural panels the scalar <A>/H/KL numbers
                        cannot resolve:
                          - per-config magnetisation M = (1/N) sum x_i
                          - axial two-point correlation G(r)/G(0)
                        G(r) reveals whether the flow reproduces critical
                        large-scale domains (slow decay, high plateau) or only
                        local noise (fast decay).  Summary scalars <|M|>,
                        G(L/2)/G(0) and xi_eff = sum_r G(r)/G(0) go to the JSON.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import source
import train

EXACT_FILE = "etc/exactz.md"
MCMC_DIR = "data/mcmc_data"


def detect_symmetry(state_dict):
    return any(k.startswith("flow.") for k in state_dict.keys())


def read_sigma(folder):
    """Flow-input scale: flow.sample() returns u = x / sigma."""
    path = os.path.join(folder, "flow_input_sigma.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["sigma"]), "flow_input_sigma.json"
    return 1.0, "DEFAULT (no marker - assuming physical scale)"


def parse_exactz(exact_path, L, T, tol=1e-3):
    """Return (lnZ_d, fix) for given L, T from etc/exactz.md, else (None, None)."""
    if not os.path.exists(exact_path):
        return None, None
    with open(exact_path) as f:
        content = f.read()
    # split on section headers "### Exact Z for Ising n = <L>"
    parts = re.split(r"### Exact Z for Ising n\s*=\s*(\d+)", content)
    for i in range(1, len(parts), 2):
        if int(parts[i]) != L:
            continue
        for line in parts[i + 1].splitlines():
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) < 3:
                continue
            try:
                t = float(cells[0])
                lnz = float(cells[1].strip("$ "))
                fix = float(cells[2])
            except ValueError:
                continue
            if abs(t - T) < tol:
                return lnz, fix
    return None, None


def find_hs_file(L, T, tol=1e-3):
    for fp in glob.glob(os.path.join(MCMC_DIR, f"hs_L{L}_T*_N*.pt")):
        m = re.search(r"_T([\d.]+)_N", fp)
        if m and abs(float(m.group(1)) - T) < tol:
            return fp
    return None


def build_flow(folder, state):
    """Reconstruct and load the MERA flow from a run folder + checkpoint dict."""
    with h5py.File(os.path.join(folder, "parameters.hdf5"), "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        T = float(np.array(f["T"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        depthMERA = int(np.array(f["depthMERA"]))
        weightTying = bool(np.array(f["weightTying"])) if "weightTying" in f else False
        haarPrior = bool(np.array(f["haarPrior"])) if "haarPrior" in f else False
    if depthMERA == -1:
        depthMERA = None

    device = torch.device("cpu")
    dtype = torch.float32
    sym_used = detect_symmetry(state)
    name = f"SymmMERA_l{nlayers}_M{nmlp}H{nhidden}_R{nrepeat}_Ising"
    sym = [lambda x: -x] if sym_used else None
    fw = train.symmetryMERAInit(
        L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, name,
        depthMERA=depthMERA, weightTying=weightTying, haarPrior=haarPrior,
    )
    fw.load(state)
    fw.eval()
    target = source.Ising(L, d, T).to(device=device, dtype=dtype)
    return fw, target, L, T, sym_used, weightTying, haarPrior


def _batch_corr(x):
    """x: (B,1,L,L) tensor -> (per-config magnetisation array, summed LxL autocorr).

    Autocorr via FFT (Wiener-Khinchin), periodic BCs:
        C[d] = (1/N) Sum_i x_i x_{i+d}
    so C[0,0] = <x^2> and C[d] -> <M^2> at large |d|.
    """
    arr = x[:, 0].detach().cpu().numpy().astype(np.float64)   # (B,L,L)
    B, L, _ = arr.shape
    M = arr.reshape(B, -1).mean(axis=1)                       # per-config magnetisation
    f = np.fft.fft2(arr, axes=(1, 2))
    cmap = np.fft.ifft2(np.abs(f) ** 2, axes=(1, 2)).real / (L * L)
    return M, cmap.sum(axis=0)


def _radial_G(cmean):
    """Axial two-point correlation G(r), r = 0..L//2, from an LxL autocorr map."""
    L = cmean.shape[0]
    return np.array([0.5 * (cmean[r, 0] + cmean[0, r]) for r in range(L // 2 + 1)])


def _render_grid(x_phys, nrow=10):
    """sigmoid(2x) greyscale grid -- same rendering as learn.py proposals_*.png."""
    from torchvision.utils import make_grid
    img = torch.sigmoid(2.0 * x_phys)                 # (B,1,L,L) -> [0,1]
    grid = make_grid(img, nrow=nrow, padding=1, pad_value=0.5)
    return grid[0].numpy()                            # single channel -> 2D


def save_flow_png(folder, x_q_vis, x_p_vis, epoch, label):
    """Side-by-side PNG: flow output (x~q) vs HS target data (x~p).

    This is the *actual* trained-flow output -- unlike proposals_*.png, which
    in a data-driven run shows the input training data, not the model.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable - skipping PNG: {e})")
        return None
    panels = [("flow output   x ~ q", x_q_vis)]
    if x_p_vis is not None:
        panels.append(("HS target data   x ~ p", x_p_vis))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 6.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, xv) in zip(axes, panels):
        ax.imshow(_render_grid(xv), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{label}    (checkpoint epoch {epoch})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(folder.rstrip("/"), "flow_samples.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def save_corr_png(folder, M_q, M_p, G_q, G_p, epoch, label):
    """Two panels: per-config magnetisation histogram, and normalised G(r)/G(0).

    These resolve structure the scalar <A>/H/KL numbers cannot: the
    magnetisation spread (global order) and the correlation length (whether
    the flow reproduces critical large-scale domains, not just local noise).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable - skipping correlation PNG: {e})")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # panel 1: per-config magnetisation  M = (1/N) Sum x_i
    mmax = np.abs(M_q).max()
    if M_p is not None:
        mmax = max(mmax, float(np.abs(M_p).max()))
    bins = np.linspace(-mmax * 1.05, mmax * 1.05, 51)
    ax1.hist(M_q, bins=bins, density=True, alpha=0.55, color="C0",
             label="flow  x ~ q")
    if M_p is not None:
        ax1.hist(M_p, bins=bins, density=True, alpha=0.55, color="C1",
                 label="HS data  x ~ p")
    ax1.set_xlabel("magnetisation  M = (1/N) sum x_i")
    ax1.set_ylabel("density")
    ax1.set_title("per-config magnetisation")
    ax1.legend(fontsize=9)

    # panel 2: normalised axial two-point correlation  G(r)/G(0)
    ax2.plot(np.arange(len(G_q)), G_q / G_q[0], "o-", color="C0",
             label="flow  x ~ q")
    if G_p is not None:
        ax2.plot(np.arange(len(G_p)), G_p / G_p[0], "s-", color="C1",
                 label="HS data  x ~ p")
    ax2.set_xlabel("lattice distance  r")
    ax2.set_ylabel("G(r) / G(0)")
    ax2.set_title("axial two-point correlation")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(f"{label}    (checkpoint epoch {epoch})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(folder.rstrip("/"), "flow_correlations.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def run_one(folder, n_samples, batch_size, seed, make_png=True):
    folder = folder.rstrip("/") + "/"

    saving_files = sorted(
        glob.glob(folder + "savings/*.saving"),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not saving_files:
        raise FileNotFoundError(f"no .saving files in {folder}savings/")
    ckpt_path = saving_files[-1]
    epoch = int(re.search(r"epoch(\d+)", ckpt_path).group(1))

    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    fw, target, L, T, sym_used, wt, hp = build_flow(folder, state)
    N = L * L

    sigma, sigma_src = read_sigma(folder)
    n_log_sigma = N * math.log(sigma)  # change-of-variables shift, 0 when sigma=1

    out = {
        "folder": folder, "checkpoint": os.path.basename(ckpt_path), "epoch": epoch,
        "L": L, "T": T, "symmetry": sym_used, "weightTying": wt, "haarPrior": hp,
        "sigma": sigma, "sigma_source": sigma_src,
    }

    # ---- exact lnZ_c -------------------------------------------------------
    lnZ_d, fix = parse_exactz(EXACT_FILE, L, T)
    lnZ_c = (lnZ_d + fix) if lnZ_d is not None else None
    out["lnZ_c"] = lnZ_c

    # ---- model-side: sample x ~ q -----------------------------------------
    torch.manual_seed(seed)
    A_chunks, logq_chunks = [], []
    n_vis = 100                                 # first 100 samples -> 10x10 PNG grid
    x_q_vis = []
    M_q, csum_q, ncfg_q = [], np.zeros((L, L)), 0
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            b = min(batch_size, n_samples - n_done)
            u, logq_u = fw.sample(b)            # u standardized; logq_u = log q_U(u)
            x = sigma * u                       # physical field
            A_chunks.append(target.energy(x).cpu().numpy().astype(np.float64))
            logq_chunks.append(logq_u.cpu().numpy().astype(np.float64))
            m_b, c_b = _batch_corr(x)           # magnetisation + autocorr
            M_q.append(m_b); csum_q += c_b; ncfg_q += x.shape[0]
            if sum(c.shape[0] for c in x_q_vis) < n_vis:
                x_q_vis.append(x[:n_vis].detach().cpu())
            n_done += b
    x_q_vis = torch.cat(x_q_vis)[:n_vis] if x_q_vis else None
    M_q = np.concatenate(M_q)
    G_q = _radial_G(csum_q / ncfg_q)
    A_q = np.concatenate(A_chunks)
    logqU_q = np.concatenate(logq_chunks)

    EA = float(A_q.mean())
    sem_EA = float(A_q.std(ddof=1) / np.sqrt(len(A_q)))
    Hq = float(-logqU_q.mean() + n_log_sigma)            # physical-space H(q_X)
    sem_Hq = float(logqU_q.std(ddof=1) / np.sqrt(len(logqU_q)))
    Fcq = EA - Hq
    sem_Fcq = float(np.sqrt(sem_EA**2 + sem_Hq**2))
    out.update({
        "n_samples_q": int(len(A_q)),
        "EA": EA, "sem_EA": sem_EA,
        "Hq": Hq, "sem_Hq": sem_Hq,
        "Fcq": Fcq, "sem_Fcq": sem_Fcq,
        "KL_qp": (Fcq + lnZ_c) if lnZ_c is not None else None,   # >= 0
    })

    # ---- target-side: samples x ~ p_HS  ->  KL(p||q) ----------------------
    x_p_vis, M_p, G_p = None, None, None
    hs_file = find_hs_file(L, T)
    if hs_file is not None and lnZ_c is not None:
        x_p = torch.load(hs_file, weights_only=True).reshape(-1, 1, L, L)
        x_p = x_p[:n_samples].to(dtype=torch.float32)
        x_p_vis = x_p[:n_vis].clone()
        A_chunks, logq_chunks = [], []
        M_p, csum_p, ncfg_p = [], np.zeros((L, L)), 0
        with torch.no_grad():
            for s in range(0, len(x_p), batch_size):
                xb = x_p[s:s + batch_size]
                A_chunks.append(target.energy(xb).cpu().numpy().astype(np.float64))
                # log q_X(x) = log q_U(x / sigma) - N log sigma
                lq = fw.logProbability(xb / sigma).cpu().numpy().astype(np.float64)
                logq_chunks.append(lq - n_log_sigma)
                m_b, c_b = _batch_corr(xb)
                M_p.append(m_b); csum_p += c_b; ncfg_p += xb.shape[0]
        A_p = np.concatenate(A_chunks)
        logqX_p = np.concatenate(logq_chunks)
        M_p = np.concatenate(M_p)
        G_p = _radial_G(csum_p / ncfg_p)

        CE = float(-logqX_p.mean())                 # forward cross-entropy = MLE loss
        sem_CE = float(logqX_p.std(ddof=1) / np.sqrt(len(logqX_p)))
        Hp_mc = float(A_p.mean() + lnZ_c)           # MC estimate of H(p_HS)
        KL_pq = CE - Hp_mc                          # = E_p[log p - log q] >= 0
        out.update({
            "n_samples_p": int(len(A_p)),
            "CE_pq": CE, "sem_CE_pq": sem_CE,
            "Hp_mc": Hp_mc,
            "KL_pq": KL_pq,
        })
    else:
        out.update({"n_samples_p": 0, "CE_pq": None, "Hp_mc": None, "KL_pq": None})
        if hs_file is None:
            print(f"  (no HS sample file for L={L}, T={T}: skipping KL(p||q))")
        if lnZ_c is None:
            print(f"  (no exactz.md entry for L={L}, T={T}: skipping KL terms)")

    # ---- structure: magnetisation + two-point correlation -----------------
    out["mag_abs_q"] = float(np.abs(M_q).mean())          # <|M|>, global order
    out["G0_q"] = float(G_q[0])                           # <x^2>
    out["Gnorm_q"] = (G_q / G_q[0]).tolist()              # G(r)/G(0), r=0..L//2
    out["g_longrange_q"] = float(G_q[-1] / G_q[0])        # plateau at r=L//2
    out["xi_q"] = float((G_q[1:] / G_q[0]).sum())         # eff. correlation length
    if M_p is not None:
        out["mag_abs_p"] = float(np.abs(M_p).mean())
        out["G0_p"] = float(G_p[0])
        out["Gnorm_p"] = (G_p / G_p[0]).tolist()
        out["g_longrange_p"] = float(G_p[-1] / G_p[0])
        out["xi_p"] = float((G_p[1:] / G_p[0]).sum())
    else:
        out["mag_abs_p"] = out["G0_p"] = out["Gnorm_p"] = None
        out["g_longrange_p"] = out["xi_p"] = None

    # ---- visualization ----------------------------------------------------
    if make_png and x_q_vis is not None:
        label = os.path.basename(folder.rstrip("/"))
        png = save_flow_png(folder, x_q_vis, x_p_vis, epoch, label)
        if png:
            out["flow_png"] = png
        cpng = save_corr_png(folder, M_q, M_p, G_q, G_p, epoch, label)
        if cpng:
            out["corr_png"] = cpng

    return out


def _fmt(v, nd=4):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", help="one or more run folders")
    p.add_argument("-n", "--n_samples", type=int, default=8000)
    p.add_argument("-b", "--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-png", action="store_true",
                   help="skip writing flow_samples.png")
    p.add_argument("--no-json", action="store_true",
                   help="do not overwrite flow_diagnostic.json "
                        "(use with small -n when only refreshing the PNG)")
    args = p.parse_args()

    for folder in args.folders:
        print(f"\n=== {folder} ===", flush=True)
        try:
            out = run_one(folder, args.n_samples, args.batch_size, args.seed,
                          make_png=not args.no_png)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        if not args.no_json:
            out_path = os.path.join(folder.rstrip("/"), "flow_diagnostic.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        print(f"  ckpt epoch {out['epoch']}, sym={out['symmetry']}, "
              f"sigma={out['sigma']:.4f} ({out['sigma_source']})")
        print(f"  model side (x~q, N={out['n_samples_q']}):")
        print(f"    <A>_q    = {_fmt(out['EA'])} +/- {out['sem_EA']:.3f}")
        print(f"    H(q)     = {_fmt(out['Hq'])} +/- {out['sem_Hq']:.3f}")
        print(f"    F_c^q    = {_fmt(out['Fcq'])} +/- {out['sem_Fcq']:.3f}")
        print(f"    KL(q||p) = {_fmt(out['KL_qp'])}")
        if out.get("KL_pq") is not None:
            print(f"  target side (x~p_HS, N={out['n_samples_p']}):")
            print(f"    CE = -E_p[log q] = {_fmt(out['CE_pq'])} +/- {out['sem_CE_pq']:.3f}")
            print(f"    H(p_HS) [MC]     = {_fmt(out['Hp_mc'])}")
            print(f"    KL(p||q)         = {_fmt(out['KL_pq'])}")
        print(f"  structure (<|M|>, G(L/2)/G(0), xi_eff):")
        print(f"    x~q : {out['mag_abs_q']:.4f}   {out['g_longrange_q']:.4f}   "
              f"{out['xi_q']:.3f}")
        if out.get("mag_abs_p") is not None:
            print(f"    x~p : {out['mag_abs_p']:.4f}   {out['g_longrange_p']:.4f}   "
                  f"{out['xi_p']:.3f}")
        if not args.no_json:
            print(f"  wrote {out_path}")
        if out.get("flow_png"):
            print(f"  wrote {out['flow_png']}")
        if out.get("corr_png"):
            print(f"  wrote {out['corr_png']}")


if __name__ == "__main__":
    main()
