"""Bridge-integrity trajectory across saved checkpoints.

For each .saving file in <folder>/savings/, load the flow, sample N
configurations, compute per-sample magnetization M_i = (1/L^2) * sum_j x_ij,
then report:
  <|M|>     -- mean absolute magnetization
  Var(M)    -- variance of M (susceptibility proxy)
  P(|M|<th) -- bridge density (fraction of samples in inter-mode region)

Compare against the HS data baseline so deviations are interpretable.
Output: <folder>/bridge_trajectory.csv + <folder>/bridge_trajectory.png

For Phase-2 finetune folders, the trajectory directly shows whether the
inherited mass-covering shape (Phase 1 hs_bignet endpoint) survives the
reverse-KL refinement or collapses to a single mode.
"""
import argparse, os, sys, glob, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

import source
import train
import flow


def build_flow_from_folder(folder):
    """Reconstruct the flow architecture from parameters.hdf5."""
    with h5py.File(os.path.join(folder, "parameters.hdf5"), "r") as f:
        L = int(f["L"][()])
        d = int(f["d"][()])
        T = float(f["T"][()])
        nlayers = int(f["nlayers"][()])
        nmlp    = int(f["nmlp"][()])
        nhidden = int(f["nhidden"][()])
        nrepeat = int(f["nrepeat"][()])
        depthMERA = int(f["depthMERA"][()])
        wt = bool(f["weightTying"][()]) if "weightTying" in f else False
        hp = bool(f["haarPrior"][()])   if "haarPrior"   in f else False
        ft = f["flowType"][()].decode() if "flowType" in f else "rnvp"
        nk = int(f["nsfBins"][()])      if "nsfBins"   in f else 8
        nb = float(f["nsfBound"][()])   if "nsfBound"  in f else 5.0
    if depthMERA == -1:
        depthMERA = None
    target = source.Ising(L, d, T)
    sym = [lambda x: -x]
    fw = train.symmetryMERAInit(L, d, nlayers, nmlp, nhidden, nrepeat,
                                sym, torch.device("cuda:0"),
                                torch.float32, "SymmMERA",
                                depthMERA=depthMERA,
                                weightTying=wt, haarPrior=hp,
                                flowType=ft, nsfBins=nk, nsfBound=nb)
    return fw, L, T


def read_sigma(folder):
    p = os.path.join(folder, "flow_input_sigma.json")
    if os.path.exists(p):
        return float(json.load(open(p)).get("sigma", 1.0))
    return 1.0


def measure_M(fw, L, sigma, n_samples=2000, batch=256):
    """Sample n_samples; return (global M per sample, per-site samples X[n,L*L])."""
    Ms = []
    Xs = []
    with torch.no_grad():
        nb = (n_samples + batch - 1) // batch
        for _ in range(nb):
            u, _ = fw.sample(batch)
            x = sigma * u if abs(sigma - 1.0) > 1e-6 else u
            x = x.reshape(batch, -1)
            Ms.append(x.mean(dim=1).cpu().numpy())
            Xs.append(x.cpu().numpy())
    M = np.concatenate(Ms)[:n_samples]
    X = np.concatenate(Xs, axis=0)[:n_samples]
    return M, X


def measure_M_data(hs_path, n_samples=2000):
    """Magnetization + per-site array from HS data samples."""
    data = torch.load(hs_path, map_location="cpu", weights_only=False)
    X = data[:n_samples].reshape(n_samples, -1).cpu().numpy()
    return X.mean(axis=1), X


def stats(M, X, threshold=0.5):
    per_site_mean = X.mean(axis=0)            # shape (L*L,)
    # sign_uniformity: 1 if all per-site means share one sign, 0 if perfectly mixed
    sign_unif = float(abs(np.sign(per_site_mean).mean()))
    return {
        "mag_abs":   float(np.mean(np.abs(M))),
        "mag_var":   float(np.var(M)),
        "bridge_p":  float(np.mean(np.abs(M) < threshold)),
        "sign_unif": sign_unif,
    }


def save_per_site(folder, epoch, X, L, X_data=None):
    """Dump per-site mean/var arrays + heatmap PNG for the given checkpoint."""
    mean_q = X.mean(axis=0).reshape(L, L)
    var_q  = X.var(axis=0).reshape(L, L)
    np.save(os.path.join(folder, f"per_site_mean_epoch{epoch}.npy"), mean_q)
    np.save(os.path.join(folder, f"per_site_var_epoch{epoch}.npy"),  var_q)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol = 4 if X_data is not None else 2
        fig, axes = plt.subplots(1, ncol, figsize=(4*ncol, 4))
        vmax = max(abs(mean_q).max(), 1e-9)
        im0 = axes[0].imshow(mean_q, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"q: <x_i>  (epoch {epoch})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(var_q, cmap="viridis")
        axes[1].set_title(f"q: Var(x_i)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        if X_data is not None:
            mean_p = X_data.mean(axis=0).reshape(L, L)
            var_p  = X_data.var(axis=0).reshape(L, L)
            vmax_p = max(abs(mean_p).max(), 1e-9)
            im2 = axes[2].imshow(mean_p, cmap="RdBu_r", vmin=-vmax_p, vmax=vmax_p)
            axes[2].set_title("data: <x_i>")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)
            im3 = axes[3].imshow(var_p, cmap="viridis")
            axes[3].set_title("data: Var(x_i)")
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
        png = os.path.join(folder, f"per_site_epoch{epoch}.png")
        plt.tight_layout(); plt.savefig(png, dpi=120); plt.close(fig)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(skipping heatmap PNG: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("-n", "--n_samples", type=int, default=2000)
    ap.add_argument("--hs", type=str, default=None,
                    help="path to HS data .pt for baseline (default: auto)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--final-epoch", type=int, default=None,
                    help="epoch number whose per-site dump to emit. "
                         "default: most-recently-modified checkpoint "
                         "(robust to phase-2 resume that resets epoch counter).")
    args = ap.parse_args()

    folder = args.folder.rstrip("/") + "/"
    fw, L, T = build_flow_from_folder(folder)
    sigma = read_sigma(folder)
    print(f"folder: {folder}  L={L} T={T:.6f} sigma={sigma:.4f}")

    # HS data baseline
    if args.hs is None:
        # try to auto-find
        cands = sorted(glob.glob(f"data/mcmc_data/hs_L{L}_T*_N*.pt"))
        for c in cands:
            m = re.search(r"_T([\d\.]+)_N", c)
            if m and abs(float(m.group(1)) - T) < 1e-5:
                args.hs = c
                break
    X_data = None
    if args.hs and os.path.exists(args.hs):
        M_data, X_data = measure_M_data(args.hs, args.n_samples)
        s_data = stats(M_data, X_data, args.threshold)
        print(f"data baseline (n={args.n_samples}, threshold={args.threshold}): "
              f"<|M|>={s_data['mag_abs']:.4f}  Var(M)={s_data['mag_var']:.4f}  "
              f"P(|M|<th)={s_data['bridge_p']:.4f}  sign_unif={s_data['sign_unif']:.4f}")
    else:
        s_data = None
        print("no HS data baseline available")

    # Iterate checkpoints (CSV row order = ascending by epoch number)
    saves = sorted(glob.glob(f"{folder}savings/*.saving"),
                   key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    # For "which checkpoint to dump per-site arrays for", default to the most
    # recently *modified* file -- this is robust to phase-2 resumes that reset
    # the epoch counter back to 0 (so a fresh epoch 1500 sorts before an
    # inherited anchor at epoch 9500). --final-epoch N overrides explicitly.
    if args.final_epoch is not None:
        target_ep = args.final_epoch
        target_path = next((p for p in saves
                            if int(re.search(r"epoch(\d+)", p).group(1)) == target_ep),
                           None)
        if target_path is None:
            raise SystemExit(f"--final-epoch {target_ep} not found in {folder}savings/")
    else:
        target_path = max(saves, key=os.path.getmtime) if saves else None
        target_ep = (int(re.search(r"epoch(\d+)", target_path).group(1))
                     if target_path else None)
    if target_path:
        print(f"per-site dump target: epoch {target_ep}  ({os.path.basename(target_path)})")

    print(f"found {len(saves)} checkpoints")
    print()
    print(f"{'epoch':>6}  {'<|M|>_q':>9}  {'Var(M)_q':>10}  {'P(|M|<th)':>10}  {'sign_unif':>9}")
    print("-" * 60)
    rows = []
    target_X = None
    for p in saves:
        ep = int(re.search(r"epoch(\d+)", p).group(1))
        state = torch.load(p, map_location="cuda:0", weights_only=False)
        fw.load(state)
        M, X = measure_M(fw, L, sigma, args.n_samples)
        s = stats(M, X, args.threshold)
        rows.append((ep, s["mag_abs"], s["mag_var"], s["bridge_p"], s["sign_unif"]))
        print(f"{ep:>6}  {s['mag_abs']:>9.4f}  {s['mag_var']:>10.4f}  "
              f"{s['bridge_p']:>10.4f}  {s['sign_unif']:>9.4f}")
        if p == target_path:
            target_X = X

    # CSV
    csv = os.path.join(folder, "bridge_trajectory.csv")
    with open(csv, "w") as f:
        f.write("epoch,mag_abs_q,mag_var_q,bridge_p_q,sign_unif_q")
        if s_data:
            f.write(",mag_abs_p,mag_var_p,bridge_p_p,sign_unif_p")
        f.write("\n")
        for ep, ma, mv, bp, su in rows:
            f.write(f"{ep},{ma},{mv},{bp},{su}")
            if s_data:
                f.write(f",{s_data['mag_abs']},{s_data['mag_var']},"
                        f"{s_data['bridge_p']},{s_data['sign_unif']}")
            f.write("\n")
    print(f"\nwrote {csv}")

    # Per-site dump for the selected checkpoint
    if target_X is not None:
        save_per_site(folder, target_ep, target_X, L, X_data=X_data)


if __name__ == "__main__":
    main()
