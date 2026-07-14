"""Post-hoc SWA: average N checkpoints from a folder, evaluate LOSS on the HS
training data. Meant for probing the drift-around-basin regime documented in
project_l32_late_training_instability: individual checkpoints bounce ±20 nat
around a basin, so a running-mean weight can undercut the single-epoch min.

Usage:
    python analyzers/swa_eval.py --folder DATAFOLDER \
        --data data/mcmc_data/hs_L64_T2.269185314213022_N200000.pt \
        --window BEST [--n 7]

    # or explicit epoch range:
    python analyzers/swa_eval.py --folder DATAFOLDER \
        --data data/mcmc_data/hs_L64_T2.269185314213022_N200000.pt \
        --epochs 200,400,600,800,1000,1200,1400
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train


def epoch_of(path):
    m = re.search(r'epoch(\d+)\.saving$', path)
    return int(m.group(1)) if m else -1


def load_params(folder):
    """Read the config that -load would use."""
    with h5py.File(os.path.join(folder, "parameters.hdf5"), "r") as f:
        p = {
            "epochs": int(np.array(f["epochs"])),
            "batch":  int(np.array(f["batch"])),
            "cuda":   int(np.array(f["cuda"])),
            "double": bool(np.array(f["double"])),
            "lr":     float(np.array(f["lr"])),
            "savePeriod": int(np.array(f["savePeriod"])),
            "nlayers": int(np.array(f["nlayers"])),
            "nmlp":    int(np.array(f["nmlp"])),
            "nhidden": int(np.array(f["nhidden"])),
            "nrepeat": int(np.array(f["nrepeat"])),
            "depthMERA": int(np.array(f["depthMERA"])),
            "L": int(np.array(f["L"])),
            "d": int(np.array(f["d"])),
            "T": float(np.array(f["T"])),
        }
        for k in ("weightTying", "haarPrior", "symmetry"):
            p[k] = bool(np.array(f[k])) if k in f else False
        p["flowType"] = (str(np.array(f["flowType"]).item().decode())
                         if "flowType" in f else "rnvp")
        p["nsfBins"] = int(np.array(f["nsfBins"])) if "nsfBins" in f else 8
        p["nsfBound"] = float(np.array(f["nsfBound"])) if "nsfBound" in f else 5.0
        p["priorType"] = (str(np.array(f["priorType"]).item().decode())
                          if "priorType" in f else "gaussian")
        p["hcgScaleShared"] = bool(np.array(f["hcgScaleShared"])) if "hcgScaleShared" in f else True
        p["hcgHidden"] = int(np.array(f["hcgHidden"])) if "hcgHidden" in f else 32
        p["hcgDilated"] = bool(np.array(f["hcgDilated"])) if "hcgDilated" in f else True
        p["hcgCircular"] = bool(np.array(f["hcgCircular"])) if "hcgCircular" in f else True
        if "hcgSharedDilations" in f:
            raw = str(np.array(f["hcgSharedDilations"]).item().decode())
            p["hcgSharedDilations"] = raw if raw else None
        else:
            p["hcgSharedDilations"] = None
    return p


def average_state_dicts(paths):
    """Weight-average state_dicts across a list of checkpoints."""
    assert paths, "no checkpoints"
    print(f"[swa] loading {len(paths)} checkpoints for averaging")
    sums = None
    n_seen = 0
    for path in paths:
        raw = torch.load(path, map_location="cpu")
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        if sums is None:
            sums = OrderedDict((k, v.clone().to(torch.float64)) for k, v in sd.items())
        else:
            for k, v in sd.items():
                sums[k].add_(v.to(torch.float64))
        n_seen += 1
    avg = OrderedDict((k, (v / n_seen).to(torch.float32)) for k, v in sums.items())
    print(f"[swa] averaged {n_seen} checkpoints, {len(avg)} tensors")
    return avg


def eval_loss(flow, source, hs_pt, sigma, batch, device, dtype, n_batches=50, log_jac_std=None):
    """Compute mean −log q(x) on HS data. Matches train/learn.py's loss formula."""
    data = torch.load(hs_pt, map_location="cpu")
    if isinstance(data, torch.Tensor):
        X = data
    elif isinstance(data, dict):
        X = data.get("samples", data.get("data"))
    else:
        X = data
    X = X.to(dtype)
    print(f"[eval] dataset shape={tuple(X.shape)}, sigma={sigma:.4f}")
    if log_jac_std is None:
        # Matches learn.py: log_jac_std = N_spins * log(sigma)
        L = X.shape[-1]
        n_spins = L * L
        log_jac_std = n_spins * math.log(sigma)
        print(f"[eval] log_jac_std = {n_spins} × log({sigma:.4f}) = {log_jac_std:.2f}")

    losses = []
    n = min(n_batches, X.shape[0] // batch)
    flow.eval()
    with torch.no_grad():
        for i in range(n):
            x_real = X[i*batch:(i+1)*batch].to(device)
            # Match "-noDeq" path (no dequantization noise)
            x_std = x_real / sigma
            log_prob = flow.logProbability(x_std) - log_jac_std
            loss = -log_prob.mean().item()
            losses.append(loss)
    print(f"[eval] ran {n} batches of {batch}")
    losses = np.array(losses)
    return float(losses.mean()), float(losses.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--data", required=True, help="HS .pt path")
    ap.add_argument("--window", choices=["best", "tail"], default="best")
    ap.add_argument("--n", type=int, default=7, help="how many checkpoints to average")
    ap.add_argument("--epochs", default=None, help="explicit CSV of epochs to average")
    ap.add_argument("--n-batches", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    folder = args.folder.rstrip('/')
    p = load_params(folder)
    print(f"[swa] folder: {folder}")
    print(f"[swa] L={p['L']} nlayers={p['nlayers']} nhidden={p['nhidden']} "
          f"nrepeat={p['nrepeat']} priorType={p['priorType']} "
          f"hcgScaleShared={p['hcgScaleShared']} hcgDilated={p['hcgDilated']}")

    # Pick checkpoint set
    savings = sorted(glob.glob(os.path.join(folder, "savings", "*.saving")),
                     key=epoch_of)
    if args.epochs:
        want = set(int(x) for x in args.epochs.split(","))
        pick = [p_ for p_ in savings if epoch_of(p_) in want]
    elif args.window == "tail":
        pick = savings[-args.n:]
    else:  # best-window
        # Find best epoch from log
        log_glob = glob.glob("logs/*.out")
        best_ep, best_loss = None, None
        for lg in log_glob:
            try:
                with open(lg, "r", errors="ignore") as f:
                    head = f.read(2048)
            except Exception:
                continue
            if folder in head or folder.lstrip("./") in head:
                with open(lg, "r", errors="ignore") as f:
                    for line in f:
                        if not line.startswith("epoch:"):
                            continue
                        toks = line.split()
                        try:
                            ep = int(toks[1])
                            for i, t in enumerate(toks):
                                if t == "L:":
                                    loss = float(toks[i+1]); break
                            else:
                                continue
                        except (ValueError, IndexError):
                            continue
                        if best_loss is None or loss < best_loss:
                            best_loss = loss; best_ep = ep
                break
        if best_ep is None:
            print("[swa] no log found — falling back to tail window")
            pick = savings[-args.n:]
        else:
            print(f"[swa] log best: ep {best_ep}, L={best_loss:.2f}")
            # Nearest saved to best, ± n//2 either side
            eps_saved = [epoch_of(p_) for p_ in savings]
            i_center = min(range(len(eps_saved)), key=lambda i: abs(eps_saved[i]-best_ep))
            half = args.n // 2
            lo = max(0, i_center - half)
            hi = min(len(savings), i_center + args.n - (i_center - lo))
            pick = savings[lo:hi]

    print(f"[swa] averaging over epochs: {[epoch_of(p_) for p_ in pick]}")

    # Reconstruct model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if p["double"] else torch.float32
    fw = train.symmetryMERAInit(
        p["L"], p["d"], p["nlayers"], p["nmlp"], p["nhidden"], p["nrepeat"],
        p.get("symmetry", True), device, dtype, "swa_eval",
        depthMERA=p["depthMERA"], weightTying=p["weightTying"],
        haarPrior=p["haarPrior"], flowType=p["flowType"],
        nsfBins=p["nsfBins"], nsfBound=p["nsfBound"],
        priorType=p["priorType"], hcgScaleShared=p["hcgScaleShared"],
        hcgHidden=p["hcgHidden"], hcgDilated=p["hcgDilated"],
        hcgCircular=p["hcgCircular"], hcgSharedDilations=p["hcgSharedDilations"],
    )

    # Read the (target) source used for training — Ising Gaussian approximation
    import source
    # main.py calls source.Ising(L, d, T) — arg order matters
    src = source.Ising(p["L"], p["d"], p["T"])
    src.to(device=device, dtype=dtype)

    # Load sigma
    with open(os.path.join(folder, "flow_input_sigma.json")) as f:
        sigma = json.load(f)["sigma"]

    # Baseline: eval on the last-tail checkpoint alone, for comparison
    print("\n== Eval single checkpoint (last one in pick) ==")
    raw = torch.load(pick[-1], map_location=device)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    fw.load_state_dict(sd, strict=False)
    mean_last, std_last = eval_loss(fw, src, args.data, sigma, p["batch"],
                                    device, dtype, n_batches=args.n_batches)
    print(f"  L(ep {epoch_of(pick[-1])}) = {mean_last:.2f} ± {std_last:.2f}")

    print("\n== SWA over the window ==")
    swa_sd = average_state_dicts(pick)
    fw.load_state_dict(swa_sd, strict=False)
    mean_swa, std_swa = eval_loss(fw, src, args.data, sigma, p["batch"],
                                  device, dtype, n_batches=args.n_batches)
    print(f"  L(SWA over {[epoch_of(p_) for p_ in pick]}) = {mean_swa:.2f} ± {std_swa:.2f}")

    print()
    print(f"SUMMARY: single-ckpt {mean_last:.2f}  →  SWA {mean_swa:.2f}  "
          f"(Δ = {mean_swa - mean_last:+.2f} nat)")


if __name__ == "__main__":
    main()
