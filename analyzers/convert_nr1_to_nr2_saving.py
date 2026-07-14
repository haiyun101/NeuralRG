"""Convert a nrepeat=1 checkpoint into a nrepeat=2 checkpoint by
identity-initializing the extra rep-1 RNVP blocks.

Layer ordering (from flow/hierarchy/mera.py):
  layerList[s * 2*nrepeat + 2*r + o]  =  (scale=s, repeat=r, offset=o)

nr=1 layerList (length 2*S):
  [ s0-r0-o0, s0-r0-o1, s1-r0-o0, s1-r0-o1, ..., s{S-1}-r0-o0, s{S-1}-r0-o1 ]

nr=2 layerList (length 4*S):
  [ s0-r0-o0, s0-r0-o1, s0-r1-o0, s0-r1-o1,
    s1-r0-o0, s1-r0-o1, s1-r1-o0, s1-r1-o1, ...]

Mapping nr=1 index X (with s=X//2, o=X%2) → nr=2:
  rep-0 slot: X_new = 4*s + o        (copy nr=1 weights here)
  rep-1 slot: X_new = 4*s + 2 + o    (copy nr=1 weights BUT identity-init)

Identity-init for RNVP block (see flow/rnvp.py):
  forward:  y = maskList*y + maskListR*(y*exp(s) + t)
  If s(y) = 0 and t(y) = 0 for all y  →  y = y (block is identity).

For this repo's SimpleMLPreshape MLPs (nmlp=3, coreSize=4):
  tList[n].layerList: [Linear, ELU, Linear, ELU, Linear, ELU, Linear]
                       0       1    2       3    4       5    6
  sList[n].layerList: [Linear, ELU, Linear, ELU, Linear, ELU, Linear, ScalableTanh]
                       0       1    2       3    4       5    6       7

To make t = 0: zero final Linear (layerList.6) weight + bias.
To make s = 0: zero ScalableTanh (layerList.7) scale
                (final Linear.6 output doesn't matter because ScalableTanh
                 multiplies by scale=0).
For safety, we also zero sList.<n>.layerList.6.weight and .bias.

Usage:
  python analyzers/convert_nr1_to_nr2_saving.py \\
      --src data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16 \\
      --epoch 15000 \\
      --dst-folder data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_fromnr1_nr2_b16

The destination folder gets:
  - parameters.hdf5 (nrepeat=2 flipped in; rest copied from src)
  - savings/<name>Saving_epoch0.saving   (state_dict only, no optimizer)

Then launch training via `main.py -load -folder <dst-folder>` — the
epoch counter picks up at ep 1 and everything else (dataDriven, priorType,
hcgHidden, vpWeight, etc.) is restored from parameters.hdf5.
"""
import argparse
import glob
import os
import re
import shutil
import sys

import h5py
import numpy as np
import torch


def find_saving(src_folder, epoch=None):
    files = sorted(
        glob.glob(os.path.join(src_folder, "savings", "*.saving")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not files:
        raise SystemExit(f"no *.saving under {src_folder}/savings/")
    if epoch is None:
        return files[-1]
    eps = [int(re.search(r"epoch(\d+)", p).group(1)) for p in files]
    idx = min(range(len(eps)), key=lambda i: abs(eps[i] - epoch))
    print(f"[convert] requested epoch {epoch}, using nearest: {eps[idx]}")
    return files[idx]


def rename_saving_file(dst_folder, src_name):
    """Rename SymmMERA_l16_M3H128_R1_Ising… → …_R2_… for the dst folder."""
    new_name = re.sub(r"(_R)1(_)", r"\g<1>2\g<2>", src_name)
    return new_name


def convert_state(nr1_state, S, nlayers):
    """Return a new state dict compatible with a fresh nr=2 model.

    nr1_state maps 2*S RNVP blocks; output maps 4*S blocks.
    Rep-1 slots are identity-initialized (s=t=0 in RNVP formula).
    """
    new_state = {}

    # Regex to match layerList indices: works whether the top prefix is
    # `flow.` (Symmetrized-wrapped) or bare.
    layer_re = re.compile(r"^(?P<prefix>.*?)layerList\.(?P<idx>\d+)\.(?P<sub>.+)$")

    # Which layerList indices are we doubling?  A saving might have keys
    # like `flow.layerList.0.tList.0…` AND `flow.prior.*` — the prior keys
    # don't match layer_re so they pass through untouched.
    n_copied = 0
    n_doubled = 0
    for k, v in nr1_state.items():
        m = layer_re.match(k)
        if m is None:
            new_state[k] = v.clone()
            n_copied += 1
            continue

        prefix = m.group("prefix")
        idx = int(m.group("idx"))
        sub = m.group("sub")
        # Only handle the flow layerList — not e.g. RNVP-internal tList/sList
        # (those are nested and don't refer to MERA scales).
        # Guard: top-level flow layerList indices are < 2*S; anything larger
        # would be a nested layerList index inside RNVP.
        # We disambiguate by requiring the prefix to end in "flow." or "".
        if not (prefix.endswith("flow.") or prefix == ""):
            new_state[k] = v.clone()
            n_copied += 1
            continue
        if idx >= 2 * S:
            new_state[k] = v.clone()
            n_copied += 1
            continue

        s_idx = idx // 2
        off = idx % 2

        rep0_new = 4 * s_idx + off
        rep1_new = 4 * s_idx + 2 + off

        new_state[f"{prefix}layerList.{rep0_new}.{sub}"] = v.clone()
        new_state[f"{prefix}layerList.{rep1_new}.{sub}"] = v.clone()
        n_doubled += 1

    print(f"[convert] passed through: {n_copied} tensors")
    print(f"[convert] doubled: {n_doubled} tensors (rep-0 + rep-1 slots)")

    # Now zero rep-1 slots' identity-init tensors.
    # We know keys look like:
    #   flow.layerList.<rep1_new>.tList.<n>.layerList.6.weight
    #   flow.layerList.<rep1_new>.tList.<n>.layerList.6.bias
    #   flow.layerList.<rep1_new>.sList.<n>.layerList.6.weight
    #   flow.layerList.<rep1_new>.sList.<n>.layerList.6.bias
    #   flow.layerList.<rep1_new>.sList.<n>.layerList.7.scale
    prefixes = set()
    for k in new_state.keys():
        m = layer_re.match(k)
        if m:
            prefixes.add(m.group("prefix"))
    if not prefixes:
        raise RuntimeError("no matching layerList prefix found — key layout unexpected")
    if len(prefixes) > 1:
        # Take the shortest — the top-level MERA layerList prefix. Nested
        # ones are longer.
        prefixes = {min(prefixes, key=len)}
    prefix = next(iter(prefixes))
    print(f"[convert] using prefix: {prefix!r}")

    n_zeroed = 0
    for s_idx in range(S):
        for off in range(2):
            rep1_new = 4 * s_idx + 2 + off
            for n in range(nlayers):
                for which in ("tList", "sList"):
                    for what in ("weight", "bias"):
                        k = f"{prefix}layerList.{rep1_new}.{which}.{n}.layerList.6.{what}"
                        if k in new_state:
                            new_state[k] = torch.zeros_like(new_state[k])
                            n_zeroed += 1
                # ScalableTanh scale in sList only
                k_st = f"{prefix}layerList.{rep1_new}.sList.{n}.layerList.7.scale"
                if k_st in new_state:
                    new_state[k_st] = torch.zeros_like(new_state[k_st])
                    n_zeroed += 1
    print(f"[convert] zeroed {n_zeroed} tensors in rep-1 slots (identity-init)")
    return new_state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source nr=1 folder")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch to convert (default: latest)")
    ap.add_argument("--dst-folder", required=True,
                    help="destination nr=2 folder to create")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing dst folder")
    args = ap.parse_args()

    src_folder = args.src.rstrip("/")
    dst_folder = args.dst_folder.rstrip("/")

    if os.path.exists(dst_folder) and not args.overwrite:
        raise SystemExit(f"dst folder exists (pass --overwrite): {dst_folder}")
    os.makedirs(os.path.join(dst_folder, "savings"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "records"), exist_ok=True)

    # (1) Read src parameters.hdf5, patch nrepeat=2, write to dst.
    src_hdf5 = os.path.join(src_folder, "parameters.hdf5")
    dst_hdf5 = os.path.join(dst_folder, "parameters.hdf5")
    if not os.path.exists(src_hdf5):
        raise SystemExit(f"missing src parameters.hdf5: {src_hdf5}")
    shutil.copyfile(src_hdf5, dst_hdf5)

    with h5py.File(dst_hdf5, "r+") as f:
        old_nr = int(np.array(f["nrepeat"])) if "nrepeat" in f else -1
        if "nrepeat" in f:
            del f["nrepeat"]
        f["nrepeat"] = 2
        # Force -load-safe: overwrite folder to match dst (main.py uses -folder
        # from CLI anyway, but keep hdf5 consistent).
        L = int(np.array(f["L"]))
        nlayers = int(np.array(f["nlayers"]))
    S = int(np.log2(L))
    print(f"[convert] L={L}, S={S}, nlayers={nlayers}")
    print(f"[convert] parameters.hdf5: nrepeat {old_nr} → 2")

    # (2) Load src saving, transform state dict, save to dst savings/.
    src_saving = find_saving(src_folder, args.epoch)
    src_epoch = int(re.search(r"epoch(\d+)", src_saving).group(1))
    print(f"[convert] source saving: {src_saving} (epoch {src_epoch})")

    src_state = torch.load(src_saving, weights_only=False, map_location="cpu")
    if isinstance(src_state, dict) and "model" in src_state:
        src_model = src_state["model"]
        print(f"[convert] source has optimizer state → dropping "
              f"(nr=2 needs fresh Adam)")
    else:
        src_model = src_state

    dst_model = convert_state(src_model, S=S, nlayers=nlayers)

    src_name = os.path.basename(src_saving)
    dst_name = rename_saving_file(dst_folder, src_name)
    # Force epoch back to 0 so training loop starts fresh but restores model.
    dst_name = re.sub(r"epoch\d+", "epoch0", dst_name)
    dst_saving = os.path.join(dst_folder, "savings", dst_name)

    # Save model-only (no optimizer); flow.load() handles both formats.
    torch.save(dst_model, dst_saving)
    print(f"[convert] wrote → {dst_saving} ({len(dst_model)} tensors)")


if __name__ == "__main__":
    main()
