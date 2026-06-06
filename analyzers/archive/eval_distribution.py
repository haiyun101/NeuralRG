import os
import sys
import glob
import argparse
import h5py
import numpy as np

# Use Agg backend for cluster compatibility (no GUI required)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

import utils
import flow
import train
import source

def main():
    # --- 1. Command line arguments parsing ---
    parser = argparse.ArgumentParser(description='Evaluate and plot distribution of x, s, and order parameter m.')
    parser.add_argument("subfolder", type=str, help="Subfolder name of the model, e.g., T_2.6")
    parser.add_argument("-base", type=str, default="./opt/", help="Base directory (default: ./opt/)")
    parser.add_argument("-batch", type=int, default=1000, help="Batch size for sampling (default: 1000)")
    parser.add_argument("-nosym", action="store_true", help="Add this flag if the model was trained without -symmetry")
    
    args = parser.parse_args()

    # Construct full path
    model_path = os.path.join(args.base, args.subfolder)
    if not model_path.endswith('/'):
        model_path += '/'

    print(f"[*] Base directory: {model_path}")

    # --- 2. Load training parameters ---
    param_file = model_path + "parameters.hdf5"
    if not os.path.exists(param_file):
        print(f"[Error] Parameter file not found: {param_file}")
        sys.exit(1)

    with h5py.File(param_file, "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        depthMERA = int(np.array(f["depthMERA"]))
        if depthMERA == -1: 
            depthMERA = None

    device = torch.device("cpu")  # Sampling and plotting on CPU is sufficient
    dtype = torch.float32

    # --- 3. Instantiate network architecture ---
    if not args.nosym:
        def op(x): return -x
        sym = [op]
        print("[*] Using symmetry")
    else:
        sym = None

    fw = train.symmetryMERAInit(L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, "SymmMERA", depthMERA=depthMERA)

    # Find the latest .saving model weights
    saving_files = glob.glob(model_path + 'savings/*.saving')
    if not saving_files:
        print(f"[Error] No .saving weights found in {model_path}savings/ !")
        sys.exit(1)
        
    latest_saving = max(saving_files, key=os.path.getctime)
    print(f"[*] Loading model from: {latest_saving}")
    
    saved = torch.load(latest_saving, map_location=device)
    fw.load(saved)

    # --- 4. Sample and calculate physical quantities ---
    print(f"[*] Generating {args.batch} samples...")
    with torch.no_grad():
        # Extract the continuous field x generated at the top layer
        x, _ = fw.sample(args.batch)
        x_flat = x.cpu().numpy().flatten()
        
        # Map to physical spins s \in {-1, 1}
        p_up = torch.sigmoid(2.0 * x)
        s = 2.0 * torch.bernoulli(p_up) - 1.0
        s_flat = s.cpu().numpy().flatten()

        # Calculate Order Parameter (Magnetization m = 1/N * \sum s_i) for each sample in the batch
        s_batch = s.view(args.batch, -1)  # Reshape to (batch_size, N_spins)
        m = s_batch.mean(dim=1).cpu().numpy()  # Mean over spins -> (batch_size,)

    # --- 5. Plot and save to the model folder ---
    print("[*] Plotting distributions...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left: Histogram of continuous variable x
    ax1.hist(x_flat, bins=100, color='royalblue', alpha=0.7, density=True)
    ax1.set_title(f"Continuous Field $x$ Distribution\n(Folder: {args.subfolder})")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)

    # Middle: Bar chart of discrete spin s
    unique, counts = np.unique(s_flat, return_counts=True)
    ax2.bar(unique, counts/len(s_flat), color='crimson', alpha=0.7, width=0.4)
    ax2.set_title(f"Sampled Physical Spin $s$\n(Folder: {args.subfolder})")
    ax2.set_xticks([-1, 1])
    ax2.set_xlabel("$s$")
    ax2.set_ylabel("Probability")
    ax2.grid(True, axis='y', alpha=0.3)

    # Right: Histogram of Order Parameter m
    # We restrict range to (-1.1, 1.1) to clearly see symmetry breaking bounds
    ax3.hist(m, bins=50, range=(-1.1, 1.1), color='forestgreen', alpha=0.7, density=True)
    ax3.set_title(f"Order Parameter $m$ Distribution\nOver {args.batch} Samples")
    ax3.set_xlabel("Magnetization $m$")
    ax3.set_ylabel("Density")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure inside the passed model_path
    save_fig_path = os.path.join(model_path, "distribution_eval.pdf")
    plt.savefig(save_fig_path)
    plt.close()
    print(f"[+] Successfully saved plot to: {save_fig_path}")

if __name__ == '__main__':
    main()