import os
import glob
import re
import argparse
import h5py
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Required for cluster environments to prevent GUI errors
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot x, s, and m distributions from HDF5 files.")
    parser.add_argument('folder', type=str, help="Subfolder name, e.g., T_2.6")
    parser.add_argument('-base', type=str, default='./opt', help="Base path, default is ./opt")
    args = parser.parse_args()

    target_path = os.path.join(args.base, args.folder)
    
    # Find all HMCresult files
    search_pattern = os.path.join(target_path, '**/*HMCresult*.hdf5')
    files = glob.glob(search_pattern, recursive=True)
    
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    files = sorted(files, key=extract_epoch)

    if not files:
        print(f"[!] No *HMCresult*.hdf5 files found in {target_path}.")
        return

    print(f"[*] Found {len(files)} sampling record files. Starting batch plotting...")

    for target_file in tqdm(files, desc="Processing Epochs"):
        epoch_num = extract_epoch(target_file)

        try:
            with h5py.File(target_file, 'r') as f:
                # 1. Read continuous variable x (usually stored as 'X' in HDF5)
                x_raw = np.array(f['X'])
                
                # Flatten x for global histogram
                x_flat = x_raw.flatten()
                
                # 2. Convert x to physical spin s
                # Using hard truncation np.sign() as in calc_Gr
                s_raw = np.sign(x_raw)
                s_flat = s_raw.flatten()
                
                # 3. Calculate macroscopic magnetization m (Order Parameter) for each sample
                # Assume the first dimension of x_raw is batch_size
                batch_size = x_raw.shape[0]
                s_batch = s_raw.reshape(batch_size, -1)
                m_values = np.mean(s_batch, axis=1)

        except Exception as e:
            print(f"\n[!] Error reading file {target_file}: {e}")
            continue

        # === Start plotting 1x3 distribution plots ===
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        # Plot 1: Distribution of continuous variable x
        ax1.hist(x_flat, bins=100, color='royalblue', alpha=0.7, density=True)
        ax1.set_title(f"Continuous Field $x$\n(Epoch: {epoch_num})")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("Density")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution of discrete physical spin s
        unique, counts = np.unique(s_flat, return_counts=True)
        ax2.bar(unique, counts / len(s_flat), color='crimson', alpha=0.7, width=0.4)
        ax2.set_title(f"Physical Spin $s$\n(Epoch: {epoch_num})")
        ax2.set_xticks([-1, 1])
        ax2.set_xlabel("$s$")
        ax2.set_ylabel("Probability")
        ax2.grid(True, axis='y', alpha=0.3)

        # Plot 3: Distribution of Order Parameter m
        ax3.hist(m_values, bins=50, range=(-1.1, 1.1), color='forestgreen', alpha=0.7, density=True)
        ax3.set_title(f"Order Parameter $m$\n(Epoch: {epoch_num})")
        ax3.set_xlabel("Magnetization $m$")
        ax3.set_ylabel("Density")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_name = f'Dist_epoch_{epoch_num:05d}.png'
        save_path = os.path.join(target_path, save_name)
        plt.savefig(save_path, dpi=150)
        
        # Must close to prevent memory leaks
        plt.close()

    print(f"\n[+] Success! All epoch distribution plots saved in {target_path}")

if __name__ == '__main__':
    main()