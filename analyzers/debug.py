import os
import glob
import re
import argparse
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    target_path = os.path.join('./opt', args.folder)
    
    # Find files
    files = glob.glob(os.path.join(target_path, '**/*IsingHMCresult*.hdf5'), recursive=True)
    if not files:
        print("ERROR: No .hdf5 files found in", target_path)
        return

    # Sort files
    files = sorted(files, key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)) if re.search(r'epoch(\d+)', x) else -1)
    
    target_file = files[-1]
    
    print("--------------------------------------------------")
    print("DIAGNOSING FILE: " + str(target_file))
    print("--------------------------------------------------")

    try:
        with h5py.File(target_file, 'r') as f:
            if 'X' not in f.keys():
                print("ERROR: 'X' not found in keys. Available keys:", list(f.keys()))
                return
                
            spins = np.array(f['X'])
            print("1. Original shape :", spins.shape)
            
            L = spins.shape[-1]
            spins = spins.reshape(-1, L, L).astype(np.float32)
            print("2. Reshaped shape :", spins.shape)
            print("3. Max value      :", np.max(spins))
            print("4. Min value      :", np.min(spins))
            
            max_r = L // 2
            G_r = np.zeros(max_r)
            
            print("\nCalculating G(r)...")
            for r in range(max_r):
                shift_x = np.roll(spins, shift=-r, axis=-1)
                shift_y = np.roll(spins, shift=-r, axis=-2)
                
                G_r[r] = (np.mean(spins * shift_x) + np.mean(spins * shift_y)) / 2.0
                
                print("  r=" + str(r).ljust(2) + "  ---> G(r) = " + str(round(G_r[r], 6)))
                
    except Exception as e:
        print("CRASHED during reading or calculation:")
        print(e)

if __name__ == "__main__":
    main()