import h5py
import sys
import numpy as np
import os

# 1. Allow filename from command line argument
#    Default to 'parameters.hdf5' if no argument is provided
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = 'parameters.hdf5'

print(f"Reading file: {file_path}")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"\n--- Keys (Datasets/Groups) ---")
        keys = list(f.keys())
        print(keys)
        
        print(f"\n--- Content Details ---")
        for key in keys:
            # f[key][()] reads the data into a numpy array/scalar
            data = f[key][()]
            
            # If it's a scalar (single number/string), print value
            if np.ndim(data) == 0:
                print(f"  {key}: {data}")
            # If it's an array, print shape and type to avoid flooding screen
            else:
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        # Check for Global Attributes (metadata often stored here)
        if len(f.attrs) > 0:
            print(f"\n--- Global Attributes ---")
            for key, val in f.attrs.items():
                print(f"  {key}: {val}")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")
