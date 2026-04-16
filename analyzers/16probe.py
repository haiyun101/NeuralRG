import torch
import numpy as np
import h5py
import itertools
import train
import source
import glob
import os
import traceback

# 1. Setup paths and device
rootFolder = "./opt/32Ising_T2.4——weightTying/"
device = torch.device("cpu")
dtype = torch.float32

# # 2. Hardcode parameters exactly as they appear in parameters.hdf5
# L = 32
# d = 2
# T = 2.5
# nlayers = 10
# nmlp = 3
# nhidden = 64
# nrepeat = 1
# depthMERA = None # Translated from -1

# print("Initializing Ising target...")
# target = source.Ising(L, d, T)
# target = target.to(device=device, dtype=dtype)

# ====================================================================
# 2. Automatically read parameters from parameters.hdf5
# ====================================================================
param_file = os.path.join(rootFolder, "parameters.hdf5")
print("Reading parameters from: " + param_file)

try:
    with h5py.File(param_file, "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        T = float(np.array(f["T"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        depthMERA = int(np.array(f["depthMERA"]))
        
        if depthMERA == -1:
            depthMERA = None
except Exception as e:
    print("Failed to read parameters.hdf5:", e)
    exit()

print("Loaded architecture: L={}, nlayers={}, nmlp={}, nhidden={}".format(L, nlayers, nmlp, nhidden))

# 3. CRITICAL FIX: Enable Symmetry
# The filename 'SymmMERA' indicates symmetry was used during training.
def op(x):
    return -x
sym = [op]

name = "SymmMERA_analysis"

print("Initializing MERA network with symmetry...")
fw = train.symmetryMERAInit(L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, name, depthMERA=depthMERA)

# 4. Find and load the weights
try:
    latest_saving = max(glob.iglob(rootFolder + 'savings/*.saving'), key=os.path.getctime)
    print("Found saving file: " + latest_saving)
except ValueError:
    print("Error: No .saving files found in savings directory.")
    exit()

print("Loading weights into memory...")
saved_weights = torch.load(latest_saving, map_location=device)

# Handle cases where weights are wrapped inside a 'state_dict' key
if 'state_dict' in saved_weights:
    saved_weights = saved_weights['state_dict']

# 5. Debugging: Print keys to verify match
print("\n--- Key Comparison ---")
saved_keys = list(saved_weights.keys())
print("Top 5 keys in saved file:")
for k in saved_keys[:5]:
    print("  " + k)

fw_keys = list(fw.state_dict().keys())
print("\nTop 5 keys in initialized network:")
for k in fw_keys[:5]:
    print("  " + k)
print("----------------------\n")

# 6. Apply weights to the network
print("Attempting to load state dict...")
try:
    fw.load_state_dict(saved_weights, strict=True)
    print("SUCCESS: Strict loading passed! The architecture matches perfectly.")
except RuntimeError as e:
    print("Strict loading failed. Falling back to strict=False...")
    try:
        fw.load_state_dict(saved_weights, strict=False)
        print("WARNING: Loaded with strict=False. Some keys were ignored.")
    except Exception as e2:
        print("FAILED completely: " + str(e2))

fw.eval()
print("Model is in eval mode and ready for RG extraction.")

# ====================================================================
# 7. The 16-State Probing Experiment
# ====================================================================
print("\n--- Starting 16-State Probing ---")

# Generate all 16 combinations of +1 and -1 for a 2x2 block
states = list(itertools.product([1.0, -1.0], repeat=4))

# Convert to tensor. Shape must be (Batch, Channel, Height, Width) -> (16, 1, 2, 2)
test_inputs = torch.tensor(states, dtype=dtype, device=device).view(16, 1, 2, 2)

# Extract the first RG layer based on the keys we found earlier
first_layer = fw.flow.layerList[0]

# # Pass the states through the first layer
# with torch.no_grad():
#     try:
#         # Most flow layers return (output, log_det_jacobian)
#         output = first_layer(test_inputs)
        
#         if isinstance(output, tuple):
#             z = output[0]  # We only care about the transformed variables
#         else:
#             z = output
            
#     except Exception as e:

#         print("\n--- FULL ERROR TRACEBACK ---")
#         traceback.print_exc()
#         print("----------------------------\n")
#         exit()

# Pass the states through the first layer explicitly
with torch.no_grad():
    try:
        # Attempt 1: Standard PyTorch convention for X -> Z
        output = first_layer.forward(test_inputs)
    except AttributeError:
        try:
            # Attempt 2: In some Flow codebases, generative is forward and inference is inverse
            output = first_layer.inverse(test_inputs)
        except AttributeError:
            # If both fail, print the "blueprint" of the layer so we can find the right name
            print("ERROR: Neither 'forward' nor 'inverse' method found.")
            print("Available methods in this layer:")
            print([m for m in dir(first_layer) if callable(getattr(first_layer, m)) and not m.startswith('_')])
            exit()
            
    # CRITICAL FIX: Extract 'z' from 'output'
    if isinstance(output, tuple):
        z = output[0]  # We only care about the transformed variables
    else:
        z = output

# Format and print the results
print("Input 2x2 State (Flattened) --> Output Latent Variables (Flattened)")
print("Format: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]")
print("-" * 75)

for i in range(16):
    # Extract the 4 input spins and 4 output variables for this state
    in_val = test_inputs[i, 0].flatten().numpy()
    out_val = z[i, 0].flatten().numpy()
    
    # Calculate the total magnetization of the 2x2 block (-4, -2, 0, +2, +4)
    mag = sum(in_val)
    
    # Format strings for clean printing
    in_str = f"[{in_val[0]:+2.0f}, {in_val[1]:+2.0f}, {in_val[2]:+2.0f}, {in_val[3]:+2.0f}]"
    out_str = f"[{out_val[0]:+6.2f}, {out_val[1]:+6.2f}, {out_val[2]:+6.2f}, {out_val[3]:+6.2f}]"
    
    print(f"State {i:2d} (Mag: {mag:+2.0f}): {in_str}  -->  {out_str}")

print("\nExperiment Complete. Analyze the output vectors to find the coarse-graining rule!")