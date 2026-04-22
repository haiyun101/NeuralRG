# import torch
# import numpy as np
# import math
# import argparse

# def wolff_step(lattice, beta):
#     L = lattice.shape[0]
#     p_add = 1.0 - math.exp(-2.0 * beta)
    
#     i, j = np.random.randint(0, L), np.random.randint(0, L)
#     cluster_spin = lattice[i, j].item()
    
#     stack = [(i, j)]
#     lattice[i, j] = -cluster_spin
    
#     while stack:
#         cx, cy = stack.pop()
#         neighbors = [
#             ((cx + 1) % L, cy), ((cx - 1) % L, cy),
#             (cx, (cy + 1) % L), (cx, (cy - 1) % L)
#         ]
        
#         for nx, ny in neighbors:
#             if lattice[nx, ny] == cluster_spin:
#                 if np.random.rand() < p_add:
#                     stack.append((nx, ny))
#                     lattice[nx, ny] = -cluster_spin

#     return lattice

# def generate_dataset(L, beta, num_samples, thermalize_steps=1000, steps_between_samples=10):
#     print(f"Generating {num_samples} samples for L={L}, beta={beta:.4f}...")
#     lattice = torch.randint(0, 2, (L, L), dtype=torch.float32) * 2 - 1
    
#     for _ in range(thermalize_steps):
#         lattice = wolff_step(lattice, beta)
        
#     samples = []
#     for _ in range(num_samples):
#         for _ in range(steps_between_samples):
#             lattice = wolff_step(lattice, beta)
#         samples.append(lattice.clone().unsqueeze(0))
        
#     return torch.stack(samples)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Generate Wolff MCMC samples')
#     parser.add_argument("-L", type=int, default=16, help="Lattice size")
#     parser.add_argument("-T", type=float, default=None, help="Temperature. Defaults to Tc if not provided.")
#     parser.add_argument("-N", "--num_samples", type=int, default=50000, help="Number of samples to generate")
#     args = parser.parse_args()
    
#     # Default to Critical Temperature if no T is provided
#     Tc = 2.0 / math.log(1.0 + math.sqrt(2.0))
#     T = args.T if args.T is not None else Tc
#     beta = 1.0 / T
    
#     dataset = generate_dataset(args.L, beta, num_samples=args.num_samples)
    
#     # Save with naming convention that main.py can auto-search
#     filename = f"./data/mcmc_data/mcmc_wolff_L{args.L}_T{T:.4f}_N{args.num_samples}.pt"
#     torch.save(dataset, filename)
#     print(f"Dataset successfully saved to {filename}")


import torch
import numpy as np
import math
import argparse
import time
from numba import njit

@njit
def numba_wolff_step(lattice, beta):
    L = lattice.shape[0]
    p_add = 1.0 - math.exp(-2.0 * beta)
    
    # Choose seed
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    cluster_spin = lattice[i, j]
    
    # Using a list as a stack (Numba handles this efficiently)
    stack = [(i, j)]
    lattice[i, j] = -cluster_spin
    
    while len(stack) > 0:
        cx, cy = stack.pop()
        
        # Manually define neighbors for Numba compatibility
        neighbors = [
            ((cx + 1) % L, cy), ((cx - 1) % L, cy),
            (cx, (cy + 1) % L), (cx, (cy - 1) % L)
        ]
        
        for nx, ny in neighbors:
            if lattice[nx, ny] == cluster_spin:
                if np.random.random() < p_add:
                    stack.append((nx, ny))
                    lattice[nx, ny] = -cluster_spin
    return lattice

def generate_dataset(L, beta, num_samples, thermalize=1000, steps_between=10):
    # Initialize lattice as a numpy array for Numba
    lattice_np = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    
    print(f"Thermalizing...")
    for _ in range(thermalize):
        lattice_np = numba_wolff_step(lattice_np, beta)
    
    samples = []
    start_time = time.time()
    for i in range(num_samples):
        for _ in range(steps_between):
            lattice_np = numba_wolff_step(lattice_np, beta)
        samples.append(torch.from_numpy(lattice_np.copy()).float().unsqueeze(0))
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i+1}/{num_samples} | Speed: {elapsed/(i+1):.4f}s per sample")
            
    return torch.stack(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, default=32)
    parser.add_argument("-T", type=float, default=2.2692)
    parser.add_argument("-N", type=int, default=50000)
    args = parser.parse_args()

    # Run a tiny N=1 trial to trigger Numba compilation before timing
    numba_wolff_step(np.ones((args.L, args.L), dtype=np.int8), 1.0/args.T)
    
    dataset = generate_dataset(args.L, 1.0/args.T, args.N)
    
    out_path = f"./data/mcmc_data/mcmc_wolff_L{args.L}_T{args.T}_N{args.N}.pt"
    torch.save(dataset, out_path)
    print(f"Saved to {out_path}")