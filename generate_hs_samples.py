"""
Convert discrete Ising MCMC samples to exact continuous HS samples.

Math:  The Hubbard-Stratonovich joint distribution is
           p(s, x) ∝ exp(−½ xᵀK⁻¹x + xᵀs)
       The conditional is a Gaussian:
           p(x | s) = N(Ks, K)
       Marginalising over s ~ p_discrete gives x ~ p_HS exactly.

Usage:
    python generate_hs_samples.py -L 8  -T 2.269 --in_path data/mcmc_data/mcmc_wolff_L8_T2.269_N200000.pt
    python generate_hs_samples.py -L 16 -T 2.269 --in_path data/mcmc_data/mcmc_wolff_L16_T2.269_N200000.pt
"""
import argparse
import os
import numpy as np
import torch
from scipy.linalg import eigh, cholesky as scipy_chol

def build_K(L, T):
    N = L * L
    Adj = np.zeros((N, N))
    for i in range(N):
        row, col = divmod(i, L)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            j = ((row + dr) % L) * L + (col + dc) % L
            Adj[i, j] = 1.0
    K = Adj / T
    w = eigh(K, eigvals_only=True)
    offset = 0.1 - w.min()
    K += np.eye(N) * offset
    return K, offset


def convert(discrete_samples, K, batch_size=2048, seed=42):
    """
    discrete_samples: (N_samples, 1, L, L) float32 tensor of ±1
    returns: (N_samples, 1, L, L) float32 tensor from p_HS
    """
    rng = np.random.default_rng(seed)
    N_samples, _, L, _ = discrete_samples.shape
    N = L * L

    # Cholesky of K  (K = LLᵀ, lower triangular)
    L_chol = scipy_chol(K, lower=True)           # (N, N)
    L_chol_t = torch.from_numpy(L_chol).float()  # for batched mm

    # Mean vector for each sample: μ = K s  →  shape (N_samples, N)
    K_t = torch.from_numpy(K).float()
    s_flat = discrete_samples.reshape(N_samples, N)  # (N_samples, N)

    out = torch.empty_like(s_flat)

    for start in range(0, N_samples, batch_size):
        end = min(start + batch_size, N_samples)
        s_b = s_flat[start:end]            # (B, N)
        mu = s_b @ K_t.T                   # (B, N)   = (K s)ᵀ row-wise
        z = torch.from_numpy(rng.standard_normal((end - start, N)).astype(np.float32))
        x = mu + z @ L_chol_t.T           # (B, N)   = μ + L z
        out[start:end] = x

    return out.reshape(N_samples, 1, L, L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, required=True)
    parser.add_argument("-T", type=float, required=True)
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/mcmc_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Building K  (L={args.L}, T={args.T}) ...")
    K, offset = build_K(args.L, args.T)
    print(f"  offset = {offset:.6f},  K shape = {K.shape},  "
          f"eigenvalue range = [{np.linalg.eigvalsh(K).min():.4f}, {np.linalg.eigvalsh(K).max():.4f}]")

    print(f"Loading discrete samples from {args.in_path} ...")
    discrete = torch.load(args.in_path, weights_only=True)
    N_samples = discrete.shape[0]
    print(f"  {N_samples} samples, shape {tuple(discrete.shape)}")

    print("Sampling x ~ N(Ks, K) for each discrete configuration ...")
    hs_samples = convert(discrete, K, seed=args.seed)

    # Quick sanity check: marginal mean and variance per site
    flat = hs_samples.reshape(N_samples, -1)
    print(f"  x mean ≈ {flat.mean():.4f}  (expect ≈ 0 for symmetric distribution)")
    print(f"  x std  ≈ {flat.std():.4f}")
    # E[x_i] = E_s[K e_i]^T s  =  K row_i · E[s] ≈ 0 (magnetisation ≈ 0 near Tc)
    # Var[x_i] should be K_ii + (Ks)_i^2 averaged

    N_str = f"N{N_samples}"
    out_name = f"hs_L{args.L}_T{args.T}_{N_str}.pt"
    out_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(hs_samples, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
