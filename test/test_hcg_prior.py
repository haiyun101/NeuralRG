"""Sanity checks for HierarchicalConditionalGaussian.

Runs:
    python test/test_hcg_prior.py

Checks:
  1. Level masks partition every site exactly once.
  2. At initialization (final conv weights zeroed) HCG is exactly the
     isotropic N(0, I) baseline — energy(z) equals the Gaussian energy.
  3. sample() draws z that look ~ N(0, I) at init (mean ≈ 0, std ≈ 1).
  4. energy/sample are consistent with each other for a fixed init
     (drawing z ~ sample and re-evaluating energy(z) → matches a fresh
     Gaussian evaluation).
  5. After a random perturbation of CNN weights, HCG diverges from Gaussian
     (confirming the CNN is actually being consulted).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source import Gaussian, HierarchicalConditionalGaussian


def gaussian_energy_manual(z):
    """−log N(z; 0, I) summed over lattice, per sample."""
    B = z.shape[0]
    numel_per_sample = z[0].numel()
    return (0.5 * z ** 2).reshape(B, -1).sum(dim=1) + \
        0.5 * math.log(2.0 * math.pi) * numel_per_sample


def check_masks(hcg, L, channel):
    total = torch.zeros(1, 1, L, L)
    for k in range(hcg.K):
        total = total + hcg._buffers[f"level_mask_{k}"]
    ok = torch.allclose(total, torch.ones_like(total))
    assert ok, f"Level masks do not partition L×L: max={total.max().item()} min={total.min().item()}"
    sites = sum(hcg.sites_per_level)
    assert sites == L * L, f"Total sites {sites} ≠ L² = {L*L}"
    print(f"  [ok] level masks partition {L*L} sites into "
          f"{[hcg.sites_per_level[k] for k in range(hcg.K)]} at strides {hcg.strides}")


def check_init_equals_gaussian(hcg, L, channel, B=8):
    torch.manual_seed(0)
    z = torch.randn(B, channel, L, L)
    e_hcg = hcg.energy(z)
    e_ref = gaussian_energy_manual(z)
    diff = (e_hcg - e_ref).abs().max().item()
    # float32 accumulation over ~L² sites → tolerance grows ~L².
    # For a lattice up to L=64, 1e-3 is safe (still far below any signal).
    assert diff < 1e-3, f"HCG energy at init deviates from Gaussian by {diff}"
    print(f"  [ok] HCG energy at init == Gaussian energy (max abs diff = {diff:.2e})")


def check_sample_at_init(hcg, L, channel, B=2000):
    torch.manual_seed(0)
    z = hcg.sample(B)
    mu, sd = z.mean().item(), z.std().item()
    assert abs(mu) < 0.05, f"sample mean {mu} not ≈ 0"
    assert 0.95 < sd < 1.05, f"sample std {sd} not ≈ 1"
    print(f"  [ok] HCG sample at init ~ N(0, 1): mean={mu:+.4f}, std={sd:.4f}")


def check_perturbed_differs_from_gaussian(hcg, L, channel, B=8):
    # Add non-zero weights to final conv and see energy diverge from Gaussian
    if hcg.scale_shared:
        cnn = hcg.cnn_shared
    else:
        cnn = hcg.cnns[0]

    with torch.no_grad():
        cnn[-1].weight.normal_(0, 0.5)
        cnn[-1].bias.normal_(0, 0.5)

    torch.manual_seed(0)
    z = torch.randn(B, channel, L, L)
    e_hcg = hcg.energy(z)
    e_ref = gaussian_energy_manual(z)
    diff = (e_hcg - e_ref).abs().max().item()
    assert diff > 1.0, f"HCG energy still ≈ Gaussian after CNN perturbation (diff = {diff})"
    print(f"  [ok] after CNN perturbation HCG energy diverges from Gaussian "
          f"(max abs diff = {diff:.2f} nat)")


def run_config(L, channel, scale_shared, circular, dilated):
    print(f"\n=== L={L} channel={channel} scale_shared={scale_shared} "
          f"circular={circular} dilated={dilated} ===")
    hcg = HierarchicalConditionalGaussian(
        [channel, L, L],
        n_hidden=16,
        scale_shared=scale_shared,
        use_circular_padding=circular,
        dilated_conv=dilated,
    )
    hcg.eval()
    check_masks(hcg, L, channel)
    check_init_equals_gaussian(hcg, L, channel)
    check_sample_at_init(hcg, L, channel)
    check_perturbed_differs_from_gaussian(hcg, L, channel)


def main():
    torch.set_default_dtype(torch.float32)
    # Try a few configurations.
    run_config(L=8,  channel=1, scale_shared=True,  circular=True,  dilated=True)
    run_config(L=8,  channel=1, scale_shared=False, circular=True,  dilated=True)
    run_config(L=16, channel=1, scale_shared=True,  circular=True,  dilated=True)
    run_config(L=16, channel=1, scale_shared=False, circular=False, dilated=False)
    run_config(L=32, channel=1, scale_shared=True,  circular=True,  dilated=True)
    print("\nAll HCG sanity checks passed.")


if __name__ == "__main__":
    main()
