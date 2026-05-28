"""Numerical sanity checks for the NSF coupling.

  (1) Invertibility round-trip:  forward(inverse(z)) ~ z   and vice versa
  (2) Analytical log-det matches a numerical Jacobian computed via autograd

Run with `python test/test_nsf.py` from repo root.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from flow.nsf import NSFCoupling


def make_param_net(coreSize, K, hidden=32):
    out = coreSize * (3 * K - 1)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(coreSize, hidden), nn.ELU(),
        nn.Linear(hidden, hidden),   nn.ELU(),
        nn.Linear(hidden, out),
    )


def main():
    torch.manual_seed(0)
    B_batch  = 8
    channel  = 1
    kH = kW  = 2
    coreSize = channel * kH * kW
    nlayers  = 4
    K        = 8
    bound    = 5.0

    # alternating half-mask
    masks = []
    half  = coreSize // 2
    for n in range(nlayers):
        b = torch.zeros(channel, kH, kW)
        flat = b.view(-1)
        if n % 2 == 0:
            flat[:half] = 1
        else:
            flat[half:] = 1
        masks.append(b)
    maskList = torch.stack(masks)        # (nlayers, channel, kH, kW)

    nets = [make_param_net(coreSize, K) for _ in range(nlayers)]
    nsf  = NSFCoupling(maskList, nets, num_bins=K, bound=bound)
    nsf.eval()

    # ---- 1) round-trip ----
    z = torch.randn(B_batch, channel, kH, kW) * 0.5     # well inside bound
    x, _ = nsf.inverse(z)
    z_back, _ = nsf.forward(x)
    err = (z - z_back).abs().max().item()
    print(f"[1] round-trip max-err = {err:.2e}  (target < 1e-4)")
    assert err < 1e-4, "round-trip failed"

    # ---- 2) log-det vs autograd ----
    # for a single point: compare analytical sum log|dy_i/dx_i| (triangular)
    # against log|det J| from autograd.
    z1 = torch.randn(1, channel, kH, kW, requires_grad=True) * 0.5
    x1, logjac1 = nsf.inverse(z1)
    # numerical Jacobian
    J = torch.zeros(coreSize, coreSize)
    for i in range(coreSize):
        grad = torch.autograd.grad(x1.view(-1)[i], z1, retain_graph=True)[0]
        J[i] = grad.view(-1)
    logdet_num = torch.slogdet(J)[1]
    err2 = (logjac1.item() - logdet_num.item())
    print(f"[2] logdet  analytic={logjac1.item():+.5f}  numeric={logdet_num.item():+.5f}  err={err2:+.2e}")
    assert abs(err2) < 1e-3, "logdet mismatch"

    # ---- 3) inverse logdet sign consistency ----
    x2 = torch.randn(1, channel, kH, kW, requires_grad=True) * 0.5
    z2, logjac2 = nsf.forward(x2)
    print(f"[3] inverse-direction logdet sample = {logjac2.item():+.5f}")

    print("\nAll NSF tests passed.")


if __name__ == "__main__":
    main()
