import math

import numpy as np
import torch
from torch import nn

from .source import Source


class ConditionalGaussian(Source):
    """Hierarchical Gaussian prior  P(z) = P(z_slow) * P(z_fast | z_slow).

    Slow positions live on a sub-lattice with stride ``slow_stride``
    (e.g. 16 for L=32 ⇒ 2×2 slow grid). Fast positions are everything
    else. Both stay marginally Gaussian, but fast modes get a
    z_slow-dependent mean and log-std produced by a small CNN. At
    initialization the CNN output is identically (0, 0), so the prior
    reduces to the isotropic ``Gaussian`` baseline ⇒ the conditional
    structure has to be learned, not assumed.

    Motivation (see analyzers/rg_fixed_point/improvements_zh.md, I.2 /
    Scheme A): the original Gaussian prior demands that 3/4 of every
    scale's field be marginally i.i.d. N(0,1), which is the trivial
    Gaussian fixed-point geometry incompatible with Ising T_c.
    Letting fast modes condition on slow modes preserves local
    physical entanglement while keeping the prior tractable.
    """

    def __init__(self, nvars, slow_stride, n_hidden=32, name="conditional_gaussian"):
        super().__init__(nvars, name)
        if len(nvars) < 3:
            raise ValueError(
                "ConditionalGaussian expects nvars=[channel, L, L]; got " + str(nvars)
            )
        channel, L = nvars[0], nvars[1]
        assert nvars[1] == nvars[2], "conditional prior assumes square lattice"
        assert L % slow_stride == 0, (
            f"slow_stride={slow_stride} must divide L={L}"
        )
        self.L = L
        self.channel = channel
        self.slow_stride = slow_stride

        # 3-layer CNN: z (with fast positions zeroed) → (mu, log_sigma)
        # at every site. We zero-init the final conv so mu=0, log_sigma=0
        # (i.e. sigma=1) at the start of training — this reproduces the
        # unconditional N(0,1) prior exactly, so any divergence from
        # baseline is attributable to learned conditional structure
        # rather than initialization noise.
        self.cnn = nn.Sequential(
            nn.Conv2d(channel, n_hidden, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(n_hidden, 2 * channel, kernel_size=3, padding=1),
        )
        with torch.no_grad():
            self.cnn[-1].weight.zero_()
            self.cnn[-1].bias.zero_()

        # Boolean mask: True at slow-grid positions.
        slow_mask = torch.zeros(L, L, dtype=torch.float32)
        slow_mask[::slow_stride, ::slow_stride] = 1.0
        # shape broadcastable to (B, C, L, L)
        self.register_buffer("slow_mask", slow_mask.reshape(1, 1, L, L))
        self.register_buffer("fast_mask", 1.0 - slow_mask.reshape(1, 1, L, L))
        # Site counts: needed for the (2π) normalization in energy().
        n_slow = int(slow_mask.sum().item())
        n_fast = L * L - n_slow
        self.n_slow_per_channel = n_slow
        self.n_fast_per_channel = n_fast

    # ------------------------------------------------------------------
    # core: (mu, log_sigma) conditioned on slow-only input
    # ------------------------------------------------------------------

    def _mu_logsig(self, z):
        """Return (mu, log_sigma) on the full LxL grid.

        The CNN receives z with fast positions zeroed out so that
        conditioning is on z_slow only. Outputs at fast positions are
        what we'll use; outputs at slow positions are ignored.
        """
        z_slow_only = z * self.slow_mask.to(z.dtype)
        out = self.cnn(z_slow_only)
        mu, log_sigma = out.chunk(2, dim=1)
        # Clamp log_sigma to keep sigma in [exp(-5), exp(5)] = [0.0067, 148].
        # Hard floor prevents 1/sigma blow-up if the CNN drifts to extremes.
        log_sigma = log_sigma.clamp(min=-5.0, max=5.0)
        return mu, log_sigma

    # ------------------------------------------------------------------
    # Source API: energy + sample
    # ------------------------------------------------------------------

    def energy(self, z):
        """Negative log p(z) summed over the lattice (per-sample).

        −log p(z) =  Σ_slow [½ z² + ½ log(2π)]
                  +  Σ_fast [½ ((z−μ)/σ)² + log σ + ½ log(2π)].
        """
        B = z.shape[0]
        mu, log_sigma = self._mu_logsig(z)
        slow = self.slow_mask.to(z.dtype)
        fast = self.fast_mask.to(z.dtype)

        e_slow = 0.5 * (z ** 2) * slow
        e_fast = (0.5 * ((z - mu) / torch.exp(log_sigma)) ** 2 + log_sigma) * fast
        const = 0.5 * math.log(2.0 * math.pi) * (slow + fast)  # i.e., 0.5 log 2π everywhere

        return (e_slow + e_fast + const).reshape(B, -1).sum(dim=1)

    def sample(self, batchSize):
        """Draw z ~ P(z_slow) P(z_fast | z_slow).

        Two-step: first sample z_slow ~ N(0,1); then evaluate (μ, σ)
        from z_slow alone and sample z_fast ~ N(μ, σ²). The CNN call
        runs under no_grad — sampling is for the q-side of the flow
        and should not back-propagate into the prior parameters here.
        """
        size = [batchSize] + list(self.nvars)
        device = self.slow_mask.device
        dtype = self.slow_mask.dtype
        slow = self.slow_mask.to(dtype)
        fast = self.fast_mask.to(dtype)

        # z_slow on the slow grid, zeros elsewhere
        eps_slow = torch.randn(size, device=device, dtype=dtype)
        z = eps_slow * slow

        # Conditional (μ, σ) — depends only on z (with fast still 0).
        with torch.no_grad():
            mu, log_sigma = self._mu_logsig(z)
        eps_fast = torch.randn(size, device=device, dtype=dtype)
        z = z + (mu + torch.exp(log_sigma) * eps_fast) * fast
        return z
