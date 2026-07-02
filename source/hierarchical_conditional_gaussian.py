"""Hierarchical Conditional Gaussian prior — multi-scale extension of i2.

Motivation
----------
The single-CNN i2 prior (`ConditionalGaussian`) lets fast modes condition on a
single slow sublattice. The V5 diagnostic shows the resulting flow off-loads
variance control to that one CNN, leaving MERA's intermediate y_s looking
non-Wilson. This is a "correct but weak" inductive bias: it doesn't force
scale-invariance of the conditional structure.

This class puts a *hierarchical* conditional Gaussian on the whole latent
z, decomposed by successive strides (coarsest → finest). At each level k
the sites at that stride are conditioned on all coarser levels via a CNN.
With scale-shared CNN weights (default), the same conditional-whitening
operation runs at every scale — a direct architectural expression of RG
scale invariance.

Math
----
Decompose z (shape channel × L × L) by strides s_0 > s_1 > ... > s_{K-1} = 1
into K non-overlapping site groups:

    level_0 = { (i, j) : i ≡ 0 mod s_0,  j ≡ 0 mod s_0 }
    level_k = { (i, j) : i ≡ 0 mod s_k,  j ≡ 0 mod s_k } \ (union of coarser levels)

Prior:
    p(z) = p_0(z_{level_0})  ·  ∏_{k=1}^{K-1} p_k(z_{level_k} | z_{levels < k})

    p_0     = ∏  N(z_i; 0, 1)              (iid standard normal)
    p_k     = ∏  N(z_i; μ_i^k, σ_i^k²)     (CNN reads coarser context)

CNN input:
    z with all sites at levels < k revealed, sites at level k and finer zeroed.

CNN output:
    (μ, log σ) at every (L × L) position (translation-equivariant).
    Only the outputs at level_k positions are used to score level_k sites.

Configuration
-------------
strides:              list of stride values, coarsest → finest. Default
                      = [L/2, L/4, ..., 1] (full log2 hierarchy).
n_hidden:             CNN hidden channels (default 32).
scale_shared:         True (default): one CNN shared across all levels.
                      False: independent CNN per level.
use_circular_padding: True (default): CNN uses padding_mode='circular' to
                      respect Ising periodic boundary conditions. i2's
                      original CNN uses zero-padding — a small architectural
                      bug that HCG fixes.
dilated_conv:         True: per-level CNN uses dilation matched to that
                      level's stride (so the 3×3 conv reaches neighboring
                      coarser sites even at deep levels). Only meaningful
                      when scale_shared=False. Capped at L/4.
"""
import math

import torch
from torch import nn

from .source import Source


class HierarchicalConditionalGaussian(Source):
    def __init__(self,
                 nvars,
                 strides=None,
                 n_hidden=32,
                 scale_shared=True,
                 use_circular_padding=True,
                 dilated_conv=True,
                 name="hierarchical_conditional_gaussian"):
        super().__init__(nvars, name)
        if len(nvars) < 3:
            raise ValueError(
                "HierarchicalConditionalGaussian expects nvars=[channel, L, L]; got "
                + str(nvars))
        channel, L = nvars[0], nvars[1]
        assert nvars[1] == nvars[2], "HCG assumes square lattice"

        # Default strides: [L/2, L/4, ..., 1] — every log2 step.
        if strides is None:
            k_max = int(math.log2(L))
            strides = [2 ** k for k in range(k_max - 1, -1, -1)]
        # Sanity: strides should be strictly decreasing, divide L, end at 1
        assert strides[-1] == 1, f"finest stride must be 1, got {strides[-1]}"
        for i in range(len(strides) - 1):
            assert strides[i] > strides[i + 1], "strides must be strictly decreasing"
        for s in strides:
            assert L % s == 0, f"stride {s} must divide L={L}"

        self.L = L
        self.channel = channel
        self.strides = strides
        self.K = len(strides)                       # number of levels
        self.scale_shared = scale_shared
        self.use_circular_padding = use_circular_padding
        self.dilated_conv = dilated_conv
        self.n_hidden = n_hidden

        # ------------------------------------------------------------------
        # Level masks
        # ------------------------------------------------------------------
        level_masks = []       # per-level: sites new to this level
        prev_covered = torch.zeros(L, L, dtype=torch.float32)
        for k, s in enumerate(strides):
            at_stride = torch.zeros(L, L, dtype=torch.float32)
            at_stride[::s, ::s] = 1.0
            new_here = at_stride * (1.0 - prev_covered)   # subtract already-covered
            level_masks.append(new_here.reshape(1, 1, L, L))
            prev_covered = prev_covered + new_here

        # Sanity: union should cover everything (since finest stride = 1)
        total = sum(m.sum().item() for m in level_masks)
        assert int(total) == L * L, \
            f"Level masks cover {int(total)} sites, expected {L*L}"

        for k, m in enumerate(level_masks):
            self.register_buffer(f"level_mask_{k}", m)

        # Context mask up to level k (inclusive) — used as CNN input for level k+1
        cumulative = torch.zeros(1, 1, L, L, dtype=torch.float32)
        for k in range(self.K):
            cumulative = cumulative + level_masks[k]
            self.register_buffer(f"context_mask_up_to_{k}", cumulative.clone())

        self.sites_per_level = [int(m.sum().item()) for m in level_masks]

        # ------------------------------------------------------------------
        # CNN builder
        # ------------------------------------------------------------------
        padding_mode = 'circular' if use_circular_padding else 'zeros'

        def make_cnn(dilation):
            # dilation applied uniformly to all conv layers; padding compensates.
            layers = [
                nn.Conv2d(channel, n_hidden, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          padding_mode=padding_mode),
                nn.ELU(),
                nn.Conv2d(n_hidden, n_hidden, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          padding_mode=padding_mode),
                nn.ELU(),
                nn.Conv2d(n_hidden, 2 * channel, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          padding_mode=padding_mode),
            ]
            cnn = nn.Sequential(*layers)
            with torch.no_grad():
                cnn[-1].weight.zero_()
                cnn[-1].bias.zero_()      # μ=0, log_sigma=0 (σ=1) at init
            return cnn

        # ------------------------------------------------------------------
        # CNN(s) — coarsest level (level 0) has NO CNN (unconditional N(0,1))
        # Levels 1..K-1 each need CNN-produced (μ, σ)
        # ------------------------------------------------------------------
        if scale_shared:
            # One CNN, dilation=1 (fixed 3×3 receptive field). Scale-invariance
            # is enforced at the parameter level: same conditional-whitening
            # operator applied at every scale.
            self.cnn_shared = make_cnn(dilation=1)
            self.cnns = None
        else:
            # Per-level CNNs. Dilation matched to that level's stride so the
            # 3×3 conv can reach the nearest coarser sites.
            cnns = []
            for k in range(1, self.K):
                if dilated_conv:
                    # coarser context lives at stride strides[k-1]; dilate to reach it
                    d = strides[k - 1]
                    d = max(1, min(d, L // 4))
                else:
                    d = 1
                cnns.append(make_cnn(dilation=d))
            self.cnns = nn.ModuleList(cnns)
            self.cnn_shared = None

    # ---------------------------------------------------------------------
    # Core: compute (μ, log_sigma) for level k conditioned on coarser context.
    # ---------------------------------------------------------------------
    def _mu_logsig_level_k(self, z, k):
        """Level k ≥ 1. z is the current sample (may have finer sites zeroed;
        we re-mask to be safe). Returns (μ, log_σ) at ALL (L×L) positions;
        caller keeps only level-k positions."""
        assert k >= 1
        context_mask = self._buffers[f"context_mask_up_to_{k-1}"]
        z_ctx = z * context_mask.to(z.dtype)
        cnn = self.cnn_shared if self.scale_shared else self.cnns[k - 1]
        out = cnn(z_ctx)
        mu, log_sigma = out.chunk(2, dim=1)
        # Same log_sigma clamp as ConditionalGaussian to avoid 1/σ blow-up.
        log_sigma = log_sigma.clamp(min=-5.0, max=5.0)
        return mu, log_sigma

    # ---------------------------------------------------------------------
    # Source API: energy + sample
    # ---------------------------------------------------------------------
    def energy(self, z):
        """−log p(z) summed over the lattice, per sample. Returns shape (B,)."""
        B = z.shape[0]
        e = torch.zeros(B, device=z.device, dtype=z.dtype)
        two_pi_log = 0.5 * math.log(2.0 * math.pi)

        # Coarsest level: iid N(0, 1) at level_0 sites
        m0 = self._buffers["level_mask_0"].to(z.dtype)
        # sum: 0.5 * z² at level 0 sites
        e_lvl0 = (0.5 * (z ** 2) * m0).reshape(B, -1).sum(dim=1)
        # const: 0.5 log(2π) per (channel, level_0 site)
        e_lvl0 = e_lvl0 + two_pi_log * self.sites_per_level[0] * self.channel
        e = e + e_lvl0

        # Levels 1..K-1: conditional Gaussians
        for k in range(1, self.K):
            mu_k, log_sigma_k = self._mu_logsig_level_k(z, k)
            mk = self._buffers[f"level_mask_{k}"].to(z.dtype)
            resid = (z - mu_k) / torch.exp(log_sigma_k)
            e_lvl_k = ((0.5 * resid ** 2 + log_sigma_k) * mk).reshape(B, -1).sum(dim=1)
            e_lvl_k = e_lvl_k + two_pi_log * self.sites_per_level[k] * self.channel
            e = e + e_lvl_k

        return e

    def sample(self, batchSize):
        """Draw z ~ p(z) via sequential (top-down) conditional sampling."""
        size = [batchSize] + list(self.nvars)
        device = self._buffers["level_mask_0"].device
        dtype = self._buffers["level_mask_0"].dtype
        z = torch.zeros(size, device=device, dtype=dtype)

        # Coarsest level: iid N(0, 1)
        m0 = self._buffers["level_mask_0"].to(dtype)
        eps0 = torch.randn(size, device=device, dtype=dtype)
        z = z + eps0 * m0

        # Levels 1..K-1: sample conditionally on already-filled context.
        # No grad through the CNN during sampling (same convention as
        # ConditionalGaussian.sample: avoids sample-side gradients pushing
        # CNN toward easily-scored latents).
        with torch.no_grad():
            for k in range(1, self.K):
                mu_k, log_sigma_k = self._mu_logsig_level_k(z, k)
                sigma_k = torch.exp(log_sigma_k)
                eps_k = torch.randn(size, device=device, dtype=dtype)
                mk = self._buffers[f"level_mask_{k}"].to(dtype)
                z = z + (mu_k + sigma_k * eps_k) * mk

        return z
