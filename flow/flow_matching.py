"""Conditional Flow Matching (CFM) for 2D Ising HS data.

Minimal implementation of Lipman et al. 2022 with OT-based (rectified-flow-
style) probability paths. Trains a neural velocity field v_t^θ(x, t) via MSE
against the conditional target u_t(x | x_1) = x_1 − x_0 where x_t = (1−t)·x_0 + t·x_1
and x_0 ~ N(0, I). Sampling is Euler ODE integration from x_0 to x_1.

Key differences from our RNVP+MERA flows:
  - No Jacobian computation during training (loss is MSE, not log-likelihood).
  - No VP penalty needed — Jacobian doesn't appear.
  - Sampling costs an ODE integration (N_steps forward passes), not a single
    flow pass. Trade compute for training simplicity + tail flexibility.

Not integrated with main.py's CLI — this module is standalone. See
`train/fm_learn.py` for the training loop and
`analyzers/fm_sample_diagnostic.py` for evaluation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────── time embedding ───────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-features time embedding, matches Nichol & Dhariwal 2021 (iDDPM).
    Input: (B,) scalar in [0, 1]. Output: (B, dim)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(1, half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────── velocity field network ───────────────────


class ResBlock(nn.Module):
    """Conv-Norm-SiLU x2 with time-embedding modulation + skip.

    Time embedding modulates via an affine (scale + shift) on the intermediate
    feature (FiLM-style)."""

    def __init__(self, ch, temb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.temb_proj = nn.Linear(temb_dim, ch * 2)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # FiLM: scale + shift from t
        gamma, beta = self.temb_proj(F.silu(temb)).chunk(2, dim=-1)
        h = h * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class VelocityUNet(nn.Module):
    """Small U-Net for L=32 or L=64 lattice velocity field.

    Input:  x  (B, 1, L, L)  and  t (B,)  in [0, 1].
    Output: v  (B, 1, L, L).

    Two downsampling stages: L → L/2 → L/4. Base channels `nhidden`.
    Symmetric decoder with skip connections. Total params ~1M for nhidden=64."""

    def __init__(self, L, nhidden=64, temb_dim=128, in_channels=1):
        super().__init__()
        self.L = L
        self.temb_dim = temb_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(temb_dim),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, nhidden, 3, padding=1)
        # Encoder
        self.enc1 = ResBlock(nhidden, temb_dim)
        self.down1 = nn.Conv2d(nhidden, nhidden * 2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(nhidden * 2, temb_dim)
        self.down2 = nn.Conv2d(nhidden * 2, nhidden * 4, 4, stride=2, padding=1)
        # Bottleneck
        self.mid = ResBlock(nhidden * 4, temb_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(nhidden * 4, nhidden * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(nhidden * 2, temb_dim)
        self.up1 = nn.ConvTranspose2d(nhidden * 2, nhidden, 4, stride=2, padding=1)
        self.dec1 = ResBlock(nhidden, temb_dim)
        # Output — zero-init so v_t^θ ≈ 0 at start (well-behaved ODE)
        self.out_conv = nn.Conv2d(nhidden, in_channels, 3, padding=1)
        with torch.no_grad():
            self.out_conv.weight.zero_()
            self.out_conv.bias.zero_()

    def forward(self, x, t):
        temb = self.time_embed(t)                    # (B, temb_dim)
        h0 = self.in_conv(x)                          # (B, C, L, L)
        h1 = self.enc1(h0, temb)                      # (B, C, L, L)
        h2 = self.down1(h1)                           # (B, 2C, L/2, L/2)
        h2 = self.enc2(h2, temb)
        h3 = self.down2(h2)                           # (B, 4C, L/4, L/4)
        h3 = self.mid(h3, temb)
        d2 = self.up2(h3)                             # (B, 2C, L/2, L/2)
        d2 = self.dec2(d2 + h2, temb)                 # skip
        d1 = self.up1(d2)                             # (B, C, L, L)
        d1 = self.dec1(d1 + h1, temb)                 # skip
        return self.out_conv(d1)


# ─────────────────── MERA-structured velocity field ───────────────────


class MERAUNet(nn.Module):
    """Velocity field with MERA-style scale hierarchy.

    log2(L) downsampling stages instead of a fixed 2 — each stage corresponds
    to one physical RG scale (L → L/2 → L/4 → … → 1). This is the "physics-
    aware flow matching" architecture: same CFM loss as VelocityUNet, but
    the network has explicit multi-scale blocks that can be layer-analyzed
    the same way as MERA's RNVP blocks.

    Channel schedule: nhidden × min(2**s, max_channel_mult) at scale s.

    For L=32, n_scales=5:  chs = [64, 128, 256, 256, 256, 256] with max_mult=4
                        → ~9 M params at nhidden=64
    For L=64, n_scales=6:  chs = [64, 128, 256, 256, 256, 256, 256] similar arch
                        → ~10 M params at nhidden=64
    """

    def __init__(self, L, nhidden=64, temb_dim=128, in_channels=1, max_channel_mult=4):
        super().__init__()
        self.L = L
        self.n_scales = int(math.log2(L))
        self.temb_dim = temb_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(temb_dim),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # Channel schedule: cap channel doubling at max_channel_mult × nhidden.
        chs = [nhidden * min(2 ** i, max_channel_mult)
               for i in range(self.n_scales + 1)]

        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # Encoder path: n_scales × (ResBlock + strided-conv downsample)
        self.enc = nn.ModuleList([ResBlock(chs[i], temb_dim) for i in range(self.n_scales)])
        self.down = nn.ModuleList([
            nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1)
            for i in range(self.n_scales)
        ])

        # Bottleneck at coarsest scale (1×1 lattice for L=32, 1×1 for L=64)
        self.mid = ResBlock(chs[self.n_scales], temb_dim)

        # Decoder path: mirror of encoder (with skip connections)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1)
            for i in range(self.n_scales - 1, -1, -1)
        ])
        self.dec = nn.ModuleList([
            ResBlock(chs[i], temb_dim) for i in range(self.n_scales - 1, -1, -1)
        ])

        # Output — zero init so v_t^θ ≈ 0 at start (well-behaved ODE start)
        self.out_conv = nn.Conv2d(chs[0], in_channels, 3, padding=1)
        with torch.no_grad():
            self.out_conv.weight.zero_()
            self.out_conv.bias.zero_()

    def forward(self, x, t):
        temb = self.time_embed(t)                                # (B, temb_dim)
        h = self.in_conv(x)                                       # (B, C0, L, L)

        # Encoder — save skips at each scale
        skips = []
        for s in range(self.n_scales):
            h = self.enc[s](h, temb)
            skips.append(h)
            h = self.down[s](h)                                   # (B, C_{s+1}, L/2^{s+1}, L/2^{s+1})

        # Bottleneck
        h = self.mid(h, temb)                                     # (B, C_max, 1, 1)

        # Decoder — mirror with skips
        for s in range(self.n_scales):
            h = self.up[s](h)
            h = h + skips[self.n_scales - 1 - s]                  # additive skip (like resnet-unet)
            h = self.dec[s](h, temb)

        return self.out_conv(h)


# ─────────────────── True MERA-structured velocity field ───────────────────


class MERAPatchBlock(nn.Module):
    """Small time-modulated MLP for 2×2 patches — one 'MERA block' worth of
    processing. Analogous to an RNVP coupling block but not invertible:
    outputs a delta (velocity contribution) with same shape as input patches.

    FiLM time modulation on the middle hidden layer. Zero-init on output so
    initial velocity ≈ 0.
    """

    def __init__(self, hidden=256, temb_dim=128, in_channels=1, kernel_size=2, n_hidden_layers=2):
        super().__init__()
        core = (kernel_size ** 2) * in_channels                     # 4 for 2×2 single ch
        self.core = core
        # Input projection
        self.in_lin = nn.Linear(core, hidden)
        # Hidden layers with time modulation
        self.mid_lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_hidden_layers)])
        self.temb_projs = nn.ModuleList([nn.Linear(temb_dim, hidden * 2) for _ in range(n_hidden_layers)])
        # Output projection (zero-init for identity-like start)
        self.out_lin = nn.Linear(hidden, core)
        with torch.no_grad():
            self.out_lin.weight.zero_()
            self.out_lin.bias.zero_()

    def forward(self, patches, temb_broadcast):
        """
        patches: (B_p, C, kernel, kernel) — B_p may be batch * num_patches
        temb_broadcast: (B_p, temb_dim) — already expanded to match B_p
        Returns delta patches of same shape as input.
        """
        B_p = patches.shape[0]
        x = patches.reshape(B_p, self.core)
        h = self.in_lin(x)
        h = F.silu(h)
        for mid_lin, temb_proj in zip(self.mid_lins, self.temb_projs):
            gamma, beta = temb_proj(F.silu(temb_broadcast)).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
            h = F.silu(mid_lin(h))
        h = self.out_lin(h)
        return h.reshape(patches.shape)


class TrueMERAVelocityField(nn.Module):
    """True MERA-structured velocity field for Flow Matching.

    Same 2×2 patch decomposition as MERA (dispatch/collect on offset-0 and
    offset-shifted index sets, log2(L) × 2 × nrepeat layers total), but each
    'block' is a small time-modulated MLP producing a velocity delta instead
    of an invertible RNVP transformation.

    Sequential accumulation: h ← h + block_l(dispatch(h, l)), and the
    accumulated deltas are the returned velocity v(x, t).

    Not invertible; not a flow. Just outputs v(x, t) for CFM MSE training.

    Params: depth × (~n_hidden_layers × hidden² + FiLM). L=64 nr=1 with
    hidden=256, n_hidden_layers=2 → depth=12, ~1.5M params (small!).
    Bump hidden or n_hidden_layers to match MERAUNet's 22M if needed.
    """

    def __init__(self, L, nrepeat=1, hidden=256, temb_dim=128,
                 in_channels=1, n_hidden_layers=2):
        super().__init__()
        from .hierarchy.im2col import getIndeices

        self.L = L
        self.n_scales = int(math.log2(L))
        self.nrepeat = nrepeat
        self.depth = 2 * self.n_scales * nrepeat
        self.kernel_size = 2

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(temb_dim),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # Build MERA-style index list (SAME as MERA class)
        # For each scale s and offset o in {0, stride}:
        #   getIndeices(shape=[L,L], height=2, width=2, stride=2×2^s,
        #               dilation=2^s, offset={0 or 2^s})
        kernelSize = 2
        shape = [L, L]
        self._layer_buffer_names = []
        for scale_idx in range(self.n_scales):
            for rep_idx in range(nrepeat):
                for offset in (0, kernelSize ** scale_idx):
                    indexI, indexJ = getIndeices(
                        shape, kernelSize, kernelSize,
                        kernelSize * (kernelSize ** scale_idx),
                        kernelSize ** scale_idx, offset,
                    )
                    ni = f"indexI_s{scale_idx}_r{rep_idx}_o{offset}"
                    nj = f"indexJ_s{scale_idx}_r{rep_idx}_o{offset}"
                    self.register_buffer(ni, torch.as_tensor(indexI, dtype=torch.long))
                    self.register_buffer(nj, torch.as_tensor(indexJ, dtype=torch.long))
                    self._layer_buffer_names.append((ni, nj))

        # One MLP block per layer
        self.blocks = nn.ModuleList([
            MERAPatchBlock(hidden=hidden, temb_dim=temb_dim,
                           in_channels=in_channels, kernel_size=kernelSize,
                           n_hidden_layers=n_hidden_layers)
            for _ in range(self.depth)
        ])

    def _get_indices(self, layer_idx):
        name_i, name_j = self._layer_buffer_names[layer_idx]
        return self._buffers[name_i], self._buffers[name_j]

    def forward(self, x, t):
        """
        x: (B, 1, L, L), t: (B,) ∈ [0,1].
        Returns velocity v: (B, 1, L, L) accumulated over all MERA-layer deltas.
        """
        from .hierarchy.im2col import dispatch, collect

        temb = self.time_embed(t)                                    # (B, temb_dim)
        B = x.shape[0]

        h = x                                                          # current state (sequential accumulation)
        v_total = torch.zeros_like(x)                                  # returned velocity

        for layer_idx in range(self.depth):
            indexI, indexJ = self._get_indices(layer_idx)

            # dispatch: extract patches at layer's index set. dispatch returns
            # (x_unchanged, x_) so we ignore the first return.
            _, patches = dispatch(indexI, indexJ, h)
            # patches shape: (B, C, H_patch, W_patch) with H_patch*W_patch =
            # (L/(stride) × 2 × 2 per patch). But it's laid out as (B, C, N_patches, 2*2 aliased).
            # Actually the raw dispatch output has shape matching the index shape.
            B_p_and_more = patches.shape                              # (B, C, H_idx, W_idx)
            # Reshape into (B * num_patches, C, kernel, kernel) — but the
            # dispatch output isn't laid out that way naturally. Let's do it
            # the way MERA does: reshape to (-1, C, kernel, kernel) after
            # extraction. See template.py:
            #     x_.reshape(-1, channelSize, *self.kernelShape)
            patches_flat = patches.reshape(-1, patches.shape[1],
                                            self.kernel_size, self.kernel_size)
            B_p = patches_flat.shape[0]
            num_patches = B_p // B                                    # patches per sample

            # Broadcast temb to (B_p, temb_dim)
            temb_expanded = temb.repeat_interleave(num_patches, dim=0)

            # Process — returns delta with same shape
            delta_flat = self.blocks[layer_idx](patches_flat, temb_expanded)
            # Reshape back to what collect expects
            delta_patches = delta_flat.reshape(patches.shape)

            # collect: put delta into a zero field at those positions →
            # sparse delta field
            zero_lattice = torch.zeros_like(h)
            delta = collect(indexI, indexJ, zero_lattice, delta_patches)

            # Sequential update + accumulate velocity contribution
            h = h + delta
            v_total = v_total + delta

        return v_total


# ─────────────────── CFM training utilities ───────────────────


def cfm_loss(v_net, x1, sigma_min=1e-4):
    """Conditional Flow Matching loss with OT (linear) probability path.

    x1: (B, 1, L, L) — data samples.
    Path: x_t = (1 − (1 − sigma_min) · t) · x_0 + t · x_1,  x_0 ~ N(0, I).
    Target velocity: u_t(x | x_1) = x_1 − (1 − sigma_min) · x_0.

    In the pure OT limit sigma_min → 0 this is the "rectified flow" objective
    (Liu et al. 2022): straight-line paths, velocity is a constant per sample."""
    B = x1.shape[0]
    device = x1.device
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)                # (B,)
    t_x = t[:, None, None, None]                     # (B, 1, 1, 1)
    x_t = (1.0 - (1.0 - sigma_min) * t_x) * x0 + t_x * x1
    u_target = x1 - (1.0 - sigma_min) * x0
    v_pred = v_net(x_t, t)
    return F.mse_loss(v_pred, u_target)


# ─────────────────── ODE sampling ───────────────────


@torch.no_grad()
def sample_euler(v_net, shape, n_steps=50, device="cuda:0", x0=None):
    """Euler ODE sampling: x_0 ~ N(0, I) → x_1 via forward Euler on v_t^θ."""
    v_net.eval()
    if x0 is None:
        x = torch.randn(*shape, device=device)
    else:
        x = x0.to(device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full((shape[0],), i * dt, device=device)
        v = v_net(x, t_val)
        x = x + v * dt
    return x


@torch.no_grad()
def sample_rk4(v_net, shape, n_steps=50, device="cuda:0", x0=None):
    """Classic RK4 ODE sampling — 4× cost per step, better accuracy."""
    v_net.eval()
    if x0 is None:
        x = torch.randn(*shape, device=device)
    else:
        x = x0.to(device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        t_v = torch.full((shape[0],), t, device=device)
        t_h = torch.full((shape[0],), t + dt / 2, device=device)
        t_e = torch.full((shape[0],), t + dt, device=device)
        k1 = v_net(x, t_v)
        k2 = v_net(x + dt / 2 * k1, t_h)
        k3 = v_net(x + dt / 2 * k2, t_h)
        k4 = v_net(x + dt * k3, t_e)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
