import math

import torch
from torch import nn
from torch.distributions import StudentT as _StudentT

from .source import Source


class StudentT(Source):
    """Diagonal Student-t prior with marginal variance fixed to 1.

    Scheme I.1 from ``analyzers/rg_fixed_point/improvements_zh.md``: the
    cheapest "non-Gaussian prior" experiment, useful as a *negation*
    test for the Wilson-Gaussian-FP-geometry critique in
    ``rg_fixed_point_report_zh.md``. Student-t has heavier tails than
    Gaussian but still factorises across sites (``P(z) = ∏_i p_t(z_i)``),
    so it can shift V5 marginal KS without touching long-range spatial
    structure. If a Student-t prior moves V5 KS but leaves V5 RMS-G
    unchanged, the marginal-shape vs spatial-structure decomposition is
    confirmed and the path forward is a *structured* prior (scheme A /
    I.2 conditional, B / I.3 EBM, C / II.2 self-similarity), not just
    heavier tails.

    Parameterisation: we scale the standard t(df) by
    ``sigma = sqrt((df - 2) / df)`` so the marginal variance equals 1
    (matching the isotropic ``Gaussian`` baseline for like-for-like
    comparison). Requires ``df > 2`` for the variance to be finite.

    For df → ∞ the distribution collapses to N(0, 1) up to scale, so
    df is the knob that controls how heavy the tails are at fixed
    variance (df=4: kurtosis = 3 + 6/(df-4)/... infinite for df<=4
    — actually df=4 has infinite excess kurtosis at the boundary;
    pick df=5 if you want bounded kurtosis. Default 4.0 keeps fat
    tails while staying within the finite-variance regime).
    """

    def __init__(self, nvars, df=4.0, name="student_t"):
        super().__init__(nvars, name)
        if df <= 2.0:
            raise ValueError(
                f"StudentT prior needs df > 2 for finite variance; got df={df}"
            )
        self.df = float(df)
        # Buffer follows .to(device, dtype) automatically.
        self.register_buffer(
            "sigma",
            torch.tensor(math.sqrt((df - 2.0) / df), dtype=torch.float32),
        )
        # Sentinel parameter so device/dtype propagate when no buffer
        # has been touched yet (mirrors the Gaussian source convention).
        self.register_parameter(
            "_anchor",
            nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
        )

    def _dist(self, ref):
        """Build a StudentT distribution object on the same device/dtype as ref."""
        return _StudentT(df=torch.tensor(self.df, dtype=ref.dtype, device=ref.device))

    def energy(self, z):
        """Negative log p(z) summed over the lattice (per-sample).

        z = sigma * T, T ~ t(df), so
            log p(z) = log p_T(z / sigma) - log(sigma)
        per dimension. Sum over the lattice (B, C, L, L) into B scalars.
        """
        B = z.shape[0]
        sigma = self.sigma.to(z)
        log_sigma = torch.log(sigma)
        dist = self._dist(z)
        log_pz = dist.log_prob(z / sigma) - log_sigma
        return -log_pz.reshape(B, -1).sum(dim=1)

    def sample(self, batchSize):
        """Sample z ~ sigma * t(df) on the prior's device/dtype."""
        sigma = self.sigma
        dist = self._dist(sigma)
        size = [batchSize] + list(self.nvars)
        # StudentT.sample returns a tensor on the same device as the df arg.
        # rsample would be needed for backprop through the sample; not
        # required here because the prior sample feeds inverse(), and
        # the gradients we need flow through inverse, not the noise draw.
        with torch.no_grad():
            t = dist.sample(torch.Size(size))
        return (sigma * t).to(self._anchor)
