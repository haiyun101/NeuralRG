"""Neural Spline Flow coupling layer (rational-quadratic spline).

Drop-in replacement for `RNVP`: same coupling structure (binary mask splits
each patch into two halves; one half is passed through, the other is
transformed conditioned on the first), but the affine map
`y = x * exp(s(x_a)) + t(x_a)` is replaced by a monotone rational-quadratic
spline on `[-bound, bound]`, with identity tails outside.

The conditioner network (one per coupling layer) is supplied externally and
must map a flat input of size `coreSize` -> a flat output of size
`coreSize * (3*K - 1)` where K = num_bins.  Raw outputs split into:

    raw_widths   (K)     -> softmax * 2*bound      -> bin widths summing to 2*bound
    raw_heights  (K)     -> softmax * 2*bound      -> bin heights summing to 2*bound
    raw_derivs   (K-1)   -> softplus + eps         -> internal-knot derivatives
                                                       (boundary derivs fixed = 1)

Both forward and inverse are analytic (the inverse solves a quadratic);
log-det is the sum of per-coordinate log derivatives.

References:
  Durkan et al. 2019, "Neural Spline Flows", https://arxiv.org/abs/1906.04032
"""
import torch
from torch import nn
import torch.nn.functional as F

from .flow import Flow

DEFAULT_MIN_BIN = 1e-3
DEFAULT_MIN_DERIV = 1e-3


def _unconstrained_rqs(x, raw_widths, raw_heights, raw_derivs,
                       bound, inverse=False,
                       min_bin_width=DEFAULT_MIN_BIN,
                       min_bin_height=DEFAULT_MIN_BIN,
                       min_derivative=DEFAULT_MIN_DERIV):
    """Apply RQS where |x| < bound, identity (with log_det=0) where |x| >= bound.

    Shapes:
      x:           (...,)
      raw_widths:  (..., K)
      raw_heights: (..., K)
      raw_derivs:  (..., K-1)
    Returns:
      y:       (...,)
      log_det: (...,)   per-coord log-derivative
    """
    inside_mask = x.abs() < bound
    out_y = x.clone()
    out_log_det = torch.zeros_like(x)
    if inside_mask.any():
        x_in = x[inside_mask]
        rw = raw_widths[inside_mask]
        rh = raw_heights[inside_mask]
        rd = raw_derivs[inside_mask]
        y_in, ld_in = _rqs(x_in, rw, rh, rd, bound, inverse,
                            min_bin_width, min_bin_height, min_derivative)
        out_y[inside_mask] = y_in
        out_log_det[inside_mask] = ld_in
    return out_y, out_log_det


def _rqs(x, raw_widths, raw_heights, raw_derivs, bound, inverse,
         min_bin_width, min_bin_height, min_derivative):
    """Core RQS on guaranteed-inside-bound inputs (1-D in last axis case only)."""
    K = raw_widths.shape[-1]

    # bin widths/heights: softmax * (2B - K*min) + min  -> positive, sum = 2B
    widths  = F.softmax(raw_widths, dim=-1)
    widths  = min_bin_width  + (2 * bound - K * min_bin_width)  * widths
    heights = F.softmax(raw_heights, dim=-1)
    heights = min_bin_height + (2 * bound - K * min_bin_height) * heights

    # internal derivatives, with d_0 = d_K = 1 (smooth identity tail merge)
    derivs_internal = min_derivative + F.softplus(raw_derivs)
    bdry = torch.ones_like(derivs_internal[..., :1])
    derivs = torch.cat([bdry, derivs_internal, bdry], dim=-1)   # (..., K+1)

    # cumulative knots, shifted so the first knot sits at -bound
    cum_w = torch.cumsum(widths,  dim=-1)
    cum_w = F.pad(cum_w, (1, 0)) - bound                        # (..., K+1)
    cum_h = torch.cumsum(heights, dim=-1)
    cum_h = F.pad(cum_h, (1, 0)) - bound                        # (..., K+1)

    # bin index: find k such that cum[k] <= x < cum[k+1]
    if inverse:
        # x is actually y; locate via y-knots
        bin_idx = torch.sum((cum_h <= x.unsqueeze(-1)).long(), dim=-1) - 1
    else:
        bin_idx = torch.sum((cum_w <= x.unsqueeze(-1)).long(), dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def g(t, idx):
        return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    x_k    = g(cum_w, bin_idx)
    y_k    = g(cum_h, bin_idx)
    w_k    = g(widths,  bin_idx)
    h_k    = g(heights, bin_idx)
    d_k    = g(derivs,  bin_idx)
    d_kp1  = g(derivs,  bin_idx + 1)
    s_k    = h_k / w_k

    if not inverse:
        xi = (x - x_k) / w_k
        xi = xi.clamp(0, 1)
        omxi = 1 - xi
        num = h_k * (s_k * xi * xi + d_k * xi * omxi)
        denom = s_k + (d_kp1 + d_k - 2 * s_k) * xi * omxi
        y = y_k + num / denom
        deriv_num = s_k * s_k * (d_kp1 * xi * xi
                                 + 2 * s_k * xi * omxi
                                 + d_k * omxi * omxi)
        log_det = torch.log(deriv_num) - 2 * torch.log(denom)
        return y, log_det
    else:
        # solve quadratic for xi (Durkan eq. (29))
        y = x
        a = h_k * (s_k - d_k) + (y - y_k) * (d_kp1 + d_k - 2 * s_k)
        b = h_k * d_k - (y - y_k) * (d_kp1 + d_k - 2 * s_k)
        c = -s_k * (y - y_k)
        disc = b * b - 4 * a * c
        disc = disc.clamp(min=0)
        # numerically stable root selection
        xi = 2 * c / (-b - torch.sqrt(disc))
        xi = xi.clamp(0, 1)
        omxi = 1 - xi
        x_back = xi * w_k + x_k
        denom = s_k + (d_kp1 + d_k - 2 * s_k) * xi * omxi
        deriv_num = s_k * s_k * (d_kp1 * xi * xi
                                 + 2 * s_k * xi * omxi
                                 + d_k * omxi * omxi)
        # inverse log-det is -log|dy/dx|
        log_det = -(torch.log(deriv_num) - 2 * torch.log(denom))
        return x_back, log_det


class NSFCoupling(Flow):
    """Coupling-layer NSF, same masking convention as `RNVP`.

    Args:
        maskList:    (nlayers, channel, kH, kW) binary mask per coupling
        paramNetList: list of nn.Modules; each maps a flat tensor of size
                      `coreSize` to size `coreSize * (3K-1)`
        num_bins:    K
        bound:       B (transform is identity outside [-B, B])
    """

    def __init__(self, maskList, paramNetList, num_bins=8, bound=5.0,
                 prior=None, name="NSF"):
        super().__init__(prior, name)
        assert len(paramNetList) == maskList.shape[0]
        self.maskList  = nn.Parameter(maskList,    requires_grad=False)
        self.maskListR = nn.Parameter(1 - maskList, requires_grad=False)
        self.paramNets = nn.ModuleList(paramNetList)
        self.K = int(num_bins)
        self.B = float(bound)

    # split raw conditioner output into (widths, heights, derivs) along the
    # spline-parameter axis
    def _split_params(self, raw):
        # raw: (..., coreSize * (3K-1)) -> (..., coreSize, 3K-1)
        shape = raw.shape
        raw = raw.reshape(*shape[:-1], -1, 3 * self.K - 1)
        rw = raw[..., :self.K]
        rh = raw[..., self.K:2 * self.K]
        rd = raw[..., 2 * self.K:]
        return rw, rh, rd

    def inverse(self, z):
        """noise z -> sample x (with forward log-det jacobian)"""
        logjac = z.new_zeros(z.shape[0])
        for i in range(self.maskList.shape[0]):
            m  = self.maskList[i]
            mR = self.maskListR[i]
            z_kept = z * m
            raw = self.paramNets[i](z_kept)              # (B, coreSize*(3K-1))
            rw, rh, rd = self._split_params(raw)         # each (B, coreSize, ...)
            # reshape to match z (B, channel, kH, kW)
            target_shape = z.shape
            rw = rw.reshape(*target_shape, self.K)
            rh = rh.reshape(*target_shape, self.K)
            rd = rd.reshape(*target_shape, self.K - 1)
            y_in, ld = _unconstrained_rqs(z, rw, rh, rd, self.B, inverse=False)
            # only transform the masked-out half; pass through the kept half
            z = z_kept + mR * y_in
            ld = ld * mR
            logjac += ld.reshape(ld.shape[0], -1).sum(dim=-1)
        return z, logjac

    def forward(self, x):
        """sample x -> noise z (with inverse log-det jacobian)"""
        logjac = x.new_zeros(x.shape[0])
        for i in reversed(range(self.maskList.shape[0])):
            m  = self.maskList[i]
            mR = self.maskListR[i]
            x_kept = x * m
            raw = self.paramNets[i](x_kept)
            rw, rh, rd = self._split_params(raw)
            target_shape = x.shape
            rw = rw.reshape(*target_shape, self.K)
            rh = rh.reshape(*target_shape, self.K)
            rd = rd.reshape(*target_shape, self.K - 1)
            y_in, ld = _unconstrained_rqs(x, rw, rh, rd, self.B, inverse=True)
            x = x_kept + mR * y_in
            ld = ld * mR
            logjac += ld.reshape(ld.shape[0], -1).sum(dim=-1)
        return x, logjac
