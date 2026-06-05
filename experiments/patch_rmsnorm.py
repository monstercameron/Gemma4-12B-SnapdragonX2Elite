"""Numerically-safe RMSNorm for fp16-on-HTP: pre-scale by S before squaring so x.pow(2)
never overflows fp16 (Gemma activations exceed 256 => 256^2 > 65504 overflows). Mathematically
identical to the original (RMSNorm is scale-invariant; eps rescaled exactly).

  original: x * rsqrt(mean(x^2) + eps)
  safe:     xs = x/S ;  x_norm = xs * rsqrt(mean(xs^2) + eps/S^2)
            = (x/S) / sqrt(mean(x^2)/S^2 + eps/S^2) = x / sqrt(mean(x^2) + eps)   [exact]
"""
import torch


def apply(M):
    """Patch Gemma4UnifiedRMSNorm._norm to a DYNAMIC per-row-max-scaled formulation.

    A fixed pre-scale (e.g. /256) trades overflow for underflow: small activations square
    below fp16's smallest normal (~6e-5) and vanish. Dividing by the per-row max instead
    keeps the scaled values in [-1, 1] (squares in [0, 1]) -> neither overflow nor underflow,
    for ANY magnitude. Exactly equivalent (eps rescaled per-row):
        x * rsqrt(mean(x^2)+eps) = xs * rsqrt(mean(xs^2)+eps/m^2),  xs = x/m
    """
    def _safe_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # divide by per-row max -> scaled row has max |.|=1, so mean(xs^2) in [1/dim, 1]:
        # never near zero, so the original +eps (1e-6) is negligible and dropped (avoids the
        # m*m fp16 overflow). Result == x / sqrt(mean(x^2)).
        m = hidden_states.abs().amax(dim=-1, keepdim=True) + 1e-12
        xs = hidden_states / m
        mean_sq = xs.pow(2).mean(-1, keepdim=True)
        return xs * torch.pow(mean_sq, -0.5)
    M.Gemma4UnifiedRMSNorm._norm = _safe_norm
    return M
