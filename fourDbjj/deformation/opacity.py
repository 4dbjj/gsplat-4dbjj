"""
Temporal opacity (Gaussian window) for anti-ghosting.

Mathematical basis:
  Standard Gaussian function — public domain mathematics.

  α_i(t) = α_base · exp( -(t - τ_i)² / (2σ_i²) )

  where:
    α_base  — the learnable base opacity (same as static 3DGS opacity)
    τ_i     — peak timestamp: when Gaussian i is most visible
    σ_i     — temporal lifespan: how long Gaussian i contributes

Why for BJJ:
  During a fast sweep, the same region of space contains a leg at t=0 and
  empty space at t=1. Without temporal opacity, Gaussians "ghost" — they
  appear semi-transparent in frames where they shouldn't exist at all.
  The Gaussian window ensures each Gaussian fades in and out naturally,
  acting as a temporal low-pass filter on opacity.

Design decisions:
  - σ_i is learnable per Gaussian (different body parts move at different
    speeds — gi fabric vs. a stationary mat)
  - σ_i is clamped to [σ_min, σ_max] during forward pass to prevent
    training instability from degenerate near-zero or near-infinite lifespans
  - τ_i is initialised to the midpoint of the sequence and learned from data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalOpacity(nn.Module):
    """
    Learnable temporal Gaussian opacity window per Gaussian.

    Args:
        num_gaussians:  N
        num_frames:     T — used to set sensible initialisations
        sigma_min:      Minimum temporal lifespan (prevents Gaussians from
                        becoming instantaneous flashes). Default: 0.05
                        (5% of total sequence duration).
        sigma_max:      Maximum temporal lifespan. Default: 0.5
                        (50% of sequence — Gaussians that span half the clip).
    """

    def __init__(
        self,
        num_gaussians: int,
        num_frames: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # τ_i: peak timestamps, initialised to sequence midpoint (0.5)
        # Shape (N,), normalised to [0, 1]
        self.tau = nn.Parameter(torch.full((num_gaussians,), 0.5))

        # σ_i: temporal lifespan in log-space for stable optimisation
        # log(0.15) ≈ -1.9 → σ ≈ 0.15 (15% of sequence), a reasonable default
        self.log_sigma = nn.Parameter(
            torch.full((num_gaussians,), -1.9)
        )

    @property
    def sigma(self) -> Tensor:
        """Clamped σ values, shape (N,)."""
        return self.log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def forward(self, t: Tensor, base_opacity: Tensor) -> Tensor:
        """
        Compute temporally-modulated opacity for each Gaussian.

        Equation:
            α_i(t) = α_base_i · exp( -(t - τ_i)² / (2σ_i²) )

        Args:
            t:            Normalised timestamp in [0, 1], shape (B,) or scalar.
            base_opacity: Static base opacity from the 3DGS model,
                          shape (N, 1) — same as gsplat's opacity parameter
                          (after sigmoid activation).

        Returns:
            modulated_opacity: shape (B, N, 1) — ready for gsplat rasteriser.
        """
        t = t.view(-1, 1)         # (B, 1)
        tau = self.tau.unsqueeze(0)    # (1, N)
        sigma = self.sigma.unsqueeze(0)  # (1, N)

        # Gaussian window: shape (B, N)
        window = torch.exp(-((t - tau) ** 2) / (2 * sigma ** 2))

        # base_opacity shape: (N, 1) → (1, N, 1) for broadcasting
        alpha_base = base_opacity.T.unsqueeze(0)  # (1, 1, N) → needs reshape
        alpha_base = base_opacity.squeeze(-1).unsqueeze(0)  # (1, N)

        modulated = alpha_base * window  # (B, N)

        return modulated.unsqueeze(-1)  # (B, N, 1)

    def at_frame(self, frame: int, base_opacity: Tensor) -> Tensor:
        """
        Convenience wrapper for single-frame evaluation.

        Returns:
            modulated_opacity: shape (N, 1).
        """
        t = torch.tensor(
            frame / max(self.num_frames - 1, 1),
            dtype=torch.float32,
            device=base_opacity.device,
        )
        return self.forward(t.unsqueeze(0), base_opacity).squeeze(0)  # (N, 1)
