"""
SLERP-based rotation splines for per-Gaussian orientation trajectories.

Mathematical basis:
  Shoemaker, K. (1985). "Animating Rotation with Quaternion Curves."
  SIGGRAPH '85 Proceedings. Public domain mathematics, all patents expired.

  Spherical Linear Interpolation (SLERP) between unit quaternions q0, q1:

    SLERP(q0, q1, t) = q0 · sin((1-t)θ) / sin(θ)  +  q1 · sin(tθ) / sin(θ)

  where θ = arccos(q0 · q1).

  For smooth rotation splines across K keyframes we apply SLERP piecewise
  between adjacent quaternion control points, analogous to how CatmullRomSpline
  handles positions. This gives C0 continuity in rotation (sufficient for
  rendering; C1 requires Squad which is left as a future improvement).

Why this matters for BJJ:
  Wrist supination during an armbar, hip rotation during a sweep, and shoulder
  rotation during a choke all involve non-trivial SO(3) motion. Without rotation
  splines, Gaussians snap between orientations at keyframes rather than rotating
  smoothly — visible as flickering on high-frequency detail like gi fabric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlerpRotationSpline(nn.Module):
    """
    Per-Gaussian piecewise SLERP rotation spline.

    Each Gaussian stores K quaternion control points (unit quaternions).
    At query time t the orientation is found by SLERP between the two
    surrounding keyframe quaternions.

    Quaternion convention: (x, y, z, w) — matches gsplat's convention.

    Args:
        num_gaussians:  N
        num_ctrl_pts:   K (must be ≥ 2)
        num_frames:     T
    """

    def __init__(self, num_gaussians: int, num_ctrl_pts: int, num_frames: int) -> None:
        super().__init__()
        assert num_ctrl_pts >= 2, "Need at least 2 quaternion control points"

        self.num_gaussians = num_gaussians
        self.num_ctrl_pts = num_ctrl_pts
        self.num_frames = num_frames

        # Learnable quaternion control points — shape (N, K, 4)
        # Initialised to identity quaternion (0, 0, 0, 1) = no rotation.
        quats = torch.zeros(num_gaussians, num_ctrl_pts, 4)
        quats[..., 3] = 1.0  # w = 1 → identity
        self.control_quats = nn.Parameter(quats)

    def forward(self, t: Tensor) -> Tensor:
        """
        Evaluate rotation quaternions for all Gaussians at timestamp(s) t.

        Args:
            t: Normalised timestamps in [0, 1], shape (B,).

        Returns:
            rotations: unit quaternions, shape (B, N, 4).
        """
        t = t.view(-1)
        B = t.shape[0]
        N, K, _ = self.control_quats.shape

        # Normalise quaternions to keep them on the unit sphere
        quats = F.normalize(self.control_quats, dim=-1)  # (N, K, 4)

        num_segments = K - 1
        scaled = t * num_segments                             # (B,)
        seg_idx = scaled.long().clamp(0, num_segments - 1)   # (B,)
        u = (scaled - seg_idx.float()).clamp(0.0, 1.0)       # (B,) local t

        # Gather start (q0) and end (q1) quaternions for each segment
        q0 = quats[:, seg_idx, :]   # (N, B, 4)
        q1 = quats[:, seg_idx + 1, :]  # (N, B, 4)

        # Transpose to (B, N, 4) for batched SLERP
        q0 = q0.permute(1, 0, 2)   # (B, N, 4)
        q1 = q1.permute(1, 0, 2)   # (B, N, 4)

        return _slerp(q0, q1, u.view(B, 1, 1))  # (B, N, 4)

    def rotation_at_frame(self, frame: int) -> Tensor:
        """Evaluate at a single integer frame. Returns (N, 4)."""
        t = torch.tensor(
            frame / max(self.num_frames - 1, 1),
            dtype=torch.float32,
            device=self.control_quats.device,
        )
        return self.forward(t.unsqueeze(0)).squeeze(0)


def _slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """
    Batched SLERP between unit quaternions.

    Args:
        q0: shape (..., 4), start quaternions.
        q1: shape (..., 4), end quaternions.
        t:  shape (..., 1) or scalar, interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternions, shape (..., 4).

    Implementation note:
        When θ ≈ 0, sin(θ) ≈ 0 causing division instability. We fall back
        to linear interpolation (LERP) + renormalisation in that region,
        which is numerically identical to SLERP for small angles.
        Threshold: θ < 0.001 rad (≈ 0.057°).
    """
    # Ensure shortest path: flip q1 if dot product is negative
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (..., 1)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))  # (..., 1)
    sin_theta = torch.sin(theta)                     # (..., 1)

    # SLERP coefficients
    coeff0 = torch.sin((1.0 - t) * theta) / sin_theta
    coeff1 = torch.sin(t * theta) / sin_theta

    # Fall back to LERP when sin_theta ≈ 0 (nearly identical quaternions)
    lerp_q = q0 + t * (q1 - q0)
    slerp_q = coeff0 * q0 + coeff1 * q1

    near_zero = (sin_theta < 1e-3).expand_as(q0)
    result = torch.where(near_zero, lerp_q, slerp_q)

    return F.normalize(result, dim=-1)
