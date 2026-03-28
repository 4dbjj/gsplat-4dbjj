"""
Catmull-Rom spline evaluation for per-Gaussian trajectory motion.

Mathematical basis:
  Catmull-Rom (1974) — public domain, no patent risk.

  Given 4 control points P0, P1, P2, P3 and local parameter u ∈ [0, 1],
  the position on the spline segment between P1 and P2 is:

    q(u) = 0.5 * [1, u, u², u³] · M_CR · [P0, P1, P2, P3]ᵀ

  where the Catmull-Rom basis matrix M_CR is:

        ⎡  0   2   0   0 ⎤
        ⎢ -1   0   1   0 ⎥
  0.5 × ⎢  2  -5   4  -1 ⎥
        ⎣ -1   3  -3   1 ⎦

  This gives C1 continuity (smooth velocity) and passes through every
  control point — critical for BJJ so rendered Gaussians sit exactly at
  their observed positions at each keyframe.

Reference: Catmull, E. & Rom, R. (1974). "A class of local interpolating splines."
  Computer Aided Geometric Design, pp. 317-326.
"""

import torch
import torch.nn as nn
from torch import Tensor


# Catmull-Rom basis matrix (equation above), shape (4, 4)
# Registered as a buffer so it moves to GPU automatically.
_CR_MATRIX = 0.5 * torch.tensor(
    [
        [ 0,  2,  0,  0],
        [-1,  0,  1,  0],
        [ 2, -5,  4, -1],
        [-1,  3, -3,  1],
    ],
    dtype=torch.float32,
)


class CatmullRomSpline(nn.Module):
    """
    Learnable per-Gaussian Catmull-Rom spline trajectories.

    Each Gaussian stores K control points in 3D space. At query time t,
    the position is evaluated as a smooth curve through the control points.

    Args:
        num_gaussians:  Number of Gaussians (N).
        num_ctrl_pts:   Number of control points per Gaussian (K ≥ 4).
        num_frames:     Total number of frames in the sequence (T).
    """

    def __init__(self, num_gaussians: int, num_ctrl_pts: int, num_frames: int) -> None:
        super().__init__()
        assert num_ctrl_pts >= 4, "Catmull-Rom requires at least 4 control points"

        self.num_gaussians = num_gaussians
        self.num_ctrl_pts = num_ctrl_pts
        self.num_frames = num_frames

        # Learnable control points — shape (N, K, 3)
        # Initialised near zero; the training loop offsets these from the
        # canonical 3DGS positions via flow-guided initialisation (Phase 3).
        self.control_points = nn.Parameter(
            torch.zeros(num_gaussians, num_ctrl_pts, 3)
        )

        # Register the basis matrix as a non-trainable buffer
        self.register_buffer("cr_matrix", _CR_MATRIX)

    def forward(self, t: Tensor) -> Tensor:
        """
        Evaluate spline positions for all Gaussians at timestamp(s) t.

        Args:
            t: Normalised timestamps in [0, 1], shape (B,) for a batch
               of B query times, or scalar for a single time.

        Returns:
            positions: shape (B, N, 3) — 3D position of each Gaussian
                       at each queried timestamp.
        """
        t = t.view(-1)  # ensure 1-D: (B,)
        B = t.shape[0]
        N, K, _ = self.control_points.shape

        # Map t ∈ [0,1] to segment index and local parameter u ∈ [0,1]
        # There are (K-1) segments between K control points, but Catmull-Rom
        # needs a phantom point on each end, so usable segments = K-3.
        num_segments = K - 3  # segments between indices 1..K-2
        assert num_segments >= 1, "Need at least 4 control points for 1 segment"

        scaled = t * num_segments                        # (B,)
        seg_idx = scaled.long().clamp(0, num_segments - 1)  # (B,) segment index
        u = (scaled - seg_idx.float()).clamp(0.0, 1.0)  # (B,) local param

        # Gather the 4 control points for each segment: P_{i-1}, Pi, P_{i+1}, P_{i+2}
        # indices into control_points for the 4 neighbours
        i0 = seg_idx          # (B,) — phantom start point index
        i1 = (seg_idx + 1)
        i2 = (seg_idx + 2)
        i3 = (seg_idx + 3).clamp(max=K - 1)

        P = self.control_points  # (N, K, 3)

        # Stack 4 control points → (B, N, 4, 3)
        pts = torch.stack(
            [P[:, i0, :], P[:, i1, :], P[:, i2, :], P[:, i3, :]],
            dim=2,
        ).permute(1, 0, 2, 3)  # (B, N, 4, 3)

        # Build polynomial basis vector [1, u, u², u³] — shape (B, 4)
        u2 = u ** 2
        u3 = u ** 3
        basis = torch.stack([torch.ones_like(u), u, u2, u3], dim=-1)  # (B, 4)

        # Apply Catmull-Rom matrix: (B, 4) × (4, 4) → (B, 4)
        cr_basis = basis @ self.cr_matrix  # (B, 4)

        # Contract with control points: (B, 4) · (B, N, 4, 3) → (B, N, 3)
        positions = torch.einsum("bk,bnk d->bnd", cr_basis, pts)

        return positions  # (B, N, 3)

    def position_at_frame(self, frame: int) -> Tensor:
        """
        Convenience: evaluate at a single integer frame index.

        Args:
            frame: Integer frame index in [0, num_frames-1].

        Returns:
            positions: shape (N, 3).
        """
        t = torch.tensor(
            frame / max(self.num_frames - 1, 1),
            dtype=torch.float32,
            device=self.control_points.device,
        )
        return self.forward(t.unsqueeze(0)).squeeze(0)  # (N, 3)
