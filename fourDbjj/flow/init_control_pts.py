"""
Flow-guided initialisation of Catmull-Rom spline control points.

Design rationale:
  Without initialisation, all control points start at zero (CatmullRomSpline
  default) so the network must learn motion from scratch via rendering loss
  alone — slow and prone to local minima, especially for fast BJJ motion.

  Instead we use RAFT optical flow + depth to estimate per-Gaussian 3D
  velocities at several anchor frames, then seed the spline control points
  so the model already approximates the observed motion before any
  gradient steps.

Initialisation strategy for a Gaussian at canonical position μ with
observed velocity v at normalised time t_anchor:

  For K control points uniformly spaced in [0, 1], and assuming approximately
  constant velocity over the clip (reasonable for a ~2s BJJ action), the
  control points are placed along the velocity direction:

    P_k = μ  +  (t_k - t_anchor) * v * T_seconds

  where T_seconds is the clip duration in seconds (e.g. 2.0 s for a 60fps clip).

  This means:
    - The spline passes through μ at the anchor frame
    - Adjacent control points are offset by v × Δt (where Δt = 1/(K-1) × T_sec)
    - Velocities of 0 m/s → all control points at μ (static Gaussian)

  After training starts, the optimiser refines these points to match the full
  non-linear trajectory. This initialisation just gives it a head start by
  pointing in roughly the right direction.

For multi-frame velocity averaging:
  We compute velocities at multiple anchor frames (e.g. 10 evenly spaced frames
  across the clip) and use their median to initialise. The median is more robust
  than the mean to outlier flows (occlusions, depth errors, motion blur).
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Optional

from fourDbjj.deformation.spline import CatmullRomSpline
from fourDbjj.flow.flow_to_3d import flow_to_3d_velocity


def init_control_points_from_flow(
    spline: CatmullRomSpline,
    means3d: Tensor,
    flows: List[Tensor],
    depths: List[Tensor],
    viewmats: List[Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fps: float = 60.0,
    anchor_frame_indices: Optional[List[int]] = None,
) -> None:
    """
    Initialise CatmullRomSpline control points from optical flow.

    This function modifies `spline.control_points` **in-place** (via
    `torch.no_grad()` data assignment). Call this before the training loop.

    Args:
        spline:               CatmullRomSpline to initialise.
        means3d:              (N, 3) canonical Gaussian positions (phase 1
                              3DGS means, in world space).
        flows:                List of T-1 flow tensors, each (H, W, 2).
                              flows[i] is the flow from frame i to frame i+1.
                              Produced by RaftEstimator.estimate_sequence().
        depths:               List of T depth maps, each (H, W). Either from
                              a monocular depth model (one depth per keyframe)
                              or repeated if depth is estimated once.
        viewmats:             List of T (4, 4) world-to-camera matrices,
                              one per frame.
        fx, fy:               Focal lengths in pixels.
        cx, cy:               Principal point in pixels.
        fps:                  Frames per second of the input video.
        anchor_frame_indices: Which frames to use for velocity estimation.
                              If None, uses up to 10 evenly spaced frames.

    Returns:
        None — modifies spline.control_points in-place.
    """
    T = len(depths)  # total frames
    N = means3d.shape[0]
    K = spline.num_ctrl_pts
    device = means3d.device

    if anchor_frame_indices is None:
        # Up to 10 evenly spaced anchor frames (excluding last — no flow at T-1)
        num_anchors = min(10, T - 1)
        anchor_frame_indices = [
            int(i * (T - 2) / max(num_anchors - 1, 1))
            for i in range(num_anchors)
        ]

    # Estimate 3D velocity at each anchor frame
    velocities_list = []  # each entry: (N, 3)

    for frame_idx in anchor_frame_indices:
        if frame_idx >= len(flows):
            continue
        flow = flows[frame_idx]       # (H, W, 2)
        depth = depths[frame_idx]     # (H, W)
        viewmat = viewmats[frame_idx].to(device=device, dtype=means3d.dtype)

        vel = flow_to_3d_velocity(
            means3d, flow, depth, viewmat, fx, fy, cx, cy
        )  # (N, 3), in world space, units: depth_units / pixel × pixels_per_frame

        # Convert from "depth units per frame" to "depth units per second"
        vel_per_second = vel * fps

        velocities_list.append(vel_per_second)

    if not velocities_list:
        # No valid anchor frames — leave control points at zero (static)
        return

    # Robust velocity estimate: median across anchor frames
    velocities_stacked = torch.stack(velocities_list, dim=0)  # (A, N, 3)
    median_velocity, _ = velocities_stacked.median(dim=0)     # (N, 3)

    # Clip duration in seconds
    T_seconds = (T - 1) / fps

    # Compute control point positions
    # Anchor is taken as the temporal midpoint (t_anchor = 0.5)
    t_anchor = 0.5

    # Normalised times for each control point: k / (K-1)
    t_ctrl = torch.linspace(0.0, 1.0, K, device=device, dtype=means3d.dtype)  # (K,)

    # Offset from midpoint: (t_k - t_anchor) × T_seconds
    dt = (t_ctrl - t_anchor) * T_seconds  # (K,)

    # Position offsets: (K,) × (N, 3) → (N, K, 3)
    # dt: (K,) unsqueezed to (1, K, 1)
    # median_velocity: (N, 3) unsqueezed to (N, 1, 3)
    offsets = dt.view(1, K, 1) * median_velocity.unsqueeze(1)  # (N, K, 3)

    # Control points = canonical position + offset
    # means3d: (N, 3) unsqueezed to (N, 1, 3)
    new_ctrl_pts = means3d.unsqueeze(1) + offsets  # (N, K, 3)

    # Write to parameter (no gradient tracking needed — this is initialisation)
    with torch.no_grad():
        spline.control_points.copy_(new_ctrl_pts)


def compute_flow_coverage(
    means3d: Tensor,
    flows: List[Tensor],
    depths: List[Tensor],
    viewmats: List[Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict:
    """
    Diagnostic: report what fraction of Gaussians have valid flow coverage.

    Returns a dict with:
        in_frustum_fraction: fraction of Gaussians visible in at least one frame
        mean_velocity_magnitude: mean 3D speed across all Gaussians (m/s or
                                 depth-units/s — whatever units depth is in)
        max_velocity_magnitude: maximum speed (useful for detecting outliers)
    """
    from fourDbjj.flow.flow_to_3d import project_gaussians_to_pixels

    T = len(depths)
    N = means3d.shape[0]
    device = means3d.device
    dtype = means3d.dtype

    ever_visible = torch.zeros(N, dtype=torch.bool, device=device)
    all_velocities = []

    num_anchors = min(5, T - 1)
    anchor_indices = [int(i * (T - 2) / max(num_anchors - 1, 1)) for i in range(num_anchors)]

    for fi in anchor_indices:
        if fi >= len(flows):
            continue
        vm = viewmats[fi].to(device=device, dtype=dtype)
        H, W = flows[fi].shape[:2]
        _, in_frustum = project_gaussians_to_pixels(
            means3d, vm, fx, fy, cx, cy, H, W
        )
        ever_visible |= in_frustum

        vel = flow_to_3d_velocity(
            means3d, flows[fi], depths[fi], vm, fx, fy, cx, cy
        )
        all_velocities.append(vel[in_frustum])

    if all_velocities:
        all_vel = torch.cat(all_velocities, dim=0)
        magnitudes = all_vel.norm(dim=-1)
        mean_mag = magnitudes.mean().item()
        max_mag = magnitudes.max().item()
    else:
        mean_mag = 0.0
        max_mag = 0.0

    return {
        "in_frustum_fraction": ever_visible.float().mean().item(),
        "mean_velocity_magnitude": mean_mag,
        "max_velocity_magnitude": max_mag,
    }
