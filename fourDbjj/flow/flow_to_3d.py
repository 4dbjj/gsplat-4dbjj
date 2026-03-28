"""
Project dense 2D optical flow + depth map → per-Gaussian 3D velocity.

Mathematical basis:
  Standard perspective camera model — textbook computer vision, public domain.

  Given a pixel (u, v) with depth d, the 3D point is:

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

  If the pixel moves by (du, dv) between frames, the displaced 3D point is:

    X' = (u + du - cx) * d' / fx
    Y' = (v + dv - cy) * d' / fy
    Z' = d'

  We assume d' ≈ d (depth is approximately constant over one frame interval),
  giving the approximate 3D velocity:

    vX = du * d / fx
    vY = dv * d / fy
    vZ = 0   (depth velocity not recoverable from monocular flow alone)

  For each Gaussian i at 3D position μ_i, we find the pixel (u_i, v_i) by
  projecting μ_i into camera space, then look up the flow (du, dv) at that
  pixel to compute the Gaussian's 3D velocity.

Limitations:
  - Depth is estimated from a monocular depth model (e.g. ZoeDepth or DPT),
    which introduces scale ambiguity. The velocity is therefore in an
    arbitrary metric scale. Control point initialisation normalises by the
    mean depth so relative velocities are correct.
  - vZ is not estimated. This is acceptable for initialisation — the training
    loop learns the full 3D trajectory via the flow + rendering losses.

Reference:
  Hartley, R. & Zisserman, A. (2003). "Multiple View Geometry in Computer
  Vision." 2nd ed., Cambridge University Press. Chapter 6 (camera model).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def project_gaussians_to_pixels(
    means3d: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor]:
    """
    Project 3D Gaussian centres into pixel coordinates.

    Args:
        means3d:      (N, 3) Gaussian centres in world space.
        viewmat:      (4, 4) world-to-camera matrix (column-major, same as
                      gsplat's convention: viewmat @ [X;Y;Z;1] → camera space).
        fx, fy:       Focal lengths in pixels.
        cx, cy:       Principal point in pixels.
        image_height: H
        image_width:  W

    Returns:
        pixels:     (N, 2) pixel coordinates (u, v), float32.
        in_frustum: (N,) bool mask — True if the Gaussian projects inside
                    the image (including a 1-pixel margin).
    """
    N = means3d.shape[0]
    device = means3d.device

    # Homogeneous world coords: (N, 4)
    ones = torch.ones(N, 1, device=device, dtype=means3d.dtype)
    pts_w = torch.cat([means3d, ones], dim=1)  # (N, 4)

    # Transform to camera space: (N, 4)
    pts_c = (viewmat @ pts_w.T).T  # (N, 4)
    X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # Avoid division by zero for points behind the camera
    valid_z = Z > 1e-6

    u = torch.where(valid_z, fx * X / Z.clamp(min=1e-6) + cx, torch.zeros_like(X))
    v = torch.where(valid_z, fy * Y / Z.clamp(min=1e-6) + cy, torch.zeros_like(Y))

    # Frustum check (1-pixel margin)
    in_frustum = (
        valid_z
        & (u >= 1) & (u < image_width - 1)
        & (v >= 1) & (v < image_height - 1)
    )

    return torch.stack([u, v], dim=1), in_frustum


def flow_to_3d_velocity(
    means3d: Tensor,
    flow: Tensor,
    depth: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tensor:
    """
    Estimate 3D velocity for each Gaussian from dense 2D optical flow.

    For each Gaussian, we:
      1. Project the Gaussian centre to pixel (u, v)
      2. Sample the flow field at (u, v) to get (du, dv)
      3. Sample depth at (u, v) to get d
      4. Compute 3D velocity in camera space:
             vX = du * d / fx
             vY = dv * d / fy
             vZ = 0
      5. Rotate vX, vY, vZ back to world space using the camera rotation

    Args:
        means3d:  (N, 3) Gaussian centres in world space.
        flow:     (H, W, 2) optical flow (du, dv) in pixels.
        depth:    (H, W) depth map in the same metric as means3d.
                  If monocular depth, scale is arbitrary (velocities are
                  used only for relative initialisation).
        viewmat:  (4, 4) world-to-camera matrix.
        fx, fy:   Focal lengths in pixels.
        cx, cy:   Principal point in pixels.

    Returns:
        velocities: (N, 3) approximate 3D velocity per Gaussian in world space.
                    Gaussians outside the frustum get zero velocity.
    """
    H, W, _ = flow.shape
    device = means3d.device
    dtype = means3d.dtype

    flow = flow.to(device=device, dtype=dtype)
    depth = depth.to(device=device, dtype=dtype)

    # Project Gaussians to pixel space
    pixels, in_frustum = project_gaussians_to_pixels(
        means3d, viewmat, fx, fy, cx, cy, H, W
    )
    # pixels: (N, 2), values in pixel units

    # Normalise pixel coords to [-1, 1] for grid_sample
    # grid_sample expects (x, y) order where x is along W, y along H
    grid_x = (pixels[:, 0] / (W - 1)) * 2 - 1  # u → [-1, 1]
    grid_y = (pixels[:, 1] / (H - 1)) * 2 - 1  # v → [-1, 1]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)          # (1, N, 1, 2)

    # Sample optical flow at each Gaussian's projected location
    # flow: (H, W, 2) → (1, 2, H, W) for grid_sample
    flow_chw = flow.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    sampled_flow = F.grid_sample(
        flow_chw, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    # sampled_flow: (1, 2, N, 1) → (N, 2)
    sampled_flow = sampled_flow.squeeze(0).squeeze(-1).T  # (N, 2)
    du = sampled_flow[:, 0]  # (N,)
    dv = sampled_flow[:, 1]  # (N,)

    # Sample depth at each Gaussian's projected location
    depth_chw = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled_depth = F.grid_sample(
        depth_chw, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    d = sampled_depth.squeeze()  # (N,)

    # 3D velocity in camera space
    vX_cam = du * d / fx
    vY_cam = dv * d / fy
    vZ_cam = torch.zeros_like(vX_cam)

    vel_cam = torch.stack([vX_cam, vY_cam, vZ_cam], dim=1)  # (N, 3)

    # Rotate velocity back to world space
    # viewmat = [R | t] where R is (3, 3) rotation, t is (3,) translation
    # world-to-camera: p_c = R @ p_w + t
    # camera-to-world rotation: R^T
    R_wc = viewmat[:3, :3].T  # (3, 3) — camera-to-world rotation
    vel_world = (R_wc @ vel_cam.T).T  # (N, 3)

    # Zero out velocities for Gaussians outside the frustum
    vel_world = vel_world * in_frustum.float().unsqueeze(1)

    return vel_world  # (N, 3)


def estimate_depth_from_monocular(
    frame: Tensor,
    model_type: str = "ZoeD_N",
) -> Tensor:
    """
    Estimate a per-pixel depth map from a single RGB frame using ZoeDepth.

    ZoeDepth is MIT-licensed. Reference:
      Bhat, S.F. et al. (2023). "ZoeDepth: Zero-shot Transfer by Combining
      Relative and Metric Depth." arXiv:2302.12288. MIT License.

    Args:
        frame:      (H, W, 3) uint8 or float32 RGB frame.
        model_type: ZoeDepth model variant. "ZoeD_N" (NYU indoor, typical BJJ gym).

    Returns:
        depth: (H, W) float32 depth in arbitrary metric units.

    Note:
        This downloads ~350MB of weights on first run. In the pipeline,
        call this once per camera per video clip (not per frame).
        The depth is assumed approximately constant across frames for the
        purpose of flow-guided initialisation.
    """
    try:
        import torch.hub
    except ImportError:
        raise ImportError("torch.hub required — included with PyTorch")

    if frame.dtype == torch.uint8:
        frame_float = frame.float() / 255.0
    else:
        frame_float = frame

    # (H, W, 3) → (1, 3, H, W)
    img = frame_float.permute(2, 0, 1).unsqueeze(0)

    model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    with torch.no_grad():
        depth = model.infer(img)  # (1, 1, H, W)

    return depth.squeeze().cpu()  # (H, W)
