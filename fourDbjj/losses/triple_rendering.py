"""
Triple-Rendering Loss for 4DBJJ dynamic Gaussian training.

Mathematical basis:

  L_total = L_RGB(t) + λ_flow · L_Flow(t → t+1) + λ_triple · L_Triple(t)

  ──────────────────────────────────────────────────────────────────────────
  1. L_RGB(t) — standard photometric reconstruction loss
  ──────────────────────────────────────────────────────────────────────────

    L_RGB(t) = (1 - λ_ssim) · L1(render_t, gt_t)  +  λ_ssim · L_DSSIM(render_t, gt_t)

    Same as vanilla 3DGS. λ_ssim = 0.2 (paper default).

  ──────────────────────────────────────────────────────────────────────────
  2. L_Flow(t → t+1) — optical flow consistency loss
  ──────────────────────────────────────────────────────────────────────────

    For each pixel (u, v) in the rendered frame at time t, the rendered flow
    is computed as the 2D displacement of the dominant Gaussian between
    frames t and t+1:

        flow_rendered(u, v) = project(μ(t+1)) - project(μ(t))

    We compare this to the RAFT-estimated flow:

        L_Flow = mean( ||flow_rendered - flow_raft||₁ )

    Using the Gaussians' 2D projections from gsplat's meta dict
    (meta["means2d"]) for both frames.

    This loss ensures the spline trajectories produce pixel-level motion
    consistent with what RAFT observed — preventing the model from learning
    a trajectory that looks photometrically correct but is physically wrong
    (e.g., two Gaussians swapping roles).

  ──────────────────────────────────────────────────────────────────────────
  3. L_Triple(t) — temporal smoothness loss
  ──────────────────────────────────────────────────────────────────────────

    Renders three timestamps: t-δ, t, t+δ where δ = 1/(T-1) (one frame gap).
    Enforces that the middle frame is consistent with the average of its
    temporal neighbours:

        L_Triple = mean( ||render(t) - 0.5·(render(t-δ) + render(t+δ))||₁ )

    This is a temporal version of the 3DGS densification smoothness prior.
    It penalises temporal flickering (high-frequency oscillations in the
    rendered video) without requiring ground-truth labels at every frame.

    For BJJ, this prevents Gaussians from jumping spatially between near-
    identical frames (a local minimum the flow loss alone cannot prevent).

Design decisions:
  - L_Flow uses L1 (not L2) — flow has some outliers (occlusions, depth
    errors) and L1 is more robust to them.
  - L_Triple uses L1 for the same reason.
  - λ_flow = 0.1, λ_triple = 0.05 — empirically start conservative; the
    training loop can anneal these.
  - L_Triple is computed on full renders (all pixels), not just Gaussian
    projections, so it captures secondary effects (shadows, reflections).

Reference:
  Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance
  Field Rendering." SIGGRAPH 2023 — RGB loss formulation.

  Wang, Z. et al. (2004). "Image Quality Assessment: From Error Visibility
  to Structural Similarity." IEEE TIP — SSIM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Component losses
# ────────────────────────────────────────────────────────────────────────────

def l1_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Per-pixel L1 loss, averaged over all pixels and channels."""
    return F.l1_loss(predicted, target)


def ssim_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """
    DSSIM loss: (1 - SSIM) / 2.

    Uses a simplified 11×11 Gaussian-weighted SSIM, matching the 3DGS paper.
    Input: (H, W, C) float in [0, 1]. Returns a scalar.
    """
    # Convert to (1, C, H, W) for F.conv2d
    pred = predicted.permute(2, 0, 1).unsqueeze(0)
    gt   = target.permute(2, 0, 1).unsqueeze(0)

    C, H, W = pred.shape[1], pred.shape[2], pred.shape[3]

    # Build 11×11 Gaussian kernel
    kernel = _gaussian_kernel_2d(11, 1.5, device=pred.device).expand(C, 1, 11, 11)

    mu1 = F.conv2d(pred, kernel, padding=5, groups=C)
    mu2 = F.conv2d(gt,   kernel, padding=5, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, kernel, padding=5, groups=C) - mu1_sq
    sig2_sq = F.conv2d(gt   * gt,   kernel, padding=5, groups=C) - mu2_sq
    sig12   = F.conv2d(pred * gt,   kernel, padding=5, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = (
        (2 * mu12 + C1) * (2 * sig12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    )

    return (1.0 - ssim_map.mean()) / 2.0


def _gaussian_kernel_2d(size: int, sigma: float, device: torch.device) -> Tensor:
    """1-channel 2D Gaussian kernel, shape (1, 1, size, size)."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d)
    return g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


# ────────────────────────────────────────────────────────────────────────────
# Flow consistency loss
# ────────────────────────────────────────────────────────────────────────────

def flow_consistency_loss(
    means2d_t: Tensor,
    means2d_t1: Tensor,
    flow_raft: Tensor,
    gaussian_ids: Optional[Tensor] = None,
) -> Tensor:
    """
    Compare rendered flow (from Gaussian 2D projection shift) to RAFT flow.

    Args:
        means2d_t:   (N, 2) projected 2D centres of all Gaussians at time t,
                     in pixel coordinates. From meta["means2d"].squeeze() after
                     gsplat.rasterization() with packed=False.
        means2d_t1:  (N, 2) projected 2D centres at time t+1.
        flow_raft:   (H, W, 2) RAFT flow field (du, dv) from frame t to t+1.
        gaussian_ids: (M,) optional subset of Gaussian indices to include.
                     If None, uses all N Gaussians.

    Returns:
        loss: scalar L1 flow consistency loss.
    """
    if gaussian_ids is not None:
        means2d_t  = means2d_t[gaussian_ids]
        means2d_t1 = means2d_t1[gaussian_ids]

    # Rendered flow per Gaussian: shape (N, 2)
    rendered_flow = means2d_t1 - means2d_t

    H, W, _ = flow_raft.shape
    device = means2d_t.device

    # Sample RAFT flow at each Gaussian's t-position using bilinear interpolation
    # Normalise to [-1, 1] for grid_sample
    u = means2d_t[:, 0]
    v = means2d_t[:, 1]
    grid_x = (u / (W - 1)) * 2 - 1
    grid_y = (v / (H - 1)) * 2 - 1

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)           # (1, N, 1, 2)

    flow_chw = flow_raft.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 2, H, W)
    sampled = F.grid_sample(
        flow_chw, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    raft_flow_at_gaussians = sampled.squeeze(0).squeeze(-1).T  # (N, 2)

    # Normalise both flows by the image's long edge so the loss is dimensionless
    # (range 0–1 per axis) and comparable to the RGB L1 loss.  Without this,
    # pixel-space values of 10–115 px overwhelm the RGB loss (~0.03) even at
    # lambda_flow=0.01.  After normalisation, lambda_flow=0.1 is appropriate.
    scale = float(max(H, W))
    return F.l1_loss(rendered_flow / scale, raft_flow_at_gaussians / scale)


# ────────────────────────────────────────────────────────────────────────────
# Triple-rendering temporal smoothness loss
# ────────────────────────────────────────────────────────────────────────────

def triple_rendering_loss(
    render_t_minus: Tensor,
    render_t: Tensor,
    render_t_plus: Tensor,
) -> Tensor:
    """
    Temporal smoothness: centre frame should approximate the mean of neighbours.

        L_Triple = mean( ||render(t) - 0.5·(render(t-δ) + render(t+δ))||₁ )

    Args:
        render_t_minus: (H, W, C) rendered image at t - δ.
        render_t:       (H, W, C) rendered image at t.
        render_t_plus:  (H, W, C) rendered image at t + δ.

    Returns:
        loss: scalar.
    """
    temporal_avg = 0.5 * (render_t_minus + render_t_plus)
    return F.l1_loss(render_t, temporal_avg.detach())
    # .detach() on the average — we want render_t to move towards the average,
    # not the reverse. Prevents the loss from collapsing all three renders to
    # the same value.


# ────────────────────────────────────────────────────────────────────────────
# Combined loss
# ────────────────────────────────────────────────────────────────────────────

def total_loss(
    # RGB loss inputs
    render_t: Tensor,
    gt_t: Tensor,
    # Flow loss inputs
    means2d_t: Tensor,
    means2d_t1: Tensor,
    flow_raft: Tensor,
    # Triple-rendering inputs
    render_t_minus: Tensor,
    render_t_plus: Tensor,
    # Loss weights
    lambda_ssim: float = 0.2,
    lambda_flow: float = 0.1,
    lambda_triple: float = 0.05,
    # Optional: restrict flow loss to visible Gaussians
    gaussian_ids: Optional[Tensor] = None,
) -> tuple[Tensor, dict]:
    """
    Compute the full 4DBJJ triple-rendering loss.

    L_total = L_RGB(t) + λ_flow · L_Flow(t→t+1) + λ_triple · L_Triple(t)

    Args:
        render_t:       (H, W, 3) rendered frame at time t, float [0,1].
        gt_t:           (H, W, 3) ground-truth frame at time t.
        means2d_t:      (N, 2) Gaussian 2D projections at t (pixels).
        means2d_t1:     (N, 2) Gaussian 2D projections at t+1 (pixels).
        flow_raft:      (H, W, 2) RAFT flow from t to t+1.
        render_t_minus: (H, W, 3) rendered frame at t - δ.
        render_t_plus:  (H, W, 3) rendered frame at t + δ.
        lambda_ssim:    Weight on DSSIM term in L_RGB.
        lambda_flow:    Weight on flow consistency loss.
        lambda_triple:  Weight on temporal smoothness loss.
        gaussian_ids:   (M,) optional subset for flow loss.

    Returns:
        loss:       Scalar total loss.
        components: Dict of individual loss values for logging.
    """
    # 1. RGB loss
    l_l1   = l1_loss(render_t, gt_t)
    l_ssim = ssim_loss(render_t, gt_t)
    l_rgb  = (1.0 - lambda_ssim) * l_l1 + lambda_ssim * l_ssim

    # 2. Flow consistency loss
    l_flow = flow_consistency_loss(means2d_t, means2d_t1, flow_raft, gaussian_ids)

    # 3. Triple-rendering temporal smoothness loss
    l_triple = triple_rendering_loss(render_t_minus, render_t, render_t_plus)

    loss = l_rgb + lambda_flow * l_flow + lambda_triple * l_triple

    return loss, {
        "loss_total":  loss.item(),
        "loss_rgb":    l_rgb.item(),
        "loss_l1":     l_l1.item(),
        "loss_ssim":   l_ssim.item(),
        "loss_flow":   l_flow.item(),
        "loss_triple": l_triple.item(),
    }
