"""
DynamicGaussians: time-aware 4D Gaussian scene for BJJ motion capture.

Architecture overview
---------------------
A standard 3DGS scene stores static parameters (means, quats, scales,
opacities, SH colours) and passes them directly to gsplat's rasterizer.

DynamicGaussians replaces three of those static tensors with time-varying
versions produced by our learned motion modules:

    means(t)     ←  CatmullRomSpline      position trajectory
    quats(t)     ←  SlerpRotationSpline   orientation trajectory
    opacities(t) ←  TemporalOpacity       Gaussian window α(t) = α_base·exp(...)

At render time the caller passes a normalised timestamp t ∈ [0, 1] and the
class evaluates each module, then delegates to gsplat.rasterization().

Static parameters (scales, SH colours) are not time-varying — a Gaussian's
size and colour are properties of the underlying physical surface, not its
pose in time.

Initialisation workflow
-----------------------
Phase 1:  Train a standard static 3DGS on the VENUE_SETUP footage (empty gym).
          This gives us initialised means, scales, sh0, shN.

Phase 2:  Run RAFT flow + flow_to_3d to estimate 3D velocities for each
          Gaussian. Call init_control_points_from_flow() to seed spline.

Phase 3:  This class — train with the triple-rendering loss so μ(t) and α(t)
          match the observed dynamic footage.

Quaternion convention note
--------------------------
Our SlerpRotationSpline stores quaternions as (x, y, z, w) — standard scipy /
PyTorch3D convention. gsplat.rasterization expects (w, x, y, z) — wxyz.
We convert with torch.roll(q, shifts=1, dims=-1) inside render().

# 4DBJJ PATCH — this file is 4DBJJ IP, not upstream gsplat code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

from gsplat import rasterization

from fourDbjj.deformation.spline import CatmullRomSpline
from fourDbjj.deformation.rotation import SlerpRotationSpline
from fourDbjj.deformation.opacity import TemporalOpacity


class DynamicGaussians(nn.Module):
    """
    4D Gaussian scene: static geometry + learned temporal deformation.

    Args:
        num_gaussians:      N — number of Gaussian primitives.
        num_ctrl_pts:       K — spline control points per Gaussian (≥ 4).
        num_frames:         T — total frames in the video clip.
        sh_degree:          Spherical harmonics degree (0 = colour only, 3 = full).
        sigma_min:          Minimum temporal lifespan (TemporalOpacity).
        sigma_max:          Maximum temporal lifespan (TemporalOpacity).
    """

    def __init__(
        self,
        num_gaussians: int,
        num_ctrl_pts: int,
        num_frames: int,
        sh_degree: int = 3,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_gaussians = num_gaussians
        self.num_frames = num_frames
        self.sh_degree = sh_degree

        # ── Static parameters ─────────────────────────────────────────────────
        # Log-scale for stability; exp() in forward gives positive scales.
        self.log_scales = nn.Parameter(torch.zeros(num_gaussians, 3))

        # SH colour coefficients.  sh0 = DC component (always present).
        num_sh_coeffs = (sh_degree + 1) ** 2
        self.sh0 = nn.Parameter(torch.zeros(num_gaussians, 1, 3))
        if sh_degree > 0:
            self.shN = nn.Parameter(
                torch.zeros(num_gaussians, num_sh_coeffs - 1, 3)
            )
        else:
            self.register_parameter("shN", None)

        # Base opacity in logit-space; sigmoid() in TemporalOpacity.forward.
        self.logit_base_opacity = nn.Parameter(
            torch.full((num_gaussians, 1), -1.0)  # sigmoid(-1) ≈ 0.27
        )

        # ── Dynamic modules ───────────────────────────────────────────────────
        self.spline = CatmullRomSpline(num_gaussians, num_ctrl_pts, num_frames)
        self.rotation_spline = SlerpRotationSpline(
            num_gaussians, num_ctrl_pts, num_frames
        )
        self.temporal_opacity = TemporalOpacity(
            num_gaussians, num_frames, sigma_min=sigma_min, sigma_max=sigma_max
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def scales(self) -> Tensor:
        """Positive scales (N, 3)."""
        return self.log_scales.exp()

    @property
    def base_opacity(self) -> Tensor:
        """Base opacity after sigmoid, shape (N, 1)."""
        return torch.sigmoid(self.logit_base_opacity)

    @property
    def sh_coeffs(self) -> Tensor:
        """
        All SH coefficients concatenated: (N, num_sh_coeffs, 3).
        sh0 (DC) is always first; shN appended for degree > 0.
        """
        if self.shN is not None:
            return torch.cat([self.sh0, self.shN], dim=1)
        return self.sh0

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluate dynamic parameters at time t
    # ──────────────────────────────────────────────────────────────────────────

    def means_at(self, t: Tensor) -> Tensor:
        """
        Gaussian centres at normalised time t.

        Args:
            t: shape () or (1,) — single timestamp in [0, 1].

        Returns:
            means: (N, 3)
        """
        t_batch = t.view(1)
        return self.spline.forward(t_batch).squeeze(0)  # (N, 3)

    def quats_at(self, t: Tensor) -> Tensor:
        """
        Quaternions at normalised time t, in gsplat's **wxyz** convention.

        Our SlerpRotationSpline stores xyzw. We convert here with roll.

        Args:
            t: shape () or (1,).

        Returns:
            quats: (N, 4) in wxyz order.
        """
        t_batch = t.view(1)
        q_xyzw = self.rotation_spline.forward(t_batch).squeeze(0)  # (N, 4) xyzw
        q_wxyz = torch.roll(q_xyzw, shifts=1, dims=-1)              # (N, 4) wxyz
        return q_wxyz

    def opacities_at(self, t: Tensor) -> Tensor:
        """
        Temporally-modulated opacity at time t.

        Args:
            t: shape () or (1,).

        Returns:
            opacities: (N,) — gsplat expects 1-D opacities.
        """
        t_batch = t.view(1)
        modulated = self.temporal_opacity.forward(t_batch, self.base_opacity)
        # modulated: (1, N, 1) → squeeze to (N,)
        return modulated.squeeze(0).squeeze(-1)  # (N,)

    # ──────────────────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────────────────

    def render(
        self,
        t: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree_to_use: Optional[int] = None,
        packed: bool = True,
        absgrad: bool = False,
        rasterize_mode: str = "antialiased",
        render_mode: str = "RGB",
        **rasterization_kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Render all cameras for a single timestamp.

        Args:
            t:                  Normalised timestamp in [0, 1], scalar or (1,).
            viewmats:           (C, 4, 4) world-to-camera matrices.
            Ks:                 (C, 3, 3) camera intrinsics.
            width:              Output image width in pixels.
            height:             Output image height in pixels.
            sh_degree_to_use:   Degree of SH to evaluate (progressive training).
                                Defaults to self.sh_degree.
            packed:             Use packed (memory-efficient) rasterization.
            absgrad:            Compute absolute-value gradients of 2D means
                                (required by MCMCStrategy densification).
            rasterize_mode:     "antialiased" (recommended) or "classic".
            render_mode:        "RGB", "RGB+D", "D", etc.
            **rasterization_kwargs: Forwarded verbatim to gsplat.rasterization.

        Returns:
            render_colors: (C, H, W, channels) rendered output.
            render_alphas: (C, H, W, 1) accumulated alpha.
            meta:          Dict of gsplat intermediate tensors.
        """
        t = t.to(device=self.logit_base_opacity.device, dtype=torch.float32)

        # ── Evaluate time-varying parameters ─────────────────────────────────
        means = self.means_at(t)          # (N, 3)
        quats = self.quats_at(t)          # (N, 4) wxyz
        opacities = self.opacities_at(t)  # (N,)

        # ── Static parameters ─────────────────────────────────────────────────
        scales = self.scales  # (N, 3)

        sh_deg = sh_degree_to_use if sh_degree_to_use is not None else self.sh_degree
        # gsplat expects colors as (N, K, 3) SH coefficients
        # Slice to the number of coeffs for sh_deg
        num_coeffs_use = (sh_deg + 1) ** 2
        colors = self.sh_coeffs[:, :num_coeffs_use, :]  # (N, num_coeffs_use, 3)

        # ── Rasterize ─────────────────────────────────────────────────────────
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_deg,
            packed=packed,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            render_mode=render_mode,
            **rasterization_kwargs,
        )

        # Attach time-varying tensors to meta for loss computation
        meta["means3d_t"] = means
        meta["quats_t"] = quats
        meta["opacities_t"] = opacities

        return render_colors, render_alphas, meta

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def load_from_static_splat(self, splats: Dict[str, Tensor]) -> None:
        """
        Initialise static parameters from a trained 3DGS splat dict.

        Expects the dict format used by gsplat's simple_trainer.py:
          splats["means"]      (N, 3) — copied into spline control points
          splats["quats"]      (N, 4) wxyz — converted to xyzw for our module
          splats["scales"]     (N, 3) — in log-space already
          splats["opacities"]  (N,) — in logit-space already
          splats["sh0"]        (N, 1, 3)
          splats["shN"]        (N, K_sh-1, 3)  [optional]

        The canonical Gaussian means are broadcast to all control points as
        a flat (stationary) initialisation. Flow-guided init should be called
        afterwards via init_control_points_from_flow().
        """
        with torch.no_grad():
            if "scales" in splats:
                self.log_scales.copy_(splats["scales"])
            if "opacities" in splats:
                self.logit_base_opacity.copy_(
                    splats["opacities"].view(self.num_gaussians, 1)
                )
            if "sh0" in splats:
                self.sh0.copy_(splats["sh0"])
            if "shN" in splats and self.shN is not None:
                self.shN.copy_(splats["shN"])

            if "means" in splats:
                # Broadcast canonical means to all K control points (static init)
                means_init = splats["means"]           # (N, 3)
                ctrl = means_init.unsqueeze(1).expand(
                    -1, self.spline.num_ctrl_pts, -1
                ).clone()
                self.spline.control_points.copy_(ctrl)

            if "quats" in splats:
                # splats["quats"] is wxyz; our module stores xyzw
                q_wxyz = splats["quats"]  # (N, 4)
                q_xyzw = torch.roll(q_wxyz, shifts=-1, dims=-1)
                # Broadcast to all K rotation control points
                K = self.rotation_spline.num_ctrl_pts
                ctrl_q = q_xyzw.unsqueeze(1).expand(-1, K, -1).clone()
                self.rotation_spline.control_quats.copy_(ctrl_q)

    @torch.no_grad()
    def canonical_means(self) -> Tensor:
        """
        Gaussian positions at t = 0.5 (sequence midpoint). Useful for
        visualising the scene at its 'average' pose and for debugging.

        Returns:
            means: (N, 3)
        """
        t = torch.tensor(0.5, device=self.logit_base_opacity.device)
        return self.means_at(t)
