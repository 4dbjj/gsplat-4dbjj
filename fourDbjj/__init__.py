"""
4DBJJ motion engine — dynamic 4D Gaussian Splatting for BJJ technique capture.

Public API
----------
  DynamicGaussians          — main scene class; wraps gsplat with time-varying μ(t)/α(t)

  Deformation modules:
    CatmullRomSpline         — per-Gaussian position trajectory (C1 spline)
    SlerpRotationSpline      — per-Gaussian orientation trajectory (SLERP)
    TemporalOpacity          — per-Gaussian Gaussian-window opacity α(t)

  Flow preprocessing:
    RaftEstimator            — dense 2D optical flow (torchvision RAFT)
    flow_to_3d_velocity      — lift 2D flow + depth → 3D velocity
    init_control_points_from_flow  — seed spline control points from flow

  Losses:
    total_loss               — L_RGB + λ_flow·L_Flow + λ_triple·L_Triple
"""

from fourDbjj.scene import DynamicGaussians
from fourDbjj.deformation.spline import CatmullRomSpline
from fourDbjj.deformation.rotation import SlerpRotationSpline
from fourDbjj.deformation.opacity import TemporalOpacity
from fourDbjj.flow import (
    RaftEstimator,
    flow_to_3d_velocity,
    init_control_points_from_flow,
)
from fourDbjj.losses import total_loss

__all__ = [
    "DynamicGaussians",
    "CatmullRomSpline",
    "SlerpRotationSpline",
    "TemporalOpacity",
    "RaftEstimator",
    "flow_to_3d_velocity",
    "init_control_points_from_flow",
    "total_loss",
]
