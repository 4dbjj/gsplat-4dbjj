"""
RAFT optical flow estimator — thin wrapper around torchvision's RAFT implementation.

License: RAFT (princeton-vl) BSD 3-Clause — commercial use permitted.
         torchvision BSD 3-Clause — commercial use permitted.

Usage in the 4DBJJ pipeline:
  For each consecutive pair of frames from a GoPro camera, RAFT produces a
  dense 2D flow field F(u, v) = (du, dv) — the pixel displacement from frame t
  to frame t+1.

  This flow field is then passed to flow_to_3d.py to estimate 3D velocities
  for each Gaussian, which seeds the spline control points.

Why RAFT over other flow methods:
  - State-of-the-art accuracy on fast non-rigid motion (BJJ involves both)
  - Handles large displacements (a sweep can move limbs 100+ pixels per frame)
  - BSD 3-Clause — no commercial license needed
  - torchvision ships a pretrained RAFT model — no manual weight download

Reference:
  Teed, Z. & Deng, J. (2020). "RAFT: Recurrent All-Pairs Field Transforms
  for Optical Flow." ECCV 2020. BSD 3-Clause License.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import List, Tuple


class RaftEstimator(nn.Module):
    """
    Wraps torchvision's pretrained RAFT model for dense optical flow estimation.

    Args:
        model_size:   "large" (more accurate) or "small" (faster, less memory).
                      For training preprocessing, use "large". For real-time
                      inference during capture review, use "small".
        device:       torch.device — should be CUDA for reasonable speed.
        iters:        Number of RAFT recurrent iterations. Default 20 (paper default).
                      Reduce to 12 for faster preprocessing with minor accuracy loss.
    """

    def __init__(
        self,
        model_size: str = "large",
        device: torch.device | None = None,
        iters: int = 20,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iters = iters

        # Load pretrained RAFT from torchvision (downloads weights on first run ~20MB)
        try:
            from torchvision.models.optical_flow import raft_large, raft_small
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
        except ImportError:
            raise ImportError(
                "torchvision >= 0.13 required for RAFT. "
                "Install with: pip install torchvision>=0.13"
            )

        if model_size == "large":
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        elif model_size == "small":
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            raise ValueError(f"model_size must be 'large' or 'small', got {model_size!r}")

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def estimate_flow(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """
        Estimate dense optical flow from frame1 to frame2.

        Args:
            frame1: RGB frame, shape (H, W, 3), uint8 or float32 in [0, 255].
            frame2: RGB frame, shape (H, W, 3), same format as frame1.

        Returns:
            flow: shape (H, W, 2) — (du, dv) pixel displacement at each location.
                  du: horizontal displacement (positive = rightward motion).
                  dv: vertical displacement (positive = downward motion).
        """
        img1 = self._preprocess(frame1)  # (1, 3, H, W)
        img2 = self._preprocess(frame2)

        # RAFT returns a list of flow predictions (one per recurrent iteration)
        # We take the final (most refined) prediction.
        flow_predictions = self.model(img1, img2, num_flow_updates=self.iters)
        flow = flow_predictions[-1]  # (1, 2, H, W)

        # Return as (H, W, 2) — more intuitive for downstream processing
        return flow.squeeze(0).permute(1, 2, 0).cpu()  # (H, W, 2)

    @torch.no_grad()
    def estimate_sequence(
        self,
        frames: List[Tensor],
        show_progress: bool = True,
    ) -> List[Tensor]:
        """
        Estimate optical flow for an entire video sequence.

        Args:
            frames: List of T RGB frames, each shape (H, W, 3).
            show_progress: Print progress to stdout.

        Returns:
            flows: List of (T-1) flow tensors, each shape (H, W, 2).
                   flows[i] is the flow from frames[i] to frames[i+1].
        """
        flows = []
        T = len(frames)

        for i in range(T - 1):
            if show_progress and i % 10 == 0:
                print(f"  RAFT: estimating flow {i+1}/{T-1}")
            flow = self.estimate_flow(frames[i], frames[i + 1])
            flows.append(flow)

        return flows

    def _preprocess(self, frame: Tensor) -> Tensor:
        """
        Convert a raw frame to RAFT's expected input format.
        RAFT expects: (1, 3, H, W), float32, values in [0, 255].
        Input may be (H, W, 3) uint8 or float32.
        """
        if frame.dtype == torch.uint8:
            frame = frame.float()
        if frame.dim() == 3:
            frame = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # RAFT requires image dimensions divisible by 8
        _, _, H, W = frame.shape
        H_pad = (8 - H % 8) % 8
        W_pad = (8 - W % 8) % 8
        if H_pad > 0 or W_pad > 0:
            frame = F.pad(frame, [0, W_pad, 0, H_pad], mode="replicate")

        return frame.to(self.device)


def load_frames_from_directory(frame_dir: Path, extension: str = "jpg") -> List[Tensor]:
    """
    Load a sequence of image frames from a directory, sorted by filename.

    Args:
        frame_dir: Directory containing frame files.
        extension: File extension (jpg, png).

    Returns:
        frames: List of (H, W, 3) uint8 tensors.
    """
    try:
        from torchvision.io import read_image
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")

    paths = sorted(frame_dir.glob(f"*.{extension}"))
    if not paths:
        raise FileNotFoundError(f"No .{extension} files found in {frame_dir}")

    frames = []
    for p in paths:
        img = read_image(str(p))          # (3, H, W) uint8
        frames.append(img.permute(1, 2, 0))  # (H, W, 3)

    return frames
