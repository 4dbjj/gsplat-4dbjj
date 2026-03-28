# Upstream Boundary Documentation

## Fork Source

| Field | Value |
|---|---|
| Upstream repo | https://github.com/nerfstudio-project/gsplat |
| Fork commit | `e60d0e3` — "Add external distortion wrappers and tests (#906)" |
| Fork date | 2026-03-27 |
| Fork owner | 4dbjj |
| License | Apache 2.0 (preserved) |

## What is upstream (do not modify without reason)

All files in `gsplat/` that existed at commit `e60d0e3` are upstream code.
Changes to upstream files should be minimal and clearly marked with `# 4DBJJ PATCH` comments.

## What is ours (4DBJJ IP)

All files under `fourDbjj/` are original 4DBJJ implementations derived from
published research papers. No code was copied from any research repository.

| Module | Derives from | Equations |
|---|---|---|
| `fourDbjj/deformation/spline.py` | Catmull & Rom (1974), Computer Aided Geometric Design | CR basis matrix, §3 |
| `fourDbjj/deformation/rotation.py` | Shoemaker (1985), SIGGRAPH '85 | SLERP eq. 2 |
| `fourDbjj/deformation/opacity.py` | Gaussian function — public domain math | α(t) = α_base · exp(-(t-τ)²/2σ²) |
| `fourDbjj/flow/raft_estimator.py` | Teed & Deng (2020), ECCV — torchvision RAFT (BSD 3-Clause) | N/A (wrapper) |
| `fourDbjj/flow/flow_to_3d.py` | Hartley & Zisserman (2003), MVG Ch. 6 | Perspective projection |
| `fourDbjj/flow/init_control_pts.py` | Original 4DBJJ design | Velocity-seeded control point init |
| `fourDbjj/scene/dynamic_gaussians.py` | Original 4DBJJ design | Wires μ(t), q(t), α(t) into gsplat.rasterization |
| `fourDbjj/losses/triple_rendering.py` | Kerbl et al. (2023) SIGGRAPH + original 4DBJJ | L_RGB + λ_flow·L_Flow + λ_triple·L_Triple |

## How to pull upstream improvements

```bash
git fetch upstream
git merge upstream/main
# resolve any conflicts, keeping our fourDbjj/ modules intact
git push origin fourDbjj/temporal-gaussians
```

## IP hygiene rules

1. Never copy code from research repos (RetimeGS, SplineGS, Inria gaussian-splatting, etc.)
2. Implement all motion logic from paper equations only — cite the paper + equation number in comments
3. All 4DBJJ additions live under `fourDbjj/` namespace — never mixed into `gsplat/` internals
4. Update the table above when new modules are added
