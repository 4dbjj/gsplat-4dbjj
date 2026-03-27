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
| `fourDbjj/deformation/spline.py` | TBD | TBD |
| `fourDbjj/deformation/field.py` | TBD | TBD |
| `fourDbjj/losses/` | TBD | TBD |

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
