# gsplat-4dbjj

> **4DBJJ fork** of [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) (Apache 2.0).
> All 4DBJJ additions live under `fourDbjj/`. Upstream gsplat code in `gsplat/` is untouched except where marked `# 4DBJJ PATCH`.
> See [UPSTREAM.md](UPSTREAM.md) for the IP boundary and module inventory.

---

## Syncing with upstream gsplat

Run this whenever you want to pull the latest gsplat improvements into our branch.
Tell Claude: **"sync upstream gsplat into fourDbjj/temporal-gaussians"** and it will follow these steps.

```bash
# 1. Make sure you are on our branch and the working tree is clean
git checkout fourDbjj/temporal-gaussians
git status                          # should show "nothing to commit"

# 2. Fetch the latest commits from nerfstudio-project/gsplat
git fetch upstream

# 3. Check how many new commits exist upstream (0 = already up to date)
git log --oneline HEAD..upstream/main

# 4. Merge upstream into our branch
git merge upstream/main --no-edit

# 5. Resolve any conflicts — our code ALWAYS wins over upstream in fourDbjj/
#    If gsplat/ files conflict, prefer upstream's version (it's their code).
#    If fourDbjj/ files conflict, keep our version.
#    Mark every conflict resolution with a comment:  # 4DBJJ PATCH

# 6. Run the syntax check to confirm nothing broke
python3 -c "
import ast
for f in ['fourDbjj/deformation/spline.py','fourDbjj/deformation/rotation.py',
          'fourDbjj/deformation/opacity.py','fourDbjj/flow/raft_estimator.py',
          'fourDbjj/flow/flow_to_3d.py','fourDbjj/flow/init_control_pts.py',
          'fourDbjj/scene/dynamic_gaussians.py','fourDbjj/losses/triple_rendering.py']:
    ast.parse(open(f).read()); print('OK', f)
"

# 7. Push
git push origin fourDbjj/temporal-gaussians
```

### Sync history

| Date | Upstream commit merged | New upstream commits | Notes |
|---|---|---|---|
| 2026-03-28 | `e60d0e3` (fork point) | 0 — already up to date | Initial fork |

Update this table each time you run a sync.

---

## 4DBJJ motion engine (`fourDbjj/`)

| Module | Purpose | Paper basis |
|---|---|---|
| `fourDbjj/deformation/spline.py` | Per-Gaussian position trajectory (Catmull-Rom) | Catmull & Rom (1974) |
| `fourDbjj/deformation/rotation.py` | Per-Gaussian rotation trajectory (SLERP) | Shoemaker, SIGGRAPH '85 |
| `fourDbjj/deformation/opacity.py` | Temporal Gaussian opacity window α(t) | Public domain math |
| `fourDbjj/flow/raft_estimator.py` | Dense 2D optical flow (torchvision RAFT) | Teed & Deng, ECCV 2020 |
| `fourDbjj/flow/flow_to_3d.py` | 2D flow + depth → 3D velocity | Hartley & Zisserman Ch.6 |
| `fourDbjj/flow/init_control_pts.py` | Seeds spline from flow-estimated velocity | Original 4DBJJ |
| `fourDbjj/scene/dynamic_gaussians.py` | Wires μ(t), q(t), α(t) into `gsplat.rasterization` | Original 4DBJJ |
| `fourDbjj/losses/triple_rendering.py` | L_RGB + λ_flow·L_Flow + λ_triple·L_Triple | Original 4DBJJ |

### Quick usage

```python
from fourDbjj import DynamicGaussians, total_loss, RaftEstimator, init_control_points_from_flow

# Build model
model = DynamicGaussians(num_gaussians=N, num_ctrl_pts=8, num_frames=T)
model.load_from_static_splat(phase1_splats)   # from phase-1 3DGS training

# Seed spline from RAFT flow
raft = RaftEstimator("large")
flows = raft.estimate_sequence(frames)
init_control_points_from_flow(model.spline, means3d, flows, depths, viewmats, fx, fy, cx, cy)

# Training loop
t = torch.tensor(frame_idx / (T - 1))
render, alpha, meta = model.render(t, viewmats, Ks, W, H)
loss, log = total_loss(render[0], gt, meta["means2d"].squeeze(0),
                       means2d_t1, flow_raft, render_minus, render_plus)
loss.backward()
```

---

## License

`gsplat/` — Apache 2.0 (nerfstudio-project/gsplat, see LICENSE).
`fourDbjj/` — Proprietary (4DBJJ Inc.). All rights reserved.

---

# Original gsplat README

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## News

[Jan 2026] [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) is integrated as an alternative way of bilateral grid to compensate the training views.

[May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).

[May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.

[April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires installing some extra dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the contributors coming from following institutes (unordered):

- UC Berkeley
- NVIDIA
- ShanghaiTech University
- Amazon
- Meta
- IIIT
- LumaAI
- SpectacularAI
- Aalto University
- CMU

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
