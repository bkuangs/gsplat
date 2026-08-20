# 3D Gaussian Splatting, From the Math Up

An educational implementation of 3D Gaussian Splatting (3DGS) aimed at graphics and
computer-vision learning. The repository deliberately provides the engineering
scaffold while leaving the central representation, rendering, optimization, and
adaptive density-control work as `TODO(student)` exercises.

The target demo is a trained scene and smooth novel-view video from images registered
by COLMAP. A readable PyTorch renderer establishes correctness; the external
[`gsplat`](https://github.com/nerfstudio-project/gsplat) CUDA backend makes full-scene
training practical.

## What you implement

| Area | Your implementation | Scaffolded for you |
| --- | --- | --- |
| Geometry | Pinhole projection, 3D covariance, covariance projection | Camera types and validation |
| Representation | Gaussian initialization, SH features, parameter activations | Typed `nn.Module` parameter container |
| Rendering | PyTorch splatting, visibility, sorting, alpha compositing | CUDA backend adapter |
| Learning | Photometric loss, optimizer groups, training loop | Config and checkpoint I/O |
| Density control | Gradient statistics, clone/split/prune, optimizer-state updates | Scheduling fields and data types |
| Data and output | Dataset sampling and final camera path integration | COLMAP loading, metrics, CLI skeleton |

This split is intentional. Writing another COLMAP parser, argument framework, or
checkpoint format adds little interview value. Deriving covariance projection and
debugging differentiable compositing does.

## Learning goals

By completion, you should be able to:

1. Explain why anisotropic Gaussians use scale plus rotation and how their covariance
   projects through a perspective camera.
2. Derive front-to-back alpha compositing and discuss its numerical and gradient behavior.
3. Explain spherical harmonics as a compact view-dependent color model.
4. Design parameter-specific optimization and adaptive clone/split/prune rules.
5. Validate a readable reference implementation against an optimized CUDA operator.
6. Diagnose camera-convention, coordinate-system, visibility, and numerical-stability bugs.

## Representation conventions

- Quaternions use WXYZ component order throughout the model and both renderers; identity
  is `[1, 0, 0, 0]`.
- Both rasterizers receive an explicit active SH degree. Stored coefficient tensors may
  contain additional bands, but renderers evaluate only the configured active degree.
- Checkpoint loading reconstructs the saved model shape and device before creating and
  restoring its optimizer.

### Image and render tensor contract

- Each `Camera` represents one unbatched image. Its `width` and `height` are the exact
  target dimensions after applying the configured downscale.
- The image loader converts source images to three-channel RGB, resizes them to
  `(width, height)`, and returns contiguous `torch.float32` tensors with shape
  `(3, height, width)` and values in `[0, 1]`.
- Image values remain in their encoded RGB color space; loading does not apply gamma or
  color-space conversion beyond conversion to RGB.
- Both renderers return unbatched RGB with shape `(3, height, width)` and alpha with
  shape `(1, height, width)`. Optional depth uses shape `(1, height, width)`.
- Renderers use the model's floating-point dtype and device. Ground-truth images move
  to that device before loss evaluation.
- Renderer RGB is not silently clamped during training. Evaluation may clamp to
  `[0, 1]` before computing display-oriented metrics.
- HWC layout exists only at external-library boundaries; adapters convert it to CHW
  before constructing `RenderOutput`.

## Setup

Use Python 3.11 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra data
uv run gsplat-learn status
uv run pytest
```

On the NVIDIA training machine, install the optimized backend too. Follow the PyTorch
installer for the CUDA build matching that machine before installing `gsplat`.

```bash
uv sync --extra dev --extra data --extra cuda --extra video
```

Run COLMAP externally, then point `configs/baseline.yaml` at its sparse model and image
directory:

```bash
uv run gsplat-learn inspect-colmap --config configs/baseline.yaml
```

The `train` command is wired but intentionally raises `NotImplementedError` until
milestone 3.

## Four-week path

### Week 1 — camera and image formation

- Implement `math/projection.py` and `rendering/compositing.py`.
- Remove the skip from `tests/student/test_milestone_1_math.py`.
- Render a handful of fixed, isotropic Gaussians on a tiny synthetic image.
- Write down coordinate conventions before debugging them.

**Exit criterion:** analytic tests pass and gradients are finite.

### Week 2 — reference renderer

- Initialize `GaussianModel` from COLMAP points and colors.
- Implement view-dependent SH color evaluation.
- Build the PyTorch rasterizer: cull, bound, depth-sort, evaluate, composite.
- Favor simple vectorized or tiled code over premature optimization.

**Exit criterion:** a tiny scene renders correctly and parameters receive gradients.

### Week 3 — optimization

- Implement L1 + SSIM, parameter-specific optimizer groups, data sampling, and training.
- Save checkpoints and report held-out PSNR.
- Compare small-scene outputs and gradients between PyTorch and CUDA backends.

**Exit criterion:** a real scene improves visibly and numerically during training.

### Week 4 — adaptive density and portfolio proof

- Accumulate screen-space position gradients and implement clone/split/prune.
- Add a deterministic novel-view camera path and MP4 export.
- Record ablations: no densification, isotropic covariance, SH degree 0 versus 3.
- Add final renders, a system diagram, results table, and lessons learned to this README.

**Exit criterion:** one reproducible command trains the chosen scene and another renders
the portfolio video.

## Repository map

```text
configs/                         experiment configuration
docs/ideas/gsplat.md             scope and assumptions
src/gaussian_splatting/
  data/colmap.py                 prebuilt COLMAP adapter
  math/projection.py             TODO: projection and covariance math
  model/gaussians.py             TODO: point-cloud initialization
  rendering/
    compositing.py               TODO: alpha compositing
    torch_backend.py             TODO: reference rasterizer
    cuda_backend.py              prebuilt optimized-backend adapter
  training/
    checkpoint.py               prebuilt checkpoint I/O
    losses.py                    TODO: photometric objective
    densification.py            TODO: adaptive density control
    trainer.py                  TODO: optimization loop
tests/scaffold/                  tests for provided infrastructure
tests/student/                   skipped acceptance tests to unlock by milestone
```

## Resume-ready evidence

Do not describe this only as “implemented 3DGS.” Preserve evidence:

- side-by-side ground truth, PyTorch, and CUDA renders;
- PSNR and timing comparisons on a fixed scene;
- an ablation table for anisotropy, SH degree, and densification;
- profiler screenshots or traces explaining the reference renderer bottleneck;
- a short section on one hard bug and the invariant/test that found it.

Those artifacts show graphics reasoning and experimental discipline better than a
feature checklist.
