# Learn 3D Gaussian Splatting

## Problem Statement

How might I build a scoped 3D Gaussian Splatting system that demonstrates graphics
and computer-vision depth while leaving the important math and rendering work visible?

## Recommended Direction

Build two interchangeable rendering backends. A readable PyTorch implementation is
the source of truth for learning and correctness; the existing `gsplat` CUDA
rasterizer makes full-scene optimization practical. This avoids pretending that a
custom production rasterizer fits into a focused 2–4 week project.

The finished proof is an end-to-end run: ingest a COLMAP reconstruction, optimize a
scene, report image metrics, and render a smooth novel-view video. Tests should compare
the reference backend with the optimized backend on tiny scenes.

## Key Assumptions to Validate

- [ ] A tiny synthetic scene catches projection and compositing mistakes.
- [ ] The PyTorch and CUDA backends can follow one camera/model contract.
- [ ] The CUDA dependency installs and trains on the target NVIDIA machine.

## MVP Scope

COLMAP ingestion, Gaussian initialization, differentiable PyTorch rendering, CUDA
backend integration, photometric optimization, adaptive density control, checkpoints,
PSNR evaluation, and offline novel-view rendering.

## Not Doing (and Why)

- SfM or feature matching — COLMAP already solves a different, substantial problem.
- Custom CUDA kernels — too risky for the first 2–4 week version.
- Interactive viewer — polish it only after the learning pipeline works.
- Distributed training — unrelated to the core learning goals.
- Exact paper reproduction — correctness and explanation matter more than leaderboard parity.

## Open Questions

- Which first dataset gives the fastest iteration loop?
- How closely does the reference renderer agree with the CUDA backend?
- Which densification ablations best demonstrate understanding?

