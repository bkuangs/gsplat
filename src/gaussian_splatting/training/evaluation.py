from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from gaussian_splatting.metrics import psnr
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def _save_grayscale(path: Path, image: torch.Tensor) -> None:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array, mode="L").save(path)


def _visualize_depth(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    visualization = torch.zeros_like(depth)
    valid_depth = depth[valid]
    if valid_depth.numel() == 0:
        return visualization
    minimum = valid_depth.min()
    maximum = valid_depth.max()
    if torch.isclose(minimum, maximum):
        visualization[valid] = 1.0
    else:
        visualization[valid] = (depth[valid] - minimum) / (maximum - minimum)
    return visualization


@torch.no_grad()
def evaluate_holdout(
    model: GaussianModel,
    renderer: nn.Module,
    cameras: Sequence[Camera],
    images: Sequence[torch.Tensor],
    holdout_indices: Sequence[int],
    background: tuple[float, float, float],
    output_dir: Path,
    stage: str,
    step: int,
) -> dict[str, Any]:
    """Render a fixed holdout and persist enough evidence to compare two stages."""
    was_training = model.training
    model.eval()
    stage_dir = output_dir / "holdout" / stage
    target_dir = output_dir / "holdout" / "targets"
    stage_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    device = model.means.device
    camera_metrics: list[dict[str, Any]] = []
    for index in holdout_indices:
        camera = cameras[index]
        if camera.image_id is None:
            raise ValueError("held-out cameras must have COLMAP image IDs")
        target = images[index].to(device)
        output = renderer(model, camera.to(device), background)
        if output.depth is None:
            raise RuntimeError("held-out evaluation requires renderer depth output")

        prediction = output.rgb.clamp(0.0, 1.0)
        alpha = output.alpha[0]
        depth = output.depth[0]
        valid = (alpha > 1e-4) & torch.isfinite(depth) & (depth > 0)
        stem = f"image_{camera.image_id:04d}"

        _save_rgb(target_dir / f"{stem}.png", target)
        _save_rgb(stage_dir / f"{stem}_rgb.png", prediction)
        _save_grayscale(stage_dir / f"{stem}_alpha.png", alpha)
        _save_grayscale(
            stage_dir / f"{stem}_depth.png",
            _visualize_depth(depth, valid),
        )
        torch.save(depth.detach().cpu(), stage_dir / f"{stem}_depth.pt")

        camera_metrics.append(
            {
                "image_id": camera.image_id,
                "image_name": camera.image_path.name if camera.image_path else None,
                "psnr": psnr(prediction, target),
                "valid_depth_coverage": valid.float().mean().item(),
            }
        )

    if was_training:
        model.train()
    return {
        "step": step,
        "mean_psnr": float(np.mean([item["psnr"] for item in camera_metrics])),
        "mean_valid_depth_coverage": float(
            np.mean([item["valid_depth_coverage"] for item in camera_metrics])
        ),
        "cameras": camera_metrics,
    }
