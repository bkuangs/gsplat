import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from gaussian_splatting.config import ExperimentConfig, config_to_dict
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.metrics import psnr
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.cuda_backend import CudaRasterizer
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.checkpoint import load_checkpoint
from gaussian_splatting.training.splits import partition_camera_indices
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


def _load_sparse_depth(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"depth sample file does not exist: {path}")
    from numpy.core.multiarray import _reconstruct

    safe_globals = [
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"depth sample file must contain a mapping: {path}")
    required = {"image_name", "depth", "coord"}
    missing = required - payload.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"depth sample file is missing {names}: {path}")
    return payload


def _sample_image_at_coordinates(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    source_width: int,
    source_height: int,
) -> torch.Tensor:
    normalized = torch.stack(
        [
            2.0 * coordinates[:, 0] / source_width - 1.0,
            2.0 * coordinates[:, 1] / source_height - 1.0,
        ],
        dim=-1,
    )
    grid = normalized.reshape(1, 1, -1, 2)
    return F.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, 0]


def sparse_depth_metrics(
    rendered_depth: torch.Tensor,
    alpha: torch.Tensor,
    camera: Camera,
    depth_path: Path,
    alpha_threshold: float = 1e-4,
) -> dict[str, float | int | None]:
    """Compare rendered depth against sparse DTU depth samples."""
    if rendered_depth.shape != alpha.shape:
        raise ValueError("rendered depth and alpha must have matching shapes")
    if camera.image_path is None:
        raise ValueError("depth evaluation requires a camera image path")

    payload = _load_sparse_depth(depth_path)
    if payload["image_name"] != camera.image_path.name:
        raise ValueError(
            f"depth samples for {payload['image_name']!r} do not match "
            f"camera image {camera.image_path.name!r}"
        )
    coordinates = torch.as_tensor(
        payload["coord"],
        device=rendered_depth.device,
        dtype=rendered_depth.dtype,
    )
    ground_truth = torch.as_tensor(
        payload["depth"],
        device=rendered_depth.device,
        dtype=rendered_depth.dtype,
    )
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("depth sample coordinates must have shape (N, 2)")
    if ground_truth.shape != (coordinates.shape[0],):
        raise ValueError("depth samples must have shape (N,)")

    with Image.open(camera.image_path) as source_image:
        source_width, source_height = source_image.size
    sampled_depth = _sample_image_at_coordinates(
        rendered_depth,
        coordinates,
        source_width,
        source_height,
    )
    sampled_alpha = _sample_image_at_coordinates(
        alpha,
        coordinates,
        source_width,
        source_height,
    )
    valid_ground_truth = (
        torch.isfinite(coordinates).all(dim=-1)
        & torch.isfinite(ground_truth)
        & (ground_truth > 0)
        & (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < source_width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < source_height)
    )
    valid_prediction = (
        valid_ground_truth
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & (sampled_alpha > alpha_threshold)
    )
    ground_truth_count = int(valid_ground_truth.count_nonzero().item())
    predicted_count = int(valid_prediction.count_nonzero().item())
    coverage = predicted_count / ground_truth_count if ground_truth_count else 0.0
    if predicted_count:
        abs_rel = (
            (sampled_depth[valid_prediction] - ground_truth[valid_prediction]).abs()
            / ground_truth[valid_prediction]
        ).mean().item()
    else:
        abs_rel = None
    return {
        "depth_abs_rel": abs_rel,
        "depth_coverage": coverage,
        "depth_samples": ground_truth_count,
        "covered_depth_samples": predicted_count,
    }


class LpipsMetric:
    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS evaluation requires the evaluation extra: "
                "uv sync --extra evaluation"
            ) from error
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction = prediction.unsqueeze(0).mul(2.0).sub(1.0)
        target = target.unsqueeze(0).mul(2.0).sub(1.0)
        return float(self.model(prediction, target).item())


def _mean_metric(
    rows: Sequence[dict[str, Any]],
    name: str,
) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def _evaluate_split(
    model: GaussianModel,
    renderer: nn.Module,
    cameras: Sequence[Camera],
    images: Sequence[torch.Tensor],
    indices: Sequence[int],
    split: str,
    background: tuple[float, float, float],
    output_dir: Path,
    depths_dir: Path | None,
    lpips_metric: LpipsMetric,
) -> dict[str, Any]:
    split_dir = output_dir / "renders" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    device = model.means.device
    rows: list[dict[str, Any]] = []
    for index in indices:
        camera = cameras[index]
        if camera.image_id is None or camera.image_path is None:
            raise ValueError("evaluation requires registered cameras with image paths")
        target = images[index].to(device)
        output = renderer(model, camera.to(device), background)
        if output.depth is None:
            raise RuntimeError("evaluation requires renderer depth output")

        prediction = output.rgb.clamp(0.0, 1.0)
        alpha = output.alpha[0]
        depth = output.depth[0]
        valid_rendered_depth = (alpha > 1e-4) & torch.isfinite(depth) & (depth > 0)
        stem = f"image_{camera.image_id:04d}"
        _save_rgb(split_dir / f"{stem}_target.png", target)
        _save_rgb(split_dir / f"{stem}_rgb.png", prediction)
        _save_grayscale(split_dir / f"{stem}_alpha.png", alpha)
        _save_grayscale(
            split_dir / f"{stem}_depth.png",
            _visualize_depth(depth, valid_rendered_depth),
        )
        torch.save(depth.detach().cpu(), split_dir / f"{stem}_depth.pt")

        row: dict[str, Any] = {
            "split": split,
            "image_id": camera.image_id,
            "image_name": camera.image_path.name,
            "psnr": psnr(prediction, target),
            "lpips": lpips_metric(prediction, target),
            "depth_abs_rel": None,
            "depth_coverage": None,
            "depth_samples": 0,
            "covered_depth_samples": 0,
        }
        if depths_dir is not None:
            depth_path = depths_dir / f"{camera.image_path.stem}.pt"
            if depth_path.is_file():
                row.update(
                    sparse_depth_metrics(
                        depth,
                        alpha,
                        camera,
                        depth_path,
                    )
                )
            elif split == "test":
                raise FileNotFoundError(
                    f"test depth sample file does not exist: {depth_path}"
                )
        rows.append(row)

    return {
        "camera_count": len(rows),
        "mean_psnr": _mean_metric(rows, "psnr"),
        "mean_lpips": _mean_metric(rows, "lpips"),
        "mean_depth_abs_rel": _mean_metric(rows, "depth_abs_rel"),
        "mean_depth_coverage": _mean_metric(rows, "depth_coverage"),
        "cameras": rows,
    }


def _training_summary(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "training.jsonl"
    if not log_path.is_file():
        return {}
    last_record = None
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            last_record = json.loads(line)
    if last_record is None:
        return {}
    return {
        "final_step": last_record["step"],
        "final_loss": last_record["loss"],
        "final_gaussians": last_record["gaussians"],
    }


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint over its fixed train/test camera split."""
    if config.render.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("the CUDA renderer requires a CUDA-capable PyTorch installation")
        device = torch.device("cuda")
        renderer: nn.Module = CudaRasterizer(config.model.sh_degree)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = TorchRasterizer(config.model.sh_degree)

    scene = load_colmap_scene(
        config.data.colmap_dir,
        config.data.images_dir,
        config.data.downscale,
    )
    train_indices, test_indices = partition_camera_indices(
        scene,
        config.data.holdout_image_ids,
        train_image_ids=config.data.train_image_ids,
        test_image_ids=config.data.test_image_ids,
    )
    if not test_indices:
        raise ValueError("checkpoint evaluation requires at least one test image ID")

    model, _, step, checkpoint_metadata = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    configured_train_ids = [
        scene.cameras[index].image_id for index in train_indices
    ]
    configured_test_ids = [
        scene.cameras[index].image_id for index in test_indices
    ]
    saved_train_ids = checkpoint_metadata.get("train_image_ids")
    saved_test_ids = checkpoint_metadata.get(
        "test_image_ids",
        checkpoint_metadata.get("holdout_image_ids"),
    )
    if saved_train_ids is not None and saved_train_ids != configured_train_ids:
        raise ValueError("checkpoint training split does not match the evaluation config")
    if saved_test_ids is not None and saved_test_ids != configured_test_ids:
        raise ValueError("checkpoint test split does not match the evaluation config")
    model.eval()
    renderer = renderer.to(device)
    lpips_metric = LpipsMetric(device)
    evaluation_dir = output_dir or config.output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": _evaluate_split(
            model,
            renderer,
            scene.cameras,
            scene.images,
            train_indices,
            "train",
            config.render.background,
            evaluation_dir,
            config.data.depths_dir,
            lpips_metric,
        ),
        "test": _evaluate_split(
            model,
            renderer,
            scene.cameras,
            scene.images,
            test_indices,
            "test",
            config.render.background,
            evaluation_dir,
            config.data.depths_dir,
            lpips_metric,
        ),
    }
    run_dir = checkpoint_path.parent
    run_metadata_path = run_dir / "run_metadata.json"
    run_metadata = (
        json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if run_metadata_path.is_file()
        else {}
    )
    result = {
        "config": config_to_dict(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "checkpoint_metadata": checkpoint_metadata,
        "seed": config.training.seed,
        "train_image_ids": configured_train_ids,
        "test_image_ids": configured_test_ids,
        "model": {
            "gaussians": model.means.shape[0],
            "training_seconds": run_metadata.get(
                "training_seconds",
                checkpoint_metadata.get("training_seconds"),
            ),
            **_training_summary(run_dir),
        },
        "splits": splits,
    }
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "split",
        "image_id",
        "image_name",
        "psnr",
        "lpips",
        "depth_abs_rel",
        "depth_coverage",
        "depth_samples",
        "covered_depth_samples",
    ]
    with (evaluation_dir / "metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "test"):
            writer.writerows(splits[split]["cameras"])
    return result


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
