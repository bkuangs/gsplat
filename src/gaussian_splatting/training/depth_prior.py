import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gaussian_splatting.config import ExperimentConfig
from gaussian_splatting.data.colmap import ColmapObservation, ColmapScene, load_colmap_scene
from gaussian_splatting.image_sampling import sample_image_at_coordinates
from gaussian_splatting.training.splits import partition_camera_indices
from gaussian_splatting.types import Camera

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass(frozen=True)
class DepthAlignment:
    scale: float
    shift: float
    polarity: int
    anchor_count: int
    inlier_count: int
    positive_fraction: float
    median_abs_rel: float


@dataclass(frozen=True)
class DepthPrior:
    depth: torch.Tensor
    mask: torch.Tensor
    alignment: DepthAlignment


class DepthAnythingPredictor:
    """Frozen relative-depth predictor used once during prior generation."""

    def __init__(
        self,
        model_id: str = DEFAULT_DEPTH_MODEL,
        device: torch.device | None = None,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "depth-prior generation requires the depth extra: "
                "uv sync --extra depth"
            ) from error
        self.model_id = model_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, camera: Camera) -> torch.Tensor:
        if camera.image_path is None:
            raise ValueError("monocular depth prediction requires an image path")
        with Image.open(camera.image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            name: value.to(self.device)
            for name, value in inputs.items()
        }
        outputs = self.model(**inputs)
        result = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(camera.height, camera.width)],
        )[0]["predicted_depth"]
        prediction = torch.as_tensor(result).detach().to(dtype=torch.float32)
        if prediction.shape != (camera.height, camera.width):
            raise RuntimeError(
                "depth model returned an unexpected shape: "
                f"{tuple(prediction.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError("depth model returned non-finite values")
        return prediction.cpu()


def _weighted_affine_fit(
    predictor_values: torch.Tensor,
    target_values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    design = torch.stack(
        [predictor_values, torch.ones_like(predictor_values)],
        dim=-1,
    )
    square_root_weights = weights.sqrt()
    weighted_design = design * square_root_weights[:, None]
    weighted_target = target_values * square_root_weights
    return torch.linalg.lstsq(weighted_design, weighted_target).solution


def fit_inverse_depth_alignment(
    predicted_inverse_depth: torch.Tensor,
    colmap_depth: torch.Tensor,
    *,
    min_anchors: int = 20,
    max_median_abs_rel: float = 0.25,
    huber_delta: float = 1.5,
    iterations: int = 5,
) -> DepthAlignment:
    """Robustly align monocular relative inverse depth to COLMAP inverse depth."""
    predicted = predicted_inverse_depth.detach().to(dtype=torch.float64).flatten()
    depth = colmap_depth.detach().to(dtype=torch.float64).flatten()
    valid = (
        torch.isfinite(predicted)
        & torch.isfinite(depth)
        & (depth > 0)
    )
    predicted = predicted[valid]
    depth = depth[valid]
    if predicted.numel() < min_anchors:
        raise ValueError(
            f"depth alignment requires at least {min_anchors} valid anchors; "
            f"received {predicted.numel()}"
        )
    if predicted.std() <= torch.finfo(predicted.dtype).eps:
        raise ValueError("monocular inverse depth has no usable spatial variation")

    target_inverse = depth.reciprocal()
    weights = torch.ones_like(predicted)
    parameters = _weighted_affine_fit(predicted, target_inverse, weights)
    for _ in range(iterations):
        residual = predicted * parameters[0] + parameters[1] - target_inverse
        residual_scale = (
            1.4826 * (residual - residual.median()).abs().median()
        ).clamp_min(torch.finfo(residual.dtype).eps)
        normalized = residual.abs() / (huber_delta * residual_scale)
        weights = torch.where(
            normalized <= 1.0,
            torch.ones_like(normalized),
            normalized.reciprocal(),
        )
        parameters = _weighted_affine_fit(predicted, target_inverse, weights)

    residual = predicted * parameters[0] + parameters[1] - target_inverse
    residual_scale = (
        1.4826 * (residual - residual.median()).abs().median()
    ).clamp_min(torch.finfo(residual.dtype).eps)
    normalized = residual.abs() / (huber_delta * residual_scale)
    weights = torch.where(
        normalized <= 1.0,
        torch.ones_like(normalized),
        normalized.reciprocal(),
    )
    scale = float(parameters[0].item())
    shift = float(parameters[1].item())
    polarity = 1
    if scale < 0:
        polarity = -1
        scale = -scale
    aligned_inverse = polarity * predicted * scale + shift
    positive = aligned_inverse > torch.finfo(aligned_inverse.dtype).eps
    positive_fraction = float(positive.float().mean().item())
    if positive_fraction < 0.9:
        raise ValueError(
            "depth alignment rejected because fewer than 90% of anchors "
            "produce positive inverse depth"
        )
    predicted_depth = aligned_inverse[positive].reciprocal()
    positive_depth = depth[positive]
    abs_rel = (predicted_depth - positive_depth).abs() / positive_depth
    median_abs_rel = float(abs_rel.median().item())
    if not math.isfinite(median_abs_rel) or median_abs_rel > max_median_abs_rel:
        raise ValueError(
            "depth alignment median AbsRel exceeds the acceptance threshold: "
            f"{median_abs_rel:.4f} > {max_median_abs_rel:.4f}"
        )
    return DepthAlignment(
        scale=scale,
        shift=shift,
        polarity=polarity,
        anchor_count=int(predicted.numel()),
        inlier_count=int((weights >= 0.5).count_nonzero().item()),
        positive_fraction=positive_fraction,
        median_abs_rel=median_abs_rel,
    )


def _camera_space_depth(points: torch.Tensor, camera: Camera) -> torch.Tensor:
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    return (homogeneous @ camera.world_to_camera.T)[:, 2]


def _alignment_anchors(
    scene: ColmapScene,
    camera: Camera,
    observations: Sequence[ColmapObservation],
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if camera.image_path is None:
        raise ValueError("depth alignment requires a camera image path")
    point_indices = torch.tensor(
        [observation.point_index for observation in observations],
        dtype=torch.long,
    )
    coordinates = torch.stack([observation.xy for observation in observations])
    sampled_prediction = sample_image_at_coordinates(
        prediction,
        coordinates.to(dtype=prediction.dtype),
        camera.width,
        camera.height,
    )
    colmap_depth = _camera_space_depth(scene.points[point_indices], camera)
    in_bounds = (
        torch.isfinite(coordinates).all(dim=-1)
        & (coordinates[:, 0] >= 0.5)
        & (coordinates[:, 0] <= camera.width - 0.5)
        & (coordinates[:, 1] >= 0.5)
        & (coordinates[:, 1] <= camera.height - 0.5)
    )
    return sampled_prediction[in_bounds], colmap_depth[in_bounds]


def align_depth_prior(
    prediction: torch.Tensor,
    colmap_depth: torch.Tensor,
    alignment: DepthAlignment,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an accepted inverse-depth alignment to a dense prediction."""
    aligned_inverse = (
        alignment.polarity * prediction.to(dtype=torch.float32) * alignment.scale
        + alignment.shift
    )
    mask = torch.isfinite(aligned_inverse) & (aligned_inverse > 1e-8)
    depth = torch.zeros_like(aligned_inverse)
    depth[mask] = aligned_inverse[mask].reciprocal()
    if colmap_depth.numel():
        valid_anchor_depth = colmap_depth[
            torch.isfinite(colmap_depth) & (colmap_depth > 0)
        ]
        if valid_anchor_depth.numel():
            minimum = valid_anchor_depth.quantile(0.01) * 0.25
            maximum = valid_anchor_depth.quantile(0.99) * 4.0
            mask &= (depth >= minimum) & (depth <= maximum)
            depth[~mask] = 0.0
    return depth, mask


def _save_depth_visualization(path: Path, depth: torch.Tensor, mask: torch.Tensor) -> None:
    visualization = torch.zeros_like(depth)
    valid = depth[mask]
    if valid.numel():
        low = valid.quantile(0.02)
        high = valid.quantile(0.98)
        if high > low:
            visualization[mask] = ((depth[mask] - low) / (high - low)).clamp(0.0, 1.0)
        else:
            visualization[mask] = 1.0
    array = (
        visualization.mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array, mode="L").save(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_tensor_hash(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def _camera_sha256(camera: Camera) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, camera.world_to_camera)
    _update_tensor_hash(digest, camera.intrinsics)
    digest.update(str((camera.width, camera.height, camera.image_id)).encode())
    return digest.hexdigest()


def _alignment_source_sha256(
    scene: ColmapScene,
    camera: Camera,
    observations: Sequence[ColmapObservation],
) -> str:
    digest = hashlib.sha256()
    digest.update(_camera_sha256(camera).encode())
    point_indices = torch.tensor(
        [observation.point_index for observation in observations],
        dtype=torch.int64,
    )
    coordinates = (
        torch.stack([observation.xy for observation in observations])
        if observations
        else torch.empty(0, 2)
    )
    _update_tensor_hash(digest, point_indices)
    _update_tensor_hash(digest, coordinates)
    _update_tensor_hash(digest, scene.points[point_indices])
    return digest.hexdigest()


def generate_depth_priors(
    config: ExperimentConfig,
    output_dir: Path,
    predictor: Callable[[Camera], torch.Tensor],
    *,
    model_id: str = DEFAULT_DEPTH_MODEL,
    min_anchors: int = 20,
    max_median_abs_rel: float = 0.25,
) -> dict[str, Any]:
    """Generate and persist aligned priors for the configured training cameras."""
    scene = load_colmap_scene(
        config.data.colmap_dir,
        config.data.images_dir,
        config.data.downscale,
    )
    train_indices, _ = partition_camera_indices(
        scene,
        config.data.holdout_image_ids,
        train_image_ids=config.data.train_image_ids,
        test_image_ids=config.data.test_image_ids,
    )
    observations_by_image: dict[int, list[ColmapObservation]] = defaultdict(list)
    for observation in scene.observations:
        observations_by_image[observation.image_id].append(observation)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failed_image_ids: list[int] = []
    for index in train_indices:
        camera = scene.cameras[index]
        if camera.image_id is None or camera.image_path is None:
            raise ValueError("depth-prior generation requires registered image paths")
        observations = observations_by_image[camera.image_id]
        if not observations:
            raise ValueError(
                f"camera {camera.image_id} has no COLMAP observations for alignment"
            )
        stem = f"image_{camera.image_id:04d}"
        prior_path = output_dir / f"{stem}.pt"
        visualization_path = output_dir / f"{stem}.png"
        try:
            prediction = predictor(camera)
            if prediction.shape != (camera.height, camera.width):
                raise ValueError(
                    f"depth predictor returned {tuple(prediction.shape)} for camera "
                    f"{camera.image_id}; expected {(camera.height, camera.width)}"
                )
            anchor_prediction, anchor_depth = _alignment_anchors(
                scene,
                camera,
                observations,
                prediction,
            )
            alignment = fit_inverse_depth_alignment(
                anchor_prediction,
                anchor_depth,
                min_anchors=min_anchors,
                max_median_abs_rel=max_median_abs_rel,
            )
            depth, mask = align_depth_prior(prediction, anchor_depth, alignment)
            if not mask.any():
                raise ValueError(
                    f"aligned depth prior is empty for camera {camera.image_id}"
                )

            _atomic_torch_save(
                prior_path,
                {
                    "image_id": camera.image_id,
                    "image_name": camera.image_path.name,
                    "width": camera.width,
                    "height": camera.height,
                    "depth": depth,
                    "mask": mask,
                    "alignment": asdict(alignment),
                    "model_id": model_id,
                },
            )
            _save_depth_visualization(visualization_path, depth, mask)
            row = {
                "image_id": camera.image_id,
                "image_name": camera.image_path.name,
                "status": "accepted",
                "valid_fraction": float(mask.float().mean().item()),
                "sha256": _sha256(prior_path),
                "source_image_sha256": _sha256(camera.image_path),
                "camera_sha256": _camera_sha256(camera),
                "alignment_source_sha256": _alignment_source_sha256(
                    scene,
                    camera,
                    observations,
                ),
                **asdict(alignment),
            }
            rows.append(row)
            print(
                f"depth prior image_id={camera.image_id} "
                f"anchors={alignment.anchor_count} "
                f"median_abs_rel={alignment.median_abs_rel:.4f} "
                f"valid_fraction={row['valid_fraction']:.4f}"
            )
        except (RuntimeError, ValueError) as error:
            prior_path.unlink(missing_ok=True)
            visualization_path.unlink(missing_ok=True)
            failed_image_ids.append(camera.image_id)
            rows.append(
                {
                    "image_id": camera.image_id,
                    "image_name": camera.image_path.name,
                    "status": "rejected",
                    "reason": str(error),
                }
            )
            print(f"depth prior image_id={camera.image_id} rejected: {error}")

    metadata = {
        "model_id": model_id,
        "train_image_ids": [
            scene.cameras[index].image_id for index in train_indices
        ],
        "min_anchors": min_anchors,
        "max_median_abs_rel": max_median_abs_rel,
        "cameras": rows,
    }
    temporary_path = output_dir / "metadata.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_dir / "metadata.json")
    if failed_image_ids:
        names = ", ".join(str(image_id) for image_id in failed_image_ids)
        raise RuntimeError(f"depth-prior alignment rejected camera(s): {names}")
    return metadata


def load_depth_priors(
    scene: ColmapScene,
    indices: Sequence[int],
    directory: Path,
    device: torch.device,
) -> tuple[dict[int, DepthPrior], dict[str, Any]]:
    """Load fixed dense depth priors for every training camera."""
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"depth prior metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_ids = [
        scene.cameras[index].image_id for index in indices
    ]
    if metadata.get("train_image_ids") != expected_ids:
        raise ValueError("depth prior metadata training split does not match")
    rows = {
        row["image_id"]: row
        for row in metadata.get("cameras", [])
        if row.get("status") == "accepted"
    }
    observations_by_image: dict[int, list[ColmapObservation]] = defaultdict(list)
    for observation in scene.observations:
        observations_by_image[observation.image_id].append(observation)
    priors: dict[int, DepthPrior] = {}
    for index in indices:
        camera = scene.cameras[index]
        if camera.image_id is None or camera.image_path is None:
            raise ValueError("depth-prior training requires registered image paths")
        path = directory / f"image_{camera.image_id:04d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"depth prior does not exist: {path}")
        row = rows.get(camera.image_id)
        if row is None:
            raise ValueError(f"depth prior metadata is missing camera {camera.image_id}")
        if row.get("sha256") != _sha256(path):
            raise ValueError(f"depth prior checksum does not match metadata: {path}")
        if row.get("source_image_sha256") != _sha256(camera.image_path):
            raise ValueError(f"depth prior source image does not match: {path}")
        if row.get("camera_sha256") != _camera_sha256(camera):
            raise ValueError(f"depth prior camera geometry does not match: {path}")
        if row.get("alignment_source_sha256") != _alignment_source_sha256(
            scene,
            camera,
            observations_by_image[camera.image_id],
        ):
            raise ValueError(f"depth prior COLMAP alignment source does not match: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["image_id"] != camera.image_id:
            raise ValueError(f"depth prior image ID does not match camera: {path}")
        if payload["image_name"] != camera.image_path.name:
            raise ValueError(f"depth prior image name does not match camera: {path}")
        if (payload["width"], payload["height"]) != (camera.width, camera.height):
            raise ValueError(f"depth prior dimensions do not match camera: {path}")
        depth = payload["depth"].to(device=device, dtype=torch.float32)
        mask = payload["mask"].to(device=device, dtype=torch.bool)
        expected_shape = (camera.height, camera.width)
        if depth.shape != expected_shape or mask.shape != expected_shape:
            raise ValueError(f"depth prior tensors have the wrong shape: {path}")
        valid = mask & torch.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError(f"depth prior contains no valid pixels: {path}")
        alignment = DepthAlignment(**payload["alignment"])
        priors[index] = DepthPrior(depth=depth, mask=valid, alignment=alignment)
    provenance = {
        "metadata": metadata,
        "metadata_sha256": _sha256(metadata_path),
    }
    return priors, provenance
