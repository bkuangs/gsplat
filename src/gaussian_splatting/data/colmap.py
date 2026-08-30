import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.types import Camera


@dataclass(frozen=True)
class ColmapObservation:
    point_index: int
    image_id: int
    xy: torch.Tensor


@dataclass(frozen=True)
class ColmapScene:
    cameras: tuple[Camera, ...]
    points: torch.Tensor
    colors: torch.Tensor
    images: tuple[torch.Tensor, ...] = ()
    observations: tuple[ColmapObservation, ...] = ()


def load_image(camera: Camera) -> torch.Tensor:
    """Load one camera's target as contiguous float32 RGB in CHW layout."""
    if camera.image_path is None:
        raise ValueError("camera must reference an image")
    if not camera.image_path.is_file():
        raise FileNotFoundError(f"registered image does not exist: {camera.image_path}")

    with Image.open(camera.image_path) as image:
        image = image.convert("RGB")
        if image.size != (camera.width, camera.height):
            image = image.resize((camera.width, camera.height), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def reprojection_errors(scene: ColmapScene) -> torch.Tensor:
    """Return pixel errors for sparse points with recorded COLMAP observations."""
    if not scene.observations:
        return scene.points.new_empty((0,))

    from gaussian_splatting.math.projection import project_points

    cameras = {camera.image_id: camera for camera in scene.cameras}
    errors: list[torch.Tensor] = []
    for observation in scene.observations:
        camera = cameras.get(observation.image_id)
        if camera is None:
            raise ValueError(
                f"observation references unregistered COLMAP image {observation.image_id}"
            )
        pixel, depth = project_points(
            scene.points[observation.point_index : observation.point_index + 1],
            camera.world_to_camera,
            camera.intrinsics,
        )
        if depth.item() <= 0.0:
            raise ValueError("COLMAP observation has non-positive camera-space depth")
        target = observation.xy.to(device=pixel.device, dtype=pixel.dtype)
        errors.append(torch.linalg.vector_norm(pixel[0] - target))
    return torch.stack(errors)


def _read_colmap_arrays(model_dir: Path) -> dict[str, np.ndarray]:
    if importlib.util.find_spec("pycolmap") is None:
        raise RuntimeError(
            "COLMAP loading requires the data extra: uv sync --extra data"
        )

    with tempfile.TemporaryDirectory(prefix="gsplat-colmap-") as directory:
        output_path = Path(directory) / "reconstruction.npz"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gaussian_splatting.data._colmap_extract",
                str(model_dir),
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            message = detail[-1] if detail else "unknown pycolmap error"
            raise RuntimeError(f"failed to load COLMAP reconstruction: {message}")
        with np.load(output_path, allow_pickle=False) as payload:
            return {name: payload[name].copy() for name in payload.files}


def load_colmap_scene(model_dir: Path, images_dir: Path, downscale: int = 1) -> ColmapScene:
    """Load registered cameras and sparse points from an existing COLMAP model."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP model directory does not exist: {model_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_dir}")
    if downscale < 1:
        raise ValueError("downscale must be at least 1")

    arrays = _read_colmap_arrays(model_dir)
    cameras: list[Camera] = []
    for index, image_id in enumerate(arrays["image_ids"]):
        source_width = int(arrays["widths"][index])
        source_height = int(arrays["heights"][index])
        width = max(1, source_width // downscale)
        height = max(1, source_height // downscale)
        intrinsics = arrays["intrinsics"][index].copy()
        intrinsics[0] *= width / source_width
        intrinsics[1] *= height / source_height
        image_path = images_dir / str(arrays["image_names"][index])
        if not image_path.is_file():
            raise FileNotFoundError(f"registered image does not exist: {image_path}")
        cameras.append(
            Camera(
                world_to_camera=torch.from_numpy(arrays["world_to_camera"][index]),
                intrinsics=torch.from_numpy(intrinsics),
                width=width,
                height=height,
                image_path=image_path,
                image_id=int(image_id),
                camera_id=int(arrays["camera_ids"][index]),
            )
        )

    points = torch.from_numpy(arrays["points"])
    colors = torch.from_numpy(arrays["colors"] / 255.0)
    if not cameras:
        raise ValueError("COLMAP reconstruction contains no registered cameras")
    if points.shape[0] == 0:
        raise ValueError("COLMAP reconstruction contains no sparse points")

    observations: list[ColmapObservation] = []
    cameras_by_image_id = {camera.image_id: camera for camera in cameras}
    source_sizes = {
        int(image_id): (int(arrays["widths"][index]), int(arrays["heights"][index]))
        for index, image_id in enumerate(arrays["image_ids"])
    }
    for point_index, image_id, xy in zip(
        arrays["observation_point_indices"],
        arrays["observation_image_ids"],
        arrays["observation_xy"],
        strict=True,
    ):
        camera = cameras_by_image_id[int(image_id)]
        source_width, source_height = source_sizes[int(image_id)]
        scaled_xy = xy * np.asarray(
            [camera.width / source_width, camera.height / source_height],
            dtype=np.float32,
        )
        observations.append(
            ColmapObservation(
                point_index=int(point_index),
                image_id=int(image_id),
                xy=torch.from_numpy(scaled_xy),
            )
        )

    camera_tuple = tuple(cameras)
    return ColmapScene(
        cameras=camera_tuple,
        points=points,
        colors=colors,
        images=tuple(load_image(camera) for camera in camera_tuple),
        observations=tuple(observations),
    )
