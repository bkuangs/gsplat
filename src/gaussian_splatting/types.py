from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Camera:
    """Pinhole camera using a world-to-camera transform."""

    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    width: int
    height: int
    image_path: Path | None = None

    def __post_init__(self) -> None:
        if self.world_to_camera.shape != (4, 4):
            raise ValueError("world_to_camera must have shape (4, 4)")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.world_to_camera.device != self.intrinsics.device:
            raise ValueError("camera tensors must be on the same device")

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(
            world_to_camera=self.world_to_camera.to(device),
            intrinsics=self.intrinsics.to(device),
            width=self.width,
            height=self.height,
            image_path=self.image_path,
        )


@dataclass(frozen=True)
class RenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor | None = None
    radii: torch.Tensor | None = None

