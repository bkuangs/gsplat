import pytest
import torch

from gaussian_splatting.types import Camera


def test_camera_validates_matrix_shapes() -> None:
    with pytest.raises(ValueError, match="world_to_camera"):
        Camera(torch.eye(3), torch.eye(3), width=100, height=100)


def test_camera_moves_both_tensors() -> None:
    camera = Camera(torch.eye(4), torch.eye(3), width=100, height=50)
    moved = camera.to("cpu")
    assert moved.world_to_camera.device.type == "cpu"
    assert moved.intrinsics.device.type == "cpu"

