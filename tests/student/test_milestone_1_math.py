import pytest
import torch

from gaussian_splatting.math.projection import covariance_3d, project_points
from gaussian_splatting.rendering.compositing import alpha_composite

pytestmark = pytest.mark.skip(reason="TODO(student): remove after implementing milestone 1")


def test_identity_camera_projects_to_principal_point() -> None:
    points = torch.tensor([[0.0, 0.0, 2.0]])
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    pixels, depth = project_points(points, torch.eye(4), intrinsics)
    torch.testing.assert_close(pixels, torch.tensor([[50.0, 40.0]]))
    torch.testing.assert_close(depth, torch.tensor([2.0]))


def test_identity_rotation_produces_diagonal_covariance() -> None:
    log_scales = torch.tensor([[0.0, torch.log(torch.tensor(2.0)), 0.0]])
    quaternion_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    covariance = covariance_3d(log_scales, quaternion_xyzw)
    torch.testing.assert_close(covariance, torch.diag(torch.tensor([1.0, 4.0, 1.0]))[None])


def test_front_to_back_alpha_compositing() -> None:
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    alphas = torch.tensor([0.5, 0.5])
    color, alpha = alpha_composite(colors, alphas)
    torch.testing.assert_close(color, torch.tensor([0.5, 0.0, 0.25]))
    torch.testing.assert_close(alpha, torch.tensor(0.75))

