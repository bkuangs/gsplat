import torch

from gaussian_splatting.model import GaussianModel


def test_point_cloud_initialization_has_valid_parameters() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model = GaussianModel.from_point_cloud(
        points,
        colors,
        sh_degree=0,
        initial_opacity=0.1,
        initial_scale=0.01,
    )
    assert model.means.shape == (2, 3)
    assert model.sh_coefficients.shape == (2, 1, 3)
    torch.testing.assert_close(
        model.quaternions,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(model.opacities, torch.full((2, 1), 0.1))
    torch.testing.assert_close(model.scales, torch.ones(2, 3))


def test_point_cloud_initialization_uses_local_spacing() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
    model = GaussianModel.from_point_cloud(
        points,
        torch.ones_like(points),
        sh_degree=0,
        initial_opacity=0.1,
        initial_scale=0.01,
    )
    expected = torch.tensor([1.0, 1.0, 3.0])[:, None].expand_as(points)
    torch.testing.assert_close(model.scales, expected)
