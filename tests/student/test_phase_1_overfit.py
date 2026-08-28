import torch

from gaussian_splatting.metrics import psnr
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.losses import photometric_loss
from gaussian_splatting.types import Camera


def test_tiny_synthetic_scene_overfits() -> None:
    torch.manual_seed(7)
    camera = Camera(
        world_to_camera=torch.eye(4),
        intrinsics=torch.tensor(
            [[30.0, 0.0, 7.5], [0.0, 30.0, 7.5], [0.0, 0.0, 1.0]]
        ),
        width=16,
        height=16,
    )
    points = torch.tensor([[-0.12, -0.04, 2.0], [0.14, 0.08, 2.2]])
    colors = torch.tensor([[0.9, 0.15, 0.1], [0.1, 0.3, 0.9]])
    renderer = TorchRasterizer(sh_degree=0)

    teacher = GaussianModel.from_point_cloud(
        points,
        colors,
        sh_degree=0,
        initial_opacity=0.85,
        initial_scale=0.08,
    )
    teacher.log_scales.data.fill_(torch.log(torch.tensor(0.08)))
    with torch.no_grad():
        target = renderer(teacher, camera, (0.0, 0.0, 0.0)).rgb

    student = GaussianModel.from_point_cloud(
        points
        + torch.tensor([[0.03, -0.02, 0.05], [-0.02, 0.02, -0.04]]),
        torch.full_like(colors, 0.5),
        sh_degree=0,
        initial_opacity=0.5,
        initial_scale=0.06,
    )
    student.log_scales.data.fill_(torch.log(torch.tensor(0.06)))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)

    with torch.no_grad():
        initial_prediction = renderer(student, camera, (0.0, 0.0, 0.0)).rgb
        initial_loss = photometric_loss(initial_prediction, target).item()

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        prediction = renderer(student, camera, (0.0, 0.0, 0.0)).rgb
        loss = photometric_loss(prediction, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_prediction = renderer(student, camera, (0.0, 0.0, 0.0)).rgb
        final_loss = photometric_loss(final_prediction, target).item()
        final_psnr = psnr(final_prediction.clamp(0.0, 1.0), target)

    assert final_loss < initial_loss / 50.0
    assert final_psnr > 45.0

