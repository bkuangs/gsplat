import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from gaussian_splatting.data.colmap import (
    ColmapObservation,
    ColmapScene,
    load_colmap_scene,
    load_image,
    reprojection_errors,
)
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.densification import (
    DensificationStats,
    update_gaussian_topology,
)
from gaussian_splatting.types import Camera


def _camera(image_path: Path | None = None) -> Camera:
    intrinsics = torch.tensor(
        [[50.0, 0.0, 4.5], [0.0, 50.0, 4.5], [0.0, 0.0, 1.0]]
    )
    return Camera(
        torch.eye(4),
        intrinsics,
        width=9,
        height=9,
        image_path=image_path,
        image_id=7,
        camera_id=3,
    )


def _model() -> GaussianModel:
    return GaussianModel.from_point_cloud(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        sh_degree=0,
        initial_opacity=0.8,
        initial_scale=0.05,
    )


def test_image_loading_resizes_to_camera_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "target.png"
    Image.new("RGB", (18, 18), color=(255, 128, 0)).save(image_path)
    image = load_image(_camera(image_path))
    assert image.shape == (3, 9, 9)
    assert image.dtype == torch.float32
    assert image.is_contiguous()
    assert image.min() >= 0 and image.max() <= 1


def test_colmap_observation_reprojects_to_recorded_pixel() -> None:
    camera = _camera()
    point = torch.tensor([[0.02, -0.04, 2.0]])
    scene = ColmapScene(
        cameras=(camera,),
        images=(torch.zeros(3, 9, 9),),
        points=point,
        colors=torch.ones_like(point),
        observations=(
            ColmapObservation(
                point_index=0,
                image_id=7,
                xy=torch.tensor([5.0, 3.5]),
            ),
        ),
    )
    torch.testing.assert_close(reprojection_errors(scene), torch.zeros(1))


def test_colmap_loader_preserves_ids_and_downscales_geometry(tmp_path: Path) -> None:
    if importlib.util.find_spec("pycolmap") is None:
        pytest.skip("pycolmap is not installed")
    model_dir = tmp_path / "sparse"
    images_dir = tmp_path / "images"
    model_dir.mkdir()
    images_dir.mkdir()
    (model_dir / "cameras.txt").write_text(
        "3 PINHOLE 18 18 50 50 9 9\n", encoding="utf-8"
    )
    (model_dir / "images.txt").write_text(
        "7 1 0 0 0 0 0 0 3 target.png\n"
        "9.5 8 11\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "11 0.02 -0.04 2 255 0 0 0.0 7 0\n", encoding="utf-8"
    )
    Image.new("RGB", (18, 18), color=(255, 0, 0)).save(images_dir / "target.png")

    scene = load_colmap_scene(model_dir, images_dir, downscale=2)

    assert len(scene.cameras) == len(scene.images) == 1
    camera = scene.cameras[0]
    assert camera.image_id == 7
    assert camera.camera_id == 3
    assert (camera.width, camera.height) == (9, 9)
    torch.testing.assert_close(
        camera.intrinsics,
        torch.tensor([[25.0, 0.0, 4.5], [0.0, 25.0, 4.5], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(reprojection_errors(scene), torch.zeros(1), atol=1e-6, rtol=0)


def test_reference_renderer_outputs_depth_and_finite_gradients() -> None:
    model = _model()
    output = TorchRasterizer(sh_degree=0)(model, _camera(), (0.0, 0.0, 0.0))
    assert output.rgb.shape == (3, 9, 9)
    assert output.alpha.shape == (1, 9, 9)
    assert output.depth is not None and output.depth.shape == (1, 9, 9)
    assert output.radii is not None and output.radii.shape == (1,)
    assert output.means_2d is not None
    output.rgb.sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert output.means_2d.grad is not None
    visible_depth = output.depth[output.alpha > 0]
    torch.testing.assert_close(visible_depth, torch.full_like(visible_depth, 2.0))


def test_reference_renderer_retains_zero_screen_gradient_when_scene_is_hidden() -> None:
    model = _model()
    model.means.data[:, 2] = -2.0
    output = TorchRasterizer(sh_degree=0)(model, _camera(), (0.1, 0.2, 0.3))
    output.rgb.sum().backward()
    assert output.means_2d is not None and output.means_2d.grad is not None
    torch.testing.assert_close(output.means_2d.grad, torch.zeros_like(output.means_2d))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_backend_agrees_with_reference_on_tiny_scene() -> None:
    pytest.importorskip("gsplat")
    from gaussian_splatting.rendering.cuda_backend import CudaRasterizer

    cpu_model = _model()
    cpu_output = TorchRasterizer(sh_degree=0)(cpu_model, _camera(), (0.0, 0.0, 0.0))
    cuda_model = _model().cuda()
    cuda_output = CudaRasterizer(sh_degree=0)(
        cuda_model, _camera().to("cuda"), (0.0, 0.0, 0.0)
    )

    torch.testing.assert_close(
        cuda_output.rgb.cpu(), cpu_output.rgb, atol=0.15, rtol=0.15
    )
    visible = (cuda_output.alpha.cpu() > 1e-4) & (cpu_output.alpha > 1e-4)
    assert cuda_output.depth is not None and cpu_output.depth is not None
    torch.testing.assert_close(
        cuda_output.depth.cpu()[visible],
        cpu_output.depth[visible],
        atol=0.1,
        rtol=0.05,
    )
    assert cuda_output.radii is not None and (cuda_output.radii > 0).any()
    cuda_output.rgb.sum().backward()
    assert cuda_output.means_2d is not None and cuda_output.means_2d.grad is not None
    assert torch.isfinite(cuda_output.means_2d.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in cuda_model.parameters()
    )


def test_clone_split_prune_preserves_optimizer_state_shapes() -> None:
    means = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
    )
    quaternions = torch.zeros(3, 4)
    quaternions[:, 0] = 1.0
    model = GaussianModel(
        means=means,
        log_scales=torch.tensor(
            [[-6.0, -6.0, -6.0], [-1.0, -1.0, -1.0], [-6.0, -6.0, -6.0]]
        ),
        quaternions=quaternions,
        opacity_logits=torch.tensor([[2.0], [2.0], [-10.0]]),
        sh_coefficients=torch.zeros(3, 1, 3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    stats = DensificationStats(
        position_gradient_accumulator=torch.tensor([1.0, 1.0, 1.0]),
        observation_count=torch.ones(3),
        max_screen_radius=torch.ones(3),
    )

    update_gaussian_topology(
        model,
        optimizer,
        stats,
        gradient_threshold=0.5,
        opacity_threshold=0.01,
        scene_extent=1.0,
    )

    assert model.means.shape == (4, 3)
    optimizer_parameters = {
        parameter for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_parameters == set(model.parameters())
    for parameter, state in optimizer.state.items():
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape
