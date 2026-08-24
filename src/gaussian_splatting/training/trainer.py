import random

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.config import ExperimentConfig
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.cuda_backend import CudaRasterizer
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.checkpoint import save_checkpoint
from gaussian_splatting.training.losses import photometric_loss
from gaussian_splatting.types import Camera


def _load_target(camera: Camera, device: torch.device) -> torch.Tensor:
    if camera.image_path is None:
        raise ValueError("training cameras must reference an image")

    with Image.open(camera.image_path) as image:
        image = image.convert("RGB")
        if image.size != (camera.width, camera.height):
            image = image.resize((camera.width, camera.height), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().to(device)


def _build_optimizer(
    model: GaussianModel,
    config: ExperimentConfig,
) -> torch.optim.Adam:
    training = config.training
    return torch.optim.Adam(
        [
            {"params": [model.means], "lr": training.learning_rate_position},
            {"params": [model.sh_coefficients], "lr": training.learning_rate_features},
            {"params": [model.opacity_logits], "lr": training.learning_rate_opacity},
            {"params": [model.log_scales], "lr": training.learning_rate_scale},
            {"params": [model.quaternions], "lr": training.learning_rate_rotation},
        ],
        eps=1e-15,
    )


def train(config: ExperimentConfig) -> None:
    """Optimize a Gaussian scene from registered images."""
    if config.training.iterations < 1:
        raise ValueError("training.iterations must be positive")

    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    if config.render.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("the CUDA renderer requires a CUDA-capable PyTorch installation")
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(config.training.seed)
        renderer = CudaRasterizer(config.model.sh_degree)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = TorchRasterizer(config.model.sh_degree)

    scene = load_colmap_scene(
        config.data.colmap_dir,
        config.data.images_dir,
        config.data.downscale,
    )
    model = GaussianModel.from_point_cloud(
        points=scene.points,
        colors=scene.colors,
        sh_degree=config.model.sh_degree,
        initial_opacity=config.model.initial_opacity,
        initial_scale=config.model.initial_scale,
    ).to(device)
    renderer = renderer.to(device)
    optimizer = _build_optimizer(model, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_every = max(1, min(100, config.training.iterations // 10))

    model.train()
    for step in range(1, config.training.iterations + 1):
        camera = random.choice(scene.cameras)
        device_camera = camera.to(device)
        target = _load_target(camera, device)

        optimizer.zero_grad(set_to_none=True)
        output = renderer(model, device_camera, config.render.background)
        loss = photometric_loss(output.rgb, target)
        loss.backward()
        optimizer.step()

        # Density control belongs here once screen-space gradient statistics are
        # exposed by both renderers (Milestone 4).
        should_densify = (
            config.training.densify_from <= step <= config.training.densify_until
            and step % config.training.densify_every == 0
        )
        if should_densify:
            pass

        if step == 1 or step % report_every == 0:
            print(
                f"step {step:>{len(str(config.training.iterations))}}/"
                f"{config.training.iterations} loss={loss.item():.6f} "
                f"gaussians={model.means.shape[0]}"
            )

    save_checkpoint(
        config.output_dir / "final.pt",
        model,
        optimizer,
        step=config.training.iterations,
        metadata={"backend": config.render.backend},
    )

