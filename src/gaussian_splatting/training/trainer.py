import json
import random

import torch

from gaussian_splatting.config import ExperimentConfig
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.cuda_backend import CudaRasterizer
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.checkpoint import save_checkpoint
from gaussian_splatting.training.densification import (
    accumulate_densification_stats,
    create_densification_stats,
    update_gaussian_topology,
)
from gaussian_splatting.training.losses import photometric_loss


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
    if config.training.densify_every < 1:
        raise ValueError("training.densify_every must be positive")
    if config.training.densify_until < config.training.densify_from:
        raise ValueError("training.densify_until must not precede densify_from")

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
    stats = create_densification_stats(model)
    scene_extent = torch.linalg.vector_norm(
        scene.points.max(dim=0).values - scene.points.min(dim=0).values
    ).item()
    if scene_extent <= 0.0:
        scene_extent = config.model.initial_scale

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_every = max(1, min(100, config.training.iterations // 10))
    log_path = config.output_dir / "training.jsonl"

    model.train()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.training.iterations + 1):
            camera_index = random.randrange(len(scene.cameras))
            camera = scene.cameras[camera_index]
            device_camera = camera.to(device)
            target = scene.images[camera_index].to(device)

            optimizer.zero_grad(set_to_none=True)
            output = renderer(model, device_camera, config.render.background)
            loss = photometric_loss(output.rgb, target)
            loss.backward()
            if step <= config.training.densify_until:
                accumulate_densification_stats(stats, output)
            optimizer.step()

            should_densify = (
                config.training.densify_from <= step <= config.training.densify_until
                and step % config.training.densify_every == 0
            )
            if should_densify:
                update_gaussian_topology(
                    model,
                    optimizer,
                    stats,
                    gradient_threshold=config.training.densify_gradient_threshold,
                    opacity_threshold=config.training.densify_opacity_threshold,
                    scene_extent=scene_extent,
                    scale_threshold=config.training.densify_scale_threshold,
                    max_screen_radius=config.training.densify_max_screen_radius,
                )
                stats = create_densification_stats(model)

            record = {
                "step": step,
                "loss": loss.item(),
                "gaussians": model.means.shape[0],
            }
            log.write(json.dumps(record) + "\n")
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
        metadata={
            "backend": config.render.backend,
            "gaussians": model.means.shape[0],
        },
    )
