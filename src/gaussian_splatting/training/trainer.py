import json
import random
import time
from dataclasses import asdict

import torch

from gaussian_splatting.config import ExperimentConfig, config_to_dict
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
from gaussian_splatting.training.evaluation import evaluate_holdout
from gaussian_splatting.training.losses import photometric_loss
from gaussian_splatting.training.splits import (
    partition_camera_indices as _partition_camera_indices,
)


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
    training_indices, holdout_indices = _partition_camera_indices(
        scene,
        config.data.holdout_image_ids,
        train_image_ids=config.data.train_image_ids,
        test_image_ids=config.data.test_image_ids,
    )
    scene_extent = torch.linalg.vector_norm(
        scene.points.max(dim=0).values - scene.points.min(dim=0).values
    ).item()
    if scene_extent <= 0.0:
        scene_extent = config.model.initial_scale

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_every = max(1, min(100, config.training.iterations // 10))
    log_path = config.output_dir / "training.jsonl"
    initial_holdout = None
    if holdout_indices:
        initial_holdout = evaluate_holdout(
            model,
            renderer,
            scene.cameras,
            scene.images,
            holdout_indices,
            config.render.background,
            config.output_dir,
            stage="initial",
            step=0,
        )
        print(
            f"holdout step 0 mean_psnr={initial_holdout['mean_psnr']:.4f} "
            f"cameras={len(holdout_indices)}"
        )

    model.train()
    training_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.training.iterations + 1):
            camera_index = random.choice(training_indices)
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
            topology_update = None
            if should_densify:
                topology_update = update_gaussian_topology(
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
            if topology_update is not None:
                record["densification"] = asdict(topology_update)
            log.write(json.dumps(record) + "\n")
            if step == 1 or step % report_every == 0:
                print(
                    f"step {step:>{len(str(config.training.iterations))}}/"
                    f"{config.training.iterations} loss={loss.item():.6f} "
                    f"gaussians={model.means.shape[0]}"
                )
    training_seconds = time.perf_counter() - training_started

    final_holdout = None
    if holdout_indices:
        final_holdout = evaluate_holdout(
            model,
            renderer,
            scene.cameras,
            scene.images,
            holdout_indices,
            config.render.background,
            config.output_dir,
            stage="final",
            step=config.training.iterations,
        )
        holdout_summary = {
            "holdout_image_ids": [
                scene.cameras[index].image_id for index in holdout_indices
            ],
            "initial": initial_holdout,
            "final": final_holdout,
            "mean_psnr_improvement": (
                final_holdout["mean_psnr"] - initial_holdout["mean_psnr"]
            ),
        }
        (config.output_dir / "holdout_metrics.json").write_text(
            json.dumps(holdout_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"holdout step {config.training.iterations} "
            f"mean_psnr={final_holdout['mean_psnr']:.4f} "
            f"improvement={holdout_summary['mean_psnr_improvement']:+.4f}"
        )

    save_checkpoint(
        config.output_dir / "final.pt",
        model,
        optimizer,
        step=config.training.iterations,
        metadata={
            "backend": config.render.backend,
            "gaussians": model.means.shape[0],
            "train_image_ids": [
                scene.cameras[index].image_id for index in training_indices
            ],
            "test_image_ids": [
                scene.cameras[index].image_id for index in holdout_indices
            ],
            "holdout_mean_psnr": (
                final_holdout["mean_psnr"] if final_holdout is not None else None
            ),
            "training_seconds": training_seconds,
        },
    )
    run_metadata = {
        "config": config_to_dict(config),
        "training_seconds": training_seconds,
        "iterations": config.training.iterations,
        "final_gaussians": model.means.shape[0],
        "train_image_ids": [
            scene.cameras[index].image_id for index in training_indices
        ],
        "test_image_ids": [
            scene.cameras[index].image_id for index in holdout_indices
        ],
        "checkpoint": str(config.output_dir / "final.pt"),
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
