import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from gaussian_splatting.config import ExperimentConfig, config_to_dict, configs_match
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.rendering.cuda_backend import CudaRasterizer
from gaussian_splatting.rendering.torch_backend import TorchRasterizer
from gaussian_splatting.training.checkpoint import (
    _fsync_directory,
    load_checkpoint,
    save_checkpoint,
)
from gaussian_splatting.training.densification import (
    DensificationStats,
    accumulate_densification_stats,
    create_densification_stats,
    update_gaussian_topology,
)
from gaussian_splatting.training.depth_prior import load_depth_priors
from gaussian_splatting.training.evaluation import evaluate_holdout
from gaussian_splatting.training.losses import depth_prior_loss, photometric_loss
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)
    _fsync_directory(path.parent)


def _truncate_training_log(path: Path, step: int) -> None:
    if not path.is_file():
        return
    records: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record["step"]) <= step:
                records.append(json.dumps(record) + "\n")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("".join(records), encoding="utf-8")
    temporary_path.replace(path)


def _checkpoint_metadata(
    config: ExperimentConfig,
    stats: DensificationStats,
    training_seconds: float,
    initial_holdout: dict[str, Any] | None,
    training_image_ids: list[int | None],
    test_image_ids: list[int | None],
    depth_prior_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "config": config_to_dict(config),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if stats.position_gradient_accumulator.device.type == "cuda"
            else None
        ),
        "densification_stats": {
            "position_gradient_accumulator": stats.position_gradient_accumulator,
            "observation_count": stats.observation_count,
            "max_screen_radius": stats.max_screen_radius,
        },
        "training_seconds": training_seconds,
        "initial_holdout": initial_holdout,
        "train_image_ids": training_image_ids,
        "test_image_ids": test_image_ids,
        "depth_prior_provenance": depth_prior_provenance,
    }


def _restore_densification_stats(
    metadata: dict[str, Any],
    model: GaussianModel,
) -> DensificationStats:
    values = metadata.get("densification_stats")
    if not isinstance(values, dict):
        raise ValueError("resume checkpoint is missing densification statistics")
    stats = DensificationStats(
        position_gradient_accumulator=values[
            "position_gradient_accumulator"
        ].to(model.means.device),
        observation_count=values["observation_count"].to(model.means.device),
        max_screen_radius=values["max_screen_radius"].to(model.means.device),
    )
    expected_shape = (model.means.shape[0],)
    if any(
        value.shape != expected_shape
        for value in (
            stats.position_gradient_accumulator,
            stats.observation_count,
            stats.max_screen_radius,
        )
    ):
        raise ValueError("resume checkpoint densification statistics have the wrong shape")
    return stats


def train(config: ExperimentConfig) -> None:
    """Optimize a Gaussian scene from registered images."""
    if config.training.iterations < 1:
        raise ValueError("training.iterations must be positive")
    if config.training.densify_every < 1:
        raise ValueError("training.densify_every must be positive")
    if config.training.densify_until < config.training.densify_from:
        raise ValueError("training.densify_until must not precede densify_from")
    if config.training.checkpoint_every < 0:
        raise ValueError("training.checkpoint_every must be non-negative")
    if config.training.depth_loss_weight < 0:
        raise ValueError("training.depth_loss_weight must be non-negative")
    if config.training.depth_loss_beta <= 0:
        raise ValueError("training.depth_loss_beta must be positive")
    if not 0.0 <= config.training.depth_loss_alpha_threshold <= 1.0:
        raise ValueError(
            "training.depth_loss_alpha_threshold must be between zero and one"
        )
    if (
        config.training.depth_loss_weight > 0
        and config.data.depth_priors_dir is None
    ):
        raise ValueError(
            "data.depth_priors_dir is required when depth regularization is enabled"
        )

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
    resume_path = config.output_dir / "latest.pt"
    training_image_ids = [
        scene.cameras[index].image_id for index in training_indices
    ]
    test_image_ids = [
        scene.cameras[index].image_id for index in holdout_indices
    ]
    depth_priors, depth_prior_provenance = (
        load_depth_priors(
            scene,
            training_indices,
            config.data.depth_priors_dir,
            device,
        )
        if config.training.depth_loss_weight > 0
        and config.data.depth_priors_dir is not None
        else ({}, None)
    )
    start_step = 0
    elapsed_before = 0.0
    initial_holdout: dict[str, Any] | None = None
    if resume_path.is_file():
        model, optimizer, start_step, resume_metadata = load_checkpoint(
            resume_path,
            optimizer_factory=lambda restored: _build_optimizer(restored, config),
            device=device,
        )
        if not configs_match(
            resume_metadata.get("config"),
            config_to_dict(config),
        ):
            raise ValueError("resume checkpoint config does not match the requested run")
        if resume_metadata.get("train_image_ids") != training_image_ids:
            raise ValueError("resume checkpoint training split does not match")
        if resume_metadata.get("test_image_ids") != test_image_ids:
            raise ValueError("resume checkpoint test split does not match")
        if (
            resume_metadata.get("depth_prior_provenance")
            != depth_prior_provenance
        ):
            raise ValueError("resume checkpoint depth priors do not match")
        if optimizer is None:
            raise RuntimeError("resume checkpoint did not restore an optimizer")
        stats = _restore_densification_stats(resume_metadata, model)
        random.setstate(resume_metadata["python_random_state"])
        torch.set_rng_state(resume_metadata["torch_rng_state"].cpu())
        cuda_rng_state = resume_metadata.get("cuda_rng_state_all")
        if cuda_rng_state is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        elapsed_before = float(resume_metadata.get("training_seconds", 0.0))
        initial_holdout = resume_metadata.get("initial_holdout")
        _truncate_training_log(log_path, start_step)
        print(
            f"resuming step {start_step}/{config.training.iterations} "
            f"gaussians={model.means.shape[0]}"
        )
    else:
        model = GaussianModel.from_point_cloud(
            points=scene.points,
            colors=scene.colors,
            sh_degree=config.model.sh_degree,
            initial_opacity=config.model.initial_opacity,
            initial_scale=config.model.initial_scale,
        ).to(device)
        optimizer = _build_optimizer(model, config)
        stats = create_densification_stats(model)

    renderer = renderer.to(device)
    if holdout_indices and initial_holdout is None:
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

    run_state_path = config.output_dir / "run_state.json"
    _write_json(
        run_state_path,
        {
            "config": config_to_dict(config),
            "status": "running",
            "checkpoint_step": start_step,
        },
    )
    if start_step == 0 and config.training.checkpoint_every > 0:
        save_checkpoint(
            resume_path,
            model,
            optimizer,
            step=0,
            metadata=_checkpoint_metadata(
                config,
                stats,
                elapsed_before,
                initial_holdout,
                training_image_ids,
                test_image_ids,
                depth_prior_provenance,
            ),
        )
    model.train()
    training_started = time.perf_counter()
    checkpoint_seconds = 0.0
    log_mode = "a" if start_step else "w"
    with log_path.open(log_mode, encoding="utf-8") as log:
        for step in range(start_step + 1, config.training.iterations + 1):
            camera_index = random.choice(training_indices)
            camera = scene.cameras[camera_index]
            device_camera = camera.to(device)
            target = scene.images[camera_index].to(device)

            optimizer.zero_grad(set_to_none=True)
            output = renderer(model, device_camera, config.render.background)
            rgb_loss = photometric_loss(output.rgb, target)
            depth_loss = None
            depth_coverage = None
            if depth_priors:
                if output.depth is None:
                    raise RuntimeError("depth regularization requires renderer depth output")
                prior = depth_priors[camera_index]
                depth_loss, depth_coverage = depth_prior_loss(
                    output.depth,
                    output.alpha,
                    prior.depth,
                    prior.mask,
                    alpha_threshold=config.training.depth_loss_alpha_threshold,
                    beta=config.training.depth_loss_beta,
                )
                loss = (
                    rgb_loss
                    + config.training.depth_loss_weight * depth_loss
                )
            else:
                loss = rgb_loss
            loss.backward()
            if step <= config.training.densify_until:
                accumulate_densification_stats(
                    stats,
                    output,
                    camera.width,
                    camera.height,
                )
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
                    max_gaussians=config.training.densify_max_gaussians,
                )
                stats = create_densification_stats(model)

            record = {
                "step": step,
                "loss": loss.item(),
                "rgb_loss": rgb_loss.item(),
                "depth_loss": (
                    depth_loss.item() if depth_loss is not None else None
                ),
                "depth_prior_coverage": depth_coverage,
                "gaussians": model.means.shape[0],
                "image_id": camera.image_id,
            }
            if topology_update is not None:
                record["densification"] = asdict(topology_update)
            log.write(json.dumps(record) + "\n")
            should_checkpoint = (
                config.training.checkpoint_every > 0
                and step % config.training.checkpoint_every == 0
            )
            if should_checkpoint:
                log.flush()
                os.fsync(log.fileno())
                checkpoint_started = time.perf_counter()
                elapsed = (
                    elapsed_before
                    + checkpoint_started
                    - training_started
                    - checkpoint_seconds
                )
                save_checkpoint(
                    resume_path,
                    model,
                    optimizer,
                    step=step,
                    metadata=_checkpoint_metadata(
                        config,
                        stats,
                        elapsed,
                        initial_holdout,
                        training_image_ids,
                        test_image_ids,
                        depth_prior_provenance,
                    ),
                )
                _write_json(
                    run_state_path,
                    {
                        "config": config_to_dict(config),
                        "status": "running",
                        "checkpoint_step": step,
                    },
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_started
            if step == 1 or step % report_every == 0:
                print(
                    f"step {step:>{len(str(config.training.iterations))}}/"
                    f"{config.training.iterations} loss={loss.item():.6f} "
                    f"gaussians={model.means.shape[0]}"
                )
    training_seconds = (
        elapsed_before
        + time.perf_counter()
        - training_started
        - checkpoint_seconds
    )

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
        _write_json(config.output_dir / "holdout_metrics.json", holdout_summary)
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
            "config": config_to_dict(config),
            "gaussians": model.means.shape[0],
            "train_image_ids": training_image_ids,
            "test_image_ids": test_image_ids,
            "holdout_mean_psnr": (
                final_holdout["mean_psnr"] if final_holdout is not None else None
            ),
            "training_seconds": training_seconds,
            "depth_prior_provenance": depth_prior_provenance,
        },
    )
    run_metadata = {
        "config": config_to_dict(config),
        "training_seconds": training_seconds,
        "iterations": config.training.iterations,
        "final_gaussians": model.means.shape[0],
        "train_image_ids": training_image_ids,
        "test_image_ids": test_image_ids,
        "checkpoint": str(config.output_dir / "final.pt"),
        "depth_prior_provenance": depth_prior_provenance,
    }
    _write_json(config.output_dir / "run_metadata.json", run_metadata)
    _write_json(
        run_state_path,
        {
            "config": config_to_dict(config),
            "status": "completed",
            "checkpoint_step": config.training.iterations,
        },
    )
    if resume_path.is_file():
        resume_path.unlink()
        _fsync_directory(resume_path.parent)
