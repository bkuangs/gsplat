import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from gaussian_splatting.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RenderConfig,
    TrainingConfig,
    config_to_dict,
    load_config,
)
from gaussian_splatting.data.colmap import ColmapObservation, ColmapScene
from gaussian_splatting.training import depth_prior, trainer
from gaussian_splatting.training.depth_prior import (
    DepthAlignment,
    fit_inverse_depth_alignment,
    generate_depth_priors,
)
from gaussian_splatting.training.losses import depth_prior_loss
from gaussian_splatting.types import Camera


def _camera(
    image_path: Path,
    image_id: int = 3,
    width: int = 6,
    height: int = 5,
) -> Camera:
    return Camera(
        world_to_camera=torch.eye(4),
        intrinsics=torch.eye(3),
        width=width,
        height=height,
        image_path=image_path,
        image_id=image_id,
    )


def test_inverse_depth_alignment_recovers_affine_transform_with_outlier() -> None:
    prediction = torch.linspace(0.2, 1.2, 40)
    inverse_depth = 1.7 * prediction + 0.3
    depth = inverse_depth.reciprocal()
    depth[0] = 100.0

    alignment = fit_inverse_depth_alignment(
        prediction,
        depth,
        min_anchors=20,
        max_median_abs_rel=0.05,
    )

    assert alignment.scale == pytest.approx(1.7, rel=1e-3)
    assert alignment.shift == pytest.approx(0.3, rel=1e-3)
    assert alignment.polarity == 1
    assert alignment.median_abs_rel < 1e-3
    assert alignment.inlier_count < alignment.anchor_count


def test_inverse_depth_alignment_rejects_poor_colmap_agreement() -> None:
    prediction = torch.linspace(0.2, 1.2, 40)
    inverse_depth = torch.where(
        torch.arange(40) % 2 == 0,
        torch.tensor(0.5),
        torch.tensor(2.0),
    )

    with pytest.raises(ValueError, match="median AbsRel exceeds"):
        fit_inverse_depth_alignment(
            prediction,
            inverse_depth.reciprocal(),
            min_anchors=20,
            max_median_abs_rel=0.05,
        )


def test_generate_depth_priors_uses_colmap_observations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (6, 5)).save(image_path)
    camera = _camera(image_path)
    prediction = torch.arange(30, dtype=torch.float32).reshape(5, 6) / 30.0 + 0.2
    coordinates = torch.tensor(
        [[x + 0.5, y + 0.5] for y in range(5) for x in range(6)]
    )
    sampled = prediction.flatten()
    depths = (2.0 * sampled + 0.5).reciprocal()
    points = torch.zeros(30, 3)
    points[:, 2] = depths
    scene = ColmapScene(
        cameras=(camera,),
        points=points,
        colors=torch.ones_like(points),
        observations=tuple(
            ColmapObservation(point_index=index, image_id=3, xy=coordinate)
            for index, coordinate in enumerate(coordinates)
        ),
    )
    monkeypatch.setattr(depth_prior, "load_colmap_scene", lambda *_args: scene)
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path,
            images_dir=tmp_path,
            train_image_ids=(3,),
        )
    )
    output_dir = tmp_path / "priors"

    metadata = generate_depth_priors(
        config,
        output_dir,
        lambda _camera: prediction,
        model_id="fake-relative-depth",
        min_anchors=20,
        max_median_abs_rel=0.01,
    )

    assert metadata["train_image_ids"] == [3]
    assert metadata["cameras"][0]["median_abs_rel"] < 1e-5
    payload = torch.load(output_dir / "image_0003.pt", weights_only=True)
    torch.testing.assert_close(payload["depth"], (2.0 * prediction + 0.5).reciprocal())
    assert payload["mask"].all()
    assert (output_dir / "image_0003.png").is_file()
    assert json.loads((output_dir / "metadata.json").read_text())["model_id"] == (
        "fake-relative-depth"
    )
    loaded, _ = depth_prior.load_depth_priors(
        scene,
        (0,),
        output_dir,
        torch.device("cpu"),
    )
    assert loaded[0].mask.all()

    changed_transform = camera.world_to_camera.clone()
    changed_transform[2, 3] = 1.0
    changed_scene = replace(
        scene,
        cameras=(replace(camera, world_to_camera=changed_transform),),
    )
    with pytest.raises(ValueError, match="camera geometry does not match"):
        depth_prior.load_depth_priors(
            changed_scene,
            (0,),
            output_dir,
            torch.device("cpu"),
        )


def test_phase4_config_changes_only_depth_intervention_and_output() -> None:
    baseline = config_to_dict(
        load_config(Path("configs/phase3/dtu_scan63_sparse_aggressive.yaml"))
    )
    intervention = config_to_dict(
        load_config(Path("configs/phase4/dtu_scan63_sparse_aggressive_depth.yaml"))
    )
    assert intervention["data"].pop("depth_priors_dir") == (
        "outputs/phase4_dtu_scan63/depth_priors"
    )
    assert baseline["data"].pop("depth_priors_dir") is None
    for name in (
        "depth_loss_weight",
        "depth_loss_beta",
        "depth_loss_alpha_threshold",
    ):
        baseline["training"].pop(name)
        intervention["training"].pop(name)
    baseline["output_dir"] = None
    intervention["output_dir"] = None

    assert intervention == baseline


def test_depth_prior_loss_is_robust_and_masks_invalid_pixels() -> None:
    prior = torch.full((2, 3), 2.0)
    rendered = torch.tensor([[2.0, 2.2, 200.0], [2.0, 0.0, 2.0]])
    alpha = torch.ones(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[0, 2] = False

    loss, coverage = depth_prior_loss(
        rendered,
        alpha,
        prior,
        mask,
        beta=0.1,
    )

    assert 0.0 < loss.item() < 0.1
    assert coverage == 0.8


def test_training_logs_depth_regularization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 12)).save(image_path)
    camera = _camera(image_path, width=12, height=12)
    scene = ColmapScene(
        cameras=(camera,),
        images=(torch.zeros(3, 12, 12),),
        points=torch.tensor([[0.0, 0.0, 2.0]]),
        colors=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(trainer, "load_colmap_scene", lambda *_args: scene)
    priors_dir = tmp_path / "priors"
    priors_dir.mkdir()
    prior_path = priors_dir / "image_0003.pt"
    torch.save(
        {
            "image_id": 3,
            "image_name": image_path.name,
            "width": 12,
            "height": 12,
            "depth": torch.full((12, 12), 1.5),
            "mask": torch.ones(12, 12, dtype=torch.bool),
            "alignment": asdict(
                DepthAlignment(
                    scale=1.0,
                    shift=0.0,
                    polarity=1,
                    anchor_count=30,
                    inlier_count=30,
                    positive_fraction=1.0,
                    median_abs_rel=0.0,
                )
            ),
            "model_id": "fake",
        },
        prior_path,
    )
    (priors_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": "fake",
                "train_image_ids": [3],
                "min_anchors": 20,
                "max_median_abs_rel": 0.25,
                "cameras": [
                    {
                        "image_id": 3,
                        "image_name": image_path.name,
                        "status": "accepted",
                        "sha256": depth_prior._sha256(prior_path),
                        "source_image_sha256": depth_prior._sha256(image_path),
                        "camera_sha256": depth_prior._camera_sha256(camera),
                        "alignment_source_sha256": (
                            depth_prior._alignment_source_sha256(scene, camera, ())
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path,
            images_dir=tmp_path,
            depth_priors_dir=priors_dir,
            train_image_ids=(3,),
        ),
        model=ModelConfig(sh_degree=0, initial_opacity=0.8, initial_scale=0.05),
        training=TrainingConfig(
            iterations=1,
            densify_from=2,
            densify_until=2,
            depth_loss_weight=0.1,
        ),
        render=RenderConfig(backend="torch"),
        output_dir=output_dir,
    )

    trainer.train(config)

    record = json.loads((output_dir / "training.jsonl").read_text())
    assert record["depth_loss"] > 0
    assert 0 < record["depth_prior_coverage"] <= 1
    assert record["loss"] > record["rgb_loss"]
