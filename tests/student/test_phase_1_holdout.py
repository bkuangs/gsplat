import json
from pathlib import Path

import torch

from gaussian_splatting.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RenderConfig,
    TrainingConfig,
)
from gaussian_splatting.data.colmap import ColmapScene
from gaussian_splatting.training import trainer
from gaussian_splatting.types import Camera


def _camera(image_id: int) -> Camera:
    return Camera(
        world_to_camera=torch.eye(4),
        intrinsics=torch.tensor(
            [[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]
        ),
        width=12,
        height=12,
        image_id=image_id,
    )


def test_training_excludes_fixed_holdout_and_saves_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = ColmapScene(
        cameras=(_camera(3), _camera(7)),
        images=(torch.zeros(3, 12, 12), torch.zeros(3, 12, 12)),
        points=torch.tensor([[0.0, 0.0, 2.0]]),
        colors=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(trainer, "load_colmap_scene", lambda *_args: scene)
    output_dir = tmp_path / "output"
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path,
            images_dir=tmp_path,
            holdout_image_ids=(7,),
        ),
        model=ModelConfig(sh_degree=0, initial_opacity=0.8, initial_scale=0.05),
        training=TrainingConfig(
            iterations=1,
            densify_from=2,
            densify_until=2,
        ),
        render=RenderConfig(backend="torch"),
        output_dir=output_dir,
    )

    trainer.train(config)

    metrics = json.loads((output_dir / "holdout_metrics.json").read_text())
    assert metrics["holdout_image_ids"] == [7]
    assert metrics["initial"]["cameras"][0]["image_id"] == 7
    assert metrics["final"]["cameras"][0]["image_id"] == 7
    for relative_path in (
        "holdout/targets/image_0007.png",
        "holdout/initial/image_0007_rgb.png",
        "holdout/initial/image_0007_depth.png",
        "holdout/initial/image_0007_depth.pt",
        "holdout/final/image_0007_rgb.png",
        "holdout/final/image_0007_depth.png",
        "holdout/final/image_0007_depth.pt",
    ):
        assert (output_dir / relative_path).is_file()


def test_holdout_requires_a_registered_image_id() -> None:
    scene = ColmapScene(
        cameras=(_camera(3),),
        images=(torch.zeros(3, 12, 12),),
        points=torch.tensor([[0.0, 0.0, 2.0]]),
        colors=torch.ones(1, 3),
    )

    try:
        trainer._partition_camera_indices(scene, (7,))
    except ValueError as error:
        assert "not registered" in str(error)
    else:
        raise AssertionError("missing holdout image ID was accepted")
