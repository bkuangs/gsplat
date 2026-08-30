import json
from pathlib import Path

import pytest
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
            densify_from=1,
            densify_until=1,
            densify_every=1,
            densify_gradient_threshold=0.0,
            densify_opacity_threshold=0.0,
        ),
        render=RenderConfig(backend="torch"),
        output_dir=output_dir,
    )

    trainer.train(config)

    metrics = json.loads((output_dir / "holdout_metrics.json").read_text())
    training_record = json.loads(
        (output_dir / "training.jsonl").read_text(encoding="utf-8")
    )
    run_metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metrics["holdout_image_ids"] == [7]
    assert metrics["initial"]["cameras"][0]["image_id"] == 7
    assert metrics["final"]["cameras"][0]["image_id"] == 7
    assert training_record["densification"] == {
        "gaussians_before": 1,
        "cloned": 0,
        "split_parents": 1,
        "split_children": 2,
        "pruned": 0,
        "gaussians_after": 2,
    }
    assert training_record["gaussians"] == 2
    assert run_metadata["training_seconds"] >= 0
    assert run_metadata["train_image_ids"] == [3]
    assert run_metadata["test_image_ids"] == [7]
    assert run_metadata["final_gaussians"] == 2
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


def test_training_resumes_from_periodic_checkpoint(
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
            iterations=2,
            densify_from=1,
            densify_until=1,
            densify_every=1,
            densify_gradient_threshold=0.0,
            densify_opacity_threshold=0.0,
            checkpoint_every=1,
        ),
        render=RenderConfig(backend="torch"),
        output_dir=output_dir,
    )
    original_loss = trainer.photometric_loss
    calls = 0

    def fail_on_second_step(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_loss(prediction, target)

    monkeypatch.setattr(trainer, "photometric_loss", fail_on_second_step)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train(config)
    assert (output_dir / "latest.pt").is_file()

    monkeypatch.setattr(trainer, "photometric_loss", original_loss)
    trainer.train(config)

    records = [
        json.loads(line)
        for line in (output_dir / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert not (output_dir / "latest.pt").exists()
    assert (output_dir / "final.pt").is_file()
