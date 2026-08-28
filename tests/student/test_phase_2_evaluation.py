import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaussian_splatting.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RenderConfig,
)
from gaussian_splatting.data.colmap import ColmapScene
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.training import evaluation
from gaussian_splatting.training.checkpoint import save_checkpoint
from gaussian_splatting.training.splits import partition_camera_indices
from gaussian_splatting.types import Camera


def _camera(image_id: int, image_path: Path) -> Camera:
    return Camera(
        world_to_camera=torch.eye(4),
        intrinsics=torch.tensor(
            [[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]
        ),
        width=12,
        height=12,
        image_path=image_path,
        image_id=image_id,
    )


def _save_depth_samples(path: Path, image_name: str, depth: float = 2.0) -> None:
    torch.save(
        {
            "image_name": image_name,
            "depth": np.asarray([depth, depth], dtype=np.float64),
            "coord": np.asarray([[5.5, 5.5], [6.5, 6.5]], dtype=np.float64),
            "error": np.zeros(2, dtype=np.float64),
            "weight": np.ones(2, dtype=np.float64),
        },
        path,
    )


def test_explicit_camera_split_preserves_configured_order(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 12)).save(image_path)
    scene = ColmapScene(
        cameras=tuple(_camera(image_id, image_path) for image_id in (1, 2, 3)),
        images=tuple(torch.zeros(3, 12, 12) for _ in range(3)),
        points=torch.tensor([[0.0, 0.0, 2.0]]),
        colors=torch.ones(1, 3),
    )

    train_indices, test_indices = partition_camera_indices(
        scene,
        train_image_ids=(3, 1),
        test_image_ids=(2,),
    )

    assert train_indices == (2, 0)
    assert test_indices == (1,)


def test_sparse_depth_metrics_report_abs_rel_and_coverage(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 12)).save(image_path)
    depth_path = tmp_path / "image.pt"
    _save_depth_samples(depth_path, image_path.name)
    rendered_depth = torch.full((12, 12), 2.2)
    alpha = torch.ones(12, 12)

    metrics = evaluation.sparse_depth_metrics(
        rendered_depth,
        alpha,
        _camera(1, image_path),
        depth_path,
    )

    assert metrics["depth_abs_rel"] == pytest.approx(0.1)
    assert metrics["depth_coverage"] == 1.0
    assert metrics["depth_samples"] == 2
    assert metrics["covered_depth_samples"] == 2


def test_checkpoint_evaluation_writes_metrics_and_renders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images_dir = tmp_path / "images"
    depths_dir = tmp_path / "depths"
    images_dir.mkdir()
    depths_dir.mkdir()
    image_paths = tuple(images_dir / f"{image_id:04d}.png" for image_id in (1, 2))
    for image_path in image_paths:
        Image.new("RGB", (12, 12), color=(32, 16, 8)).save(image_path)
    _save_depth_samples(depths_dir / "0002.pt", "0002.png")
    scene = ColmapScene(
        cameras=tuple(
            _camera(image_id, image_path)
            for image_id, image_path in zip((1, 2), image_paths, strict=True)
        ),
        images=tuple(torch.full((3, 12, 12), 0.1) for _ in range(2)),
        points=torch.tensor([[0.0, 0.0, 2.0]]),
        colors=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(evaluation, "load_colmap_scene", lambda *_args: scene)
    monkeypatch.setattr(
        evaluation,
        "LpipsMetric",
        lambda _device: lambda prediction, target: float(
            (prediction - target).abs().mean().item()
        ),
    )

    model = GaussianModel.from_point_cloud(
        scene.points,
        scene.colors,
        sh_degree=0,
        initial_opacity=0.8,
        initial_scale=0.05,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "run" / "final.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=5,
        metadata={"holdout_image_ids": [2]},
    )
    (checkpoint_path.parent / "training.jsonl").write_text(
        '{"step": 5, "loss": 0.25, "gaussians": 1}\n',
        encoding="utf-8",
    )
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path,
            images_dir=images_dir,
            depths_dir=depths_dir,
            train_image_ids=(1,),
            test_image_ids=(2,),
        ),
        model=ModelConfig(sh_degree=0),
        render=RenderConfig(backend="torch"),
        output_dir=checkpoint_path.parent,
    )

    result = evaluation.evaluate_checkpoint(config, checkpoint_path)

    assert result["train_image_ids"] == [1]
    assert result["test_image_ids"] == [2]
    assert result["model"]["final_loss"] == 0.25
    assert result["splits"]["train"]["camera_count"] == 1
    assert result["splits"]["train"]["mean_depth_abs_rel"] is None
    assert result["splits"]["test"]["mean_depth_abs_rel"] is not None
    metrics_path = checkpoint_path.parent / "evaluation" / "metrics.json"
    assert json.loads(metrics_path.read_text())["checkpoint_step"] == 5
    with (checkpoint_path.parent / "evaluation" / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["split"] for row in rows] == ["train", "test"]
    for split, image_id in (("train", 1), ("test", 2)):
        render_dir = checkpoint_path.parent / "evaluation" / "renders" / split
        assert (render_dir / f"image_{image_id:04d}_rgb.png").is_file()
        assert (render_dir / f"image_{image_id:04d}_depth.pt").is_file()
