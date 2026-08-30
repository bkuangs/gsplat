import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaussian_splatting import plotting
from gaussian_splatting.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RenderConfig,
)
from gaussian_splatting.data.colmap import ColmapScene
from gaussian_splatting.image_sampling import sample_image_at_coordinates
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.training import evaluation
from gaussian_splatting.training.checkpoint import save_checkpoint
from gaussian_splatting.training.splits import partition_camera_indices
from gaussian_splatting.types import Camera


def _camera(
    image_id: int,
    image_path: Path,
    width: int = 12,
    height: int = 12,
) -> Camera:
    return Camera(
        world_to_camera=torch.eye(4),
        intrinsics=torch.tensor(
            [[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]
        ),
        width=width,
        height=height,
        image_path=image_path,
        image_id=image_id,
    )


def _save_depth_samples(
    path: Path,
    image_name: str,
    depth: float = 2.0,
    depths: np.ndarray | None = None,
    coordinates: np.ndarray | None = None,
) -> None:
    sample_depths = (
        depths
        if depths is not None
        else np.asarray([depth, depth], dtype=np.float64)
    )
    sample_coordinates = (
        coordinates
        if coordinates is not None
        else np.asarray([[5.5, 5.5], [6.5, 6.5]], dtype=np.float64)
    )
    torch.save(
        {
            "image_name": image_name,
            "depth": sample_depths,
            "coord": sample_coordinates,
            "error": np.zeros(len(sample_depths), dtype=np.float64),
            "weight": np.ones(len(sample_depths), dtype=np.float64),
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


def test_depth_sampling_uses_half_integer_pixel_centers() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    coordinates = torch.tensor([[0.5, 0.5], [2.5, 1.5], [3.5, 2.5]])

    sampled = sample_image_at_coordinates(
        image,
        coordinates,
        source_width=4,
        source_height=3,
    )

    assert sampled.tolist() == pytest.approx([0.0, 6.0, 11.0], abs=1e-6)


def test_sparse_depth_metrics_map_source_coordinates_to_downscaled_render(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image_path)
    depth_path = tmp_path / "image.pt"
    _save_depth_samples(
        depth_path,
        image_path.name,
        depths=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        coordinates=np.asarray(
            [[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]],
            dtype=np.float64,
        ),
    )
    rendered_depth = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    alpha = torch.ones(2, 2)

    metrics = evaluation.sparse_depth_metrics(
        rendered_depth,
        alpha,
        _camera(1, image_path, width=2, height=2),
        depth_path,
    )

    assert metrics["depth_abs_rel"] == pytest.approx(0.0)
    assert metrics["depth_coverage"] == 1.0
    assert metrics["depth_samples"] == 4
    assert metrics["covered_depth_samples"] == 4


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


def test_optional_metric_plot_omits_missing_values() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    metrics = {
        "splits": {
            "train": {
                "mean_lpips": 0.2,
                "mean_depth_abs_rel": None,
                "mean_depth_coverage": 0.8,
            },
            "test": {
                "mean_lpips": None,
                "mean_depth_abs_rel": 0.1,
                "mean_depth_coverage": None,
            },
        }
    }

    plotting._plot_optional_metrics(axis, metrics)

    assert sorted(patch.get_height() for patch in axis.patches) == [0.1, 0.2, 0.8]
    assert [text.get_text() for text in axis.texts] == ["N/A", "N/A", "N/A"]
    plt.close(figure)


def test_plot_run_supports_custom_evaluation_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    evaluation_dir = tmp_path / "custom-evaluation"
    run_dir.mkdir()
    evaluation_dir.mkdir()
    (run_dir / "training.jsonl").write_text(
        '{"step": 1, "loss": 0.5, "gaussians": 10}\n',
        encoding="utf-8",
    )
    metrics = {
        "splits": {
            split: {
                "mean_psnr": 10.0,
                "mean_lpips": None,
                "mean_depth_abs_rel": None,
                "mean_depth_coverage": None,
            }
            for split in ("train", "test")
        }
    }
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    output_path = plotting.plot_run(run_dir, evaluation_dir)

    assert output_path == evaluation_dir / "summary.png"
    assert output_path.is_file()
