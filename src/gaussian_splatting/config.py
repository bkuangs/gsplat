from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    colmap_dir: Path
    images_dir: Path
    downscale: int = 1
    depths_dir: Path | None = None
    train_image_ids: tuple[int, ...] = ()
    test_image_ids: tuple[int, ...] = ()
    holdout_image_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    sh_degree: int = 3
    initial_opacity: float = 0.1
    initial_scale: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 30_000
    learning_rate_position: float = 0.00016
    learning_rate_features: float = 0.0025
    learning_rate_opacity: float = 0.05
    learning_rate_scale: float = 0.005
    learning_rate_rotation: float = 0.001
    densify_from: int = 500
    densify_until: int = 15_000
    densify_every: int = 100
    densify_gradient_threshold: float = 0.0002
    densify_opacity_threshold: float = 0.005
    densify_scale_threshold: float = 0.01
    densify_max_screen_radius: float = 100.0
    seed: int = 42


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "torch"
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output_dir: Path = Path("outputs/default")


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = values.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section} configuration field(s): {names}")


def _image_ids(values: dict[str, Any], name: str) -> tuple[int, ...]:
    raw = values.get(name, [])
    if not isinstance(raw, list):
        raise ValueError(f"data.{name} must be a list")
    image_ids = tuple(int(value) for value in raw)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"data.{name} must not contain duplicates")
    return image_ids


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an experiment config into a JSON-serializable mapping."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(asdict(config))


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    _reject_unknown(raw, {"data", "model", "training", "render", "output_dir"}, "root")
    if "data" not in raw or not isinstance(raw["data"], dict):
        raise ValueError("configuration requires a data mapping")

    data_values = raw["data"]
    _reject_unknown(
        data_values,
        {
            "colmap_dir",
            "images_dir",
            "downscale",
            "depths_dir",
            "train_image_ids",
            "test_image_ids",
            "holdout_image_ids",
        },
        "data",
    )
    if "colmap_dir" not in data_values or "images_dir" not in data_values:
        raise ValueError("data requires colmap_dir and images_dir")

    model_values = raw.get("model", {})
    training_values = raw.get("training", {})
    render_values = raw.get("render", {})
    for name, values in (
        ("model", model_values),
        ("training", training_values),
        ("render", render_values),
    ):
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")

    _reject_unknown(model_values, set(ModelConfig.__dataclass_fields__), "model")
    _reject_unknown(training_values, set(TrainingConfig.__dataclass_fields__), "training")
    _reject_unknown(render_values, set(RenderConfig.__dataclass_fields__), "render")

    background = render_values.get("background", (0.0, 0.0, 0.0))
    if len(background) != 3:
        raise ValueError("render.background must have exactly three values")

    train_image_ids = _image_ids(data_values, "train_image_ids")
    test_image_ids = _image_ids(data_values, "test_image_ids")
    holdout_image_ids = _image_ids(data_values, "holdout_image_ids")
    if test_image_ids and holdout_image_ids:
        raise ValueError(
            "data.test_image_ids and data.holdout_image_ids cannot both be set"
        )
    evaluation_ids = test_image_ids or holdout_image_ids
    overlap = sorted(set(train_image_ids) & set(evaluation_ids))
    if overlap:
        names = ", ".join(str(image_id) for image_id in overlap)
        raise ValueError(f"training and test image IDs overlap: {names}")

    data = DataConfig(
        colmap_dir=Path(data_values["colmap_dir"]),
        images_dir=Path(data_values["images_dir"]),
        downscale=int(data_values.get("downscale", 1)),
        depths_dir=(
            Path(data_values["depths_dir"])
            if data_values.get("depths_dir") is not None
            else None
        ),
        train_image_ids=train_image_ids,
        test_image_ids=test_image_ids,
        holdout_image_ids=holdout_image_ids,
    )
    if data.downscale < 1:
        raise ValueError("data.downscale must be at least 1")

    render = RenderConfig(
        backend=str(render_values.get("backend", "torch")),
        background=tuple(float(value) for value in background),
    )
    if render.backend not in {"torch", "cuda"}:
        raise ValueError("render.backend must be 'torch' or 'cuda'")

    return ExperimentConfig(
        data=data,
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
        render=render,
        output_dir=Path(raw.get("output_dir", "outputs/default")),
    )
