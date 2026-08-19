from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    colmap_dir: Path
    images_dir: Path
    downscale: int = 1


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


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    _reject_unknown(raw, {"data", "model", "training", "render", "output_dir"}, "root")
    if "data" not in raw or not isinstance(raw["data"], dict):
        raise ValueError("configuration requires a data mapping")

    data_values = raw["data"]
    _reject_unknown(data_values, {"colmap_dir", "images_dir", "downscale"}, "data")
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

    data = DataConfig(
        colmap_dir=Path(data_values["colmap_dir"]),
        images_dir=Path(data_values["images_dir"]),
        downscale=int(data_values.get("downscale", 1)),
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

