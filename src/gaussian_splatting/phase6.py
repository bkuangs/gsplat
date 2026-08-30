import csv
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import yaml

from gaussian_splatting.config import (
    ExperimentConfig,
    config_to_dict,
    configs_match,
    load_config,
)
from gaussian_splatting.data.colmap import ColmapScene, load_colmap_scene
from gaussian_splatting.phase3 import (
    _resolve_config_paths,
    _resolve_path,
    _stratified_sparse_ids,
)
from gaussian_splatting.plotting import plot_run
from gaussian_splatting.training.depth_prior import (
    DEFAULT_DEPTH_MODEL,
    DepthAnythingPredictor,
    generate_depth_priors,
    load_depth_priors,
)
from gaussian_splatting.training.evaluation import evaluate_checkpoint
from gaussian_splatting.training.splits import partition_camera_indices
from gaussian_splatting.training.trainer import train

PHASE6_CONDITIONS = ("rgb", "depth")
PHASE6_EFFECT_METRICS = (
    "train_psnr",
    "test_psnr",
    "train_test_psnr_gap",
    "test_lpips",
    "test_depth_abs_rel",
    "test_depth_coverage",
    "final_gaussians",
    "training_seconds",
)
PHASE6_PRIMARY_DIRECTIONS = {
    "test_psnr": "positive",
    "test_lpips": "negative",
    "test_depth_abs_rel": "negative",
    "train_test_psnr_gap": "negative",
}


@dataclass(frozen=True)
class Phase6Scene:
    name: str
    configs: dict[str, ExperimentConfig]
    config_paths: dict[str, Path]


@dataclass(frozen=True)
class Phase6Manifest:
    base_dir: Path
    output_dir: Path
    depth_model: str
    test_count: int
    sparse_fraction: float
    sparse_seed: int
    training_seeds: tuple[int, ...]
    scenes: dict[str, Phase6Scene]


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = values.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section} field(s): {names}")


def load_phase6_manifest(path: Path) -> Phase6Manifest:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Phase 6 manifest root must be a mapping")
    _reject_unknown(
        raw,
        {
            "base_dir",
            "output_dir",
            "depth_model",
            "training_seeds",
            "selection",
            "scenes",
        },
        "Phase 6 manifest",
    )
    if "base_dir" not in raw or "output_dir" not in raw:
        raise ValueError("Phase 6 manifest requires base_dir and output_dir")
    selection = raw.get("selection")
    scenes_raw = raw.get("scenes")
    if not isinstance(selection, dict) or not isinstance(scenes_raw, dict):
        raise ValueError("Phase 6 manifest requires selection and scenes mappings")
    _reject_unknown(
        selection,
        {"test_count", "sparse_fraction", "sparse_seed"},
        "Phase 6 selection",
    )
    test_count = int(selection.get("test_count", 0))
    sparse_fraction = float(selection.get("sparse_fraction", 0.0))
    sparse_seed = int(selection.get("sparse_seed", -1))
    if test_count < 1:
        raise ValueError("Phase 6 test_count must be positive")
    if not 0.0 < sparse_fraction < 1.0:
        raise ValueError("Phase 6 sparse_fraction must be between zero and one")
    if sparse_seed < 0:
        raise ValueError("Phase 6 sparse_seed must be non-negative")
    seeds_raw = raw.get("training_seeds", [])
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("Phase 6 training_seeds must be a non-empty list")
    training_seeds = tuple(int(seed) for seed in seeds_raw)
    if any(seed < 0 for seed in training_seeds):
        raise ValueError("Phase 6 training seeds must be non-negative")
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Phase 6 training seeds must not contain duplicates")

    base_dir = (path.parent / Path(raw["base_dir"])).resolve()
    scenes: dict[str, Phase6Scene] = {}
    for name, values in scenes_raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"Phase 6 scene {name!r} must be a mapping")
        _reject_unknown(values, set(PHASE6_CONDITIONS), f"Phase 6 scene {name}")
        if set(values) != set(PHASE6_CONDITIONS):
            raise ValueError(f"Phase 6 scene {name!r} requires rgb and depth configs")
        config_paths = {
            condition: (path.parent / Path(values[condition])).resolve()
            for condition in PHASE6_CONDITIONS
        }
        scenes[name] = Phase6Scene(
            name=name,
            configs={
                condition: _resolve_config_paths(
                    load_config(config_paths[condition]),
                    base_dir,
                )
                for condition in PHASE6_CONDITIONS
            },
            config_paths=config_paths,
        )
    manifest = Phase6Manifest(
        base_dir=base_dir,
        output_dir=_resolve_path(base_dir, Path(raw["output_dir"])),
        depth_model=str(raw.get("depth_model", DEFAULT_DEPTH_MODEL)),
        test_count=test_count,
        sparse_fraction=sparse_fraction,
        sparse_seed=sparse_seed,
        training_seeds=training_seeds,
        scenes=scenes,
    )
    validate_phase6_manifest(manifest)
    return manifest


def _pair_invariants(config: ExperimentConfig) -> dict[str, Any]:
    values = config_to_dict(config)
    values["output_dir"] = None
    values["data"]["depth_priors_dir"] = None
    values["training"]["depth_loss_weight"] = None
    values["training"]["depth_loss_beta"] = None
    values["training"]["depth_loss_alpha_threshold"] = None
    return values


def _cross_scene_invariants(config: ExperimentConfig) -> dict[str, Any]:
    values = config_to_dict(config)
    values["output_dir"] = None
    values["data"] = {"downscale": values["data"]["downscale"]}
    return values


def _normalize_config_paths(
    values: Any,
    base_dir: Path,
) -> Any:
    if not isinstance(values, dict):
        return values
    normalized = deepcopy(values)
    data = normalized.get("data")
    if isinstance(data, dict):
        for name in ("colmap_dir", "images_dir", "depths_dir", "depth_priors_dir"):
            value = data.get(name)
            if value is not None:
                data[name] = str(_resolve_path(base_dir, Path(value)))
    output_dir = normalized.get("output_dir")
    if output_dir is not None:
        normalized["output_dir"] = str(_resolve_path(base_dir, Path(output_dir)))
    return normalized


def _configs_match_phase6(
    stored: Any,
    config: ExperimentConfig,
    base_dir: Path,
) -> bool:
    return configs_match(
        _normalize_config_paths(stored, base_dir),
        _normalize_config_paths(config_to_dict(config), base_dir),
    )


def validate_phase6_manifest(manifest: Phase6Manifest) -> None:
    if len(manifest.scenes) != 3 or "scan63" not in manifest.scenes:
        raise ValueError("Phase 6 requires scan63 and exactly two additional scenes")
    output_dirs: list[Path] = []
    for name, scene in manifest.scenes.items():
        rgb = scene.configs["rgb"]
        depth = scene.configs["depth"]
        if _pair_invariants(rgb) != _pair_invariants(depth):
            raise ValueError(
                f"{name} RGB/depth configs differ outside the depth intervention"
            )
        if rgb.training.depth_loss_weight != 0 or rgb.data.depth_priors_dir is not None:
            raise ValueError(f"{name} RGB config must not enable depth supervision")
        if depth.training.depth_loss_weight <= 0 or depth.data.depth_priors_dir is None:
            raise ValueError(f"{name} depth config must enable one depth intervention")
        output_dirs.extend([rgb.output_dir, depth.output_dir])
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("Phase 6 conditions require distinct output directories")
    base_seed = manifest.scenes["scan63"].configs["rgb"].training.seed
    if base_seed not in manifest.training_seeds:
        raise ValueError(
            f"Phase 6 training_seeds must include the base config seed {base_seed}"
        )

    reference_rgb = _cross_scene_invariants(
        manifest.scenes["scan63"].configs["rgb"]
    )
    reference_depth = _cross_scene_invariants(
        manifest.scenes["scan63"].configs["depth"]
    )
    for name, scene in manifest.scenes.items():
        if _cross_scene_invariants(scene.configs["rgb"]) != reference_rgb:
            raise ValueError(f"{name} changes RGB training settings across scenes")
        if _cross_scene_invariants(scene.configs["depth"]) != reference_depth:
            raise ValueError(f"{name} changes depth training settings across scenes")
    seeded_output_dirs = [
        _config_for_seed(manifest, scene_name, condition, seed).output_dir
        for seed in manifest.training_seeds
        for scene_name in manifest.scenes
        for condition in PHASE6_CONDITIONS
    ]
    if len(set(seeded_output_dirs)) != len(seeded_output_dirs):
        raise ValueError("Phase 6 seed conditions require distinct output directories")


def _config_for_seed(
    manifest: Phase6Manifest,
    scene_name: str,
    condition: str,
    seed: int,
) -> ExperimentConfig:
    source = manifest.scenes[scene_name].configs[condition]
    if seed == source.training.seed:
        return source
    condition_name = "rgb_only" if condition == "rgb" else "rgb_depth"
    return replace(
        source,
        training=replace(source.training, seed=seed),
        output_dir=(
            manifest.output_dir
            / "seeds"
            / f"seed_{seed}"
            / scene_name
            / condition_name
        ),
    )


def _selected_training_seeds(
    manifest: Phase6Manifest,
    seed: int | str | None,
) -> tuple[int, ...]:
    base_seed = manifest.scenes["scan63"].configs["rgb"].training.seed
    if seed is None:
        return (base_seed,)
    if seed == "all":
        return manifest.training_seeds
    selected_seed = int(seed)
    if selected_seed not in manifest.training_seeds:
        allowed = ", ".join(str(value) for value in manifest.training_seeds)
        raise ValueError(
            f"unknown Phase 6 training seed {selected_seed}; configured seeds: {allowed}"
        )
    return (selected_seed,)


def phase6_run_targets(
    manifest: Phase6Manifest,
    scene_name: str,
    condition: str,
    seed: int | str | None = None,
) -> tuple[tuple[str, str, int], ...]:
    selected_scenes = (
        tuple(manifest.scenes)
        if scene_name == "all"
        else (scene_name,)
    )
    if any(name not in manifest.scenes for name in selected_scenes):
        raise ValueError(f"unknown Phase 6 scene: {scene_name}")
    selected_conditions = (
        PHASE6_CONDITIONS if condition == "both" else (condition,)
    )
    if any(name not in PHASE6_CONDITIONS for name in selected_conditions):
        raise ValueError(f"unknown Phase 6 condition: {condition}")
    return tuple(
        (selected_scene, selected_condition, selected_seed)
        for selected_seed in _selected_training_seeds(manifest, seed)
        for selected_scene in selected_scenes
        for selected_condition in selected_conditions
    )


def _validate_phase6_run_state(
    config: ExperimentConfig,
    base_dir: Path,
) -> bool:
    checkpoint_path = config.output_dir / "final.pt"
    resume_path = config.output_dir / "latest.pt"
    state_path = config.output_dir / "run_state.json"
    if checkpoint_path.is_file():
        metadata_path = config.output_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "completed checkpoint is in an ambiguous state because run metadata "
                f"is missing: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _configs_match_phase6(metadata.get("config"), config, base_dir):
            raise ValueError(
                f"completed checkpoint config does not match: {config.output_dir}"
            )
        return True
    if resume_path.is_file():
        if not state_path.is_file():
            raise ValueError(f"resume checkpoint is missing run state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not _configs_match_phase6(state.get("config"), config, base_dir):
            raise ValueError(f"resume checkpoint config does not match: {config.output_dir}")
        return False
    if config.output_dir.is_dir() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"incomplete output directory must be moved or removed: {config.output_dir}"
        )
    return False


def _validate_evaluation_state(
    config: ExperimentConfig,
    base_dir: Path,
) -> bool:
    metrics_path = config.output_dir / "evaluation" / "metrics.json"
    if not metrics_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not _configs_match_phase6(metrics.get("config"), config, base_dir):
        raise ValueError(
            f"stored evaluation config does not match: {config.output_dir}"
        )
    stored_train_ids = metrics.get("train_image_ids")
    stored_test_ids = metrics.get("test_image_ids")
    if stored_train_ids is None or stored_test_ids is None:
        raise ValueError(
            f"stored evaluation is missing split metadata: {config.output_dir}"
        )
    if stored_train_ids != list(config.data.train_image_ids):
        raise ValueError(
            f"stored evaluation training split does not match: {config.output_dir}"
        )
    if stored_test_ids != list(config.data.test_image_ids):
        raise ValueError(
            f"stored evaluation test split does not match: {config.output_dir}"
        )
    return True


def _stratified_test_ids(
    registered_ids: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    if count >= len(registered_ids):
        raise ValueError("test_count must be smaller than the registered camera count")
    quotient, remainder = divmod(len(registered_ids), count)
    selected: list[int] = []
    start = 0
    for index in range(count):
        size = quotient + (index < remainder)
        selected.append(registered_ids[start])
        start += size
    return tuple(selected)


def _validate_scene_split(
    manifest: Phase6Manifest,
    scene_config: ExperimentConfig,
) -> ColmapScene:
    scene = load_colmap_scene(
        scene_config.data.colmap_dir,
        scene_config.data.images_dir,
        scene_config.data.downscale,
    )
    registered_ids = tuple(
        camera.image_id for camera in scene.cameras if camera.image_id is not None
    )
    expected_test_ids = _stratified_test_ids(registered_ids, manifest.test_count)
    if scene_config.data.test_image_ids != expected_test_ids:
        raise ValueError(
            "Phase 6 test cameras do not match the deterministic stratified split: "
            f"expected {list(expected_test_ids)}"
        )
    eligible_train_ids = tuple(
        image_id for image_id in registered_ids if image_id not in expected_test_ids
    )
    expected_train_ids = _stratified_sparse_ids(
        eligible_train_ids,
        manifest.sparse_fraction,
        manifest.sparse_seed,
    )
    if scene_config.data.train_image_ids != expected_train_ids:
        raise ValueError(
            "Phase 6 training cameras do not match the deterministic sparse split: "
            f"expected {list(expected_train_ids)}"
        )
    return scene


def generate_phase6_priors(
    manifest: Phase6Manifest,
    scene_name: str,
) -> list[Path]:
    selected = (
        tuple(manifest.scenes)
        if scene_name == "all"
        else (scene_name,)
    )
    if any(name not in manifest.scenes for name in selected):
        raise ValueError(f"unknown Phase 6 scene: {scene_name}")
    missing: list[tuple[Phase6Scene, ColmapScene]] = []
    completed: list[Path] = []
    for name in selected:
        phase_scene = manifest.scenes[name]
        config = phase_scene.configs["depth"]
        scene = _validate_scene_split(manifest, config)
        train_indices, _ = partition_camera_indices(
            scene,
            train_image_ids=config.data.train_image_ids,
            test_image_ids=config.data.test_image_ids,
        )
        directory = config.data.depth_priors_dir
        if directory is None:
            raise ValueError(f"{name} depth config has no prior directory")
        if (directory / "metadata.json").is_file():
            load_depth_priors(scene, train_indices, directory, torch.device("cpu"))
            completed.append(directory)
        else:
            if directory.is_dir() and any(directory.iterdir()):
                raise FileExistsError(
                    f"incomplete depth-prior directory must be moved or removed: "
                    f"{directory}"
                )
            missing.append((phase_scene, scene))
    if missing:
        predictor = DepthAnythingPredictor(manifest.depth_model)
        for phase_scene, _ in missing:
            config = phase_scene.configs["depth"]
            directory = config.data.depth_priors_dir
            if directory is None:
                raise RuntimeError("validated depth config lost its output directory")
            generate_depth_priors(
                config,
                directory,
                predictor,
                model_id=manifest.depth_model,
            )
            completed.append(directory)
    return completed


def run_phase6(
    manifest: Phase6Manifest,
    scene_name: str,
    condition: str,
    seed: int | str | None = None,
) -> list[Path]:
    targets = phase6_run_targets(manifest, scene_name, condition, seed)
    for name in dict.fromkeys(target[0] for target in targets):
        scene = manifest.scenes[name]
        _validate_scene_split(manifest, scene.configs["rgb"])
    selected_configs = [
        _config_for_seed(manifest, name, selected_condition, selected_seed)
        for name, selected_condition, selected_seed in targets
    ]
    completed = {
        config.output_dir: _validate_phase6_run_state(config, manifest.base_dir)
        for config in selected_configs
    }
    evaluated = {
        config.output_dir: _validate_evaluation_state(config, manifest.base_dir)
        for config in selected_configs
    }
    depth_scenes = tuple(
        dict.fromkeys(
            name
            for name, selected_condition, _ in targets
            if selected_condition == "depth"
        )
    )
    for name in depth_scenes:
        generate_phase6_priors(manifest, name)

    output_dirs: list[Path] = []
    for config in selected_configs:
        checkpoint_path = config.output_dir / "final.pt"
        if not completed[config.output_dir]:
            train(config)
        if (
            not completed[config.output_dir]
            or not evaluated[config.output_dir]
        ):
            evaluate_checkpoint(config, checkpoint_path)
            plot_run(config.output_dir)
        elif (
            config.output_dir.is_relative_to(manifest.output_dir)
            and not (config.output_dir / "evaluation" / "summary.png").is_file()
        ):
            plot_run(config.output_dir)
        output_dirs.append(config.output_dir)
    return output_dirs


def _read_metrics(
    scene_name: str,
    condition: str,
    seed: int,
    config: ExperimentConfig,
    base_dir: Path,
) -> dict[str, Any]:
    _validate_evaluation_state(config, base_dir)
    path = config.output_dir / "evaluation" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 6 metrics do not exist for {scene_name}/{condition}: {path}"
        )
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not _configs_match_phase6(
        metrics.get("config"),
        config,
        base_dir,
    ):
        raise ValueError(f"stored evaluation config does not match {scene_name}/{condition}")
    train = metrics["splits"]["train"]
    test = metrics["splits"]["test"]
    train_psnr = _required_metric(
        train["mean_psnr"],
        scene_name,
        condition,
        "train PSNR",
    )
    test_psnr = _required_metric(
        test["mean_psnr"],
        scene_name,
        condition,
        "test PSNR",
    )
    return {
        "scene": scene_name,
        "seed": seed,
        "condition": "rgb_only" if condition == "rgb" else "rgb_depth",
        "train_camera_count": train["camera_count"],
        "test_camera_count": test["camera_count"],
        "train_psnr": train_psnr,
        "test_psnr": test_psnr,
        "train_test_psnr_gap": train_psnr - test_psnr,
        "test_lpips": _required_metric(
            test["mean_lpips"],
            scene_name,
            condition,
            "test LPIPS",
        ),
        "test_depth_abs_rel": _required_metric(
            test["mean_depth_abs_rel"],
            scene_name,
            condition,
            "test depth AbsRel",
        ),
        "test_depth_coverage": _required_metric(
            test["mean_depth_coverage"],
            scene_name,
            condition,
            "test depth coverage",
        ),
        "final_gaussians": metrics["model"]["gaussians"],
        "training_seconds": metrics["model"]["training_seconds"],
    }


def _required_metric(
    value: float | int | None,
    scene_name: str,
    condition: str,
    metric_name: str,
) -> float:
    if value is None:
        raise ValueError(
            f"{scene_name}/{condition} is missing required {metric_name}"
        )
    return float(value)


def _is_favorable(value: float, direction: str) -> bool:
    return value > 0 if direction == "positive" else value < 0


def _effect_statistics(
    effects: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    for metric in PHASE6_EFFECT_METRICS:
        values = [float(effect[metric]) for effect in effects]
        metric_statistics: dict[str, Any] = {
            "mean": mean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }
        direction = PHASE6_PRIMARY_DIRECTIONS.get(metric)
        if direction is not None:
            metric_statistics["favorable_direction"] = direction
            metric_statistics["favorable_count"] = sum(
                _is_favorable(value, direction) for value in values
            )
        statistics[metric] = metric_statistics
    return statistics


def build_phase6_report(manifest: Phase6Manifest) -> dict[str, Any]:
    rows = [
        _read_metrics(
            scene_name,
            condition,
            seed,
            _config_for_seed(manifest, scene_name, condition, seed),
            manifest.base_dir,
        )
        for seed in manifest.training_seeds
        for scene_name in manifest.scenes
        for condition in PHASE6_CONDITIONS
    ]
    effects: list[dict[str, Any]] = []
    for seed in manifest.training_seeds:
        for scene_name in manifest.scenes:
            rgb = next(
                row for row in rows
                if (
                    row["seed"] == seed
                    and row["scene"] == scene_name
                    and row["condition"] == "rgb_only"
                )
            )
            depth = next(
                row for row in rows
                if (
                    row["seed"] == seed
                    and row["scene"] == scene_name
                    and row["condition"] == "rgb_depth"
                )
            )
            effects.append(
                {
                    "scene": scene_name,
                    "seed": seed,
                    **{
                        name: _required_metric(
                            depth[name],
                            scene_name,
                            "depth",
                            name,
                        )
                        - _required_metric(
                            rgb[name],
                            scene_name,
                            "rgb",
                            name,
                        )
                        for name in PHASE6_EFFECT_METRICS
                    },
                }
            )
    consistency = {
        metric: {
            "favorable_direction": direction,
            "favorable_run_count": sum(
                _is_favorable(effect[metric], direction)
                for effect in effects
            ),
            "run_count": len(effects),
        }
        for metric, direction in PHASE6_PRIMARY_DIRECTIONS.items()
    }
    scene_statistics = [
        {
            "scene": scene_name,
            "seed_count": len(manifest.training_seeds),
            **{
                f"{metric}_{name}": value
                for metric, values in _effect_statistics(
                    [
                        effect for effect in effects
                        if effect["scene"] == scene_name
                    ]
                ).items()
                for name, value in values.items()
            },
        }
        for scene_name in manifest.scenes
    ]
    all_primary_metrics_favorable_on_every_run = all(
        item["favorable_run_count"] == item["run_count"]
        for item in consistency.values()
    )
    all_primary_mean_effects_favorable_on_every_scene = all(
        _is_favorable(
            scene_statistics_row[f"{metric}_mean"],
            direction,
        )
        for scene_statistics_row in scene_statistics
        for metric, direction in PHASE6_PRIMARY_DIRECTIONS.items()
    )
    return {
        "training_seeds": list(manifest.training_seeds),
        "runs": rows,
        "effects_rgb_depth_minus_rgb_only": effects,
        "scene_effect_statistics": scene_statistics,
        "overall_effect_statistics": _effect_statistics(effects),
        "consistency": consistency,
        "all_primary_metrics_favorable_on_every_scene": (
            all_primary_mean_effects_favorable_on_every_scene
        ),
        "all_primary_metrics_favorable_on_every_run": (
            all_primary_metrics_favorable_on_every_run
        ),
        "notes": [
            "Positive PSNR and negative LPIPS, depth AbsRel, and train-test-gap "
            "effects favor the depth intervention.",
            "Reported standard deviations are sample standard deviations across "
            "training seeds.",
        ],
    }


def _plot_phase6_report(report: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Phase 6 plotting requires the evaluation extra: "
            "uv sync --extra evaluation"
        ) from error

    statistics = report["scene_effect_statistics"]
    labels = [row["scene"] for row in statistics]
    panels = (
        ("test_psnr", "Test PSNR effect", "dB"),
        ("test_lpips", "Test LPIPS effect", "LPIPS"),
        ("test_depth_abs_rel", "Depth AbsRel effect", "AbsRel"),
        ("train_test_psnr_gap", "Generalization-gap effect", "dB"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        values = [row[f"{metric}_mean"] for row in statistics]
        errors = [row[f"{metric}_sample_std"] for row in statistics]
        colors = [
            "#54a24b"
            if _is_favorable(value, PHASE6_PRIMARY_DIRECTIONS[metric])
            else "#e45756"
            for value in values
        ]
        axis.bar(labels, values, yerr=errors, capsize=4, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_phase6_report(manifest: Phase6Manifest) -> tuple[dict[str, Any], Path]:
    report = build_phase6_report(manifest)
    manifest.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest.output_dir / "phase6_metrics.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = report["runs"]
    with (manifest.output_dir / "phase6_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effects = report["effects_rgb_depth_minus_rgb_only"]
    with (manifest.output_dir / "phase6_effects.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    scene_statistics = report["scene_effect_statistics"]
    with (manifest.output_dir / "phase6_scene_statistics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_statistics[0]))
        writer.writeheader()
        writer.writerows(scene_statistics)
    plot_path = manifest.output_dir / "phase6_summary.png"
    _plot_phase6_report(report, plot_path)
    return report, plot_path
