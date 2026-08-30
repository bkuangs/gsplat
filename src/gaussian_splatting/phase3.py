import csv
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from gaussian_splatting.config import (
    ExperimentConfig,
    config_to_dict,
    configs_match,
    load_config,
)
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.plotting import plot_run
from gaussian_splatting.training.evaluation import evaluate_checkpoint
from gaussian_splatting.training.trainer import train

PHASE3_RUN_NAMES = (
    "full_baseline",
    "full_aggressive",
    "sparse_baseline",
    "sparse_aggressive",
)


@dataclass(frozen=True)
class Phase3Manifest:
    base_dir: Path
    output_dir: Path
    sparse_fraction: float
    sparse_seed: int
    configs: dict[str, ExperimentConfig]
    config_paths: dict[str, Path]


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = values.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section} field(s): {names}")


def load_phase3_manifest(path: Path) -> Phase3Manifest:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Phase 3 manifest root must be a mapping")
    _reject_unknown(
        raw,
        {"base_dir", "output_dir", "sparse_selection", "runs"},
        "Phase 3 manifest",
    )
    if "base_dir" not in raw:
        raise ValueError("Phase 3 manifest requires base_dir")
    if "output_dir" not in raw:
        raise ValueError("Phase 3 manifest requires output_dir")
    sparse_selection = raw.get("sparse_selection")
    if not isinstance(sparse_selection, dict):
        raise ValueError("Phase 3 manifest requires a sparse_selection mapping")
    _reject_unknown(
        sparse_selection,
        {"method", "fraction", "seed"},
        "Phase 3 sparse_selection",
    )
    if sparse_selection.get("method") != "stratified":
        raise ValueError("Phase 3 sparse selection method must be 'stratified'")
    sparse_fraction = float(sparse_selection.get("fraction", 0.0))
    sparse_seed = int(sparse_selection.get("seed", -1))
    if not 0.0 < sparse_fraction < 1.0:
        raise ValueError("Phase 3 sparse fraction must be between zero and one")
    if sparse_seed < 0:
        raise ValueError("Phase 3 sparse seed must be non-negative")
    runs = raw.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Phase 3 manifest requires a runs mapping")
    missing = set(PHASE3_RUN_NAMES) - runs.keys()
    extra = runs.keys() - set(PHASE3_RUN_NAMES)
    if missing or extra:
        raise ValueError(
            "Phase 3 runs must be exactly: " + ", ".join(PHASE3_RUN_NAMES)
        )

    base_dir = (path.parent / Path(raw["base_dir"])).resolve()
    config_paths = {
        name: (path.parent / Path(runs[name])).resolve()
        for name in PHASE3_RUN_NAMES
    }
    configs = {
        name: _resolve_config_paths(load_config(config_paths[name]), base_dir)
        for name in PHASE3_RUN_NAMES
    }
    manifest = Phase3Manifest(
        base_dir=base_dir,
        output_dir=_resolve_path(base_dir, Path(raw["output_dir"])),
        sparse_fraction=sparse_fraction,
        sparse_seed=sparse_seed,
        configs=configs,
        config_paths=config_paths,
    )
    validate_phase3_manifest(manifest)
    return manifest


def _resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_config_paths(
    config: ExperimentConfig,
    base_dir: Path,
) -> ExperimentConfig:
    return replace(
        config,
        data=replace(
            config.data,
            colmap_dir=_resolve_path(base_dir, config.data.colmap_dir),
            images_dir=_resolve_path(base_dir, config.data.images_dir),
            depths_dir=(
                _resolve_path(base_dir, config.data.depths_dir)
                if config.data.depths_dir is not None
                else None
            ),
            depth_priors_dir=(
                _resolve_path(base_dir, config.data.depth_priors_dir)
                if config.data.depth_priors_dir is not None
                else None
            ),
        ),
        output_dir=_resolve_path(base_dir, config.output_dir),
    )


def _stratified_sparse_ids(
    full_ids: tuple[int, ...],
    fraction: float,
    seed: int,
) -> tuple[int, ...]:
    count = max(1, round(fraction * len(full_ids)))
    quotient, remainder = divmod(len(full_ids), count)
    bins: list[tuple[int, ...]] = []
    start = 0
    for index in range(count):
        size = quotient + (index < remainder)
        bins.append(full_ids[start : start + size])
        start += size
    generator = random.Random(seed)
    return tuple(generator.choice(values) for values in bins)


def _comparison_invariants(config: ExperimentConfig) -> dict[str, Any]:
    values = config_to_dict(config)
    values["output_dir"] = None
    values["data"]["train_image_ids"] = None
    values["training"]["densify_gradient_threshold"] = None
    return values


def validate_phase3_manifest(manifest: Phase3Manifest) -> None:
    configs = manifest.configs
    if set(configs) != set(PHASE3_RUN_NAMES):
        raise ValueError("Phase 3 manifest does not contain the required four runs")

    reference = _comparison_invariants(configs["full_baseline"])
    for name in PHASE3_RUN_NAMES[1:]:
        if _comparison_invariants(configs[name]) != reference:
            raise ValueError(
                f"{name} changes settings outside supervision and gradient threshold"
            )

    full_ids = configs["full_baseline"].data.train_image_ids
    sparse_ids = configs["sparse_baseline"].data.train_image_ids
    if configs["full_aggressive"].data.train_image_ids != full_ids:
        raise ValueError("full-supervision runs must use identical training cameras")
    if configs["sparse_aggressive"].data.train_image_ids != sparse_ids:
        raise ValueError("sparse-supervision runs must use identical training cameras")
    if not full_ids or not sparse_ids:
        raise ValueError("Phase 3 runs require explicit non-empty training camera IDs")
    if not set(sparse_ids).issubset(full_ids):
        raise ValueError("sparse training cameras must be a subset of the full pool")
    expected_sparse_ids = _stratified_sparse_ids(
        full_ids,
        manifest.sparse_fraction,
        manifest.sparse_seed,
    )
    if sparse_ids != expected_sparse_ids:
        raise ValueError(
            "sparse training cameras do not match the manifest's stratified selection: "
            f"expected {list(expected_sparse_ids)}"
        )

    test_ids = configs["full_baseline"].data.test_image_ids
    if not test_ids:
        raise ValueError("Phase 3 requires an explicit held-out test split")
    if any(config.data.test_image_ids != test_ids for config in configs.values()):
        raise ValueError("all Phase 3 runs must use identical test cameras")

    baseline_threshold = configs["full_baseline"].training.densify_gradient_threshold
    aggressive_threshold = configs["full_aggressive"].training.densify_gradient_threshold
    if (
        configs["sparse_baseline"].training.densify_gradient_threshold
        != baseline_threshold
    ):
        raise ValueError("baseline runs must use the same gradient threshold")
    if (
        configs["sparse_aggressive"].training.densify_gradient_threshold
        != aggressive_threshold
    ):
        raise ValueError("aggressive runs must use the same gradient threshold")
    if aggressive_threshold >= baseline_threshold:
        raise ValueError("aggressive threshold must be lower than baseline")

    output_dirs = [config.output_dir for config in configs.values()]
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("each Phase 3 run requires a distinct output directory")


def _validate_registered_camera_partition(manifest: Phase3Manifest) -> None:
    config = manifest.configs["full_baseline"]
    scene = load_colmap_scene(
        config.data.colmap_dir,
        config.data.images_dir,
        config.data.downscale,
    )
    registered_ids = {
        camera.image_id for camera in scene.cameras if camera.image_id is not None
    }
    configured_ids = set(config.data.train_image_ids) | set(config.data.test_image_ids)
    if configured_ids != registered_ids:
        missing = sorted(registered_ids - configured_ids)
        extra = sorted(configured_ids - registered_ids)
        raise ValueError(
            "full Phase 3 train/test split must partition registered cameras; "
            f"missing={missing}, extra={extra}"
        )


def _validate_run_state(config: ExperimentConfig) -> bool:
    checkpoint_path = config.output_dir / "final.pt"
    resume_path = config.output_dir / "latest.pt"
    state_path = config.output_dir / "run_state.json"
    if checkpoint_path.is_file():
        metadata_path = config.output_dir / "run_metadata.json"
        if not metadata_path.is_file():
            if resume_path.is_file():
                return False
            raise ValueError(
                f"completed checkpoint is missing run metadata: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not configs_match(metadata.get("config"), config_to_dict(config)):
            raise ValueError(
                f"completed checkpoint config does not match: {config.output_dir}"
            )
        return True
    if resume_path.is_file():
        if not state_path.is_file():
            raise ValueError(f"resume checkpoint is missing run state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not configs_match(state.get("config"), config_to_dict(config)):
            raise ValueError(f"resume checkpoint config does not match: {config.output_dir}")
        return False
    if config.output_dir.is_dir() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"incomplete output directory must be moved or removed: {config.output_dir}"
        )
    return False


def _preflight_config(
    manifest: Phase3Manifest,
    name: str,
) -> ExperimentConfig:
    source = manifest.configs[name]
    return replace(
        source,
        data=replace(source.data, downscale=max(4, source.data.downscale)),
        training=replace(
            source.training,
            iterations=500,
            densify_from=500,
            densify_until=500,
            densify_every=500,
        ),
        output_dir=manifest.output_dir / "preflight" / name,
    )


def _last_densification_event(config: ExperimentConfig) -> dict[str, int]:
    event = None
    with (config.output_dir / "training.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line).get("densification", event)
    if event is None:
        raise RuntimeError(f"pre-flight did not perform density control: {config.output_dir}")
    return event


def run_phase3_preflight(manifest: Phase3Manifest) -> dict[str, Any]:
    """Verify that the two thresholds produce distinct growth before the pilot."""
    configs = {
        name: _preflight_config(manifest, name)
        for name in ("full_baseline", "full_aggressive")
    }
    completed = {
        name: _validate_run_state(config)
        for name, config in configs.items()
    }
    for name, config in configs.items():
        if not completed[name]:
            train(config)
    events = {
        name: _last_densification_event(config)
        for name, config in configs.items()
    }
    additions = {
        name: event["cloned"] + event["split_children"]
        for name, event in events.items()
    }
    if additions["full_baseline"] <= 0:
        raise RuntimeError("baseline pre-flight produced no clone or split children")
    if additions["full_aggressive"] <= additions["full_baseline"]:
        raise RuntimeError(
            "aggressive pre-flight did not produce more clone/split additions "
            "than baseline"
        )
    result = {
        "downscale": configs["full_baseline"].data.downscale,
        "iterations": configs["full_baseline"].training.iterations,
        "events": events,
    }
    manifest.output_dir.mkdir(parents=True, exist_ok=True)
    (manifest.output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_phase3(
    manifest: Phase3Manifest,
    run_name: str,
) -> list[Path]:
    """Train, evaluate, and plot one or all controlled Phase 3 conditions."""
    if run_name != "all" and run_name not in PHASE3_RUN_NAMES:
        raise ValueError(f"unknown Phase 3 run: {run_name}")
    _validate_registered_camera_partition(manifest)
    selected = PHASE3_RUN_NAMES if run_name == "all" else (run_name,)
    already_completed = {
        name: _validate_run_state(manifest.configs[name])
        for name in selected
    }
    run_phase3_preflight(manifest)
    output_dirs: list[Path] = []
    for name in selected:
        config = manifest.configs[name]
        checkpoint_path = config.output_dir / "final.pt"
        if not already_completed[name]:
            train(config)
        evaluate_checkpoint(config, checkpoint_path)
        plot_run(config.output_dir)
        output_dirs.append(config.output_dir)
    return output_dirs


def _read_run_metrics(name: str, config: ExperimentConfig) -> dict[str, Any]:
    path = config.output_dir / "evaluation" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 3 metrics do not exist for {name}: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not configs_match(metrics.get("config"), config_to_dict(config)):
        raise ValueError(f"stored evaluation config does not match {name}")
    expected_train_ids = list(config.data.train_image_ids)
    expected_test_ids = list(config.data.test_image_ids)
    if metrics.get("train_image_ids") != expected_train_ids:
        raise ValueError(f"stored training split does not match {name}")
    if metrics.get("test_image_ids") != expected_test_ids:
        raise ValueError(f"stored test split does not match {name}")

    train_metrics = metrics["splits"]["train"]
    test_metrics = metrics["splits"]["test"]
    train_psnr = train_metrics["mean_psnr"]
    test_psnr = test_metrics["mean_psnr"]
    gaussian_budget = config.training.densify_max_gaussians
    budget_reached = False
    if gaussian_budget is not None:
        with (config.output_dir / "training.jsonl").open(encoding="utf-8") as stream:
            budget_reached = any(
                json.loads(line)["gaussians"] >= gaussian_budget
                for line in stream
            )
    return {
        "run": name,
        "supervision": "full" if name.startswith("full_") else "sparse",
        "densification": "baseline" if name.endswith("_baseline") else "aggressive",
        "train_camera_count": len(expected_train_ids),
        "test_camera_count": len(expected_test_ids),
        "gradient_threshold": config.training.densify_gradient_threshold,
        "train_psnr": train_psnr,
        "test_psnr": test_psnr,
        "train_test_psnr_gap": (
            train_psnr - test_psnr
            if train_psnr is not None and test_psnr is not None
            else None
        ),
        "test_lpips": test_metrics["mean_lpips"],
        "test_depth_abs_rel": test_metrics["mean_depth_abs_rel"],
        "test_depth_coverage": test_metrics["mean_depth_coverage"],
        "final_gaussians": metrics["model"]["gaussians"],
        "gaussian_budget": gaussian_budget,
        "gaussian_budget_reached": budget_reached,
        "training_seconds": metrics["model"]["training_seconds"],
    }


def build_phase3_report(manifest: Phase3Manifest) -> dict[str, Any]:
    rows = [
        _read_run_metrics(name, manifest.configs[name])
        for name in PHASE3_RUN_NAMES
    ]
    rows_by_name = {row["run"]: row for row in rows}
    effect_metrics = (
        "test_psnr",
        "test_lpips",
        "train_test_psnr_gap",
        "test_depth_abs_rel",
        "test_depth_coverage",
        "final_gaussians",
    )
    effects: dict[str, dict[str, float | int | None]] = {}
    for supervision in ("full", "sparse"):
        baseline = rows_by_name[f"{supervision}_baseline"]
        aggressive = rows_by_name[f"{supervision}_aggressive"]
        effects[supervision] = {}
        for metric in effect_metrics:
            baseline_value = baseline[metric]
            aggressive_value = aggressive[metric]
            effects[supervision][metric] = (
                aggressive_value - baseline_value
                if baseline_value is not None and aggressive_value is not None
                else None
            )
    return {
        "runs": rows,
        "effects_aggressive_minus_baseline": effects,
        "notes": [
            "Depth AbsRel can cover different samples in each run; interpret its "
            "effect together with the depth-coverage effect."
        ],
    }


def _plot_report(report: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 plotting requires the evaluation extra: "
            "uv sync --extra evaluation"
        ) from error

    rows = report["runs"]
    labels = ["Full\nbase", "Full\naggr.", "Sparse\nbase", "Sparse\naggr."]
    panels = (
        ("test_psnr", "Held-out PSNR", "dB"),
        ("test_lpips", "Held-out LPIPS", "LPIPS"),
        ("test_depth_abs_rel", "Held-out depth AbsRel", "AbsRel"),
        ("final_gaussians", "Final Gaussian count", "Gaussians"),
    )
    colors = ["#4c78a8", "#f58518", "#4c78a8", "#f58518"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for index, row in enumerate(rows):
            value = row[metric]
            if value is None:
                axis.text(
                    index,
                    0.02,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    transform=axis.get_xaxis_transform(),
                )
            else:
                axis.bar(index, value, color=colors[index])
        axis.set_xticks(range(len(labels)), labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_phase3_report(manifest: Phase3Manifest) -> tuple[dict[str, Any], Path]:
    report = build_phase3_report(manifest)
    manifest.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest.output_dir / "phase3_metrics.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rows = report["runs"]
    csv_path = manifest.output_dir / "phase3_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = manifest.output_dir / "phase3_summary.png"
    _plot_report(report, plot_path)
    return report, plot_path
