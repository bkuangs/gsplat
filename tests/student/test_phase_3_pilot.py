import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gaussian_splatting.config import (
    DataConfig,
    ExperimentConfig,
    TrainingConfig,
    config_to_dict,
)
from gaussian_splatting.phase3 import (
    PHASE3_RUN_NAMES,
    Phase3Manifest,
    _validate_run_state,
    load_phase3_manifest,
    validate_phase3_manifest,
    write_phase3_report,
)


def test_scan63_phase3_manifest_defines_controlled_four_run_matrix() -> None:
    manifest = load_phase3_manifest(Path("configs/phase3/dtu_scan63.yaml"))

    full_baseline = manifest.configs["full_baseline"]
    full_aggressive = manifest.configs["full_aggressive"]
    sparse_baseline = manifest.configs["sparse_baseline"]
    sparse_aggressive = manifest.configs["sparse_aggressive"]

    assert len(full_baseline.data.train_image_ids) == 39
    assert len(sparse_baseline.data.train_image_ids) == 4
    assert sparse_baseline.data.train_image_ids == (3, 14, 32, 43)
    assert set(sparse_baseline.data.train_image_ids).issubset(
        full_baseline.data.train_image_ids
    )
    assert all(
        config.data.test_image_ids == full_baseline.data.test_image_ids
        for config in manifest.configs.values()
    )
    assert full_baseline.training.densify_gradient_threshold == 0.001
    assert sparse_baseline.training.densify_gradient_threshold == 0.001
    assert full_aggressive.training.densify_gradient_threshold == 0.0005
    assert sparse_aggressive.training.densify_gradient_threshold == 0.0005
    assert all(
        config.training.densify_max_gaussians == 250_000
        for config in manifest.configs.values()
    )
    assert all(config.training.seed == 42 for config in manifest.configs.values())


def test_phase3_manifest_rejects_uncontrolled_optimizer_change() -> None:
    manifest = load_phase3_manifest(Path("configs/phase3/dtu_scan63.yaml"))
    configs = dict(manifest.configs)
    config = configs["sparse_aggressive"]
    configs["sparse_aggressive"] = replace(
        config,
        training=replace(config.training, learning_rate_position=0.001),
    )

    with pytest.raises(
        ValueError,
        match="changes settings outside supervision and gradient threshold",
    ):
        validate_phase3_manifest(replace(manifest, configs=configs))


def test_phase3_rejects_stale_completed_checkpoint(tmp_path: Path) -> None:
    manifest = _phase3_test_manifest(tmp_path)
    config = manifest.configs["full_baseline"]
    config.output_dir.mkdir()
    (config.output_dir / "final.pt").touch()
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps({"config": {"stale": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint config does not match"):
        _validate_run_state(config)


def _phase3_test_manifest(tmp_path: Path) -> Phase3Manifest:
    full_ids = tuple(range(1, 11))
    sparse_ids = (2,)
    test_ids = (11, 12)
    configs: dict[str, ExperimentConfig] = {}
    for name in PHASE3_RUN_NAMES:
        full = name.startswith("full_")
        aggressive = name.endswith("_aggressive")
        configs[name] = ExperimentConfig(
            data=DataConfig(
                colmap_dir=tmp_path,
                images_dir=tmp_path,
                train_image_ids=full_ids if full else sparse_ids,
                test_image_ids=test_ids,
            ),
            training=TrainingConfig(
                densify_gradient_threshold=0.0001 if aggressive else 0.0002,
            ),
            output_dir=tmp_path / name,
        )
    return Phase3Manifest(
        base_dir=tmp_path,
        output_dir=tmp_path / "summary",
        sparse_fraction=0.1,
        sparse_seed=42,
        configs=configs,
        config_paths={name: tmp_path / f"{name}.yaml" for name in PHASE3_RUN_NAMES},
    )


def test_phase3_report_writes_metrics_effects_and_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    manifest = _phase3_test_manifest(tmp_path)
    validate_phase3_manifest(manifest)
    values = {
        "full_baseline": (20.0, 18.0, 0.30, 0.10, 1000),
        "full_aggressive": (20.5, 18.5, 0.28, 0.09, 1200),
        "sparse_baseline": (24.0, 14.0, 0.45, 0.20, 900),
        "sparse_aggressive": (25.0, 12.0, 0.55, 0.30, 1800),
    }
    for name, config in manifest.configs.items():
        train_psnr, test_psnr, lpips, depth_abs_rel, gaussians = values[name]
        evaluation_dir = config.output_dir / "evaluation"
        evaluation_dir.mkdir(parents=True)
        metrics = {
            "config": config_to_dict(config),
            "train_image_ids": list(config.data.train_image_ids),
            "test_image_ids": list(config.data.test_image_ids),
            "model": {
                "gaussians": gaussians,
                "training_seconds": 60.0,
            },
            "splits": {
                "train": {"mean_psnr": train_psnr},
                "test": {
                    "mean_psnr": test_psnr,
                    "mean_lpips": lpips,
                    "mean_depth_abs_rel": depth_abs_rel,
                    "mean_depth_coverage": 1.0,
                },
            },
        }
        (evaluation_dir / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )

    report, plot_path = write_phase3_report(manifest)

    assert report["effects_aggressive_minus_baseline"]["full"]["test_psnr"] == 0.5
    assert report["effects_aggressive_minus_baseline"]["sparse"]["test_psnr"] == -2.0
    assert (
        report["effects_aggressive_minus_baseline"]["sparse"]["final_gaussians"]
        == 900
    )
    assert (manifest.output_dir / "phase3_metrics.json").is_file()
    with (manifest.output_dir / "phase3_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run"] for row in rows] == list(PHASE3_RUN_NAMES)
    assert plot_path == manifest.output_dir / "phase3_summary.png"
    assert plot_path.is_file()
