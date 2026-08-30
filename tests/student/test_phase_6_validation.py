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
from gaussian_splatting.phase6 import (
    Phase6Manifest,
    Phase6Scene,
    _config_for_seed,
    _configs_match_phase6,
    _stratified_test_ids,
    _validate_evaluation_state,
    _validate_phase6_run_state,
    load_phase6_manifest,
    phase6_run_targets,
    validate_phase6_manifest,
    write_phase6_report,
)


def test_phase6_manifest_defines_three_scene_paired_validation() -> None:
    manifest = load_phase6_manifest(Path("configs/phase6/dtu_validation.yaml"))

    assert tuple(manifest.scenes) == ("scan63", "scan24", "scan110")
    assert manifest.training_seeds == (42, 7, 123)
    assert manifest.scenes["scan24"].configs["rgb"].data.train_image_ids == (
        3,
        14,
        32,
        43,
    )
    assert manifest.scenes["scan110"].configs["rgb"].data.train_image_ids == (
        13,
        16,
        27,
        45,
        57,
    )
    assert all(
        scene.configs["rgb"].training.depth_loss_weight == 0.0
        and scene.configs["depth"].training.depth_loss_weight == 0.1
        for scene in manifest.scenes.values()
    )
    assert manifest.scenes["scan63"].configs["rgb"].output_dir.name == (
        "sparse_aggressive"
    )
    assert manifest.scenes["scan63"].configs["depth"].output_dir.name == (
        "sparse_aggressive_depth"
    )


def test_phase6_stratified_test_split_matches_finalized_camera_ids() -> None:
    assert _stratified_test_ids(tuple(range(1, 50)), 10) == (
        1,
        6,
        11,
        16,
        21,
        26,
        31,
        36,
        41,
        46,
    )
    assert _stratified_test_ids(tuple(range(1, 65)), 10) == (
        1,
        8,
        15,
        22,
        29,
        35,
        41,
        47,
        53,
        59,
    )


def test_phase6_manifest_rejects_uncontrolled_pair_change() -> None:
    manifest = load_phase6_manifest(Path("configs/phase6/dtu_validation.yaml"))
    scenes = dict(manifest.scenes)
    scan24 = scenes["scan24"]
    configs = dict(scan24.configs)
    depth = configs["depth"]
    configs["depth"] = replace(
        depth,
        training=replace(depth.training, learning_rate_position=0.001),
    )
    scenes["scan24"] = replace(scan24, configs=configs)

    with pytest.raises(
        ValueError,
        match="differ outside the depth intervention",
    ):
        validate_phase6_manifest(replace(manifest, scenes=scenes))


def test_phase6_additional_seed_changes_only_seed_and_output() -> None:
    manifest = load_phase6_manifest(Path("configs/phase6/dtu_validation.yaml"))
    base = manifest.scenes["scan24"].configs["depth"]
    seeded = _config_for_seed(manifest, "scan24", "depth", 7)

    assert seeded.training == replace(base.training, seed=7)
    assert seeded.data == base.data
    assert seeded.model == base.model
    assert seeded.render == base.render
    assert seeded.output_dir == (
        manifest.output_dir / "seeds/seed_7/scan24/rgb_depth"
    )


def test_phase6_batch_expands_to_isolated_single_run_targets() -> None:
    manifest = load_phase6_manifest(Path("configs/phase6/dtu_validation.yaml"))

    targets = phase6_run_targets(manifest, "all", "both", "all")

    assert len(targets) == 18
    assert targets[0] == ("scan63", "rgb", 42)
    assert targets[-1] == ("scan110", "depth", 123)
    assert len(set(targets)) == len(targets)


def test_phase6_config_matching_reuses_relative_scan63_artifacts(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path / "data/scene/sparse/0",
            images_dir=tmp_path / "data/scene/images",
            depths_dir=tmp_path / "data/scene/depths",
            depth_priors_dir=tmp_path / "outputs/priors",
            train_image_ids=(2,),
            test_image_ids=(1,),
        ),
        training=TrainingConfig(depth_loss_weight=0.1),
        output_dir=tmp_path / "outputs/run",
    )
    stored = config_to_dict(config)
    stored["data"]["colmap_dir"] = "data/scene/sparse/0"
    stored["data"]["images_dir"] = "data/scene/images"
    stored["data"]["depths_dir"] = "data/scene/depths"
    stored["data"]["depth_priors_dir"] = "outputs/priors"
    stored["output_dir"] = "outputs/run"

    assert _configs_match_phase6(stored, config, tmp_path)


def test_phase6_rejects_completed_run_without_metadata(tmp_path: Path) -> None:
    config = ExperimentConfig(
        data=DataConfig(colmap_dir=tmp_path, images_dir=tmp_path),
        output_dir=tmp_path / "run",
    )
    config.output_dir.mkdir()
    (config.output_dir / "final.pt").touch()

    with pytest.raises(ValueError, match="ambiguous state"):
        _validate_phase6_run_state(config, tmp_path)


def test_phase6_reuses_matching_evaluation_and_rejects_stale_split(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        data=DataConfig(
            colmap_dir=tmp_path,
            images_dir=tmp_path,
            train_image_ids=(2,),
            test_image_ids=(1,),
        ),
        output_dir=tmp_path / "run",
    )
    evaluation_dir = config.output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    metrics = {
        "config": config_to_dict(config),
        "train_image_ids": [2],
        "test_image_ids": [1],
    }
    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    assert _validate_evaluation_state(config, tmp_path)

    del metrics["train_image_ids"]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="missing split metadata"):
        _validate_evaluation_state(config, tmp_path)

    metrics["train_image_ids"] = [2]
    metrics["test_image_ids"] = [3]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="test split does not match"):
        _validate_evaluation_state(config, tmp_path)


def _test_manifest(tmp_path: Path) -> Phase6Manifest:
    scenes: dict[str, Phase6Scene] = {}
    for name in ("scan63", "scan24", "scan110"):
        data = DataConfig(
            colmap_dir=tmp_path / name / "sparse/0",
            images_dir=tmp_path / name / "images",
            depths_dir=tmp_path / name / "depths",
            train_image_ids=(2,),
            test_image_ids=(1,),
        )
        rgb = ExperimentConfig(
            data=data,
            output_dir=tmp_path / name / "rgb_only",
        )
        depth = replace(
            rgb,
            data=replace(
                data,
                depth_priors_dir=tmp_path / name / "depth_priors",
            ),
            training=replace(rgb.training, depth_loss_weight=0.1),
            output_dir=tmp_path / name / "rgb_depth",
        )
        scenes[name] = Phase6Scene(
            name=name,
            configs={"rgb": rgb, "depth": depth},
            config_paths={
                "rgb": tmp_path / f"{name}_rgb.yaml",
                "depth": tmp_path / f"{name}_depth.yaml",
            },
        )
    return Phase6Manifest(
        base_dir=tmp_path,
        output_dir=tmp_path / "report",
        depth_model="fake-depth-model",
        test_count=1,
        sparse_fraction=0.1,
        sparse_seed=42,
        training_seeds=(42,),
        scenes=scenes,
    )


def test_phase6_report_writes_scene_effects_and_consistency_gate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    manifest = _test_manifest(tmp_path)
    validate_phase6_manifest(manifest)
    for scene in manifest.scenes.values():
        for condition, config in scene.configs.items():
            is_depth = condition == "depth"
            evaluation_dir = config.output_dir / "evaluation"
            evaluation_dir.mkdir(parents=True)
            metrics = {
                "config": config_to_dict(config),
                "train_image_ids": list(config.data.train_image_ids),
                "test_image_ids": list(config.data.test_image_ids),
                "model": {
                    "gaussians": 90 if is_depth else 100,
                    "training_seconds": 12.0 if is_depth else 10.0,
                },
                "splits": {
                    "train": {
                        "camera_count": 1,
                        "mean_psnr": 25.0 if is_depth else 30.0,
                    },
                    "test": {
                        "camera_count": 1,
                        "mean_psnr": 12.0 if is_depth else 10.0,
                        "mean_lpips": 0.4 if is_depth else 0.5,
                        "mean_depth_abs_rel": 0.08 if is_depth else 0.1,
                        "mean_depth_coverage": 0.85 if is_depth else 0.8,
                    },
                },
            }
            (evaluation_dir / "metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )

    report, plot_path = write_phase6_report(manifest)

    assert len(report["runs"]) == 6
    assert report["all_primary_metrics_favorable_on_every_scene"] is True
    assert all(
        effect["test_psnr"] == pytest.approx(2.0)
        and effect["test_lpips"] == pytest.approx(-0.1)
        and effect["test_depth_abs_rel"] == pytest.approx(-0.02)
        and effect["train_test_psnr_gap"] == pytest.approx(-7.0)
        for effect in report["effects_rgb_depth_minus_rgb_only"]
    )
    with (manifest.output_dir / "phase6_metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with (manifest.output_dir / "phase6_effects.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert plot_path.is_file()


def test_phase6_report_aggregates_three_seed_effect_statistics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    manifest = replace(_test_manifest(tmp_path), training_seeds=(42, 7, 123))
    effects_by_seed = {42: 1.0, 7: 2.0, 123: 3.0}
    for seed, test_psnr_effect in effects_by_seed.items():
        for scene_name in manifest.scenes:
            for condition in ("rgb", "depth"):
                config = _config_for_seed(
                    manifest,
                    scene_name,
                    condition,
                    seed,
                )
                is_depth = condition == "depth"
                evaluation_dir = config.output_dir / "evaluation"
                evaluation_dir.mkdir(parents=True)
                metrics = {
                    "config": config_to_dict(config),
                    "train_image_ids": list(config.data.train_image_ids),
                    "test_image_ids": list(config.data.test_image_ids),
                    "model": {
                        "gaussians": 100,
                        "training_seconds": 10.0,
                    },
                    "splits": {
                        "train": {
                            "camera_count": 1,
                            "mean_psnr": 29.0 if is_depth else 30.0,
                        },
                        "test": {
                            "camera_count": 1,
                            "mean_psnr": (
                                10.0 + test_psnr_effect
                                if is_depth
                                else 10.0
                            ),
                            "mean_lpips": (
                                0.5 - 0.01 * test_psnr_effect
                                if is_depth
                                else 0.5
                            ),
                            "mean_depth_abs_rel": (
                                0.1 - 0.001 * test_psnr_effect
                                if is_depth
                                else 0.1
                            ),
                            "mean_depth_coverage": 1.0,
                        },
                    },
                }
                (evaluation_dir / "metrics.json").write_text(
                    json.dumps(metrics),
                    encoding="utf-8",
                )

    report, _ = write_phase6_report(manifest)

    assert len(report["runs"]) == 18
    assert len(report["effects_rgb_depth_minus_rgb_only"]) == 9
    assert report["all_primary_metrics_favorable_on_every_scene"] is True
    assert report["all_primary_metrics_favorable_on_every_run"] is True
    for row in report["scene_effect_statistics"]:
        assert row["seed_count"] == 3
        assert row["test_psnr_mean"] == pytest.approx(2.0)
        assert row["test_psnr_sample_std"] == pytest.approx(1.0)
        assert row["test_psnr_favorable_count"] == 3
    assert (
        manifest.output_dir / "phase6_scene_statistics.csv"
    ).is_file()
