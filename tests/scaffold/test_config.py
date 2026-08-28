from pathlib import Path

import pytest

from gaussian_splatting.config import load_config


def test_loads_baseline_config() -> None:
    config = load_config(Path("configs/baseline.yaml"))
    assert config.model.sh_degree == 3
    assert config.render.backend == "torch"
    assert config.training.iterations == 30_000


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "data:\n  colmap_dir: sparse\n  images_dir: images\nsurprise: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surprise"):
        load_config(config_path)


def test_loads_fixed_holdout_image_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "holdout.yaml"
    config_path.write_text(
        "data:\n"
        "  colmap_dir: sparse\n"
        "  images_dir: images\n"
        "  holdout_image_ids: [7, 11]\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.holdout_image_ids == (7, 11)


def test_rejects_duplicate_holdout_image_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "holdout.yaml"
    config_path.write_text(
        "data:\n"
        "  colmap_dir: sparse\n"
        "  images_dir: images\n"
        "  holdout_image_ids: [7, 7]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicates"):
        load_config(config_path)
