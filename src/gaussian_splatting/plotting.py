import json
from pathlib import Path
from typing import Any


def _load_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"training log does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _plot_optional_metrics(axis: Any, metrics: dict[str, Any]) -> None:
    metric_names = ["mean_lpips", "mean_depth_abs_rel", "mean_depth_coverage"]
    labels = ["LPIPS", "Depth AbsRel", "Depth coverage"]
    x_positions = list(range(len(metric_names)))
    width = 0.35
    for offset, split, label in (
        (-width / 2, "train", "Train"),
        (width / 2, "test", "Test"),
    ):
        positions: list[float] = []
        values: list[float] = []
        for position, name in zip(x_positions, metric_names, strict=True):
            value = metrics["splits"][split][name]
            shifted_position = position + offset
            if value is None:
                axis.text(
                    shifted_position,
                    0.02,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    transform=axis.get_xaxis_transform(),
                )
            else:
                positions.append(shifted_position)
                values.append(value)
        if values:
            axis.bar(positions, values, width, label=label)
    axis.set_xticks(x_positions, labels)
    axis.set_title("Final perceptual and depth metrics")
    if axis.get_legend_handles_labels()[0]:
        axis.legend()


def plot_run(run_dir: Path, evaluation_dir: Path | None = None) -> Path:
    """Plot the minimal Phase 2 training and evaluation summary."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires the evaluation extra: uv sync --extra evaluation"
        ) from error

    records = _load_training_records(run_dir / "training.jsonl")
    evaluation_dir = evaluation_dir or run_dir / "evaluation"
    metrics_path = evaluation_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"evaluation metrics do not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    steps = [record["step"] for record in records]
    losses = [record["loss"] for record in records]
    gaussian_counts = [record["gaussians"] for record in records]
    split_names = ["train", "test"]
    split_labels = ["Train", "Test"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(steps, losses, linewidth=0.8)
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Photometric loss")

    axes[0, 1].plot(steps, gaussian_counts, linewidth=0.8)
    axes[0, 1].set_title("Gaussian count")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Gaussians")

    psnr_values = [
        metrics["splits"][split]["mean_psnr"] for split in split_names
    ]
    axes[1, 0].bar(split_labels, psnr_values)
    axes[1, 0].set_title("Final rendering quality")
    axes[1, 0].set_ylabel("PSNR (dB)")

    _plot_optional_metrics(axes[1, 1], metrics)

    figure.tight_layout()
    output_path = evaluation_dir / "summary.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
