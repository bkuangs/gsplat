import json
from pathlib import Path
from typing import Any


def _load_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"training log does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def plot_run(run_dir: Path) -> Path:
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
    metrics_path = run_dir / "evaluation" / "metrics.json"
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

    metric_names = ["mean_lpips", "mean_depth_abs_rel", "mean_depth_coverage"]
    labels = ["LPIPS", "Depth AbsRel", "Depth coverage"]
    x_positions = list(range(len(metric_names)))
    width = 0.35
    for offset, split, label in (
        (-width / 2, "train", "Train"),
        (width / 2, "test", "Test"),
    ):
        values = [
            metrics["splits"][split][name]
            if metrics["splits"][split][name] is not None
            else 0.0
            for name in metric_names
        ]
        axes[1, 1].bar(
            [position + offset for position in x_positions],
            values,
            width,
            label=label,
        )
    axes[1, 1].set_xticks(x_positions, labels)
    axes[1, 1].set_title("Final perceptual and depth metrics")
    axes[1, 1].legend()

    figure.tight_layout()
    output_path = run_dir / "evaluation" / "summary.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
