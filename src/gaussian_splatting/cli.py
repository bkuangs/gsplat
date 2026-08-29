import argparse
from pathlib import Path

import torch

from gaussian_splatting import __version__
from gaussian_splatting.config import load_config
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.phase3 import (
    PHASE3_RUN_NAMES,
    load_phase3_manifest,
    run_phase3,
    run_phase3_preflight,
    write_phase3_report,
)
from gaussian_splatting.plotting import plot_run
from gaussian_splatting.training.evaluation import evaluate_checkpoint
from gaussian_splatting.training.trainer import train


def _status(_: argparse.Namespace) -> None:
    print(f"learn-3d-gaussian-splatting {__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Phase 2: fixed camera splits, checkpoint metrics, depth evaluation, and plots.")


def _inspect_colmap(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    scene = load_colmap_scene(
        config.data.colmap_dir,
        config.data.images_dir,
        config.data.downscale,
    )
    print(f"Registered cameras: {len(scene.cameras)}")
    print(f"Sparse points: {scene.points.shape[0]}")


def _train(args: argparse.Namespace) -> None:
    train(load_config(args.config))


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    checkpoint = args.checkpoint or config.output_dir / "final.pt"
    result = evaluate_checkpoint(config, checkpoint, args.output_dir)
    train_metrics = result["splits"]["train"]
    test_metrics = result["splits"]["test"]
    print(
        f"train psnr={_format_metric(train_metrics['mean_psnr'])} "
        f"lpips={_format_metric(train_metrics['mean_lpips'])}"
    )
    print(
        f"test psnr={_format_metric(test_metrics['mean_psnr'])} "
        f"lpips={_format_metric(test_metrics['mean_lpips'])} "
        f"depth_abs_rel={_format_metric(test_metrics['mean_depth_abs_rel'])} "
        f"depth_coverage={_format_metric(test_metrics['mean_depth_coverage'])}"
    )


def _plot(args: argparse.Namespace) -> None:
    print(plot_run(args.run_dir, args.evaluation_dir))


def _phase3_run(args: argparse.Namespace) -> None:
    manifest = load_phase3_manifest(args.manifest)
    for output_dir in run_phase3(manifest, args.run):
        print(output_dir)


def _phase3_preflight(args: argparse.Namespace) -> None:
    manifest = load_phase3_manifest(args.manifest)
    result = run_phase3_preflight(manifest)
    for name, event in result["events"].items():
        print(
            f"{name}: cloned={event['cloned']} "
            f"split_children={event['split_children']} "
            f"pruned={event['pruned']} "
            f"gaussians_after={event['gaussians_after']}"
        )


def _phase3_report(args: argparse.Namespace) -> None:
    manifest = load_phase3_manifest(args.manifest)
    report, plot_path = write_phase3_report(manifest)
    for row in report["runs"]:
        print(
            f"{row['run']}: train_psnr={_format_metric(row['train_psnr'])} "
            f"test_psnr={_format_metric(row['test_psnr'])} "
            f"test_lpips={_format_metric(row['test_lpips'])} "
            f"depth_abs_rel={_format_metric(row['test_depth_abs_rel'])} "
            f"gaussians={row['final_gaussians']}"
        )
    print(plot_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsplat-learn",
        description="Educational 3D Gaussian Splatting implementation",
    )
    subparsers = parser.add_subparsers(required=True)

    status_parser = subparsers.add_parser("status", help="show environment and next milestone")
    status_parser.set_defaults(handler=_status)

    inspect_parser = subparsers.add_parser(
        "inspect-colmap", help="validate and summarize a COLMAP reconstruction"
    )
    inspect_parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    inspect_parser.set_defaults(handler=_inspect_colmap)

    train_parser = subparsers.add_parser("train", help="train a Gaussian scene")
    train_parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    train_parser.set_defaults(handler=_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate a checkpoint on its fixed train/test camera split",
    )
    evaluate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        help="write evaluation artifacts directly to this directory",
    )
    evaluate_parser.set_defaults(handler=_evaluate)

    plot_parser = subparsers.add_parser(
        "plot",
        help="plot training and evaluation metrics for one run",
    )
    plot_parser.add_argument("run_dir", type=Path)
    plot_parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="read metrics from and write the summary to a custom evaluation directory",
    )
    plot_parser.set_defaults(handler=_plot)

    phase3_run_parser = subparsers.add_parser(
        "phase3-run",
        help="train, evaluate, and plot one or all Phase 3 pilot conditions",
    )
    phase3_run_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase3/dtu_scan63.yaml"),
    )
    phase3_run_parser.add_argument(
        "--run",
        choices=("all", *PHASE3_RUN_NAMES),
        default="all",
    )
    phase3_run_parser.set_defaults(handler=_phase3_run)

    phase3_preflight_parser = subparsers.add_parser(
        "phase3-preflight",
        help="verify that Phase 3 density thresholds produce distinct growth",
    )
    phase3_preflight_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase3/dtu_scan63.yaml"),
    )
    phase3_preflight_parser.set_defaults(handler=_phase3_preflight)

    phase3_report_parser = subparsers.add_parser(
        "phase3-report",
        help="aggregate the completed Phase 3 pilot",
    )
    phase3_report_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase3/dtu_scan63.yaml"),
    )
    phase3_report_parser.set_defaults(handler=_phase3_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
