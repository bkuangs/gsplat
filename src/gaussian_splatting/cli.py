import argparse
import subprocess
import sys
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
from gaussian_splatting.phase6 import (
    PHASE6_CONDITIONS,
    generate_phase6_priors,
    load_phase6_manifest,
    phase6_run_targets,
    run_phase6,
    write_phase6_report,
)
from gaussian_splatting.plotting import plot_run
from gaussian_splatting.training.depth_prior import (
    DEFAULT_DEPTH_MODEL,
    DepthAnythingPredictor,
    generate_depth_priors,
)
from gaussian_splatting.training.evaluation import evaluate_checkpoint
from gaussian_splatting.training.trainer import train


def _status(_: argparse.Namespace) -> None:
    print(f"learn-3d-gaussian-splatting {__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        "Phase 6: three-scene sparse-view validation of COLMAP-aligned "
        "monocular depth regularization."
    )


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


def _generate_depth_priors(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = args.output_dir or config.data.depth_priors_dir
    if output_dir is None:
        raise ValueError(
            "set data.depth_priors_dir or pass --output-dir for depth priors"
        )
    predictor = DepthAnythingPredictor(args.model)
    metadata = generate_depth_priors(
        config,
        output_dir,
        predictor,
        model_id=args.model,
        min_anchors=args.min_anchors,
        max_median_abs_rel=args.max_median_abs_rel,
    )
    print(
        f"generated {len(metadata['cameras'])} aligned priors in {output_dir}"
    )


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


def _phase6_priors(args: argparse.Namespace) -> None:
    manifest = load_phase6_manifest(args.manifest)
    for output_dir in generate_phase6_priors(manifest, args.scene):
        print(output_dir)


def _phase6_run(args: argparse.Namespace) -> None:
    manifest = load_phase6_manifest(args.manifest)
    targets = phase6_run_targets(
        manifest,
        args.scene,
        args.condition,
        args.seed,
    )
    if len(targets) > 1:
        manifest_path = args.manifest.resolve()
        for scene, condition, seed in targets:
            print(
                f"starting scene={scene} condition={condition} seed={seed}",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gaussian_splatting.cli",
                    "phase6-run",
                    "--manifest",
                    str(manifest_path),
                    "--scene",
                    scene,
                    "--condition",
                    condition,
                    "--seed",
                    str(seed),
                ],
                check=True,
            )
        return
    for output_dir in run_phase6(
        manifest,
        args.scene,
        args.condition,
        args.seed,
    ):
        print(output_dir)


def _phase6_report(args: argparse.Namespace) -> None:
    manifest = load_phase6_manifest(args.manifest)
    report, plot_path = write_phase6_report(manifest)
    for effect in report["effects_rgb_depth_minus_rgb_only"]:
        print(
            f"{effect['scene']} seed={effect['seed']}: "
            f"test_psnr={effect['test_psnr']:+.4f} "
            f"test_lpips={effect['test_lpips']:+.4f} "
            f"depth_abs_rel={effect['test_depth_abs_rel']:+.4f} "
            f"psnr_gap={effect['train_test_psnr_gap']:+.4f}"
        )
    for row in report["scene_effect_statistics"]:
        print(
            f"{row['scene']} mean over {row['seed_count']} seeds: "
            f"test_psnr={row['test_psnr_mean']:+.4f}"
            f"±{row['test_psnr_sample_std']:.4f} "
            f"test_lpips={row['test_lpips_mean']:+.4f}"
            f"±{row['test_lpips_sample_std']:.4f} "
            f"depth_abs_rel={row['test_depth_abs_rel_mean']:+.4f}"
            f"±{row['test_depth_abs_rel_sample_std']:.4f}"
        )
    consistent = report["all_primary_metrics_favorable_on_every_scene"]
    print(f"all primary mean effects favorable on every scene: {consistent}")
    every_run = report["all_primary_metrics_favorable_on_every_run"]
    print(f"all primary metrics favorable on every individual run: {every_run}")
    print(plot_path)


def _phase6_seed(value: str) -> int | str:
    if value == "all":
        return value
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be an integer or 'all'"
        ) from error


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

    depth_priors_parser = subparsers.add_parser(
        "generate-depth-priors",
        help="generate COLMAP-aligned monocular depth priors for training cameras",
    )
    depth_priors_parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    depth_priors_parser.add_argument("--output-dir", type=Path)
    depth_priors_parser.add_argument("--model", default=DEFAULT_DEPTH_MODEL)
    depth_priors_parser.add_argument("--min-anchors", type=int, default=20)
    depth_priors_parser.add_argument(
        "--max-median-abs-rel",
        type=float,
        default=0.25,
    )
    depth_priors_parser.set_defaults(handler=_generate_depth_priors)

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

    phase6_priors_parser = subparsers.add_parser(
        "phase6-priors",
        help="generate or validate Phase 6 depth priors",
    )
    phase6_priors_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase6/dtu_validation.yaml"),
    )
    phase6_priors_parser.add_argument(
        "--scene",
        default="all",
        help="manifest scene name or 'all'",
    )
    phase6_priors_parser.set_defaults(handler=_phase6_priors)

    phase6_run_parser = subparsers.add_parser(
        "phase6-run",
        help="train, evaluate, and plot Phase 6 validation conditions",
    )
    phase6_run_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase6/dtu_validation.yaml"),
    )
    phase6_run_parser.add_argument(
        "--scene",
        required=True,
        help="manifest scene name or 'all'",
    )
    phase6_run_parser.add_argument(
        "--condition",
        choices=("both", *PHASE6_CONDITIONS),
        default="both",
    )
    phase6_run_parser.add_argument(
        "--seed",
        type=_phase6_seed,
        help="configured training seed, or 'all'; defaults to the base seed",
    )
    phase6_run_parser.set_defaults(handler=_phase6_run)

    phase6_report_parser = subparsers.add_parser(
        "phase6-report",
        help="aggregate the three-scene Phase 6 comparison",
    )
    phase6_report_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase6/dtu_validation.yaml"),
    )
    phase6_report_parser.set_defaults(handler=_phase6_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
