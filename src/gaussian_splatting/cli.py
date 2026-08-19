import argparse
from pathlib import Path

import torch

from gaussian_splatting import __version__
from gaussian_splatting.config import load_config
from gaussian_splatting.data.colmap import load_colmap_scene
from gaussian_splatting.training.trainer import train


def _status(_: argparse.Namespace) -> None:
    print(f"learn-3d-gaussian-splatting {__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Next milestone: implement projection and compositing, then unskip milestone 1 tests.")


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

