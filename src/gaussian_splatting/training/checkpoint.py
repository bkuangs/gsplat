import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from gaussian_splatting.model import GaussianModel


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(
    path: Path,
    model: GaussianModel,
    optimizer: Optimizer,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "metadata": metadata or {},
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)
    _fsync_directory(path.parent)


def load_checkpoint(
    path: Path,
    optimizer_factory: Callable[[GaussianModel], Optimizer] | None = None,
    *,
    device: torch.device | str = "cpu",
) -> tuple[GaussianModel, Optimizer | None, int, dict[str, Any]]:
    """Reconstruct the saved model before creating and restoring its optimizer."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    model = GaussianModel.from_state_dict(payload["model"])
    optimizer = optimizer_factory(model) if optimizer_factory is not None else None
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, int(payload["step"]), dict(payload["metadata"])
