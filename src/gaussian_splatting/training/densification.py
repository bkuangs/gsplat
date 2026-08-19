from dataclasses import dataclass

import torch

from gaussian_splatting.model import GaussianModel


@dataclass(frozen=True)
class DensificationStats:
    position_gradient_accumulator: torch.Tensor
    observation_count: torch.Tensor
    max_screen_radius: torch.Tensor


def update_gaussian_topology(
    model: GaussianModel,
    stats: DensificationStats,
    gradient_threshold: float,
    opacity_threshold: float,
    scene_extent: float,
) -> GaussianModel:
    """Clone, split, and prune Gaussians based on optimization statistics."""
    # TODO(student): implement adaptive density control and preserve optimizer state.
    raise NotImplementedError("Milestone 4: implement densification and pruning")

