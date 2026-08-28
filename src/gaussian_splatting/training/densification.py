from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from gaussian_splatting.math.projection import quaternion_to_rotation
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import RenderOutput


@dataclass(frozen=True)
class DensificationStats:
    position_gradient_accumulator: torch.Tensor
    observation_count: torch.Tensor
    max_screen_radius: torch.Tensor


@dataclass(frozen=True)
class TopologyUpdate:
    gaussians_before: int
    cloned: int
    split_parents: int
    split_children: int
    pruned: int
    gaussians_after: int


def create_densification_stats(model: GaussianModel) -> DensificationStats:
    count = model.means.shape[0]
    return DensificationStats(
        position_gradient_accumulator=model.means.new_zeros(count),
        observation_count=model.means.new_zeros(count),
        max_screen_radius=model.means.new_zeros(count),
    )


def accumulate_densification_stats(
    stats: DensificationStats,
    output: RenderOutput,
) -> None:
    """Accumulate visible screen-space gradient magnitude and maximum radius."""
    if output.means_2d is None or output.radii is None:
        raise ValueError("renderer output must include means_2d and radii")
    if output.means_2d.grad is None:
        raise RuntimeError("screen-space gradients must be retained before accumulation")
    count = stats.observation_count.shape[0]
    if output.means_2d.shape != (count, 2) or output.radii.shape != (count,):
        raise ValueError("renderer statistics do not match the current Gaussian count")

    visibility = (
        output.visibility
        if output.visibility is not None
        else output.radii > 0
    )
    gradients = torch.linalg.vector_norm(output.means_2d.grad.detach(), dim=-1)
    with torch.no_grad():
        stats.position_gradient_accumulator[visibility] += gradients[visibility]
        stats.observation_count[visibility] += 1
        stats.max_screen_radius.copy_(
            torch.maximum(stats.max_screen_radius, output.radii.detach())
        )


_PARAMETER_NAMES = (
    "means",
    "log_scales",
    "quaternions",
    "opacity_logits",
    "sh_coefficients",
)


def _replace_parameter(
    model: GaussianModel,
    optimizer: Optimizer,
    name: str,
    values: torch.Tensor,
    kept_indices: torch.Tensor,
) -> None:
    old_parameter = getattr(model, name)
    new_parameter = nn.Parameter(values.detach(), requires_grad=old_parameter.requires_grad)

    matching_groups = [
        group
        for group in optimizer.param_groups
        if any(parameter is old_parameter for parameter in group["params"])
    ]
    if len(matching_groups) != 1:
        raise ValueError(f"optimizer must contain model parameter {name!r} exactly once")
    group = matching_groups[0]
    group["params"] = [
        new_parameter if parameter is old_parameter else parameter
        for parameter in group["params"]
    ]

    old_state = optimizer.state.pop(old_parameter, None)
    if old_state is not None:
        new_state: dict[object, object] = {}
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor) and value.shape == old_parameter.shape:
                remapped = torch.zeros_like(new_parameter)
                remapped[: kept_indices.numel()] = value[kept_indices]
                new_state[key] = remapped
            else:
                new_state[key] = value
        optimizer.state[new_parameter] = new_state
    setattr(model, name, new_parameter)


@torch.no_grad()
def update_gaussian_topology(
    model: GaussianModel,
    optimizer: Optimizer,
    stats: DensificationStats,
    gradient_threshold: float,
    opacity_threshold: float,
    scene_extent: float,
    *,
    scale_threshold: float = 0.01,
    max_screen_radius: float = 100.0,
    split_count: int = 2,
) -> TopologyUpdate:
    """Mutate Gaussian topology and return counts for each density-control action."""
    count = model.means.shape[0]
    expected_shape = (count,)
    for name, value in (
        ("position_gradient_accumulator", stats.position_gradient_accumulator),
        ("observation_count", stats.observation_count),
        ("max_screen_radius", stats.max_screen_radius),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if gradient_threshold < 0 or opacity_threshold < 0:
        raise ValueError("densification thresholds must be non-negative")
    if scene_extent <= 0 or scale_threshold <= 0 or max_screen_radius <= 0:
        raise ValueError("scene and size thresholds must be positive")
    if split_count < 2:
        raise ValueError("split_count must be at least 2")

    average_gradient = torch.where(
        stats.observation_count > 0,
        stats.position_gradient_accumulator
        / stats.observation_count.clamp_min(1),
        torch.zeros_like(stats.position_gradient_accumulator),
    )
    prune = (model.opacities[:, 0] < opacity_threshold) | (
        stats.max_screen_radius > max_screen_radius
    )
    high_gradient = (
        (stats.observation_count > 0)
        & (average_gradient >= gradient_threshold)
        & ~prune
    )
    small = model.scales.max(dim=-1).values <= scale_threshold * scene_extent
    clone = high_gradient & small
    split = high_gradient & ~small

    kept_indices = (~prune & ~split).nonzero(as_tuple=False).squeeze(-1)
    clone_indices = clone.nonzero(as_tuple=False).squeeze(-1)
    split_indices = split.nonzero(as_tuple=False).squeeze(-1)
    if (
        kept_indices.numel() == count
        and clone_indices.numel() == 0
        and split_indices.numel() == 0
    ):
        return TopologyUpdate(
            gaussians_before=count,
            cloned=0,
            split_parents=0,
            split_children=0,
            pruned=0,
            gaussians_after=count,
        )

    split_sources = split_indices.repeat_interleave(split_count)
    split_means = model.means.new_empty((0, 3))
    if split_indices.numel() > 0:
        parent_scales = model.scales[split_indices]
        samples = torch.randn(
            split_indices.numel(),
            split_count,
            3,
            device=model.means.device,
            dtype=model.means.dtype,
        )
        local_offsets = samples * parent_scales[:, None, :]
        rotations = quaternion_to_rotation(model.quaternions[split_indices])
        world_offsets = torch.einsum("nij,nkj->nki", rotations, local_offsets)
        split_means = (
            model.means[split_indices, None, :] + world_offsets
        ).reshape(-1, 3)

    values: dict[str, torch.Tensor] = {}
    for name in _PARAMETER_NAMES:
        parameter = getattr(model, name)
        pieces = [parameter[kept_indices], parameter[clone_indices]]
        if name == "means":
            pieces.append(split_means)
        elif name == "log_scales":
            child_scales = (
                model.scales[split_sources] / (0.8 * split_count)
            ).clamp_min(torch.finfo(parameter.dtype).tiny)
            pieces.append(child_scales.log())
        else:
            pieces.append(parameter[split_sources])
        values[name] = torch.cat(pieces, dim=0)

    for name in _PARAMETER_NAMES:
        _replace_parameter(model, optimizer, name, values[name], kept_indices)
    return TopologyUpdate(
        gaussians_before=count,
        cloned=clone_indices.numel(),
        split_parents=split_indices.numel(),
        split_children=split_sources.numel(),
        pruned=prune.count_nonzero().item(),
        gaussians_after=model.means.shape[0],
    )
