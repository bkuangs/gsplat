import math
from collections.abc import Mapping

import torch
from torch import nn


class GaussianModel(nn.Module):
    """Trainable parameterization of anisotropic 3D Gaussians using WXYZ quaternions."""

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacity_logits: torch.Tensor,
        sh_coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        count = means.shape[0]
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "quaternions": (count, 4),
            "opacity_logits": (count, 1),
        }
        values = {
            "means": means,
            "log_scales": log_scales,
            "quaternions": quaternions,
            "opacity_logits": opacity_logits,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if sh_coefficients.ndim != 3 or sh_coefficients.shape[0] != count:
            raise ValueError("sh_coefficients must have shape (N, coefficients, channels)")
        if sh_coefficients.shape[-1] != 3:
            raise ValueError("spherical harmonic coefficients must have three color channels")

        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.sh_coefficients = nn.Parameter(sh_coefficients)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "GaussianModel":
        """Construct a model at the Gaussian count encoded in a saved state dictionary."""
        parameter_names = (
            "means",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "sh_coefficients",
        )
        try:
            parameters = {name: state_dict[name].detach().clone() for name in parameter_names}
        except KeyError as error:
            raise ValueError(f"model state is missing parameter {error.args[0]!r}") from error

        model = cls(**parameters)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_point_cloud(
        cls,
        points: torch.Tensor,
        colors: torch.Tensor,
        sh_degree: int,
        initial_opacity: float,
        initial_scale: float,
    ) -> "GaussianModel":
        """Initialize Gaussian parameters from a COLMAP sparse point cloud."""
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if colors.shape != points.shape:
            raise ValueError("colors must have shape (N, 3)")
        if not points.is_floating_point():
            raise ValueError("points must be floating point")
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if not 0.0 < initial_opacity < 1.0:
            raise ValueError("initial_opacity must be between 0 and 1")
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError("initial_scale must be finite and positive")

        count = points.shape[0]
        means = points.detach().clone()     # .detach() -> stop gradient computation
                                            # PyTorch treats the tensor as a fixed constant 
                                            # rather than a variable with computational history

        # Scales (size) s = (s_x, s_y​, s_z​)
        # .full_like(a, b) -> create a new Tensor of (shape, datatype)
        scales = torch.full_like(points, initial_scale)
        log_scales = scales.log()

        # Identity quaternions
        quaternions = points.new_zeros((count, 4))
        quaternions[:, 0] = 1.0

        # Opacity alpha
        # .new_full() -> create a new Tensor of (size, fill_value)
        opacities = points.new_full((count, 1), initial_opacity)
        opacity_logits = torch.logit(opacities)

        # SH coefficients
        coefficient_count = (sh_degree + 1) ** 2
        sh_coefficients = points.new_zeros((count, coefficient_count, 3))
        colors = colors.detach().to(device=points.device, dtype=points.dtype)
        sh_coefficients[:, 0, :] = (colors - 0.5) / 0.28209479177387814

        return cls(
            means=means,
            log_scales=log_scales,
            quaternions=quaternions,
            opacity_logits=opacity_logits,
            sh_coefficients=sh_coefficients,
        )

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()

    @property
    def normalized_quaternions(self) -> torch.Tensor:
        """Return unit quaternions in WXYZ component order."""
        return torch.nn.functional.normalize(self.quaternions, dim=-1)
